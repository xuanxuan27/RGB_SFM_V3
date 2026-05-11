#!/usr/bin/env python3
"""
依 plots/vlm_analysis/stage{S}_block{B}/repr_pos{P}_part{K}.png 批次送進 VLM，
將每張 cluster 代表圖的敘述另存為 JSONL。

支援模型：
  - Qwen2-VL 系列  （預設：Qwen/Qwen2-VL-7B-Instruct）
  - InternVL2 系列  （例如：OpenGVLab/InternVL2-8B）

用法（專案根目錄）:
  python vit_analysis_vlm/run_caption_analysis.py
  python vit_analysis_vlm/run_caption_analysis.py --model OpenGVLab/InternVL2-8B
  python vit_analysis_vlm/run_caption_analysis.py --plots-root /path/to/plots/vlm_analysis --limit 3

若尚未載過模型，首次會從 Hugging Face 下載。
"""
from __future__ import annotations

import argparse
import json
import re
import types
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Iterable

# 專案根（vit_analysis_vlm/ 的上一層）
_ROOT = Path(__file__).resolve().parent.parent

import torch

DIR_PATTERN = re.compile(r"^stage(\d+)_block(\d+)$")
FILE_PATTERN = re.compile(r"^repr_pos(\d+)_part(\d+)\.png$", re.IGNORECASE)
INFER_STAGE_FILE_PATTERN = re.compile(r"^img(\d+)_s(\d+)_b(\d+)\.png$", re.IGNORECASE)
INFER_REPR_FILE_PATTERN = re.compile(
    r"^img(\d+)_s(\d+)_b(\d+)_repr_pos(\d+)_cluster(\d+)\.png$",
    re.IGNORECASE,
)


# ---------------------------------------------------------------------------
# 資料結構
# ---------------------------------------------------------------------------

@dataclass
class PlotRecord:
    image_path: str
    stage: int
    block: int
    pos: int | None
    part: int | None
    img: int | None
    cluster: int | None
    rel_key: str


# ---------------------------------------------------------------------------
# 圖片掃描
# ---------------------------------------------------------------------------

def discover_repr_images(plots_root: Path) -> list[PlotRecord]:
    """掃描 stage*_block* 子資料夾內符合命名的 png。"""
    plots_root = plots_root.resolve()
    if not plots_root.is_dir():
        raise FileNotFoundError(f"找不到圖片根目錄: {plots_root}")

    records: list[PlotRecord] = []
    for sub in sorted(plots_root.iterdir()):
        if not sub.is_dir():
            continue
        dm = DIR_PATTERN.match(sub.name)
        if not dm:
            continue
        stage, block = int(dm.group(1)), int(dm.group(2))
        for png in sorted(sub.glob("repr_pos*_part*.png")):
            fm = FILE_PATTERN.match(png.name)
            if not fm:
                continue
            pos, part = int(fm.group(1)), int(fm.group(2))
            rel = f"stage{stage}_block{block}/repr_pos{pos}_part{part}"
            records.append(
                PlotRecord(
                    image_path=str(png.resolve()),
                    stage=stage,
                    block=block,
                    pos=pos,
                    part=part,
                    img=None,
                    cluster=None,
                    rel_key=rel,
                )
            )

    records.sort(key=lambda r: (r.stage, r.block, r.pos, r.part, r.image_path))
    return records


def discover_inference_stage_images(plots_root: Path) -> list[PlotRecord]:
    """掃描 inference 目錄內的 img{I}_s{S}_b{B}.png。"""
    plots_root = plots_root.resolve()
    if not plots_root.is_dir():
        raise FileNotFoundError(f"找不到圖片根目錄: {plots_root}")

    records: list[PlotRecord] = []
    for png in sorted(plots_root.glob("img*_s*_b*.png")):
        fm = INFER_STAGE_FILE_PATTERN.match(png.name)
        if not fm:
            continue
        img_idx, stage, block = int(fm.group(1)), int(fm.group(2)), int(fm.group(3))
        rel = f"img{img_idx}_s{stage}_b{block}"
        records.append(
            PlotRecord(
                image_path=str(png.resolve()),
                stage=stage,
                block=block,
                pos=None,
                part=None,
                img=img_idx,
                cluster=None,
                rel_key=rel,
            )
        )

    records.sort(key=lambda r: (r.img if r.img is not None else -1, r.stage, r.block, r.image_path))
    return records


def discover_inference_repr_images(plots_root: Path) -> list[PlotRecord]:
    """掃描 inference 目錄內的 img{I}_s{S}_b{B}_repr_pos{P}_cluster{C}.png。"""
    plots_root = plots_root.resolve()
    if not plots_root.is_dir():
        raise FileNotFoundError(f"找不到圖片根目錄: {plots_root}")

    records: list[PlotRecord] = []
    for png in sorted(plots_root.glob("img*_s*_b*_repr_pos*_cluster*.png")):
        fm = INFER_REPR_FILE_PATTERN.match(png.name)
        if not fm:
            continue
        img_idx = int(fm.group(1))
        stage = int(fm.group(2))
        block = int(fm.group(3))
        pos = int(fm.group(4))
        cluster = int(fm.group(5))
        rel = f"img{img_idx}_s{stage}_b{block}_repr_pos{pos}_cluster{cluster}"
        records.append(
            PlotRecord(
                image_path=str(png.resolve()),
                stage=stage,
                block=block,
                pos=pos,
                part=None,
                img=img_idx,
                cluster=cluster,
                rel_key=rel,
            )
        )

    records.sort(
        key=lambda r: (
            r.img if r.img is not None else -1,
            r.stage,
            r.block,
            r.pos if r.pos is not None else -1,
            r.cluster if r.cluster is not None else -1,
            r.image_path,
        )
    )
    return records


# ---------------------------------------------------------------------------
# Prompt
# ---------------------------------------------------------------------------

def build_default_prompt() -> str:
    return (
        "這張圖包含多張來自同一個 K-means cluster 的 image patch，"
        "這些 patch 是從 ViT attention 特徵空間中聚類後，最接近群心的代表樣本。\n\n"
        "請依以下幾個面向分析這些 patch 的共同視覺特徵：\n"
        "1. 顏色與對比：主要色調、前景背景關係\n"
        "2. 形狀與結構：幾何特徵、筆畫方向、邊緣特性\n"
        "3. 位置線索：特徵集中在 patch 的哪個區域（上/下/左/右/中）\n"
        "4. 跨樣本一致性：這幾張 patch 之間相似在哪、差異在哪\n\n"
        "最後用一句話總結：這個 cluster 最可能在捕捉圖像中的什麼局部視覺結構。\n"
        "請用繁體中文回答。"
    )


# ---------------------------------------------------------------------------
# 模型偵測
# ---------------------------------------------------------------------------

def _model_primary_device(model: torch.nn.Module) -> torch.device:
    d = getattr(model, "device", None)
    if d is not None:
        return d
    return next(model.parameters()).device


def detect_model_family(model_id: str) -> str:
    """根據 model id 判斷模型系列。回傳 'qwen2vl' 或 'internvl'。"""
    lower = model_id.lower()
    if "internvl" in lower:
        return "internvl"
    if "qwen2-vl" in lower or "qwen2vl" in lower:
        return "qwen2vl"
    # 預設嘗試 qwen2vl
    return "qwen2vl"


# ---------------------------------------------------------------------------
# InternVL：vision_config 在部分 transformers 載入路徑會變成 meta 張量，
# 導致 torch.linspace(0, drop_path_rate, num_hidden_layers).tolist() 失敗。
# 從 repo（或本機）的 config.json 讀回純量並寫回 config，再以 config= 載入權重。
# ---------------------------------------------------------------------------

def _internvl_config_json_path(model_id: str) -> Path | None:
    mid = str(model_id)
    p = Path(mid).expanduser()
    if p.is_dir() and (p / "config.json").is_file():
        return p / "config.json"
    try:
        from huggingface_hub import hf_hub_download

        resolved = hf_hub_download(repo_id=mid, filename="config.json")
        return Path(resolved)
    except Exception:
        return None


def _internvl_vision_defaults_from_disk(model_id: str) -> dict[str, float | int]:
    path = _internvl_config_json_path(model_id)
    if path is None or not path.is_file():
        return {"drop_path_rate": 0.0, "num_hidden_layers": 24}
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {"drop_path_rate": 0.0, "num_hidden_layers": 24}
    vc: dict[str, Any] = raw.get("vision_config") or {}
    return {
        "drop_path_rate": float(vc.get("drop_path_rate", 0.0)),
        "num_hidden_layers": int(vc.get("num_hidden_layers", 24)),
    }


def _internvl_materialize_vision_scalars(config, defaults: dict[str, float | int]) -> None:
    vc = getattr(config, "vision_config", None)
    if vc is None:
        return

    def coerce(name: str, *, as_int: bool) -> None:
        fb = defaults[name]
        v = getattr(vc, name, fb)
        if isinstance(v, torch.Tensor):
            if v.device.type == "meta":
                setattr(vc, name, int(fb) if as_int else float(fb))
                return
            x = v.detach().cpu().squeeze()
            setattr(
                vc,
                name,
                int(x.item()) if as_int else float(x.item()),
            )
            return
        setattr(vc, name, int(v) if as_int else float(v))

    coerce("drop_path_rate", as_int=False)
    coerce("num_hidden_layers", as_int=True)


def _internvl_apply_linspace_patch(defaults: dict[str, float | int]):
    """
    from_pretrained 內部有時仍會讓 InternVisionEncoder 在 meta 語意下呼叫 linspace。
    載入期間暫時取代 torch.linspace：強制 CPU，且 meta 張量改用 config.json 純量。
    回傳 (restore_fn)，呼叫後恢復原函式。
    """
    _orig = torch.linspace

    def _to_float(x, fallback: float) -> float:
        if isinstance(x, torch.Tensor):
            if x.device.type == "meta":
                return float(fallback)
            return float(x.detach().cpu().reshape(-1)[0].item())
        return float(x)

    def _to_int(x, fallback: int) -> int:
        if isinstance(x, torch.Tensor):
            if x.device.type == "meta":
                return int(fallback)
            return int(x.detach().cpu().reshape(-1)[0].item())
        return int(x)

    def patched(start, end, steps, *args, **kwargs):
        kw = dict(kwargs)
        kw["device"] = torch.device("cpu")
        s = _to_float(start, 0.0)
        e = _to_float(end, defaults["drop_path_rate"])
        n = _to_int(steps, defaults["num_hidden_layers"])
        return _orig(s, e, n, *args, **kw)

    torch.linspace = patched  # type: ignore[misc]

    def _restore() -> None:
        torch.linspace = _orig  # type: ignore[misc]

    return _restore


def _internvl_apply_mark_tied_patch():
    """
    InternVLChatModel 未呼叫 PreTrainedModel.post_init()，新版 transformers 在
    _finalize_model_loading 會存取 all_tied_weights_keys 而噴錯。
    載入期間若缺少該屬性則補成空 dict，再交還原版邏輯。
    """
    import transformers.modeling_utils as modeling_utils

    _orig = modeling_utils.PreTrainedModel.mark_tied_weights_as_initialized

    def _wrapped(self, loading_info):
        if not hasattr(self, "all_tied_weights_keys"):
            self.all_tied_weights_keys = {}
        return _orig(self, loading_info)

    modeling_utils.PreTrainedModel.mark_tied_weights_as_initialized = _wrapped

    def _restore():
        modeling_utils.PreTrainedModel.mark_tied_weights_as_initialized = _orig

    return _restore


def _internvl_apply_get_keys_to_not_convert_patch():
    """
    bitsandbytes 量化預處理會呼叫 quantizers.base.get_keys_to_not_convert(model)，
    內部直接存取 model.all_tied_weights_keys；InternVL 未跑 post_init 會再噴錯。
    """
    import transformers.quantizers.base as quant_base

    _orig = quant_base.get_keys_to_not_convert

    def _wrapped(model):
        if not hasattr(model, "all_tied_weights_keys"):
            model.all_tied_weights_keys = {}
        return _orig(model)

    quant_base.get_keys_to_not_convert = _wrapped  # type: ignore[assignment]

    def _restore():
        quant_base.get_keys_to_not_convert = _orig  # type: ignore[assignment]

    return _restore


def _maybe_polyfill_torch_set_submodule():
    """
    transformers + bitsandbytes 會呼叫 model.set_submodule(...)；
    較舊 PyTorch 的 nn.Module 無此方法。若缺則補上與新版類似的實作。
    """
    import torch.nn as nn

    if hasattr(nn.Module, "set_submodule"):
        return lambda: None

    def set_submodule(self, target: str, module: nn.Module) -> None:
        if not target:
            raise ValueError("set_submodule: empty target")
        atoms = target.split(".")
        name = atoms.pop(-1)
        obj: nn.Module = self
        for item in atoms:
            if not hasattr(obj, item):
                raise AttributeError(
                    f"{type(obj).__name__!r} has no attribute {item!r} (path {target!r})"
                )
            child = getattr(obj, item)
            if child is None:
                raise ValueError(f"path {target!r}: {item!r} is None")
            if not isinstance(child, nn.Module):
                raise TypeError(f"{item!r} is not an nn.Module")
            obj = child
        setattr(obj, name, module)

    nn.Module.set_submodule = set_submodule  # type: ignore[attr-defined, assignment]
    print(
        "  提示：此 PyTorch 無 nn.Module.set_submodule，已套用 polyfill 以支援 bitsandbytes 量化"
    )

    def _restore():
        delattr(nn.Module, "set_submodule")

    return _restore


def _patch_internvl_language_model_generate(model: torch.nn.Module) -> None:
    """
    transformers >= 4.50：PreTrainedModel 不再繼承 GenerationMixin。
    InternVL 的 generate() 會呼叫 self.language_model.generate()。
    - 在 language_model 類別上追加 GenerationMixin。
    - __init__ 時因 can_generate() 為 False 而不會設 generation_config；補上後才跑得動 generate。
    """
    lm = getattr(model, "language_model", None)
    if lm is None:
        return
    from transformers import GenerationConfig
    from transformers.generation.utils import GenerationMixin

    cls = type(lm)
    if not issubclass(cls, GenerationMixin):
        if GenerationMixin not in cls.__bases__:
            try:
                cls.__bases__ = cls.__bases__ + (GenerationMixin,)
                print("  提示：已為 language_model 類別掛上 GenerationMixin（相容 transformers>=4.50）")
            except TypeError as e:
                raise RuntimeError(
                    "無法為 InternLM2 注入 GenerationMixin（MRO 衝突）。"
                    "可嘗試將 transformers 降到 4.49.x，或等待 InternVL 遠端碼更新。"
                ) from e

    if not hasattr(lm, "generation_config") or getattr(lm, "generation_config", None) is None:
        lm.generation_config = GenerationConfig.from_model_config(lm.config)
        print("  提示：已補上 language_model.generation_config")


def _internlm2_cache_to_legacy_tuple(cache) -> Any:
    """把 transformers 的 Cache（DynamicCache）轉成 InternLM2 內部用的 ( (k,v), … )；空 cache → None。"""
    from transformers.cache_utils import Cache

    if not isinstance(cache, Cache) or cache.get_seq_length() == 0:
        return None
    return tuple((cache.layers[i].keys, cache.layers[i].values) for i in range(len(cache.layers)))


def _internlm2_write_legacy_into_cache(legacy_tuple: Any, cache) -> None:
    """將 InternLM2 forward 產生的 per-layer (k,v) 寫回同一個 DynamicCache（供下一步 generate 使用）。"""
    if legacy_tuple is None:
        return
    for i, layer in enumerate(cache.layers):
        k, v = legacy_tuple[i]
        layer.keys = k
        layer.values = v
        layer.is_initialized = True


def _internlm2_prepare_inputs_for_generation_fixed(
    input_ids: Any,
    past_key_values: Any = None,
    attention_mask: Any = None,
    inputs_embeds: Any = None,
    **kwargs: Any,
) -> dict[str, Any]:
    """
    InternLM2 原版用 past_key_values[0][0]；對 DynamicCache 需改為 get_seq_length()，
    且空 cache 時仍應走 inputs_embeds 第一次餵入分支。
    """
    from transformers.cache_utils import Cache

    past_length = 0
    if past_key_values is not None:
        if isinstance(past_key_values, Cache):
            past_length = past_key_values.get_seq_length()
        else:
            past_length = past_key_values[0][0].shape[2]

    if past_length > 0:
        if input_ids.shape[1] > past_length:
            remove_prefix_length = past_length
        else:
            remove_prefix_length = input_ids.shape[1] - 1
        input_ids = input_ids[:, remove_prefix_length:]

    position_ids = kwargs.get("position_ids", None)
    if attention_mask is not None and position_ids is None:
        position_ids = attention_mask.long().cumsum(-1) - 1
        position_ids.masked_fill_(attention_mask == 0, 1)
        if past_length > 0:
            position_ids = position_ids[:, -input_ids.shape[1] :]

    cache_empty = past_key_values is None or (
        isinstance(past_key_values, Cache) and past_key_values.get_seq_length() == 0
    )
    if inputs_embeds is not None and cache_empty:
        model_inputs: dict[str, Any] = {"inputs_embeds": inputs_embeds}
    else:
        model_inputs = {"input_ids": input_ids}

    model_inputs.update(
        {
            "position_ids": position_ids,
            "past_key_values": past_key_values,
            "use_cache": kwargs.get("use_cache"),
            "attention_mask": attention_mask,
        }
    )
    return model_inputs


def _patch_internlm2_dynamic_cache_bridge(model: torch.nn.Module) -> None:
    """
    transformers 5.x：generate 對 decoder + inputs_embeds 會強制 use_cache=True 並傳入 DynamicCache；
    InternLM2 仍以 legacy tuple 下標使用 KV，導致 'DynamicCache' object is not subscriptable。
    在 language_model 上橋接：forward 前 Cache→tuple，forward 後把 KV 寫回原 Cache 並回傳該 Cache。
    """
    lm = getattr(model, "language_model", None)
    if lm is None or getattr(lm, "_internlm2_dynamic_cache_patched", False):
        return

    from transformers.cache_utils import Cache
    from transformers.modeling_outputs import ModelOutput

    _orig_forward = lm.forward

    def _forward_bridge(self, *args: Any, **kwargs: Any):
        cache_holder: Any = None
        pk = kwargs.get("past_key_values")
        if isinstance(pk, Cache):
            cache_holder = pk
            kwargs = dict(kwargs)
            kwargs["past_key_values"] = _internlm2_cache_to_legacy_tuple(pk)

        out = _orig_forward(*args, **kwargs)

        if cache_holder is not None:
            legacy_past = None
            if isinstance(out, ModelOutput):
                legacy_past = out.past_key_values
            elif isinstance(out, tuple) and len(out) > 1:
                legacy_past = out[1]
            if legacy_past is not None:
                _internlm2_write_legacy_into_cache(legacy_past, cache_holder)
            if isinstance(out, ModelOutput):
                out.past_key_values = cache_holder
        return out

    lm.forward = types.MethodType(_forward_bridge, lm)  # type: ignore[method-assign]

    def _prep_bridge(
        self,
        input_ids: Any,
        past_key_values: Any = None,
        attention_mask: Any = None,
        inputs_embeds: Any = None,
        **kwargs: Any,
    ):
        # 簽名贵 `kwargs` 且保留與 InternLM2 相同具名參數，供 transformers _validate_model_kwargs 通過 inspect
        return _internlm2_prepare_inputs_for_generation_fixed(
            input_ids, past_key_values, attention_mask, inputs_embeds, **kwargs
        )

    lm.prepare_inputs_for_generation = types.MethodType(_prep_bridge, lm)  # type: ignore[method-assign]

    lm._internlm2_dynamic_cache_patched = True
    print("  提示：已為 InternLM2 接上 DynamicCache↔legacy KV 轉接（相容 transformers 5.x + inputs_embeds）")


# ---------------------------------------------------------------------------
# 模型載入
# ---------------------------------------------------------------------------

def load_model(
    model_id: str,
    *,
    internvl_quant: str | None = None,
    internvl_cpu_only: bool = False,
):
    """
    根據 model_id 自動選擇載入方式。
    回傳 (processor_or_tokenizer, model, model_family_str)

    internvl_quant: None | \"4bit\" | \"8bit\"（需 bitsandbytes，16GB 單卡建議 4bit）
    internvl_cpu_only: 全程 CPU（極慢，僅作備援）
    """
    family = detect_model_family(model_id)

    if family == "internvl":
        from transformers import AutoConfig, AutoTokenizer, AutoModel
        print(f"  偵測到 InternVL 系列，使用 AutoModel + trust_remote_code=True")
        vdefaults = _internvl_vision_defaults_from_disk(model_id)
        config = AutoConfig.from_pretrained(
            model_id,
            trust_remote_code=True,
        )
        _internvl_materialize_vision_scalars(config, vdefaults)
        tokenizer = AutoTokenizer.from_pretrained(
            model_id,
            trust_remote_code=True,
            low_cpu_mem_usage=False,
        )
        device = "cuda:0" if torch.cuda.is_available() else "cpu"
        prev_default = None
        if hasattr(torch, "set_default_device"):
            try:
                prev_default = torch.get_default_device()
            except Exception:
                prev_default = None
            torch.set_default_device("cpu")

        fp_kwargs = dict(
            pretrained_model_name_or_path=model_id,
            config=config,
            trust_remote_code=True,
            low_cpu_mem_usage=False,
        )

        _restore_linspace = _internvl_apply_linspace_patch(vdefaults)
        _restore_mark_tied = _internvl_apply_mark_tied_patch()
        _restore_get_keys = _internvl_apply_get_keys_to_not_convert_patch()
        _restore_subm_poly = lambda: None
        try:
            if internvl_cpu_only:
                print("  InternVL：--internvl-cpu-only，權重留在 CPU")
                model = AutoModel.from_pretrained(
                    torch_dtype=torch.bfloat16,
                    **fp_kwargs,
                )
            elif internvl_quant in ("4bit", "8bit"):
                try:
                    from transformers import BitsAndBytesConfig
                except ImportError as e:
                    raise RuntimeError(
                        "使用 --internvl-quant 需安裝 bitsandbytes：pip install bitsandbytes"
                    ) from e
                if internvl_quant == "4bit":
                    qconf = BitsAndBytesConfig(
                        load_in_4bit=True,
                        bnb_4bit_compute_dtype=torch.bfloat16,
                        bnb_4bit_use_double_quant=True,
                        bnb_4bit_quant_type="nf4",
                    )
                else:
                    qconf = BitsAndBytesConfig(load_in_8bit=True)
                print(f"  InternVL：{internvl_quant} 量化 + device_map=auto")
                _restore_subm_poly = _maybe_polyfill_torch_set_submodule()
                model = AutoModel.from_pretrained(
                    quantization_config=qconf,
                    device_map="auto",
                    **fp_kwargs,
                )
            else:
                model = AutoModel.from_pretrained(
                    torch_dtype=torch.bfloat16,
                    **fp_kwargs,
                )
        finally:
            _restore_subm_poly()
            _restore_get_keys()
            _restore_mark_tied()
            _restore_linspace()
            if hasattr(torch, "set_default_device"):
                if prev_default is not None:
                    torch.set_default_device(prev_default)
                else:
                    torch.set_default_device(
                        "cuda" if torch.cuda.is_available() else "cpu"
                    )

        if internvl_cpu_only:
            model.eval()
        elif internvl_quant in ("4bit", "8bit"):
            model.eval()
        else:
            import gc

            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            p0 = next(model.parameters(), None)
            if p0 is not None and p0.device.type == "cuda":
                print("  InternVL 權重已在 GPU，略過 .to()（避免 OOM 尖峰）")
                model.eval()
            elif p0 is not None and p0.device.type == "cpu" and torch.cuda.is_available():
                try:
                    model = model.to(device).eval()
                except torch.OutOfMemoryError as e:
                    raise RuntimeError(
                        "GPU 顯存不足（InternVL2-8B bf16 約需略高於 16GB 餘裕）。"
                        "請改用其一：\n"
                        "  --internvl-quant 4bit    （單卡 16GB 建議，需 bitsandbytes）\n"
                        "  --internvl-quant 8bit\n"
                        "  --internvl-cpu-only      （極慢）\n"
                        "或兩張 GPU 時不要設 CUDA_VISIBLE_DEVICES=單卡，並搭配 --internvl-quant 4bit"
                    ) from e
            else:
                model.eval()

        _patch_internvl_language_model_generate(model)
        _patch_internlm2_dynamic_cache_bridge(model)

        print(
            f"  InternVL 就緒 (vision drop_path={vdefaults['drop_path_rate']}, "
            f"layers={vdefaults['num_hidden_layers']})"
        )
        return tokenizer, model, family

    else:  # qwen2vl
        from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
        print(f"  偵測到 Qwen2-VL 系列，使用 Qwen2VLForConditionalGeneration")
        processor = AutoProcessor.from_pretrained(model_id)
        model = Qwen2VLForConditionalGeneration.from_pretrained(
            model_id,
            torch_dtype="auto",
            device_map="auto",
        ).eval()
        return processor, model, family


# ---------------------------------------------------------------------------
# InternVL2 圖片前處理
# ---------------------------------------------------------------------------

def _build_internvl_transform(input_size: int = 448):
    import torchvision.transforms as T
    from torchvision.transforms.functional import InterpolationMode
    IMAGENET_MEAN = (0.485, 0.456, 0.406)
    IMAGENET_STD  = (0.229, 0.224, 0.225)
    return T.Compose([
        T.Lambda(lambda img: img.convert("RGB") if img.mode != "RGB" else img),
        T.Resize((input_size, input_size), interpolation=InterpolationMode.BICUBIC),
        T.ToTensor(),
        T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])

_INTERNVL_TRANSFORM = None  # lazy init

def _load_image_for_internvl(image_path: str, input_size: int = 448) -> "torch.Tensor":
    global _INTERNVL_TRANSFORM
    from PIL import Image
    if _INTERNVL_TRANSFORM is None:
        _INTERNVL_TRANSFORM = _build_internvl_transform(input_size)
    img = Image.open(image_path).convert("RGB")
    return _INTERNVL_TRANSFORM(img).unsqueeze(0)  # (1, C, H, W)


# ---------------------------------------------------------------------------
# 單張圖推理（統一介面）
# ---------------------------------------------------------------------------

def caption_one_image(
    *,
    processor,          # AutoProcessor（Qwen2-VL）或 AutoTokenizer（InternVL）
    model,
    model_family: str,  # 'qwen2vl' | 'internvl'
    image_path: str,    # 本機絕對路徑（非 URI）
    prompt: str,
    max_new_tokens: int,
) -> str:

    if model_family == "internvl":
        return _caption_internvl(
            tokenizer=processor,
            model=model,
            image_path=image_path,
            prompt=prompt,
            max_new_tokens=max_new_tokens,
        )
    else:
        return _caption_qwen2vl(
            processor=processor,
            model=model,
            image_path=image_path,
            prompt=prompt,
            max_new_tokens=max_new_tokens,
        )


def _caption_qwen2vl(
    *,
    processor,
    model,
    image_path: str,
    prompt: str,
    max_new_tokens: int,
) -> str:
    from qwen_vl_utils import process_vision_info

    image_uri = Path(image_path).as_uri()
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image_uri},
                {"type": "text",  "text": prompt},
            ],
        }
    ]
    text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    ).to(model.device)

    with torch.inference_mode():
        generated_ids = model.generate(**inputs, max_new_tokens=max_new_tokens)

    trimmed = [
        out_ids[len(in_ids):]
        for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    out = processor.batch_decode(
        trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )
    return out[0].strip() if out else ""


def _internvl_pixel_dtype(model: torch.nn.Module) -> torch.dtype:
    cfg = getattr(model, "config", None)
    dt = getattr(cfg, "torch_dtype", None) if cfg is not None else None
    if isinstance(dt, str):
        dt = getattr(torch, dt.removeprefix("torch."), torch.bfloat16)
    if isinstance(dt, torch.dtype) and dt.is_floating_point:
        return dt
    for p in model.parameters():
        if p.dtype.is_floating_point:
            return p.dtype
    return torch.bfloat16


def _caption_internvl(
    *,
    tokenizer,
    model,
    image_path: str,
    prompt: str,
    max_new_tokens: int,
) -> str:
    device = _model_primary_device(model)
    pv_dtype = _internvl_pixel_dtype(model)
    pixel_values = _load_image_for_internvl(image_path).to(
        dtype=pv_dtype, device=device
    )
    generation_config = dict(max_new_tokens=max_new_tokens, do_sample=False)
    with torch.inference_mode():
        response = model.chat(tokenizer, pixel_values, prompt, generation_config)
    return response.strip() if response else ""



def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="對 cluster 代表圖批次跑 VLM 敘述（支援 Qwen2-VL / InternVL2）")
    p.add_argument(
        "--plots-root",
        type=Path,
        default=_ROOT / "plots" / "vlm_analysis",
        help=(
            "圖片根目錄。\n"
            "  scan-mode=repr: 內含 stage{S}_block{B}/repr_pos{P}_part{K}.png\n"
            "  scan-mode=inference: 內含 img{I}_s{S}_b{B}.png"
        ),
    )
    p.add_argument(
        "--scan-mode",
        choices=("repr", "inference", "inference-repr-pos"),
        default="repr",
        help=(
            "掃描模式：\n"
            "  repr: stage{S}_block{B}/repr_pos{P}_part{K}.png\n"
            "  inference: img{I}_s{S}_b{B}.png\n"
            "  inference-repr-pos: img{I}_s{S}_b{B}_repr_pos{P}_cluster{C}.png"
        ),
    )
    p.add_argument(
        "--output-dir",
        type=Path,
        default=_ROOT / "plots" / "vlm_analysis",
        help="輸出 JSONL 目錄（預設: <repo>/plots/vlm_analysis）",
    )
    p.add_argument(
        "--output-template",
        type=str,
        default="captions_stage{stage}.jsonl",
        help=(
            "輸出檔名樣板，可用欄位：{stage} {img} {block} {pos} {part} {cluster}。\n"
            "範例（每張圖每個 stage 一檔）: captions_img{img}_stage{stage}.jsonl"
        ),
    )
    p.add_argument(
        "--model",
        type=str,
        default="Qwen/Qwen2-VL-7B-Instruct",
        help=(
            "Hugging Face 模型 id 或本機路徑。\n"
            "  Qwen2-VL 系列：Qwen/Qwen2-VL-7B-Instruct（預設）\n"
            "  InternVL2 系列：OpenGVLab/InternVL2-8B"
        ),
    )
    p.add_argument("--max-new-tokens", type=int, default=512)
    p.add_argument(
        "--limit",
        type=int,
        default=None,
        help="只處理前 N 張圖（除錯用）",
    )
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="只列出將處理的檔案，不載入模型",
    )
    p.add_argument(
        "--prompt-file",
        type=Path,
        default=None,
        help="自訂 prompt 文字檔（UTF-8），若未指定則用內建繁中說明",
    )
    p.add_argument(
        "--skip-existing",
        action="store_true",
        help="若 output JSONL 中已有相同 rel_key，則跳過（簡易續跑）",
    )
    p.add_argument(
        "--stage-start",
        type=int,
        default=3,
        help="起始 stage（預設 3）",
    )
    p.add_argument(
        "--stage-end",
        type=int,
        default=1,
        help="結束 stage（預設 1）",
    )
    p.add_argument(
        "--internvl-quant",
        choices=("none", "4bit", "8bit"),
        default="none",
        help="僅 InternVL：量化載入（單卡 16GB 建議 4bit，需安裝 bitsandbytes）",
    )
    p.add_argument(
        "--internvl-cpu-only",
        action="store_true",
        help="僅 InternVL：權重與推理全用 CPU（極慢，顯存不足時備援）",
    )
    return p.parse_args()


# ---------------------------------------------------------------------------
# 工具函式
# ---------------------------------------------------------------------------

def load_done_rel_keys(path: Path) -> set[str]:
    done: set[str] = set()
    if not path.is_file():
        return done
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
                k = obj.get("rel_key") or obj.get("meta", {}).get("rel_key")
                if k:
                    done.add(str(k))
            except json.JSONDecodeError:
                continue
    return done


def stage_range_inclusive(start: int, end: int) -> list[int]:
    step = -1 if start >= end else 1
    return list(range(start, end + step, step))


def filter_and_sort_records_by_stage(
    records: Iterable[PlotRecord], stage_start: int, stage_end: int
) -> list[PlotRecord]:
    stages = set(stage_range_inclusive(stage_start, stage_end))
    reverse = stage_start > stage_end
    out = [r for r in records if r.stage in stages]
    out.sort(
        key=lambda r: (
            -r.stage if reverse else r.stage,
            r.img if r.img is not None else -1,
            r.block,
            r.pos if r.pos is not None else -1,
            r.part if r.part is not None else -1,
            r.cluster if r.cluster is not None else -1,
            r.image_path,
        )
    )
    return out


def render_output_relpath(template: str, rec: PlotRecord) -> str:
    """依 record 套用 output template。"""
    data: dict[str, Any] = {
        "stage": rec.stage,
        "img": rec.img if rec.img is not None else -1,
        "block": rec.block,
        "pos": rec.pos if rec.pos is not None else -1,
        "part": rec.part if rec.part is not None else -1,
        "cluster": rec.cluster if rec.cluster is not None else -1,
    }
    try:
        return template.format(**data)
    except KeyError as e:
        raise ValueError(
            f"--output-template 使用了未知欄位 {e!s}，可用欄位: "
            "{stage} {img} {block} {pos} {part} {cluster}"
        ) from e


# ---------------------------------------------------------------------------
# 主程式
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()
    if not any(tok in args.output_template for tok in ("{stage}", "{img}", "{block}", "{pos}", "{part}", "{cluster}")):
        raise ValueError(
            "--output-template 至少要包含一個欄位：{stage} {img} {block} {pos} {part} {cluster}"
        )

    if args.scan_mode == "repr":
        records = discover_repr_images(args.plots_root)
    elif args.scan_mode == "inference":
        records = discover_inference_stage_images(args.plots_root)
    else:
        records = discover_inference_repr_images(args.plots_root)
    records = filter_and_sort_records_by_stage(
        records, stage_start=args.stage_start, stage_end=args.stage_end
    )
    if args.limit is not None:
        records = records[: args.limit]

    if args.prompt_file is not None:
        prompt = args.prompt_file.read_text(encoding="utf-8").strip()
    else:
        prompt = build_default_prompt()

    family = detect_model_family(args.model)

    if args.dry_run:
        print(f"plots_root  = {args.plots_root.resolve()}")
        print(f"scan_mode   = {args.scan_mode}")
        print(f"model       = {args.model}  （family: {family}）")
        print(
            f"stage 範圍  : {args.stage_start} -> {args.stage_end}（含）"
            f"，共 {len(records)} 張，prompt 長度 {len(prompt)} 字元"
        )
        for r in records:
            print(r.rel_key, "->", r.image_path)
        return

    if not records:
        if args.scan_mode == "repr":
            pattern_hint = "repr_pos*_part*.png"
        elif args.scan_mode == "inference":
            pattern_hint = "img*_s*_b*.png"
        else:
            pattern_hint = "img*_s*_b*_repr_pos*_cluster*.png"
        print(
            f"在 {args.plots_root.resolve()} 底下找不到符合 stage 範圍 "
            f"{args.stage_start}->{args.stage_end} 的 {pattern_hint}，結束。"
        )
        return

    print(f"載入模型 {args.model}  （family: {family}）…")
    q = args.internvl_quant if args.internvl_quant != "none" else None
    processor, model, model_family = load_model(
        args.model,
        internvl_quant=q,
        internvl_cpu_only=args.internvl_cpu_only,
    )

    out_to_records: dict[Path, list[PlotRecord]] = {}
    for rec in records:
        rel_out = render_output_relpath(args.output_template, rec)
        out_path = args.output_dir / rel_out
        out_to_records.setdefault(out_path, []).append(rec)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    written_files: list[Path] = []

    for out_path, stage_records in out_to_records.items():
        out_path.parent.mkdir(parents=True, exist_ok=True)
        done_keys = load_done_rel_keys(out_path) if args.skip_existing else set()
        written_files.append(out_path)

        stage_info = (
            f"{stage_records[0].stage}" if len({r.stage for r in stage_records}) == 1 else "mixed"
        )
        img_info = (
            f"{stage_records[0].img}" if len({r.img for r in stage_records}) == 1 else "mixed"
        )
        print(
            f"\n=== Stage {stage_info} / Img {img_info}："
            f"{len(stage_records)} 張，輸出 -> {out_path.resolve()} ==="
        )
        with out_path.open("a", encoding="utf-8") as out_f:
            for i, rec in enumerate(stage_records):
                if rec.rel_key in done_keys:
                    print(f"[{i+1}/{len(stage_records)}] 略過（已存在） {rec.rel_key}")
                    continue
                print(f"[{i+1}/{len(stage_records)}] {rec.rel_key} …", flush=True)
                try:
                    text = caption_one_image(
                        processor=processor,
                        model=model,
                        model_family=model_family,
                        image_path=rec.image_path,
                        prompt=prompt,
                        max_new_tokens=args.max_new_tokens,
                    )
                except Exception as e:
                    text = f"[ERROR] {e!s}"
                    print(f"     失敗: {e}", flush=True)

                row = {
                    **asdict(rec),
                    "caption": text,
                    "prompt": prompt,
                    "model": args.model,
                    "model_family": model_family,
                    "max_new_tokens": args.max_new_tokens,
                }
                out_f.write(json.dumps(row, ensure_ascii=False) + "\n")
                out_f.flush()

    print("\n完成，結果輸出檔案：")
    for p in written_files:
        print(" -", p.resolve())


if __name__ == "__main__":
    main()
