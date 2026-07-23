#!/usr/bin/env python3
"""
單一圖片路徑的 Qwen2.5-VL 簡易測試腳本。

用法（專案根目錄）:
  conda run -n SFM python vit_analysis_vlm/qwen_example.py
  conda run -n SFM python vit_analysis_vlm/qwen_example.py path/to/image.png
  conda run -n SFM python vit_analysis_vlm/qwen_example.py \
      --prompt-template default --max-new-tokens 256
"""
from __future__ import annotations

import argparse
from pathlib import Path

import torch
from qwen_vl_utils import process_vision_info
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

_ROOT = Path(__file__).resolve().parent.parent

DEFAULT_MODEL = "Qwen/Qwen2.5-VL-7B-Instruct"

img_path = (
    _ROOT
    / "plots/kmeans/experiment/caltech101/img0/s3_b0_pos4_top02.png"
)

prompt_templates = {
    "default": (
        "這張圖包含一列影像區塊（patch）。\n"
        "最左邊第一格是從某張圖片的特定空間位置裁出的輸入樣本；"
        "右側四格是與它歸屬於同一個 K-means cluster 的代表樣本，"
        "代表了模型在這個位置學習到的典型視覺特徵。\n\n"
        "請依以下面向分析這五張 patch 的共同視覺特徵：\n"
        "1. 顏色與對比：主要色調、亮暗分布、前景背景關係\n"
        "2. 形狀與結構：幾何特徵、紋理方向、邊緣特性\n"
        "3. 位置線索：視覺特徵集中在 patch 的哪個區域（上/下/左/右/中）\n"
        "4. 具體特徵：具體的顏色、方向、位置、形狀、筆畫、結構、物件 \n"
        "5. 跨樣本一致性：五張 patch 之間相似在哪、差異在哪\n\n"
        "最後，請用一句話具體總結這個位置的 cluster 最可能捕捉的局部視覺結構。"
        "請避免使用空泛詞彙（例如『局部視覺特徵』『邊緣特徵』『視覺結構』等籠統說法），"
        "並依照以下句型填空作答（將方括號替換為具體判斷，不要保留方括號）：\n"
        "此位置的 cluster 主要捕捉 [具體位置] 處 [顏色] 形成的 [具體形狀]。"
        "輸入樣本與代表樣本相比，於上述特徵上呈現「[契合程度：高度吻合／部分吻合／明顯偏離]」，"
        "具體而言，[說明契合或偏離之處，同樣具有什麼樣的顏色、方向、位置、形狀、筆畫、結構、物件，或是具體差異]。\n\n"
        "請用繁體中文回答。"
    ),
    "bloodmnist": (
        "【影像結構說明】\n"
        "這張圖包含一列影像區塊（patch）。\n"
        "最左邊第一格是從某張圖片的特定空間位置裁出的輸入樣本；"
        "右側四格是與它歸屬於同一個 K-means cluster 的代表樣本，"
        "代表了模型在這個位置學習到的典型視覺特徵。\n\n"
        "請依以下面向分析這五張 patch 的共同視覺特徵，並以一段連續文字"
        "（不要使用標題、項目符號或條列格式）依序說明：\n"
        "1. 顏色與對比：主要色調、亮暗分布、前景背景關係。\n"
        "2. 細胞核與細胞質型態：請描述代表圖之間共同呈現的細胞核外形與細胞質狀態，"
        "包含核的邊緣輪廓、核與質之間由邊緣到中心的顏色漸層或質地變化。"
        "你可以視觀察情況使用以下形態學詞彙（僅為可能的描述用詞參考，"
        "並非需要對應到特定類別，請僅在實際觀察到對應形狀時才使用，"
        "不需要也不應該指出這是哪一種細胞）：\n"
        "核形狀類：圓形核緣、分葉核塊、雙葉核、腎形折角、馬蹄形折角、"
        "不規則鬆散圓弧、緻密圓核；\n"
        "質地類：緻密顆粒狀、細緻均勻、大型顆粒、少量細胞質、"
        "豐富細胞質、無核碎片狀。\n"
        "【重要】若五張圖的色塊呈現連續、均勻的漸層變化，找不到可以明確指認"
        "「這是核、這是質」的分界或獨立結構，請如實說明這是色塊或漸層層次的觀察，"
        "不要勉強套用上述形態詞彙或編造核／質的分界。\n"
        "3. 額外結構：色塊範圍內是否存在獨立於主體之外的額外小色塊、斑點或不連續結構。\n"
        "4. 位置線索：視覺特徵集中在 patch 的哪個區域（上/下/左/右/中）。\n"
        "5. 跨樣本一致性：五張 patch 之間相似在哪、差異在哪。\n"
        "6. 契合度：說明輸入樣本與代表樣本相比，於上述特徵上呈現高度吻合／"
        "部分吻合／明顯偏離，並具體說明契合或偏離之處。\n\n"
        "全程不需要也不應該推測或指出對應的細胞種類或醫學診斷名稱。"
        "請用繁體中文回答。"
    ),
    "caltech101": (
    "【任務背景與角色】\n"
            "你現在是一個計算機視覺與深度學習特徵空間（Latent Space）分析師。目前正在分析 MergingViT 網路在 Caltech 101 物體分類資料集中的特徵分群結果。\n"
            "【影像結構說明】\n"
            "本次輸入的圖片包含 5 個區塊：input：當前待推論的未知局部影像區塊（Patch）。repr1 到 repr4：該特徵群聚中心附近的 4 個代表性影像區塊。它們共同定義了這個特徵群聚的局部視覺概念。\n"
            "【重要：請依據目前影像的清晰度（Stage）進行動態分析】\n"
            "步驟一：多尺度視覺共性分析\n"
            "（觀察 repr1 到 repr4）如果影像極度像素化/呈粗糙色塊（低階層 Stage 0-2）：請著重描述代表圖之間共同的「色彩組合」、「亮暗梯度分佈（例如：左暗右亮、橫向帶狀明暗）」以及「是否有特定方向的色彩邊界」。如果影像細節清晰/具備具體形狀（高階層 Stage 3）：請著重描述其共同展現的「人工製品局部結構（如：整齊窗戶、輪胎弧度、金屬線條）」或「自然物體紋理（如：毛髮、葉片）」，並注意其背景環境。\n"
            "步驟二：待推論物件（input）的契合度比對\n"
            "對比 input 與 repr1~repr4。無論在「色塊梯度（Stage 0-2）」還是「結構幾何（Stage 3）」上，input 是否都完美契合並融入了這個特徵群聚中心？請指出它們最一致的視覺特徵。\n"
            "步驟三：群聚語意與潛在物體類別推論\n"
            "結合上述視覺證據，嘗試推測這個特徵群聚最可能是在捕捉 Caltech 101 中哪一種常見物體的局部？請給出 Top-1（最可能） 與 Top-2（次可能） 的預測，並詳細說明理由。(備註：若目前影像屬於 Stage 0-2 且幾何特徵太模糊，請主要依據「色彩與梯度搭配（如：上方藍天色、下方金屬色）」進行類別聯想，並在理由中說明這是基於低階特徵的推測。)\n"
        "請用繁體中文回答。"
    ),
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="對單一圖片跑 Qwen2.5-VL 推理（效果測試）")
    p.add_argument(
        "image",
        type=Path,
        nargs="?",
        default=img_path,
        help=f"本機圖片路徑（預設: {img_path}）",
    )
    p.add_argument(
        "--prompt-template",
        type=str,
        default="default",
        choices=sorted(prompt_templates.keys()),
        help="使用內建 prompt_templates 的 key（預設: default）",
    )
    p.add_argument(
        "--prompt",
        type=str,
        default=None,
        help="直接指定文字 prompt（覆蓋 --prompt-template）",
    )
    p.add_argument(
        "--prompt-file",
        type=Path,
        default=None,
        help="從 UTF-8 文字檔讀取 prompt（優先於 --prompt / --prompt-template）",
    )
    p.add_argument(
        "--model",
        type=str,
        default=DEFAULT_MODEL,
        help=f"Hugging Face 模型 id 或本機路徑（預設: {DEFAULT_MODEL}）",
    )
    p.add_argument("--max-new-tokens", type=int, default=512)
    return p.parse_args()


def resolve_prompt(args: argparse.Namespace) -> str:
    if args.prompt_file is not None:
        path = args.prompt_file.expanduser().resolve()
        if not path.is_file():
            raise FileNotFoundError(f"找不到 prompt 檔: {path}")
        text = path.read_text(encoding="utf-8").strip()
        if not text:
            raise ValueError(f"prompt 檔是空的: {path}")
        return text
    if args.prompt is not None and args.prompt.strip():
        return args.prompt.strip()
    return prompt_templates[args.prompt_template]


def caption_image(
    *,
    image_path: Path,
    prompt: str,
    model_id: str,
    max_new_tokens: int,
) -> str:
    print(f"載入模型: {model_id}")
    # Qwen2.5-VL 必須用 Qwen2_5_VLForConditionalGeneration；
    # 若誤用 Qwen2VLForConditionalGeneration 會出現 bias shape 不符錯誤。
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        model_id,
        torch_dtype="auto",
        device_map="auto",
    ).eval()
    processor = AutoProcessor.from_pretrained(model_id)

    # Qwen VL 習慣用 file:// URI
    image_uri = image_path.resolve().as_uri()
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image_uri},
                {"type": "text", "text": prompt},
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
    )
    device = next(model.parameters()).device
    inputs = inputs.to(device)

    with torch.inference_mode():
        generated_ids = model.generate(**inputs, max_new_tokens=max_new_tokens)

    trimmed = [
        out_ids[len(in_ids) :]
        for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    output_text = processor.batch_decode(
        trimmed,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )
    return output_text[0].strip()


def main() -> None:
    args = parse_args()
    image_path = Path(args.image).expanduser().resolve()
    if not image_path.is_file():
        raise FileNotFoundError(f"找不到圖片: {image_path}")

    prompt = resolve_prompt(args)
    print(f"image  = {image_path}")
    print(f"prompt = {prompt[:120]}{'…' if len(prompt) > 120 else ''}")
    print(f"tokens = {args.max_new_tokens}")
    print("-" * 60)

    result = caption_image(
        image_path=image_path,
        prompt=prompt,
        model_id=args.model,
        max_new_tokens=args.max_new_tokens,
    )
    print(result)


if __name__ == "__main__":
    main()
