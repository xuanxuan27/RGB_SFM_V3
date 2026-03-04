
from medmnist import INFO, Evaluator, PathMNIST

from torch.utils.data import DataLoader

from .APROS_2019 import APROS_2019Dataset
from .CustomerMedMNIST import CustomerMedMNIST, CustomerDermaMNIST, CustomerPathMNIST, CustomerRetinaMNIST, \
    CustomerBloodMNIST, CustomerRetinaMNIST_224, CustomerPathMNIST_224
from .HeartCalcification import *
from .MNIST import MNISTDataset
from .MNISTWeight import MNISTWeightDataset
from .MultiColorShapes import MultiColorShapesDataset
from .FaceDataset import FaceDataset
from .Malaria import MalariaCellDataset
from .PreprocessedRetinaMNIST224 import PreprocessedRetinaMNIST224
from .RGB_circle import RGBCircle
from .MultiGrayShapes import MultiGrayShapesDataset
from .MultiEdgeShapes import MultiEdgeShapesDataset
from .Colored_MNIST import Colored_MNIST
from .Colored_FashionMNIST import Colored_FashionMNIST
from .AnotherColored_MNIST import AnotherColored_MNIST
from .AnotherColored_FashionMNIST import AnotherColored_FashionMNIST
from .CIFAR10 import CIFAR10
from .Colorful_MNIST import Colorful_MNIST
from torchvision import transforms
from .Caltech101 import Caltech101Dataset
dataset_classes = {
    'mnist': MNISTDataset,
    'MultiColor_Shapes_Database': MultiColorShapesDataset,
    'face_dataset': FaceDataset,
    'malaria':MalariaCellDataset,
    'RGB_Circle':RGBCircle,
    'MultiGrayShapesDataset': MultiGrayShapesDataset,
    'MultiEdgeShapes': MultiEdgeShapesDataset,
    'Colored_MNIST':Colored_MNIST,
    'Colored_FashionMNIST':Colored_FashionMNIST,
    'AnotherColored_MNIST':AnotherColored_MNIST,
    'AnotherColored_FashionMNIST':AnotherColored_FashionMNIST,
    'CIFAR10': CIFAR10,
    'Colorful_MNIST':Colorful_MNIST,
    'HeartCalcification_Color': HeartCalcificationColor,
    'HeartCalcification_Gray': HeartCalcificationGray,
    "MinstWeight" : MNISTWeightDataset,
    "PathMNIST" :  CustomerPathMNIST,
    "PathMNIST_224" :  CustomerPathMNIST_224,
    "DermaMNIST" : CustomerDermaMNIST,
    "RetinaMNIST" : CustomerRetinaMNIST,
    "RetinaMNIST_224" : CustomerRetinaMNIST_224,
    "BloodMNIST" :  CustomerBloodMNIST,
    'APROS_2019' : APROS_2019Dataset,
    'PreprocessedRetinaMNIST224' : PreprocessedRetinaMNIST224,
    'Caltech101' : Caltech101Dataset
}



def get_dataloader(dataset, root: str = '.', batch_size=32, input_size: tuple = (28, 28)):
    if dataset in dataset_classes:
        # 對 Caltech101 先統一轉成 RGB，避免 batch 內出現 [1, H, W] / [3, H, W] 混合
        if dataset == 'Caltech101':
            common_transforms = [
                transforms.Resize([*input_size]),
                transforms.Lambda(lambda img: img.convert('RGB')),
                transforms.ToTensor(),
                transforms.ConvertImageDtype(torch.float),
            ]
        else:
            common_transforms = [
                transforms.Resize([*input_size]),
                transforms.ToTensor(),
                transforms.ConvertImageDtype(torch.float),
            ]

        train_transform = transforms.Compose(common_transforms)

        test_transform = transforms.Compose(common_transforms)

        print(dataset_classes[dataset])

        train_dataset = dataset_classes[dataset](root, train = True, transform = train_transform)
        test_dataset = dataset_classes[dataset](root, train = False, transform = test_transform)

        train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)

        return train_dataloader, test_dataloader
    else:
        raise ValueError(f"Unknown dataset: {dataset}")