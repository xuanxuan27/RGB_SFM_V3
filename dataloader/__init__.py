from .get_dataloader import get_dataloader
from .MNIST import MNISTDataset
from .MultiColorShapes import MultiColorShapesDataset
from .FaceDataset import FaceDataset
from .Malaria import MalariaCellDataset
from .Caltech101 import Caltech101Dataset
__all__ = ['get_dataloader', 
           'MNISTDataset', 'MultiColorShapesDataset', "FaceDataset",
           "MalariaCellDataset", "AnotherColored_MNIST", "Caltech101Dataset"]