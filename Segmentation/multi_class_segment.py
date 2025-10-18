# imports
import torch, torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms.functional as TF
import numpy as np, cv2, os, matplotlib.pyplot as plt
import albumentations as A
import segmentation_models_pytorch as smp
from tqdm import tqdm
from sklearn.model_selection import train_test_split

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
NUM_CLASSES = 3
IMG_SIZE = 128

os.makedirs(
