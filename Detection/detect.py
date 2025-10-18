import torch, torchvision
from torch.utils.data import Datset, DataLoader
from torchvision.models.detection.faster_rcnn import FasterRCNN_ResNet50_FPN_Weights
import torchvision.transforms as T
import numpy as np, cv2, os, matplotlib.pyplot as plt 
from tqdm import tqdm 

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
NUM_CLASSES = 3
IMG_SIZE = 256

os.makedirs("data_det/images", exist_ok=True)
os.makedirs("data_det/labels",exist_ok=True)

for i in range(15):
    img = np.zeros((IMG_SIZE, IMG_SIZE, 3), np.uint8)
    num_objs = np.random.randint(1,4)
    boxes , labels = [], []
    for j in range(num_objs):
        x1, y1 = np.random.randint(20,180,2)
        x2, y2 = x1 + np.random.randint(30,60), y1 + np.random.randint(30,60)
        color = (0,255,0


