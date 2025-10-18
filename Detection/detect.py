# imports and config
import torch, torchvision
from torch.utils.data import Datset, DataLoader
from torchvision.models.detection.faster_rcnn import FasterRCNN_ResNet50_FPN_Weights
import torchvision.transforms as T
from torchvision.ops import box_iou
import numpy as np, cv2, os, matplotlib.pyplot as plt 
from tqdm import tqdm 

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
NUM_CLASSES = 3
IMG_SIZE = 256


# dummy dataset
os.makedirs("data_det/images", exist_ok=True)
os.makedirs("data_det/labels",exist_ok=True)

for i in range(15):
    img = np.zeros((IMG_SIZE, IMG_SIZE, 3), np.uint8)
    num_objs = np.random.randint(1,4)
    boxes , labels = [], []
    for j in range(num_objs):
        x1, y1 = np.random.randint(20,180,2)
        x2, y2 = x1 + np.random.randint(30,60), y1 + np.random.randint(30,60)
        color = (0,255,0)

# dataset class for detection 
class DetectionDataset(Dataset):
    def __init__():

    def __len__():

    def __getitem__():

# transform and dataloader
transform = T.Compose([T.ToTensor()])
train_ds = DetectionDataset("data_det/images","data_det/labels",transform)
train_loader = DataLoader(train_ds, batch_size = 2, shuffle = True, collate_fn = lambda x: tuple(zip(zip(*x))))

# model 
model = torch.vision.models.detection.fasterrcnn_resnet50_fpn(weights = FasterRCNN_ResNet50_FPN_Weights.COCO_V1)
in_features = model.roi_heads.box_predictor.cls_score.in_feature
model.roi_heads.box_predictor = torchvision.models.detection.faster_rcnn.FastRCNNPredictor(in_features,NUM_CLASSES)

model.to(DEVICE)


# optimizer
params = [p for p in model.parameters() if p.requires_grad]
optimizer = torch.optim.Adam(params, lr=1e-4)

# training loop 
EPOCHS = 5 
for epoch in range(EPOCHS):
    model.train()
    total_loss = 0
    for imgs, targets in tqdm(train_loader):
        imgs =
        targets = 
        loss_dict = 
        losses = 
        optimizer.zero_grad
        losses.backward
        optimizer.step
        total_loss
    print()


# inference and visualization
model.eval()
img, _ = train_ds
with torch.no_grad():
    pred = model([img.to(DEVICE)])

img_np = img.permute(1,2,0).cpu().numpy().copy()
for box, label, score in zip:
    if score >
        x1,y1,x2,y2
        color
        cv2
        cv2
plt.imshow()


        


# inference and visualization

pred_boxes = pred
true_boxes
iou
print

# evaluation




