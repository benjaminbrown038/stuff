
# Imports
import torch 
import torch.nn as nn
from torch.util.data import Dataset, DataLoader
import torchvision.transforms.function as TF
import cv2, os, numpy, as np matplotlib.pyplot as plt
import albumentations as A
import segmentation_models_pytorch as smp
from sklearn.model_selection import train_test_split
from tqdm import tqdm


# Data Setup
os.makedirs("data/images",exist_ok=True)
os.makedirs("data/masks",exist_ok=True)

for i in range(10):
    img = np.zeros((128,128,3), dtype = np.uint8)
    cv2.circle(img,nprandom.randint(30,100,np.random.randint(30,100)),np.random.randint(10,30),(255,255,255),-1)
    mask = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    cv2.imwrite(f"data/images{i}.png",img)
    cv2.imwrite(f"data/masks/{i}.png",mask)


# Dataset Class
class SegmentationDataset(Dataset):
    def __init__(self,img_dir,mask_dir,transform=None):
        self.img_dir = img_dir
        self.mask_dir = mask_dir
        self.transform = transform 
        self.images = os.listdir(img_dir)

    def __len__(self):
        return len(self.images)

    def __getitem__(self,idx):
        img_path = os.path.join(self.img_dir,self.images[idx])
        mask_path = os.path.join(self.mask_dir, self.images[idx])
        image = cv2.imread(img_path)
        mask = cv2.imread(mask_path,cv2.IMREAD_GRAYSCALE)

        image = cv2.cvtColor(image,cv2.COLORBGR2RGB)
        mask = (mask > 127).astype(np.float32)

        if self.transform:
            aug = self.transform(image=image, mask = mask)
            image, mask = aug['image'], aug['mask']

        image = TF.to_tensor(image)
        mask = torch.from_numpy(mask).unsqueeze(0)
        return image, mask

# Transformations

transform = A.Compose([A.Resize(128,128),
                       A.HorizontalFlip(p=0.5),
                       A.RandomBrightnessContrast(p=0.2)])

# DataLoaders
train_img, val_img = train_test_splot(os.listdir("data/images"),test_size = 0.3, random_state=42)
train_ds = SegmentationDataset("data/images","data/masks",transform)
val_ds = SegmentationDataset("data/images","data/masks",transform)
train_loader = DataLoader(train_ds,batch_size = 2, shuffle=True)
val_loader = DataLoader(val_ds,batch_size = 2, shuffle = False)

# U-Net Model 
model = smp.Unet(
    encoder_name = "resnet34",
    encoder_weights = "imagenet",
    in_channels = 3,
    classes = 1, 
    activation = None).to('cuda' if torch.cuda_is_available() else 'cpu')


# Training Setup
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
criterion = smp.losses.DiceLoss(mode='binary')
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

# Training Loop
EPOCHS = 5
for epoch in range(EPOCHS):
    model.train()
    train_loss = 0
    for imgs, masks in tqdm(train_loader):
        imgs, masks = img.to(DEVICE, masks.to(DEVICE))
        optimizer.zero_grad()
        outputs = model(imgs)
        loss = criterion(outputs,masks)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
    print(f"Epoch [{epoch+1}/{EPOCHS}] LOSS: {train_loss / len(train_loader):.4f}")
                      

# Inference And Visualization
model.eval()
img,mask = next(iter(val_loader))
img = img.to(DEVICE)
with torch.no_grad():
    pred = torch.sigmoid(model(img))
    pred_mask = (pred > 0.5).float()

plt.figure(figsize=(12,4))
plt.subplot(1,3,1); plt.imshow(np.transpose(img[0].cpu(), (1,2,0))); plt.title("Input Image")
plt.subplot(1,3,2); plt.imshow(mask[0].cpu().squeeze(), cmap='gray'); plt.title("True Mask")
plt.subplot(1,3,3); plt.imshow(pred_mask[0].cpu().squeeze(), cmap='gray'); plt.title("Predicted Mask")
plt.show()

        
