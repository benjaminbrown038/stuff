# Key Components

# backbone 
# extract rich features


# neck 
# multi scale detection 


# detection heads
# allows localization 


# anchor boxes 



# loss functions


# Dataset And Loader Implementation
class DetectionDataset(torch.utils.data.Dataset):
    def __init__(self,image_paths, annotations, transforms=None):
        self.image_paths  = image_paths
        self.annotations = annotations
        self.transforms = transforms

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self,idx):
        img = read_image(self,image_paths[idx])
        boxes , class_ids = self.annotations[idx]['boxes'],self.annotations[idx]['classes']
        if self.transforms: 
            img, boxes = self.transforms(img,boxes)
        return img, boxes, class_ids


for images, boxes, classes in train_loader:
    preds = model(images)
    loss = compute_detection_loss(preds,boxes,classes)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

boxes, scores, labels = postprocess(preds,score_thresh=0.5,iou_threshold=0.45)


''' 
A Minimal Working Example
'''
import torch 
import torch.nn as nn 
import torchvision

backbone = torchvision.models.resnet50(pretrained=True)
backbone = nn.Sequential(*list(backbone.children())[:-2])

class SimpleFPN(nn.Module):
    def __init__(self,in_channels,out_channels=256):
        super().__init__()
        self.conv1 = nn.Conv2d
        self.conv1 = nn.Conv2d

    def forward(self,x):
        x = self.conv1(x)
        x = self.conv2(x)
        return x 

neck = SimpleFPN(in_channels=2048,out_channels=256)

class DetectionHead(nn.Module):
    def __init__(self,in_channels, num_anchors,num_classes):
        super().__init__()
        self.cls_conv
        self.reg_conv
        self.obj_conv
    
    def forward(self,x):
        cls_logits
        bbox_regs
        obj_logits
        return cls_logits, bbox_regs, obj_logits

num_anchors
num_classes
head = DetectionHead(in_channels,num_anchors 
