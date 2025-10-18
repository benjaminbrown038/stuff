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
