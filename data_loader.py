from torch.utils.data import Dataset
import pandas as pd
# import cv2
from PIL import Image
# from torchvision.io import read_image


class EgoObjectDataset(Dataset):
    def __init__(self, dataset_path, transform):
        self.data = pd.read_csv(dataset_path)
        self.anchor_images = self.data['anchor'].values
        self.positive_images = self.data['positive'].values
        self.negative_images = self.data['negative'].values
        self.transform = transform
    
    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, idx):
        anchor_image = Image.open(self.anchor_images[idx])
        positive_image = Image.open(self.positive_images[idx])
        negative_image = Image.open(self.negative_images[idx])
        
        anchor_image = self.transform(anchor_image)
        positive_image = self.transform(positive_image)
        negative_image = self.transform(negative_image)

        return anchor_image, positive_image, negative_image
        