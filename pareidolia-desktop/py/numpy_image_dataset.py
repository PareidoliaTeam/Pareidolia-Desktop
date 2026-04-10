# Cell 2: DataModule
import json
import os
import cv2
import numpy as np
import torch
import pytorch_lightning as pl
from PIL import Image
from torch.utils.data import DataLoader, Dataset, random_split
from torchvision import transforms, datasets

class NumpyImageDataset(Dataset):
    def __init__(self, images, labels, transform=None):
        self.images = images
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = Image.fromarray((self.images[idx] * 255).astype(np.uint8))
        label = int(self.labels[idx])
        if self.transform is not None:
            image = self.transform(image)
        return image, label