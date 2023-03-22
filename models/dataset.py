import os
import time

import numpy as np
import cv2 as cv
import pandas as pd
from importlib import reload
import torch as tc
from torchvision.io import read_image
from torchvision.transforms import ToTensor, Compose, RandomResizedCrop
from torch.utils.data import Dataset
from PIL import Image


class OriginalImageDataset(Dataset):
    def __init__(self, img_dir, transform=None, target_transform=None):
        file_list = []
        for file in os.listdir(img_dir):
            if file.endswith(('.jpg', '.jpeg', '.png')):
                file_list.append(os.path.join(img_dir, file))
        assert len(file_list) > 0, 'img_dir:{} has no images'.format(img_dir)
        self.img_dir = img_dir
        self.file_list = file_list
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        image = Image.open(self.file_list[idx])
        # image = read_image(self.file_list[idx])
        if self.transform:
            image = self.transform(image)
        return image

