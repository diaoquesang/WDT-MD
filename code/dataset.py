from torch.utils.data import Dataset
import pandas as pd
import cv2 as cv
import os
from config import config
import numpy as np


class myTrainDataset(Dataset):  # Define dataset classes
    def __init__(self, filelist, image_dir, mask_dir,
                 transform1=None,
                 transform2=None):  # Incoming parameters (label path, image path, image preprocessing method, label preprocessing method)
        self.image_dir = image_dir  # Read the image path
        self.mask_dir = mask_dir  # Read the image path

        self.transform1 = transform1  # Read image pre-processing method
        self.transform2 = transform2  # Read image pre-processing method
        self.filelist = pd.read_csv(filelist, sep="\t", header=None)  # Read the list of filenames

    def __len__(self):
        return len(self.filelist)  # Read the number of filenames as the length of the dataset

    def __getitem__(self, idx):  # Remove data from the dataset
        file = self.filelist.iloc[idx, 0]  # Read the filename
        if os.path.exists(os.path.join(self.image_dir, file)):
            image = cv.imread(os.path.join(self.image_dir, file))
        else:
            raise ValueError("Image not found")
        if os.path.exists(os.path.join(self.mask_dir, file)):
            mask = cv.imread(os.path.join(self.mask_dir, file))
        else:
            mask = np.zeros_like(image)
        mask = cv.cvtColor(mask, cv.COLOR_BGR2GRAY)

        hsv_image = cv.cvtColor(image, cv.COLOR_BGR2HSV)
        h, s, v = cv.split(hsv_image)
        v_enhanced = v.copy()
        if os.path.exists(os.path.join(self.mask_dir, file)) and config.inpaint:
            v_enhanced = cv.inpaint(v_enhanced, mask, inpaintRadius=3, flags=cv.INPAINT_TELEA)

        if self.transform1:
            image = self.transform1(image)
            mask = self.transform1(mask)
            v = self.transform1(v)
            v_enhanced = self.transform1(v_enhanced)
        return image, mask, file, v, v_enhanced


class myTestDataset(Dataset):  # Define dataset classes
    def __init__(self, filelist, image_dir,
                 transform=None):  # Incoming parameters (label path, image path, image preprocessing method, label preprocessing method)
        self.image_dir = image_dir  # Read the image path

        self.transform = transform  # Read image pre-processing method
        self.filelist = pd.read_csv(filelist, sep="\t", header=None)  # Read the list of filenames

    def __len__(self):
        return len(self.filelist)  # Read the number of filenames as the length of the dataset

    def __getitem__(self, idx):  # Remove data from the dataset
        file = self.filelist.iloc[idx, 0]  # Read the filename
        if os.path.exists(os.path.join(self.image_dir, file)):
            image = cv.imread(os.path.join(self.image_dir, file))
        else:
            raise ValueError("Image not found")

        hsv_image = cv.cvtColor(image, cv.COLOR_BGR2HSV)
        h, s, v = cv.split(hsv_image)

        if self.transform:
            image = self.transform(image)
            v = self.transform(v)
        return image, file, v
