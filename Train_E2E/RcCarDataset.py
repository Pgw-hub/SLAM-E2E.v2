"""
Self-driving car image pair Dataset.

@author: Zhenye Na - https://github.com/Zhenye-Na
@reference: "End to End Learning for Self-Driving Cars", arXiv:1604.07316
"""

from torch.utils import data

import cv2
import numpy as np
import os.path
# # use skimage if you do not have cv2 installed
# from skimage import io


def augment(dataroot, imgName, angle,velocity, test):
    """Data augmentation."""
    #name = dataroot + 'IMG/' + imgName.split('\\')[-1]
    name = imgName
    current_image = cv2.imread(name)
    if current_image is None:
        print("%s is not in the dataset" %name)
    current_image = current_image[65:-25, :, :]
    
    if test == False:
        if np.random.rand() < 0.5:
            current_image = cv2.flip(current_image,1)
            angle = angle * -1.0
    return current_image, angle, velocity

class TripletDataset(data.Dataset):
    """Image pair dataset."""

    def __init__(self, dataroot, samples, transform=None, test=False):
        """Initialization."""
        self.samples = samples
        self.dataroot = dataroot
        self.transform = transform
        self.test = test

    def __getitem__(self, index):
        """Get image."""
        batch_samples  = self.samples[index]
        steering_angle = float(batch_samples[1])
        velocity = float(batch_samples[2])

        center_img, steering_angle_center, velocity = augment(self.dataroot, batch_samples[0], steering_angle, velocity, self.test)  
        center_img = self.transform(center_img)

        return (center_img, steering_angle_center,velocity)

    def __len__(self):
        """Length of dataset."""
        return len(self.samples)
