# to read in data in dataloader and apply different data augmentations
# script based on U-Net implementaion on https://github.com/aladdinpersson/Machine-Learning-Collection
import os
from torch.utils.data import Dataset
import xarray as xr
import torch
import random


# define class
class BIADataset(Dataset):
    def __init__(self, image_dir, target_dir, transform_p=0.0, normalize_elevation=False):
        self.image_dir = image_dir
        self.target_dir = target_dir
        self.transform_p = transform_p
        self.images = os.listdir(image_dir)
        self.normalize_elevation = normalize_elevation

        
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, index):
        img_path = os.path.join(self.image_dir, self.images[index])
        if self.target_dir is not None:
            target_path = os.path.join(self.target_dir, self.images[index])
        image = torch.tensor(xr.open_dataset(img_path)[['DEM_norm','radar_norm','modis_norm','MOD_B1_norm','MOD_B2_norm',
                                                        'MOD_B3_norm','MOD_B4_norm','MOD_B5_norm','MOD_B6_norm','MOD_B7_norm']].to_array().values).float()
        if self.target_dir is not None:
            target = torch.tensor(xr.open_dataset(target_path)['target'].values).float()
        lowerleft = self.images[index]
     
        if self.transform_p>0:
            if torch.rand(1) < self.transform_p:
                k = random.choice([0,1,2,3])
                image = torch.rot90(image, k=k, dims=[-2,-1])
                if self.target_dir is not None:
                    target = torch.rot90(target, k=k, dims=[-2,-1])

                if torch.rand(1) < 0.5:
                    image = torch.flip(image,dims=[-1,-2])
                    if self.target_dir is not None:
                        target = torch.flip(target,dims=[-1,-2])

                if torch.rand(1) < 0.5:
                    image = torch.flip(image,dims=[-2,-1])
                    if self.target_dir is not None:
                        target = torch.flip(target,dims=[-2,-1])
        
        if self.normalize_elevation==True:
            image[0] = image[0] - image[0].mean()

        if self.target_dir is None:
            return image, lowerleft
        
        else:
            return image, target, lowerleft