from config import *
import torch
from torchvision import transforms

from dataset import KITTIDataset

# A list of transformations to apply to both the source data and ground truth
basic_trans = transforms.Compose([
    transforms.RandomCrop(CROP_SIZE),
    transforms.RandomVerticalFlip(0.5),
])

# A list of transformations to apply ONLY to the source data
source_trans = transforms.Compose([
    transforms.RandomApply(torch.nn.ModuleList([transforms.GaussianBlur(5, sigma=(0.05, 0.15)),
                                                transforms.ColorJitter()]))
])

# Load the KITTI dataset
kitti = KITTIDataset(r'C:\Users\andre\Dropbox\~DGMD E-17\Project\Datasets\KITTI\2011_09_26_drive_0001_sync',
                     r'C:\Users\andre\Dropbox\~DGMD E-17\Project\Datasets\KITTI\data_depth_annotated\train\2011_09_26_drive_0001_sync',
                     basic_trans, source_trans)

# Test
source, truth = kitti[0]
print(len(kitti))