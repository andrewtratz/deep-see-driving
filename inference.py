from config import *
import os
import torch
import torch.nn as nn
import numpy as np
from torchvision import transforms
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from backbones import ResNetLike
from PIL import Image
from tqdm import tqdm

from dataset import KITTIDataset

if torch.cuda.is_available():
    device = torch.device("cuda")
    print("Running on GPU")
else:
    device = torch.device("cpu")
    print("Running on CPU")

# Load the source image and crop
left_image_path = r'D:\KITTI\2011_09_26_drive_0113_sync_2\2011_09_26\2011_09_26_drive_0113_sync\image_02\data\0000000018.png'
right_image_path = left_image_path.replace('image_02', 'image_03')
depth_path = r'C:\Users\andre\Dropbox\~DGMD E-17\Project\Datasets\KITTI\data_depth_annotated\val\2011_09_26_drive_0113_sync\proj_depth\groundtruth\image_02\0000000018.png'

crop_pattern = (0, 120, 1242, 375)
left_image = Image.open(left_image_path).crop(crop_pattern)
right_image = Image.open(right_image_path).crop(crop_pattern)
depth_image = Image.open(depth_path).crop(crop_pattern)
valid_mask = np.asarray(depth_image) > 0

# Create overlapping patchwork of different crops of the image
def create_patchwork(left_image, right_image):
    w, h = left_image.size
    crop_ul = [] # List of upper-left points of each patch

    # Create set of overlapping patch upper-left points
    for y in range(0, h - CROP_SIZE, 20):
        for x in range(0, w - CROP_SIZE, 20):
            crop_ul.append((y, x))
        crop_ul.append((y, w - CROP_SIZE))
    for x in range(0, w - CROP_SIZE, 20):
        crop_ul.append((h - CROP_SIZE, x))
    crop_ul.append((h - CROP_SIZE, w - CROP_SIZE))
    crop_ul = list(set(crop_ul)) # Make unique

    # Set up return data structures
    patch_count = len(crop_ul)
    patches = torch.from_numpy(np.ndarray((patch_count, 6, CROP_SIZE, CROP_SIZE), dtype=np.float32))
    overlaps = np.zeros((h, w), dtype=np.uint8)

    # Create a transform function to convert into pytorch tensors
    tensor_conv = transforms.ToTensor()

    # Create the individual patches
    for patch_id, ul in zip(range(len(patches)), crop_ul):
        y, x = ul
        crop = (x, y, x+CROP_SIZE, y+CROP_SIZE)
        patches[patch_id, :3] = tensor_conv(left_image.crop(crop))
        patches[patch_id, 3:] = tensor_conv(right_image.crop(crop))
        overlaps[y:y+CROP_SIZE, x:x+CROP_SIZE] = np.add(overlaps[y:y+CROP_SIZE, x:x+CROP_SIZE], 1)

    return patches, crop_ul, overlaps

# Instantiate our model
model = ResNetLike().to(device)
model.load_state_dict(torch.load(r'./models/model19.pth'))
model.eval()

patches, crop_ul, overlaps = create_patchwork(left_image, right_image)

w, h = left_image.size
output_data = np.zeros((h, w), dtype=np.float32)

# Do individual patch inference
for patch_id, ul in tqdm(zip(range(patches.shape[0]), crop_ul)):
    y, x = ul
    with torch.no_grad():
        out = model(patches[patch_id].unsqueeze(0).to(device))
    output_data[y:y+CROP_SIZE, x:x+CROP_SIZE] += out.squeeze(0).cpu().numpy()

# Divide by the number of overlapping patches per pixel to get average output
output_data = np.divide(output_data, overlaps)

# Convert the output from normalized form into actual output
output_data *= 3186
output_data += 4582
output_data = output_data.astype(np.int32)

print('Done')



