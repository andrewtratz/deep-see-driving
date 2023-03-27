import os
from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision import transforms

# pyTorch Dataset for the KITTI autonomous driving data
# Will recursively search img_dir and depth_dir to build up the dataset and apply specified transformations
class KITTIDataset(Dataset):

    # Initialization with directories and transforms specified
    def __init__(self, img_root_dir, depth_root_dir, img_dirs, transform=None, source_additional_transform=None):
        self.img_root_dir = img_root_dir
        self.depth_root_dir = depth_root_dir
        self.img_dirs = img_dirs
        self.transform = transform
        self.source_additional_transform = source_additional_transform

        # Populate the list of file paths and depth paths
        self.file_paths = []
        self.depth_paths = []

        for img_dir in img_dirs:
            walk = os.walk(os.path.join(self.img_root_dir, img_dir + '_2'))
            for entry in walk:
                dir, subdir, files = entry
                if 'image_02' in dir:
                    for file in files:
                        if file[-4:] == '.png':
                            # First several frames of each video have no ground truth data, skip them
                            if file.split('.')[0] in ['0000000000', '0000000001', '0000000002', '0000000003', '0000000004']:
                                continue
                            self.file_paths.append(dir + '\\' + file)

        # Populate the list of depth paths (ground truth labels)

            walk = os.walk(os.path.join(self.depth_root_dir, img_dir))
            for entry in walk:
                dir, subdir, files = entry
                if 'image_02' in dir:
                    for file in files:
                        if file[-4:] == '.png':
                            self.depth_paths.append(dir + '\\' + file)
                            # Get rid of file paths which don't exist in the depth ground truth
                            # assert(self.depth_paths[-1].split('\\')[-1] == self.file_paths[len(self.depth_paths)-1].split('\\')[-1])
                            while self.depth_paths[-1].split('\\')[-1] != self.file_paths[len(self.depth_paths)-1].split('\\')[-1]:
                                print('Discarding ' + self.file_paths[len(self.depth_paths)-1])
                                self.file_paths.remove(self.file_paths[len(self.depth_paths)-1])

            # Truncate file paths, since some of the final depths may not exist
            self.file_paths = self.file_paths[0:len(self.depth_paths)]


    # Return the total length of the Dataset
    def __len__(self):
        return len(self.file_paths)

    # Retrieve a single item (defined as img*2 and depth label) from the Dataset
    def __getitem__(self, idx):

        # Treat the original path as left, get revised path for right image
        left_image_path = self.file_paths[idx]
        right_image_path = left_image_path.replace('image_02', 'image_03')
        depth_path = self.depth_paths[idx]

        # Apply a crop to the data since the LiDAR scans don't provide info for the top third of the image
        crop_pattern = (0, 120, 1242, 375)
        left_image = Image.open(left_image_path).crop(crop_pattern)
        right_image = Image.open(right_image_path).crop(crop_pattern)
        depth_image = Image.open(depth_path).crop(crop_pattern)

        # Debugging code to show comparative frames
        # dst = Image.new('RGB', (left_image.width, left_image.height + left_image.height))
        # dst.paste(left_image, (0, 0))
        # dst.paste(depth_image, (0, left_image.height))
        # dst.show()

        # Apply transform augmentations to the data

        # Convert images into pyTorch tensor data format, which will be used for analysis
        tensor_conv = transforms.ToTensor()
        left_image = tensor_conv(left_image)
        right_image = tensor_conv(right_image)
        depth_image = tensor_conv(depth_image).to(torch.float32).squeeze(0)

        # Create a Boolean mask of locations where depth information is provided
        valid_mask = depth_image > 0

        # Normalize the depth image
        depth_image = torch.div(torch.subtract(depth_image, 4582.0), 3186) # Subtract mean and divide by std

        # Stack the data into a single tensor so we apply the same random augmentations to everything
        full_data = torch.vstack((left_image, right_image, depth_image.unsqueeze(0),
                                  valid_mask.to(dtype=torch.uint8).unsqueeze(0)))

        # Apply the data augmentations
        if self.transform:
            full_data = self.transform(full_data)

        # Split the data back into our source data and ground truth
        source_data = full_data[0:full_data.shape[0]-2]
        ground_truth = full_data[-2:-1].squeeze(0)
        valid_mask = full_data[-1:].squeeze(0).to(dtype=torch.bool)

        # Apply additional augmentations only to the source data (things we don't want to affect ground truth)
        if self.source_additional_transform:
            source_data = self.source_additional_transform(source_data)

        return source_data, ground_truth, valid_mask



