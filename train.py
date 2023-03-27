from config import *
import os
import torch
import torch.nn as nn
from torchvision import transforms
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from backbones import ResNetLike

from dataset import KITTIDataset

if torch.cuda.is_available():
    device = torch.device("cuda")
    print("Running on GPU")
else:
    device = torch.device("cpu")
    print("Running on CPU")


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

# List of train and validation data (this split is provided by the KITTI dataset creators)
train_folders = os.listdir(r'C:\Users\andre\Dropbox\~DGMD E-17\Project\Datasets\KITTI\data_depth_annotated\train')
val_folders = os.listdir(r'C:\Users\andre\Dropbox\~DGMD E-17\Project\Datasets\KITTI\data_depth_annotated\val')

# Debug code
# train_folders = [train_folders[0]]
# val_folders = [val_folders[0]]

# Load the KITTI dataset
kitti = KITTIDataset(r'D:\KITTI', r'C:\Users\andre\Dropbox\~DGMD E-17\Project\Datasets\KITTI\data_depth_annotated\train', train_folders,
                     basic_trans, source_trans)

kitti_cv = KITTIDataset(r'D:\KITTI', r'C:\Users\andre\Dropbox\~DGMD E-17\Project\Datasets\KITTI\data_depth_annotated\val', val_folders,
                     basic_trans, source_trans)

# DataLoader objects for the two datasets
train_loader = DataLoader(kitti, batch_size=BATCH_SIZE, shuffle=True)
cv_loader = DataLoader(kitti_cv, batch_size=BATCH_SIZE, shuffle=False)

# Create a TensorBoard logging writer
writer = SummaryWriter()

# Instantiate our model and optimizer
epochs = 20
model = ResNetLike().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)

# Our loss function - using mean squared error over all pixels where we have LiDAR depth data
def compute_loss(output, truth, valid_mask):
    loss = nn.MSELoss()
    q = output[valid_mask]
    r = truth[valid_mask]
    return loss(q, r)

# Run a big loop multiple times throughout the entire dataset
iter = 0
for epoch in range(epochs):

    # Training loop
    for i, (source, truth, valid_mask) in enumerate(train_loader):
        iter += 1

        # Run the model on the data and get its outputs
        outputs = model(source.to(device))

        # Compute the loss function for the output data
        loss = compute_loss(outputs, truth.to(device), valid_mask.to(device))

        # Print out loss term every 10 batches
        if iter % 10 == 0:
            loss_copy = loss.item()
            print('Epoch: %d\tBatch: %d\tLoss: %.2f' % (epoch, i+1, loss_copy))
            writer.add_scalar('loss/train', loss_copy, iter)

        # Backward gradient propagation to learn and improve the models
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    # Cross validation loop - the same, but we are testing, not learning
    losses = []
    for i, (source, truth, valid_mask) in enumerate(cv_loader):
        # torch.no_grad makes it so the model won't learn anything new
        with torch.no_grad():
            outputs = model(source.to(device))
            loss = compute_loss(outputs, truth.to(device), valid_mask.to(device))
            losses.append(loss.item())

    # Compute the average loss over the validation set and print it
    avg_loss = sum(losses) / len(losses)

    print('-------------------------------')
    print('Epoch: %d\tLoss: %.2f' % (epoch, avg_loss))
    print('-------------------------------\n')
    writer.add_scalar('loss/val', avg_loss, iter)

    # Save an intermediate copy of the model weights
    torch.save(model.state_dict(), 'model' + str(epoch) + '.pth')
    torch.save(optimizer.state_dict(), 'optimizer' + str(epoch) + '.pth')



