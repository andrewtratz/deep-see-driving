'''
Takes Velodyne PCap data as an input and converts to point cloud.
Sample data found here: https://data.kitware.com/#collection/5b7f46f98d777f06857cb206/folder/5b7fff608d777f06857cb539

Dependencies:
veldodyne decoder (pip install velodyne-decoder)
rosbag (pip install rosbag --extra-index-url https://rospypi.github.io/simple/)
'''

import velodyne_decoder as vd
import numpy as np
import math
import cv2

# Configuration parameters
model_type = 'VLP-16' # VLP default settings will get overwritten by Puck Hi-Res configuration file
rpm = 600
calibration_file = r'Puck Hi-Res.yml'

# Data input
pcap_file = r'velo2.pcap'

# Read first frame of the pcap
config = vd.Config(model=model_type, rpm=rpm, calibration_file=calibration_file)
pcap_file = pcap_file
cloud_arrays = []
x = vd.read_pcap(pcap_file, config)
for stamp, points in vd.read_pcap(pcap_file, config):
    cloud_arrays.append(points)
    break # Only get first set of data points

# Add distance as a column to the dataset
data = np.array(cloud_arrays[0])
dist = np.sqrt(data[:,0]**2 + data[:,1]**2 + data[:,2]**2)
data = np.hstack((data, np.expand_dims(dist, 1)))

# Function to filter out points which don't fall in a particular field of view (in degrees)
def filter_fov(points, fov_horiz):
    return points[math.radians(fov_horiz/2) < abs(np.arctan(points[:,0] / points[:,1]))]

# Filter all points behind the view
data = data[data[:,0] >= 0.0]

# Filter out FOV
data = filter_fov(data, 160)

# Reorder and reorient axes
newdata = np.hstack([np.expand_dims(dim, 1) for dim in (-data[:,2], -data[:,1], data[:,0], data[:,6])])

fl = 0.90 # Focal length
od = 1280 # Output image dimensions

# Camera matrix
C = np.array([[fl,0,0,0],
              [0,fl,0,0],
              [0,0,1,0]])

# Format data into Nx4 matrix of [X, Y, Z, 1]
vec = newdata[:,0:3].T
t = np.ones((1, vec.shape[1]))
vec = np.vstack((vec, t))

# Project into camera space
projected = np.matmul(C, vec).T

# Add the distance information back to the projected points
projected = np.hstack((projected, np.expand_dims(newdata[:,3], 1)))

# Resize to output size
projected[:,0:2] = np.round((projected[:,0:2] * (od / 2)) + (od/2), 0)

# Create a test image
img = np.zeros((od, od, 3), dtype=np.uint8)

# Populate the image with color-coded pixel values
for i in range(projected.shape[0]):
    row = projected[i]
    dist = row[3]

    if dist >= 0.75:
        color = 1
    elif dist <= 0.5:
        color = 0
    else:
        color = (dist - 0.5) / 0.25

    if row[0] > 0 and row[0] < od and row[1] > 0 and row[1] < od:
        img[int(row[0]), int(row[1]), 0] = (1 - color) * 255
        img[int(row[0]), int(row[1]), 2] = (color) * 255

# Display the test image
cv2.imshow("Test", img)
cv2.waitKey(0)


print(f'File: {str(pcap_file)} | Number of point cloud frames: {len(cloud_arrays)}')