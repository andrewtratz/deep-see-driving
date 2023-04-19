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

# Focal length
fl = 1.15

# Define field of view
fov = 70

# x_axis_rotation = math.radians(0)
# y_axis_rotation = math.radians(0)

y_axis_rotation = math.radians(-3)
x_axis_rotation = math.radians(-10)

x_translation = 0.35
y_translation = 0.0
z_translation = 0

# Define coordinates of principal point
px = 0
py = 0

crop_bottom = 0

# Load camera image
cam = cv2.imread("cups_photo.jpg")
cam = cv2.rotate(cam, cv2.ROTATE_90_CLOCKWISE)
cam = cv2.rotate(cam, cv2.ROTATE_90_CLOCKWISE)
cam = cv2.cvtColor(cam, cv2.COLOR_BGR2RGB)
cam = cv2.cvtColor(cam, cv2.COLOR_BGR2GRAY)
cam = cam // 10
cam = cv2.cvtColor(cam, cv2.COLOR_GRAY2RGB)

# Extract left image only
cam = cam[:, :cam.shape[1]//2, :]

# Crop bottom of image
cam = cam[crop_bottom:, :, :]

h, w, c = cam.shape

# cv2.imshow("Cam", cam)
# cv2.waitKey(0)

# Configuration parameters
model_type = 'VLP-16' # VLP default settings will get overwritten by Puck Hi-Res configuration file
rpm = 600
calibration_file = r'Puck Hi-Res.yml'

# Data input
pcap_file = r'velo.pcap'

# Read first frame of the pcap
config = vd.Config(model=model_type, rpm=rpm, calibration_file=calibration_file)
config.min_angle = 360 - fov/2
config.max_angle = fov/2
pcap_file = pcap_file
cloud_arrays = []

i = 0
for stamp, points in vd.read_pcap(pcap_file, config):
    i += 1
    cloud_arrays.append(points)
    if i == 10:
        break # Only get first set of data points

# Add distance as a column to the dataset
data = np.array(cloud_arrays[i-1])
dist = np.sqrt(data[:,0]**2 + data[:,1]**2 + data[:,2]**2)
data = np.hstack((data, np.expand_dims(dist, 1)))

# Filter all points behind the view
data = data[data[:,0] >= 0.0]

# Reorder and reorient axes
newdata = np.hstack([np.expand_dims(dim, 1) for dim in (-data[:,2], -data[:,1], data[:,0], data[:,6])])


translation_matrix = np.array([[1, 0, 0, x_translation],
                               [0, 1, 0, y_translation],
                               [0, 0, 1, z_translation],
                               [0, 0, 0, 1]])

y_axis_rotation_matrix = np.array([[1, 0, 0, 0],
                                   [0, math.cos(y_axis_rotation), -math.sin(y_axis_rotation), 0],
                                   [0, math.sin(y_axis_rotation), math.cos(y_axis_rotation), 0],
                                   [0, 0, 0, 1]])

x_axis_rotation_matrix = np.array([[math.cos(x_axis_rotation), 0, math.sin(x_axis_rotation), 0],
                                   [0, 1, 0, 0],
                                   [0, -math.sin(x_axis_rotation), math.cos(x_axis_rotation), 0],
                                   [0, 0, 0, 1]])

# Camera matrix
C = np.array([[fl,0,py,0],
              [0,fl,px,0],
              [0,0,1,0]])

# Format data into Nx4 matrix of [X, Y, Z, 1]
vec = newdata[:,0:3].T
t = np.ones((1, vec.shape[1]))
vec = np.vstack((vec, t))

# Rotation
vec = np.matmul(x_axis_rotation_matrix, vec)
vec = np.matmul(y_axis_rotation_matrix, vec)

# Translation
vec = np.matmul(translation_matrix, vec)


# Project into camera space
projected = np.matmul(C, vec)

# Transpose
projected = projected.T

# Add the distance information back to the projected points
projected = np.hstack((projected, np.expand_dims(newdata[:,3], 1)))

# Resize to output size
projected[:,0] = np.round((projected[:,0] * (h / 2)) + (h/2), 0)
projected[:,1] = np.round((projected[:,1] * (w / 2)) + (w/2), 0)

# Create a test image
# img = np.zeros((h, w, 3), dtype=np.uint8)
img = cam.copy()

# Populate the image with color-coded pixel values
for i in range(projected.shape[0]):
    row = projected[i]
    dist = row[3]

    max_clip = 2.5
    min_clip = 0.75

    if dist >= max_clip:
        color = 1
    elif dist <= min_clip:
        color = 0
    else:
        color = (dist - min_clip) / (max_clip - min_clip)

    if row[0] > 0 and row[0] < h and row[1] > 0 and row[1] < w:
        img[int(row[0]), int(row[1]), 0] = (1 - color) * 255
        img[int(row[0]), int(row[1]), 2] = (color) * 255

# Display the test image
cv2.imshow("Test", img)
cv2.imwrite('output.jpg', img)
cv2.waitKey(0)



print(f'File: {str(pcap_file)} | Number of point cloud frames: {len(cloud_arrays)}')