import math

##########
# Multi-sensor calibration settings
##########

# Focal length
fl = 2.8

# Define field of view
fov = 60

# Rotation transformation to apply
y_axis_rotation = math.radians(2)
x_axis_rotation = math.radians(1)

# Translation transformation to apply
x_translation = -0.05
y_translation = 0.14
z_translation = 0

# Define coordinates of principal point
px = 0
py = 0

# Should we auto-crop the bottom?
crop_bottom = 0

# Velodyne decoder settings
model_type = 'VLP-16' # VLP default settings will get overwritten by Puck Hi-Res configuration file
rpm = 600
calibration_file = r'Puck Hi-Res.yml'

# Fixed image dimensions
image_height = 480
image_width = 640
