'''
Camera_calibration.py takes in stereo images and resizes and recalibrates them 
according to calibration data obtained through StereoPi calibration (https://github.com/realizator/stereopi-fisheye-robot/blob/master/4_calibration_fisheye.py)

Rectified photos are outputted in a 'Rectified_photos' folder.
'''

# Imports
import cv2
import os
import numpy as np
import json


# Global variables preset
calibration_data = r'./calibration_data/stereo_camera_calibration.npz'
raw_image_folder = r'./Photos/'
rectified_image_folder = r'./Rectified_photos/'
photo_width = 1280
photo_height = 480
image_width = 320
image_height = 240
dim = (photo_width, photo_height)
image_size = (image_width,image_height)

def correct_image(image):
	image = cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
	image = cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
	image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
	return image


# Takes in unmodified StereoPi image, resizes it, uses calibration data to rectify it
def rectify_image(image):
	# Read image
	pair_img = cv2.imread(image)

	# Correct channels and rotation
	pair_img = correct_image(pair_img)

	# Resize image
	# pair_img_resized = cv2.resize(pair_img, dim, interpolation = cv2.INTER_AREA)
	pair_img_resized = cv2.resize(pair_img, (image_width*2, image_height), interpolation=cv2.INTER_AREA)

	# Read image and split it in a stereo pair
	imgLeft = pair_img_resized[0:image_height,0:image_width] #Y+H and X+W
	imgRight = pair_img_resized[0:image_height,image_width:] #Y+H and X+W

	# Implementing calibration data
	try:
		npzfile = np.load(calibration_data)
	except:
		print("Camera calibration data not found in cache, file " & './calibration_data/stereo_camera_calibration.npz')
		exit(0)

	imageSize = tuple(npzfile['imageSize'])
	leftMapX = npzfile['leftMapX']
	leftMapY = npzfile['leftMapY']
	rightMapX = npzfile['rightMapX']
	rightMapY = npzfile['rightMapY']

	width_left, height_left = imgLeft.shape[:2]
	width_right, height_right = imgRight.shape[:2]

	if 0 in [width_left, height_left, width_right, height_right]:
		print("Error: Can't remap image.")

	imgL = cv2.remap(imgLeft, leftMapX, leftMapY, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
	imgR = cv2.remap(imgRight, rightMapX, rightMapY, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)

	# Recombine images
	rectified_image = np.concatenate((imgL, imgR), axis=1)
	rectified_image = cv2.resize(rectified_image, (photo_width, photo_height), interpolation=cv2.INTER_AREA)

	# Return rectified image
	return rectified_image


def main():
	for image in os.listdir(raw_image_folder):
		img = os.path.join(raw_image_folder, image)
		print(f'Rectifying: {image}')
		rectified_image = rectify_image(img)

		save_path = os.path.join(rectified_image_folder, 'rectified_' + str(image))
		cv2.imwrite(save_path, rectified_image)


if __name__ == "__main__":
	main()