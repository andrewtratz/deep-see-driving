import os
import cv2
from tqdm import tqdm

# Camera image preprocessing script

data_dir = r'D:\DeepSeeData\Raw'
output_dir = r'D:\DeepSeeData\Processed'

# Make output directory if needed

walk = os.walk(data_dir)
for entry in tqdm(walk):
    dir, subdir, files = entry

    for file in files:
        if file[-4:] == '.jpg':

            subdirectory_path = dir[len(data_dir):]

            # Fix timestamp offset
            timestamp_str = file[-17:-4]
            corrected_timestamp = str(int(float(timestamp_str) + 4 * 60 * 60 * 1000))

            if not os.path.exists(output_dir + subdirectory_path[:-1]):
                os.makedirs(output_dir + subdirectory_path[:-1])

            # Make corrections to the image
            img = cv2.imread(os.path.join(dir, file))
            img = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
            img = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            width = img.shape[1] // 2
            left = img[:, :width, :]
            right = img[:, width:, :]

            # Split and save out both left and right images
            filename_left = os.path.join(output_dir + subdirectory_path[:-1], file[:-17] + corrected_timestamp + '_left.jpg')
            filename_right = os.path.join(output_dir + subdirectory_path[:-1], file[:-17] + corrected_timestamp + '_right.jpg')

            cv2.imwrite(filename_left, left)
            cv2.imwrite(filename_right, right)