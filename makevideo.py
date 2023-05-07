from config import *
import os
import numpy as np
from PIL import Image
from tqdm import tqdm
import cv2

base_path = r'D:\KITTI\2011_09_26_drive_0005_sync_2\2011_09_26\2011_09_26_drive_0005_sync\image_02\data'
#base_path = r'D:\run3\camera\run3'
depth_path = r'D:\out'

img_list = []

print('Combining files')
walk = os.walk(base_path)
for entry in walk:
    dir, subdir, files = entry
    for file in files:
        if '.png' in file or '_left':
            base = np.asarray(Image.open(os.path.join(base_path, file)))
            if os.path.exists(os.path.join(depth_path, file)):
                depth = cv2.GaussianBlur(np.asarray(Image.open(os.path.join(depth_path, file))),
                                         (5, 5), cv2.BORDER_DEFAULT)

                mixed = ((base / 2) + (depth / 3)).astype(np.uint8)
                size = (base.shape[1] * 3 // 4, base.shape[0] * 9 // 4)
                out = cv2.resize(cv2.cvtColor(np.concatenate((base, mixed, depth), axis=0), cv2.COLOR_BGR2RGB),
                                 size, cv2.INTER_AREA)

                img_list.append(out)
                # cv2.imshow("Video Test", out)
                # cv2.waitKey(0)

print('Writing video...')
vid = cv2.VideoWriter(r'D:\project0005_v2.avi', cv2.VideoWriter_fourcc(*'DIVX'), 10, size)
for i in range(len(img_list)):
    vid.write(img_list[i])
vid.release()
