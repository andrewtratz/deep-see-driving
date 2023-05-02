'''
Samples Pcap files and finds mean and standard deviation of non-zero elements.
Returns average mean and standard deviation of sample.
'''
import numpy as np
import os
import random
from tqdm import tqdm


def depth_mean(npz):
	npz = np.load(npz)
	depth_map = npz['arr_0']
	depth_map_nonzero = depth_map[np.nonzero(depth_map)] # Get non-zero elements
	std = np.std(depth_map_nonzero)
	mean = np.mean(depth_map_nonzero)
	return std, mean

data_dir = r'../DeepSeeData/Processed/lidar/run7'

stdevs = []
means = []
sample = 1 # Out of 1

random.seed(1023)
for file in tqdm(os.listdir(data_dir)):
	if file[-4:] == '.npz':
		if random.random() >= (1-sample):
			npz_path = os.path.join(data_dir, file)
			std, mean = depth_mean(npz_path)
			stdevs.append(std)
			means.append(mean)
print(f'Average mean: {sum(means)/len(means)} | Average std: {sum(stdevs)/len(stdevs)}')
