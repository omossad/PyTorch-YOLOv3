

from __future__ import print_function
import numpy as np
import csv
import pickle
import math
import os
from shapely.geometry import Polygon
import torch
import sys, os
import re
########################################
### THIS FILE GENERATE FIXATIONS     ###
### FOR TS FRAMES AS DICTIONARY      ###
########################################
## VARIABLES ###
# input folder is where the selected data is located #
game='fifa'
num_objects = 3
input_folder = 'C:\\Users\\omossad\\Desktop\\recorded_samples\\fifa\\model_data\\filenames\\'
objects_folder = 'C:\\Users\\omossad\\Desktop\\recorded_samples\\fifa\\model_data\\objects\\'
output_folder =  'C:\\Users\\omossad\\Desktop\\recorded_samples\\fifa\\model_data\\tiled_objects\\'


[W,H] = [2560,1440]
num_tiles = 8
[ts, t_overlap, fut] = [10,2,2]
radius = 135
intersection_threshold = 0.1

def fixation_to_tile(x,y):
	n_tiles = num_tiles
	X = min(n_tiles - 1, x * n_tiles)
	Y = min(n_tiles - 1, y * n_tiles)
	return [int(X), int(Y)]

input_path = input_folder
path, dirs, files = next(os.walk(input_path))
output_path = output_folder

files.sort(key=lambda var:[int(x) if x.isdigit() else x for x in re.findall(r'[^0-9]|[0-9]+', var)])
file_count = len(files)
objects_arr = torch.zeros([file_count, ts, 2, num_tiles, num_objects], dtype=torch.float)
#objects_arr_y = torch.zeros([file_count, ts, 2, num_tiles, 3], dtype=torch.int32)
for fidx in range(file_count):
	f = open(input_path + files[fidx], "r")
	#print(files[fidx])
	for l in range(ts):
		frame_name = f.readline()
		frame_name = frame_name.replace('\n','')
		#frame_name = line.split(',')[0]
		temp = torch.load(objects_folder + frame_name + '.pt')
		#print(frame_name)
		if temp is None:
			continue
		#print(temp)
		for obj in range(len(temp)):
			x1 = temp[obj][0]/W
			y1 = temp[obj][1]/H
			x2 = temp[obj][2]/W
			y2 = temp[obj][3]/H
			[X,Y] = fixation_to_tile((x1+x2)/2.0,(y1+y2)/2.0)
			objects_arr[fidx][l][0][X][int(temp[obj][6])] += float(temp[obj][4])
			objects_arr[fidx][l][1][Y][int(temp[obj][6])] += float(temp[obj][4])
#print(objects_arr)
torch.save(objects_arr, output_folder + 'fifa.pt')
