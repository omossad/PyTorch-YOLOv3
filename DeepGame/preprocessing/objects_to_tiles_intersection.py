

from __future__ import print_function
import numpy as np
import csv
import pickle
import math
import os
from shapely.geometry import Polygon
import torch
import sys, os
sys.path.append(os.path.abspath(os.path.join('..')))
import utils
########################################
### THIS FILE GENERATE FIXATIONS     ###
### FOR TS FRAMES AS DICTIONARY      ###
########################################
## VARIABLES ###
# input folder is where the selected data is located #
input_folder = 'C:\\Users\\omossad\\Desktop\\dataset\\model_data\\filenames\\'
objects_folder = 'C:\\Users\\omossad\\Desktop\\dataset\\model_data\\objects\\'
output_folder =  'C:\\Users\\omossad\\Desktop\\dataset\\model_data\\tiled_objects_intersection\\'

[W,H] = utils.get_img_dim()
num_tiles = utils.get_num_tiles()
[ts, t_overlap, fut] = utils.get_model_conf()

### READ NUMBER OF FILES and NAMES ###
num_files = utils.get_no_files()
file_names = utils.get_files_list(num_files)


for i in range(num_files):
	print(file_names[i])
	input_path = input_folder + file_names[i] + '\\'
	path, dirs, files = next(os.walk(input_path))
	output_path = output_folder + file_names[i] + '\\'
	#try:
	#	os.mkdir(output_path)
	#except:
	#	print('directories already exist')
	file_count = len(files)
	objects_arr = torch.zeros([file_count, ts, 2, num_tiles, 3], dtype=torch.float)
	#objects_arr_y = torch.zeros([file_count, ts, 2, num_tiles, 3], dtype=torch.int32)
	for fidx in range(file_count):
		f = open(input_path + files[fidx], "r")
		for l in range(ts):
			frame_name = f.readline()
			frame_name = frame_name.replace('\n','')
			#frame_name = line.split(',')[0]
			temp = torch.load(objects_folder + file_names[i] + '\\' + frame_name + '.pt')
			for obj in range(len(temp)):
				#print(temp)
				x1 = temp[obj][0]
				y1 = temp[obj][1]
				x2 = temp[obj][2]
				y2 = temp[obj][3]
				arr = utils.object_to_tile_intersection(x1,y1,x2,y2)
				#print('intersection array for : ' + str(l))
				#print(arr)
				for t in range(num_tiles):
					objects_arr[fidx][l][0][t][int(temp[obj][6])] += arr[0][t]
					objects_arr[fidx][l][1][t][int(temp[obj][6])] += arr[1][t]
					#print('Object inter : ' + str(int(temp[obj][6])) + ' ' + str(arr[0][t]) + ' ' + str( arr[1][t]))
					#print(objects_arr[fidx][l][0])
	torch.save(objects_arr, output_folder + file_names[i] + '.pt')
				#objects_arr_x[fidx][l][X][int(temp[obj][6])] += 1
				#objects_arr_y[fidx][l][Y][int(temp[obj][6])] += 1
	#torch.save(objects_arr_x, output_folder + file_names[i] + '_x.pt')
	#torch.save(objects_arr_y, output_folder + file_names[i] + '_y.pt')
	#print(objects_arr_x)
