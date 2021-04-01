

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
import re
########################################
### THIS FILE GENERATE FIXATIONS     ###
### FOR TS FRAMES AS DICTIONARY      ###
########################################
## VARIABLES ###
# input folder is where the selected data is located #
game='fifa'
num_objects = 3
input_folder = 'C:\\Users\\omossad\\Desktop\\dataset\\model_data\\' + game + '\\filenames\\'
objects_folder = 'C:\\Users\\omossad\\Desktop\\dataset\\model_data\\' + game + '\\objects\\'
output_folder =  'C:\\Users\\omossad\\Desktop\\dataset\\model_data\\' + game + '\\tiled_objects\\'

[W,H] = utils.get_img_dim()
num_tiles = utils.get_num_tiles()
[ts, t_overlap, fut] = utils.get_model_conf()

### READ NUMBER OF FILES and NAMES ###
num_files = utils.get_no_files(game)
print(num_files)
file_names = utils.get_files_list(num_files, game)
print(file_names)

for i in range(num_files):
	print(file_names[i])
	input_path = input_folder + file_names[i] + '\\'
	path, dirs, files = next(os.walk(input_path))
	output_path = output_folder + file_names[i] + '\\'
	#try:
	#	os.mkdir(output_path)
	#except:
	#	print('directories already exist')



	#print(files)
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
			temp = torch.load(objects_folder + file_names[i] + '\\' + frame_name + '.pt')
			#print(frame_name)
			if temp is None:
				continue
			#print(temp)
			for obj in range(len(temp)):
				x1 = temp[obj][0]/W
				y1 = temp[obj][1]/H
				x2 = temp[obj][2]/W
				y2 = temp[obj][3]/H
				[X,Y] = utils.fixation_to_tile((x1+x2)/2.0,(y1+y2)/2.0)
				objects_arr[fidx][l][0][X][int(temp[obj][6])] += float(temp[obj][4])
				objects_arr[fidx][l][1][Y][int(temp[obj][6])] += float(temp[obj][4])
	#print(objects_arr)
	torch.save(objects_arr, output_folder + file_names[i] + '.pt')
				#objects_arr[fidx][l][X][int(temp[obj][6])] += 1
				#objects_arr_y[fidx][l][Y][int(temp[obj][6])] += 1
	#torch.save(objects_arr, output_folder + file_names[i] + '_x.pt')
	#torch.save(objects_arr_y, output_folder + file_names[i] + '_y.pt')
	#print(objects_arr)
