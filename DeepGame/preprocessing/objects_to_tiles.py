

from __future__ import print_function
import numpy as np
import csv
import pickle
import math
import os
from shapely.geometry import Polygon
import torch
########################################
### THIS FILE GENERATE FIXATIONS     ###
### FOR TS FRAMES AS DICTIONARY      ###
########################################
## VARIABLES ###
# input folder is where the selected data is located #
input_folder = 'C:\\Users\\omossad\\Desktop\\dataset\\model_data\\filenames\\'
objects_folder = 'C:\\Users\\omossad\\Desktop\\dataset\\model_data\\objects\\'
output_folder =  'C:\\Users\\omossad\\Desktop\\dataset\\model_data\\tiled_objects\\'
W = 1920
H = 1080
num_tiles = 8
ts = 10


def fixation_to_tile(x,y):
	#X = x*W
	#Y = y*H
	#tile_width  = W/num_tiles
	#tile_height = H/num_tiles
	X = min(num_tiles - 1, x * num_tiles)
	Y = min(num_tiles - 1, y * num_tiles)
	return [int(X), int(Y)]


### READ NUMBER OF FILES and NAMES ###
num_files = 0
with open('..\\frames_info', 'r') as f:
    for line in f:
        num_files += 1
# number of files is the number of files to be processed #
num_files = num_files - 1
print("Total number of files is:", num_files)

### READ NUMBER OF FRAMES in each FILE ###
frame_time = np.zeros((num_files,1))
file_names = []

with open('..\\frames_info') as csv_file:
	csv_reader = csv.reader(csv_file, delimiter=',')
	line_count = 0
	for row in csv_reader:
		if line_count == 0:
			line_count += 1
		elif line_count < num_files+1:
			file_names.append(row[0])
			frame_time[line_count-1] = int(row[5])
			line_count += 1
		else:
			break
print('Files read in order are')
print(file_names)


for i in range(num_files):
	input_path = input_folder + file_names[i] + '\\'
	path, dirs, files = next(os.walk(input_path))
	output_path = output_folder + file_names[i] + '\\'
	#try:
	#	os.mkdir(output_path)
	#except:
	#	print('directories already exist')
	file_count = len(files)
	objects_arr_x = torch.zeros([file_count, ts, num_tiles, 3], dtype=torch.int32)
	objects_arr_y = torch.zeros([file_count, ts, num_tiles, 3], dtype=torch.int32)
	for fidx in range(file_count):
		f = open(input_path + files[fidx], "r")
		for l in range(ts):
			line = f.readline()
			frame_name = line.split(',')[0]
			temp = torch.load(objects_folder + file_names[i] + '\\' + frame_name + '.pt')
			for obj in range(len(temp)):
				x = temp[obj][0]/W
				y = temp[obj][1]/H
				[X,Y] = fixation_to_tile(x,y)
				objects_arr_x[fidx][l][X][int(temp[obj][6])] += 1
				objects_arr_y[fidx][l][Y][int(temp[obj][6])] += 1
	torch.save(objects_arr_x, output_folder + file_names[i] + '_x.pt')
	torch.save(objects_arr_y, output_folder + file_names[i] + '_y.pt')
	#print(objects_arr_x)
