

from __future__ import print_function
import numpy as np
import csv
import pickle
import math
import os
from shapely.geometry import Polygon

########################################
### THIS FILE GENERATE FIXATIONS     ###
### FOR TS FRAMES AS DICTIONARY      ###
########################################
## VARIABLES ###
# input folder is where the selected data is located #
input_folder = 'C:\\Users\\omossad\\Desktop\\dataset\\model_data\\filenames\\'
output_folder =  'C:\\Users\\omossad\\Desktop\\dataset\\model_data\\tiled_labels\\'
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
with open('frames_info', 'r') as f:
    for line in f:
        num_files += 1
# number of files is the number of files to be processed #
num_files = num_files - 1
print("Total number of files is:", num_files)

### READ NUMBER OF FRAMES in each FILE ###
frame_time = np.zeros((num_files,1))
file_names = []

with open('frames_info') as csv_file:
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
	targets_x = np.zeros((file_count, num_tiles))
	targets_y = np.zeros((file_count, num_tiles))
	for fidx in range(file_count):
		f = open(input_path + files[fidx], "r")
		tiles_array_x = np.zeros((num_tiles))
		tiles_array_y = np.zeros((num_tiles))
		for l in range(ts):
			line = f.readline()
			fixations = line.split(',')[1]
			fixations = fixations.replace('[','')
			fixations = fixations.replace(']','')
			x = float(fixations.split()[0])
			y = float(fixations.split()[1])
			[X,Y] = fixation_to_tile(x,y)
			tiles_array_x[X] = 1
			tiles_array_y[Y] = 1
		targets_x[fidx] = tiles_array_x
		targets_y[fidx] = tiles_array_y
	np.savetxt(output_folder + file_names[i] + '_x.txt', targets_x)
	np.savetxt(output_folder + file_names[i] + '_y.txt', targets_y)

			#f.readline()
			#print(f.readline().split())
