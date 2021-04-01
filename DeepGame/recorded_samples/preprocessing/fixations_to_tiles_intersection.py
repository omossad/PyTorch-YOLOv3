

from __future__ import print_function
import numpy as np
import csv
import pickle
import math
import os
import torch
from shapely.geometry import Polygon
import sys, os
import re
########################################
### THIS FILE GENERATE FIXATIONS     ###
### FOR TS FRAMES AS DICTIONARY      ###
########################################
## VARIABLES ###
# input folder is where the selected data is located #
input_folder = 'C:\\Users\\omossad\\Desktop\\recorded_samples\\fifa\\model_data\\filenames\\'
output_folder =  'C:\\Users\\omossad\\Desktop\\recorded_samples\\fifa\\model_data\\tiles\\'

[W,H] = [2560,1440]
num_tiles = 8
[ts, t_overlap, fut] = [10,2,2]
radius = 135
intersection_threshold = 0.1


def object_to_tile_intersection(x1,y1,x2,y2):
	n_tiles = num_tiles
	arr_x = np.zeros((n_tiles))
	arr_y = np.zeros((n_tiles))
	tile_w = W/n_tiles
	tile_h = H/n_tiles
	object_poly = Polygon([(x1, y1), (x2, y1), (x2, y2), (x1, y2)])
	for i in range(n_tiles):
		tile_poly_x = Polygon([(i*tile_w, y1), ((i+1)*tile_w, y1), ((i+1)*tile_w, y2), (i*tile_w, y2)])
		tile_poly_y = Polygon([(x1, i*tile_h), (x2, i*tile_h), (x2, (i+1)*tile_h), (x1, (i+1)*tile_h)])
		intersection = object_poly.intersection(tile_poly_x)
		arr_x[i] = intersection.area/(tile_w*(y2-y1))
		intersection = object_poly.intersection(tile_poly_y)
		arr_y[i] = intersection.area/(tile_h*(x2-x1))
	return [arr_x, arr_y]




input_path = input_folder
path, dirs, files = next(os.walk(input_path))
output_path = output_folder

files.sort(key=lambda var:[int(x) if x.isdigit() else x for x in re.findall(r'[^0-9]|[0-9]+', var)])
file_count = len(files)
#targets = np.zeros((file_count, fut, 2, num_tiles))
targets = torch.zeros([file_count, fut, 2, num_tiles], dtype=torch.int32)

for fidx in range(file_count):
	#print('Processing : ' + files[fidx])
	f = open(input_path + files[fidx], "r")
	#tiles_array_x = np.zeros((num_tiles))
	#tiles_array_y = np.zeros((num_tiles))
	for s in range(ts):
		f.readline()
	for l in range(fut):
		fixations = f.readline()
		#fixations = line.split(',')[1]
		fixations = fixations.replace('[','')
		fixations = fixations.replace(']','')
		x = float(fixations.split()[0])
		y = float(fixations.split()[1])

		x1 = x*W - radius
		x2 = x*W + radius
		y1 = y*H - radius
		y2 = y*H + radius
		[arr_x, arr_y] = object_to_tile_intersection(x1,y1,x2,y2)

		for k in range(len(arr_x)):
			if arr_x[k] > intersection_threshold:
				targets[fidx][l][0][k] = 1
			if arr_y[k] > intersection_threshold:
				targets[fidx][l][1][k] = 1
	if sum(targets[fidx][l][0]) > 0:
		previous_arr_x = targets[fidx][l][0]
	else:
		targets[fidx][l][0] = previous_arr_x
		print('problem')
	if sum(arr_y) > 0:
		previous_arr_y = targets[fidx][l][1]
	else:
		targets[fidx][l][1] = previous_arr_y
		print('problem')

torch.save(targets, output_folder + 'fifa.pt')
