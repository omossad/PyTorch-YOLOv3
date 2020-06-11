
from __future__ import print_function
import numpy as np
import csv
import pickle
from collections import namedtuple
Rectangle = namedtuple('Rectangle', 'xmin ymin xmax ymax')


########################################
### THIS FILE GENERATE TILE DATA     ###
### INTO GAZES & OBJECT OCCURENCES   ###
########################################
normalize=False
players = []
cfg_file = open('name_count', 'r') 
for line in cfg_file:
	tmp = line.split(' ')
	players.append([tmp[0], int(tmp[1])])
	 


W_ref = 1920.0
H_ref = 1080.0
W_tiles = 8.0
H_tiles  = 8.0
tile_width = W_ref / W_tiles
tile_height = H_ref / H_tiles
num_tiles = int(W_tiles * H_tiles)
distinct_obj = 2

def get_tile(x,y):
	x = x*W_ref
	y = y*H_ref
	x_cor = np.minimum(x, W_ref-1)
	y_cor = np.minimum(y, H_ref-1)
	x_dis = int(x_cor / tile_width)
	y_dis = int(y_cor / tile_height)
	tile = int(x_dis + y_dis * W_tiles)
	return tile

def get_tile_coor(tile_no):
	tile_y = np.floor(tile_no / W_tiles)
	tile_x = tile_no % W_tiles
	x_min = tile_x * tile_width/W_ref
	x_max = (tile_x + 1) * tile_width/W_ref
	y_min = tile_y * tile_height/H_ref
	y_max = (tile_y + 1) * tile_height/H_ref
	return [x_min, y_min, x_max, y_max]

def overlap_area(a, b):  # returns None if rectangles don't intersect
	dx = min(a.xmax, b.xmax) - max(a.xmin, b.xmin)
	dy = min(a.ymax, b.ymax) - max(a.ymin, b.ymin)
	if (dx>=0) and (dy>=0):
		return dx*dy





def process_gaze(filename, data):
	with open(filename, newline='') as csvfile:
		reader = csv.reader(csvfile, delimiter=',')
		for row in reader:
			temp = []
			data[row[0]] = [get_tile(float(row[1]), float(row[2])), temp]


def process_obj(filename, data):
        obj_count = []
        with open(filename, newline='') as csvfile:
                reader = csv.reader(csvfile, delimiter=',')
                for row in reader:
                        #s_tile = get_tile(float(row[2]), float(row[3]))
                        obj_type = int(row[1])
                        obj_x = float(row[2])
                        obj_y = float(row[3])
                        obj_w = float(row[4])
                        obj_h = float(row[5])
                        #if int(row[1]) == 32:
                        #       obj_type = 1
                        data[row[0]][1].append([obj_type, obj_x, obj_y, obj_w, obj_h])
                        obj_count.append(len(data[row[0]][1]))
                        #print(data[row[0]])
        print(np.amax(obj_count))





def process_obj(filename, data):
	obj_count = []
	with open(filename, newline='') as csvfile:
		reader = csv.reader(csvfile, delimiter=',')
		for row in reader:
			s_tile = get_tile(float(row[2]), float(row[3]))
			obj_type = int(row[1])
			obj_x = float(row[2])
			obj_y = float(row[3])
			obj_w = float(row[4])
			obj_h = float(row[5])
			data[row[0]][1].append([obj_type, obj_x, obj_y, obj_w, obj_h])
			obj_count.append(len(data[row[0]][1]))



counter = 0
for p in players:
	name = p[0]
	indices = [i for i in range(p[1]+1)]
	for i in indices:
		directory = '/home/omossad/scratch/Gaming-Dataset/processed/'+ name +'/fifa/'+ str(i) +'/'
		out_directory = '/home/omossad/projects/def-hefeeda/omossad/roi_detection/codes/raw_data/fifa/'
		data_file = {}
		csv_file = directory + 'gaze.txt'
		process_gaze(csv_file, data_file)
		csv_file = directory + 'label.txt'
		process_obj(csv_file, data_file)
		print('Processed ', ' ' + name + ' ' + str(i) + ' as ' + str(counter))
		pickle_out = open("../data/raw/data_" + str(counter) + ".pickle","wb")
		pickle.dump(data_file, pickle_out)
		pickle_out.close()
		counter = counter + 1
	#np.save('data_' + str(i+1) + '.npy', data_file) 
