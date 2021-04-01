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

[W,H] = [2560,1440]
num_tiles = 8
[ts, t_overlap, fut] = [10,2,2]
radius = 135
intersection_threshold = 0.1


inpt_data_path = 'C:\\Users\\omossad\\Desktop\\recorded_samples\\fifa\\model_data\\filenames\\'
outp_data_path = 'C:\\Users\\omossad\\Desktop\\recorded_samples\\fifa\\encoding\\gt\\'
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


path, dirs, files = next(os.walk(inpt_data_path))
files.sort(key=lambda var:[int(x) if x.isdigit() else x for x in re.findall(r'[^0-9]|[0-9]+', var)])
file_count = len(files)

def rindex(lst, value):
    lst.reverse()
    i = lst.index(value)
    lst.reverse()
    return len(lst) - i - 1

def tile_to_enc(tgt_x, tgt_y,fname):
    print(tgt_x)
    print(tgt_y)
    x_tiles = np.where(tgt_x == 1)[0]
    y_tiles = np.where(tgt_y == 1)[0]
    start_x = x_tiles[0]
    end_x = x_tiles[-1]
    start_y = y_tiles[0]
    end_y = y_tiles[-1]
    tile_w = tile_h = 1/num_tiles
    top_x = start_x * tile_w
    top_y = start_y * tile_h
    w = (end_x - start_x + 1)* tile_w
    h = (end_y - start_y + 1)* tile_h
    f = open(fname,'w')
    f.write("0 " + str(top_x) + " " + str(top_y) + " " + str(w) + " " + str(h))
    f.close()


for fidx in range(file_count):
    #print('Processing : ' + files[fidx])
    f = open(inpt_data_path + files[fidx], "r")
    #tiles_array_x = np.zeros((num_tiles))
    #tiles_array_y = np.zeros((num_tiles))
    for s in range(ts-1):
        f.readline()
    curr_frame =int(f.readline().replace('frame_',''))
    for l in range(fut):
        targets = np.zeros((2, num_tiles))
        fixations = f.readline()
        #fixations = line.split(',')[1]
        fixations = fixations.replace('[','')
        fixations = fixations.replace(']','')
        x = float(fixations.split()[0])
        y = float(fixations.split()[1])
        #print(x)
        #print(y)
        #x1 = x*W - W/(num_tiles*2)
        #x2 = x*W + W/(num_tiles*2)
        #y1 = y*H - H/(num_tiles*2)
        #y2 = y*H + H/(num_tiles*2)
        x1 = x*W - radius
        x2 = x*W + radius
        y1 = y*H - radius
        y2 = y*H + radius
        #print('tile coor : ' + str(x1) + ' ' + str(x2) + ' ' + str(y1) + ' ' + str(y2))
        [arr_x, arr_y] = object_to_tile_intersection(x1,y1,x2,y2)
        #print(arr_x)
        #print(arr_y)
        #print(arr)
        for k in range(len(arr_x)):
            if arr_x[k] > intersection_threshold:
                #print(k)
                #tiles_array_x[X] = 1
                #tiles_array_y[Y] = 1
                targets[0][k] = 1
            if arr_y[k] > intersection_threshold:
                #print(k)
                targets[1][k] = 1
        filename = outp_data_path + 'roi' + str(curr_frame + l + 1 - 202) + '.txt'
        print(filename)
        tile_to_enc(targets[0], targets[1], filename)
