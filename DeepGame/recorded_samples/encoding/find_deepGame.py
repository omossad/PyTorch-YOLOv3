from __future__ import print_function
import numpy as np
import csv
import pickle
import math
import os
import torch
from shapely.geometry import Polygon
import sys, os
sys.path.append(os.path.abspath(os.path.join('..')))
import utils
import re


[W,H] = [2560,1440]
num_tiles = 8
[ts, t_overlap, fut] = [10,2,2]
radius = 135
intersection_threshold = 0.1


inpt_data_path = 'C:\\Users\\omossad\\Desktop\\recorded_samples\\fifa\\model_data\\filenames\\'
outp_data_path = 'C:\\Users\\omossad\\Desktop\\recorded_samples\\fifa\\encoding\\deepGame\\'


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


predicted_x = np.loadtxt('predicted_x.txt')
predicted_y = np.loadtxt('predicted_y.txt')

for fidx in range(file_count):
    #print('Processing : ' + files[fidx])
    f = open(inpt_data_path + files[fidx], "r")
    #tiles_array_x = np.zeros((num_tiles))
    #tiles_array_y = np.zeros((num_tiles))
    for s in range(ts-1):
        f.readline()
    curr_frame =int(f.readline().replace('frame_',''))
    for l in range(fut):
        filename = outp_data_path + 'roi' + str(curr_frame + l + 1 - 202) + '.txt'
        print(filename)
        tile_to_enc(predicted_x[fidx], predicted_y[fidx], filename)
