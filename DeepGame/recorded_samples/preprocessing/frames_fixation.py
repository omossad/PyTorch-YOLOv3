

from __future__ import print_function
import numpy as np
import csv
import pickle
import math
import os
import sys, os

########################################
### THIS FILE GENERATE FIXATIONS     ###
### FOR TS FRAMES AS DICTIONARY      ###
########################################
## VARIABLES ###
# frames folder is where the selected frames are located #
game = 'fifa'
frames_folder = 'C:\\Users\\omossad\\Desktop\\recorded_samples\\fifa\\gazepoint\\selected_frames\\'
# fixations folder is where the frame fixations are located#
fixations_folder = 'C:\\Users\\omossad\\Desktop\\recorded_samples\\fifa\\fixations\\'
output_folder = 'C:\\Users\\omossad\\Desktop\\recorded_samples\\fifa\\model_data\\filenames\\'

[ts, t_overlap, fut] = [10,2,2]

### READ NUMBER OF FILES and NAMES ###
def create_overlap(list_, n_items, n_overlap):
	return [list_[i:i+n_items] for i in range(0, len(list_), n_overlap)]

def collate_ts_frames():
	'''
	Create the sample input consisting of ts frames
	The next sample will have t_overlap with the previous one
	'''
	dir = frames_folder
	list_dir = os.listdir(dir)
	file_list = sorted(list_dir)
	collated_list = create_overlap(file_list, ts, t_overlap)
	collated_list = collated_list[:-t_overlap]
	return collated_list


def collate_ts_labels(labels_file, frames_list):
	labels = np.loadtxt(labels_file)
	collated_list = []
	frame_idx = frames_list[-1].replace('frame_','')
	frame_idx = frame_idx.replace('.jpg','')
	frame_no = int(frame_idx)
	for i in range(fut):
		collated_list.append(labels[frame_no + 1 + i])
	return collated_list




### COLLATE TS FRAMES  ###

output_path = output_folder
labels_file = 'C:\\Users\\omossad\\Desktop\\recorded_samples\\fifa\\fixations\\fifa.txt'
collated_frames = collate_ts_frames()
index = 0

for j in collated_frames:
	collated_ts_labels = collate_ts_labels(labels_file, j)
	#collated_labels.append(collated_ts_labels)
	f = open(output_path + str(index) + '.txt', "w")
	for k in range(len(j)):
		f.write(j[k].replace('.jpg',''))
		f.write('\n')
	for m in range(len(collated_ts_labels)):
		f.write(str(collated_ts_labels[m]))
		f.write('\n')
	index = index + 1
	if index >= len(collated_frames) - fut - t_overlap:
		break
f.close()
