

from __future__ import print_function
import numpy as np
import csv
import pickle
import math
import os
import sys, os
sys.path.append(os.path.abspath(os.path.join('..')))
import utils

########################################
### THIS FILE GENERATE FIXATIONS     ###
### FOR TS FRAMES AS DICTIONARY      ###
########################################
## VARIABLES ###
# frames folder is where the selected frames are located #
frames_folder = 'C:\\Users\\omossad\\Desktop\\dataset\\model_data\\selected_frames\\'
# fixations folder is where the frame fixations are located#
fixations_folder = 'C:\\Users\\omossad\\Desktop\\dataset\\model_data\\labels\\text_labels\\'
output_folder = 'C:\\Users\\omossad\\Desktop\\dataset\\model_data\\filenames\\'

[ts, t_overlap, fut] = utils.get_model_conf()


### READ NUMBER OF FILES and NAMES ###
num_files = utils.get_no_files()
file_names = utils.get_files_list(num_files)

def create_overlap(list_, n_items, n_overlap):
	return [list_[i:i+n_items] for i in range(0, len(list_), n_items-n_overlap)]

def collate_ts_frames(input_folder):
	'''
	Create the sample input consisting of ts frames
	The next sample will have t_overlap with the previous one
	'''
	dir = frames_folder + input_folder + '\\'
	list_dir = os.listdir(dir)
	file_list = sorted(list_dir)
	collated_list = create_overlap(file_list, ts, t_overlap)
	collated_list = collated_list[:-t_overlap]
	return collated_list


def collate_ts_labels(labels_file, frames_list):
	labels = np.loadtxt(labels_file)
	collated_list = []
	frame_idx = frames_list[-1].replace('frame_','')
	frame_idx = frame_idx.replace('.png','')
	frame_no = int(frame_idx)
	for i in range(fut):
		collated_list.append(labels[frame_no + i])
	'''
	for frame_name in frames_list:
		frame_idx = frame_name.replace('frame_','')
		frame_idx = frame_idx.replace('.png','')
		frame_no = int(frame_idx)
		frame_no = frame_no + 10
		#print(frame_no)
		#print(labels[frame_no-1])
		#print(labels[frame_no-1+10])
		collated_list.append(labels[frame_no-1+10])
	'''
	return collated_list




### COLLATE TS FRAMES  ###
#num_files = 1
for i in range(num_files):
	print("Processing " + file_names[i])
	output_path = output_folder + file_names[i] + '\\'
	try:
		os.mkdir(output_path)
	except:
		print('folder exists, skipping')
	labels_file = "C:\\Users\\omossad\\Desktop\\dataset\\model_data\\labels\\text_labels\\"
	labels_file = labels_file + file_names[i]
	labels_file = labels_file + '.txt'
	collated_frames = collate_ts_frames(file_names[i])
	#collated_labels = []
	index = 0

	for j in collated_frames:
		collated_ts_labels = collate_ts_labels(labels_file, j)
		#collated_labels.append(collated_ts_labels)
		f = open(output_path + str(index) + '.txt', "w")
		for k in range(len(j)):
			f.write(j[k].replace('.png',''))
			f.write('\n')
		for m in range(len(collated_ts_labels)):
			f.write(str(collated_ts_labels[m]))
			f.write('\n')
		index = index + 1
	f.close()
	'''
		f = open(output_path + str(index) + '.txt', "w")
		for k in range(len(j)):
			f.write(j[k].replace('.png',''))
			f.write(',')
			f.write(str(collated_ts_labels[k]))
			f.write('\n')
		index = index + 1
	f.close()
	'''
	#print(collated_frames[0])
	#print(collated_labels[0])
