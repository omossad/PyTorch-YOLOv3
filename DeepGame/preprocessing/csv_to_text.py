

from __future__ import print_function
import numpy as np
import csv
import pickle
import math
########################################
### THIS FILE CONVERTS FIXATION      ###
### FROM RAW TO FRAMES IN .TXT       ###
########################################
## VARIABLES ###
# input folder is where the fixation points are located #
input_folder = 'C:\\Users\\omossad\\Desktop\\dataset\\model_data\\labels\\csv_labels\\'
# output folder is where the frame fixations will be stored#
output_folder = 'C:\\Users\\omossad\\Desktop\\dataset\\model_data\\labels\\text_labels\\'
fps = 10

def find_nearest(array,value):
	'''
	function to find the nearest fixation point to the current frame
	takes as input an array of fixations and the frame index (time offset in this case)
	'''
	idx = np.searchsorted(array, value, side="left")
	if idx > 0 and (idx == len(array) or math.fabs(value - array[idx-1]) < math.fabs(value - array[idx])):
		return idx-1
	else:
		return idx

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


### EXTRACT FIXATION PER FRAME ###
for i in range(num_files):
	fixations_time = []
	fixations_val  = []
	with open(input_folder + file_names[i]+ '.csv') as csv_file:
		csv_reader = csv.reader(csv_file, delimiter=',')
		line_count = 0
		for row in csv_reader:
			if line_count == 0:
				line_count += 1
			else:
				fixations_time.append(float(row[0]))
				fixations_val.append([float(row[2]), float(row[3])])
				line_count += 1
		print(f'Real Fixations are {line_count}')

	labels = []
	frames = 0

	total_frames = int(frame_time[i])
	for j in range(total_frames):
		current_frame_time = j * 1/fps
		idx = find_nearest(fixations_time, current_frame_time)
		labels.append(fixations_val[idx])
	np.savetxt(output_folder+file_names[i]+'.txt', labels)
