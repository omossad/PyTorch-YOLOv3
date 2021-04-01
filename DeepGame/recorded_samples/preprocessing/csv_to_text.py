

from __future__ import print_function
import numpy as np
import csv
import pickle
import math
import sys, os
########################################
### THIS FILE CONVERTS FIXATION      ###
### FROM RAW TO FRAMES IN .TXT       ###
########################################
## VARIABLES ###
# input folder is where the fixation points are located #
input_folder = 'C:\\Users\\omossad\\Desktop\\recorded_samples\\fifa\\fixations\\'
# output folder is where the frame fixations will be stored#
output_folder = 'C:\\Users\\omossad\\Desktop\\recorded_samples\\fifa\\fixations\\'
fps = 10
# LAST FRAME STARTING COUNT FROM 1
total_frames = 1250

### READ NUMBER OF FILES and NAMES ###



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


### READ NUMBER OF FRAMES in each FILE ###
fixations = []
with open(input_folder + 'fifa.csv') as csv_file:
	csv_reader = csv.reader(csv_file, delimiter=',')
	line_count = 0
	for row in csv_reader:
		if line_count == 0:
			line_count += 1
		else:
			fixation_end_time = float(row[0])
			fixation_x = float(row[2])
			fixation_y = float(row[3])
			fixations.append([fixation_end_time, fixation_x, fixation_y])

labels = []
for i in range(total_frames):
	frame_time = i * 1/fps
	for j in range(len(fixations)):
		if frame_time <= fixations[j][0]:
			labels.append([fixations[j][1], fixations[j][2]])
			print(frame_time)
			print(fixations[j][0])
			print(labels[i])
			break
	#print(frame_time)
	#print(fixations[i])

np.savetxt(output_folder+ 'fifa.txt', labels)
'''

### EXTRACT FIXATION PER FRAME ###
for i in range(num_files):
	fixations_time = []
	fixations_val  = []
	with open(input_folder + 'fifa.csv') as csv_file:
		csv_reader = csv.reader(csv_file, delimiter=',')
		previous_x = float(0)
		previous_y = float(0)
		line_count = 0
		for row in csv_reader:
			if line_count == 0:
				line_count += 1
			else:
				fixations_time.append(float(row[0]))
				if float(row[2]) < 0 or float(row[3]) < 0:
					if float(row[2]) < 0 and float(row[3]) >= 0:
						fixations_val.append([previous_x, float(row[3])])
						previous_y = float(row[3])
					elif float(row[2]) >= 0 and float(row[3]) < 0:
						fixations_val.append([float(row[2]), previous_y])
						previous_x = float(row[2])
					else:
						fixations_val.append([previous_x, previous_y])
				else:
					fixations_val.append([float(row[2]), float(row[3])])
					previous_x = float(row[2])
					previous_y = float(row[3])
				line_count += 1
		print(f'Real Fixations are {line_count}')

	labels = []
	frames = 0

	total_frames = int(frame_time[i])
	for j in range(total_frames):
		current_frame_time = (j) * 1/fps
		idx = find_nearest(fixations_time, current_frame_time)
		#print(fixations_time[idx])
		#print(current_frame_time)
		labels.append(fixations_val[idx])
	np.savetxt(output_folder+file_names[i]+'.txt', labels)
'''
