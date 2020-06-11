

from __future__ import print_function
import numpy as np
import csv
import pickle
import math
########################################
### THIS FILE GENERATE TILE DATA     ###
### FROM RAW FRAMES AND FIXATION     ###
########################################

num_files = 10
base_dir = '/home/omossad/scratch/temp/fixations/'
out_dir = '/home/omossad/scratch/temp/labels/'

#def find_nearest(array, value):
#	array = np.asarray(array)
#	idx = (np.abs(array - value)).argmin()
#	return array[idx]

def find_nearest(array,value):
	idx = np.searchsorted(array, value, side="left")
	if idx > 0 and (idx == len(array) or math.fabs(value - array[idx-1]) < math.fabs(value - array[idx])):
		return idx-1
	else:
		return idx


frame_time = np.zeros((num_files,1))
frame_info = np.zeros((num_files,2))
with open('frames_info') as csv_file:
	csv_reader = csv.reader(csv_file, delimiter=',')
	line_count = 0
	for row in csv_reader:
		if line_count == 0:
			line_count += 1
		elif line_count < num_files+1:
			frame_time[line_count-1] = int(row[5])
			frame_info[line_count-1] = [int(row[2]), int(row[3])]
			line_count += 1
		else:
			break

for i in range(num_files):
	fixations_time = []
	fixations_val  = []
	with open(base_dir + 'User '+str(i)+'_fixations.csv') as csv_file:
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
	start_frame = int(frame_info[i][0]) 
	end_frame = int(frame_info[i][1])
#	start_frame = 0
#	end_frame = 734
#	print(start_frame)
#	print(end_frame)
	for j in range(start_frame, end_frame + 1):
		current_frame_time = j * 1/10
		idx = find_nearest(fixations_time, current_frame_time)
#		print(idx)
#		print(fixations_val[idx])
		labels.append(fixations_val[idx])			
	np.savetxt(out_dir+'labels_'+str(i)+'.txt', labels)
