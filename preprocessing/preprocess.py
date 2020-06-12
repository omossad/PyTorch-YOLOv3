import numpy as np
import csv
##### CONVERT LABEL DATA TO ROI DATA FOR EVALUATION ######

W_ref = 1920.0
H_ref = 1080.0
W_tiles = 8.0
H_tiles  = 8.0
tile_width = W_ref / W_tiles
tile_height = H_ref / H_tiles
num_tiles = int(W_tiles * H_tiles)

input_folder = '/home/omossad/scratch/Gaming-Dataset/processed/raw_frame_labels/fifa/'
output_folder_64 = '/home/omossad/scratch/Gaming-Dataset/processed/labels_64/fifa/'
output_folder_8x8 = '/home/omossad/scratch/Gaming-Dataset/processed/labels_8x8/fifa/'

#base_dir  = '/home/omossad/scratch/temp/labels/ha_labels/'
#out_dir   = '/home/omossad/scratch/temp/roi/'
#frame_dir = '/home/omossad/scratch/temp/frames/'

def get_tile(x,y):
        x = x*W_ref
        y = y*H_ref
        x_cor = np.minimum(x, W_ref-1)
        y_cor = np.minimum(y, H_ref-1)
        x_dis = int(x_cor / tile_width)
        y_dis = int(y_cor / tile_height)
        tile = int(x_dis + y_dis * W_tiles)
        return tile

def get_tileXY(x,y):
        x = x*W_ref
        y = y*H_ref
        x_cor = np.minimum(x, W_ref-1)
        y_cor = np.minimum(y, H_ref-1)
        #x_dis = int(np.minimum(round(x_cor / tile_width),W_tiles-1))
        #y_dis = int(np.minimum(round(y_cor / tile_height), H_tiles-1))
        x_dis = int(x_cor / tile_width)
        y_dis = int(y_cor / tile_height)
        #tile = int(x_dis + y_dis * W_tiles)
        return x_dis,y_dis



num_files = 0
with open('frames_info', 'r') as f:
    for line in f:
        num_files += 1
# number of files is the number of files to be processed #
num_files = num_files - 1
print("Total number of files is:", num_files)


frame_info = np.zeros((num_files,2))
file_names = []
#with open('../ROI-detection/frames_info') as csv_file:
with open('frames_info') as csv_file:
	csv_reader = csv.reader(csv_file, delimiter=',')
	line_count = 0
	for row in csv_reader:
		if line_count == 0:
			line_count += 1
		elif line_count < num_files+1:
            file_names.append(row[0])
			frame_info[line_count-1] = [int(row[2]), int(row[3])]
			line_count += 1
        else:
			break
print(file_names)

f = open("move_files.sh", "w")
f.write("#!/bin/bash \n")


for i in range(num_files):
    labels = np.loadtxt(input_folder + file_names[i] + '.txt')
    start_frame = int(frame_info[i][0])
    end_frame = int(frame_info[i][1])
    current_frame = start_frame
    #current_tile = -1
    #counter = 0
    #counters = []
    for lbl in labels:
        #previous_tile = current_tile
        s_tile = get_tile(lbl[0],lbl[1])
        #current_tile = s_tile
        #if current_tile == previous_tile:
        #    counter += 1
        #else:
        #    counters.append(counter)
        #    counter = 0
        sx_tile, sy_tile = get_tileXY(lbl[0],lbl[1])
        filename = file_names[i] + '_frame_' + str(current_frame).zfill(5)
        #command = 'cp ' + frame_dir + 'ha_' + str(i) + '/frame_' + str(current_frame).zfill(5) + '.jpg ' + out_dir + 'images/'
        #filelist = "echo '" + out_dir + 'images/' + filename + ".jpg' >> " + out_dir
        #if i < num_files - 2:
        #    filelist = filelist + 'train.txt \n'
        #else:
        #    filelist = filelist + 'valid.txt \n'
        #command = command + filename + '.jpg \n'
        #command = command + "echo '"+ str(s_tile) +" 0.1 0.1 0.1 0.1' > " + out_dir + 'labels/'
        command = "echo '"+ str(s_tile) + "' > " + output_folder_64
        command = command + filename + '.txt \n'
        command = command + "echo '"+ str(sx_tile) +' '+ str(sy_tile) + "' > " + output_folder_8x8
        command = command + filename + '.txt \n'
        #command = command + filelist
        f.write(command)
        current_frame = current_frame + 1
    #print(counters)
f.close()
