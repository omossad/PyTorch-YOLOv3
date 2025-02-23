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
        x_dis = int(x_cor / tile_width)
        y_dis = int(y_cor / tile_height)
        #tile = int(x_dis + y_dis * W_tiles)
        return x_dis,y_dis

num_files = 10
base_dir  = '/home/omossad/scratch/temp/labels/ha_labels/'
out_dir   = '/home/omossad/scratch/temp/roi/'
frame_dir = '/home/omossad/scratch/temp/frames/'

frame_info = np.zeros((num_files,2))
with open('../ROI-detection/frames_info') as csv_file:
	csv_reader = csv.reader(csv_file, delimiter=',')
	line_count = 0
	for row in csv_reader:
		if line_count == 0:
			line_count += 1
		elif line_count < num_files+1:
			frame_info[line_count-1] = [int(row[2]), int(row[3])]
			line_count += 1
		else:
			break


f = open("move_files.sh", "w")
f.write("#!/bin/bash \n")


for i in range(num_files):
    labels = np.loadtxt(base_dir + 'labels_' + str(i) + '.txt')
    start_frame = int(frame_info[i][0])
    end_frame = int(frame_info[i][1])
    current_frame = start_frame
    for lbl in labels:
        #s_tile = get_tile(lbl[0],lbl[1])
        sx_tile, sy_tile = get_tileXY(lbl[0],lbl[1])
        if(sx_tile > 7):
            print('Problem')
        if(sy_tile > 7):
            print('Problem')
        filename = 'ha_' + str(i) + '_frame_' + str(current_frame).zfill(5)
        #command = 'cp ' + frame_dir + 'ha_' + str(i) + '/frame_' + str(current_frame).zfill(5) + '.jpg ' + out_dir + 'images/'
        #filelist = "echo '" + out_dir + 'images/' + filename + ".jpg' >> " + out_dir
        #if i < num_files - 2:
        #    filelist = filelist + 'train.txt \n'
        #else:
        #    filelist = filelist + 'valid.txt \n'
        #command = command + filename + '.jpg \n'
        #command = command + "echo '"+ str(s_tile) +" 0.1 0.1 0.1 0.1' > " + out_dir + 'labels/'
        command = "echo '0 "+ str(sx_tile*1/W_tiles) +' '+ str(sy_tile*1/H_tiles) + ' 0.124 0.124' + "' > " + out_dir + 'labels/'
        command = command + filename + '.txt \n'
        #command = command + filelist
        f.write(command)
        current_frame = current_frame + 1
f.close()
