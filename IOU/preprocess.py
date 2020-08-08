import numpy as np
import csv
import os
##### CONVERT LABEL DATA TO ROI DATA FOR EVALUATION ######

W_ref = 1920.0
H_ref = 1080.0
W_tiles = 8.0
H_tiles  = 8.0
tile_width = W_ref / W_tiles
tile_height = H_ref / H_tiles
num_tiles = int(W_tiles * H_tiles)

input_folder = '/home/omossad/scratch/Gaming-Dataset/labels_txt/fifa/'
#output_folder_64 = '/home/omossad/projects/def-hefeeda/omossad/roi_detection/temporary_data/ha_0_labels_64/'
output_folder = '/home/omossad/scratch/Gaming-Dataset/frame_labels/fifa/'

#base_dir  = '/home/omossad/scratch/temp/labels/ha_labels/'
#out_dir   = '/home/omossad/scratch/temp/roi/'
#frame_dir = '/home/omossad/scratch/temp/frames/'
def iou(boxA, boxB):
	# determine the (x, y)-coordinates of the intersection rectangle
	xA = max(boxA[0], boxB[0])
	yA = max(boxA[1], boxB[1])
	xB = min(boxA[2], boxB[2])
	yB = min(boxA[3], boxB[3])
	# compute the area of intersection rectangle
	interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)
	# compute the area of both the prediction and ground-truth
	# rectangles
	boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
	boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)
	# compute the intersection over union by taking the intersection
	# area and dividing it by the sum of prediction + ground-truth
	# areas - the interesection area
	iou = interArea / float(boxAArea + boxBArea - interArea)
	# return the intersection over union value
	return iou


def get_tileXY(x,y):
        x_cor = np.minimum(x, W_ref-1)
        y_cor = np.minimum(y, H_ref-1)
        x_cor = np.maximum(x, 0)
        y_cor = np.maximum(y, 0)
        #x_dis = int(np.minimum(round(x_cor / tile_width),W_tiles-1))
        #y_dis = int(np.minimum(round(y_cor / tile_height), H_tiles-1))
        x_dis = int(x_cor / tile_width)
        y_dis = int(y_cor / tile_height)
        #tile = int(x_dis + y_dis * W_tiles)
        return x_dis,y_dis

def get_box_r(x, y):
        x1 = x * tile_width
        x2 = (x+1) * tile_width
        y1 = y * tile_height
        y2 = (y+1) * tile_height
        return [x1, y1, x2, y2]

def get_box_c(c_x, c_y, d):
        x1 = c_x - d/2
        x2 = c_x + d/2
        y1 = c_y - d/2
        y2 = c_y + d/2
        return [x1, y1, x2, y2]


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

iou_tot = 0
for i in range(num_files):
    labels = np.loadtxt(input_folder + file_names[i] + '.txt')
    start_frame = int(frame_info[i][0])
    end_frame = int(frame_info[i][1])
    current_frame = start_frame
    #current_tile = -1
    #
    #counters = []
    print(len(labels))
    thre = len(labels)*0.57
    counter = 0
    iou_avg = 0

    for lbl in labels:
        lbl_x = lbl[0] * W_ref
        lbl_y = lbl[1] * H_ref
        sx_tile, sy_tile = get_tileXY(lbl_x,lbl_y)
        if counter > thre:
                sx_tile = 4
                sy_tile = 4
        box_1 = get_box_r(sx_tile, sy_tile)
        box_2 = get_box_c(lbl_x, lbl_y, 70)
        iou_avg = iou_avg + iou(box_1, box_2)
        counter =  counter + 1
    iou_avg = iou_avg / counter
    iou_tot = iou_tot + iou_avg
    print(iou_avg)
iou_tot = iou_tot / num_files
print(iou_tot)
		#print(lbl_x)
        #print(lbl_y)
		#print(sx_tile)
        #print(sy_tile)
		#print(box_1)
		#print(box_2)
        #print(iou(box_1, box_2))
        #current_tile = s_tile
        #if current_tile == previous_tile:
        #    counter += 1
        #else:
        #    counters.append(counter)
        #    counter = 0
        #if counter == 1:
        #    break




    #print(counters)
