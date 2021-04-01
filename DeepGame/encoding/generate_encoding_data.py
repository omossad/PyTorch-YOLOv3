from __future__ import print_function
import numpy as np
import pickle
import math
import os
import torch
import sys, os
sys.path.append(os.path.abspath(os.path.join('..')))
import utils
import re

num_tiles = utils.get_num_tiles()
games = ['fifa','csgo','nba','nhl']
game ='nhl'

# FIFA
#start_frame_gzpt = [215, 217,219, 221, 1194 ]
#start_frame_rec =  [52 , 66 , 78, 91 , 5926 ]

start_frame_gzpt = [215, 33, 53, 35]
start_frame_rec= [52, 60, 28, 79]
last_frame_gzpt = [1194, 632, 1152, 998 ]
last_frame_rec = [5929, 3597, 3318, 2968]


gzpt_fps = 10
rec_fps = 30
frames_to_jump = (rec_fps/gzpt_fps)*2
#[W,H] = [2560, 1440]
[W,H] = [1920,1080]

pre_arr = np.loadtxt('..\\visualize\\' + game + '_predicted.txt')
gt_arr  = np.loadtxt('..\\visualize\\' + game + '_labels.txt')

out_path = 'D:\\Encoding\\encoding_files\\' + game + '\\rois\\'
obj_path = 'D:\\Encoding\\encoding_files\\' + game + '\\model\\objects\\'

tile_w = 1/num_tiles
tile_h = 1/num_tiles


def tile_to_roi(arr):
    to_write = []
    for i in range(len(arr)):
        if arr[i] == 1:
            hor = (i%num_tiles)/num_tiles
            ver = (i//num_tiles)/num_tiles
            to_write.append('0 ' + str(hor-tile_w/2) + ' ' + str(ver-tile_h/2) + ' ' + str(tile_w) + ' ' + str(tile_h) + '\n')
            #print(i, hor, ver)
    #print(to_write)
    return to_write

def load_obj(obj_id):
    obj_tensor = torch.load(obj_path + 'frame_' + str(obj_id).zfill(5) + '.pt')
    obj_arr = obj_tensor.cpu().detach().numpy()
    to_write = []
    for i in range(len(obj_arr)):
        [x1, y1, x2, y2] = [obj_arr[i][0], obj_arr[i][1], obj_arr[i][2], obj_arr[i][3]]
        to_write.append('0 ' + str(x1/W) + ' ' + str(y1/H) + ' ' + str((x2-x1)/W) + ' ' + str((y2-y1)/H) + '\n')
    return to_write



print(len(pre_arr))
#base_enc = '0 ' + str(0.5-tile_w/2) + ' ' + str(0.5-tile_h/2) + ' ' + str(tile_w) + ' ' + str(tile_h)
base_enc = '0 0.01 0.01 0.9 9.9'
frame_counter = 0
pred_counter = 0
to_write_dg = tile_to_roi(pre_arr[pred_counter])
to_write_gt = tile_to_roi(gt_arr[pred_counter])



for i in range(last_frame_rec[games.index(game)]):
    if i < start_frame_rec[games.index(game)]:
        f = open(out_path + 'dg_rois\\roi' + str(i+1) + '.txt', "w")
        f.write(base_enc)
        f.close()

        f = open(out_path + 'gt_rois\\roi' + str(i+1) + '.txt', "w")
        f.writelines(to_write_gt)
        f.close()

    else:
        if frame_counter < frames_to_jump-1:
            f = open(out_path + 'dg_rois\\roi' + str(i+1) + '.txt', "w")
            f.writelines(to_write_dg)
            f.close()

            f = open(out_path + 'gt_rois\\roi' + str(i+1) + '.txt', "w")
            f.writelines(to_write_gt)
            f.close()

            frame_counter = frame_counter + 1
        else:
            frame_counter = 0
            pred_counter = pred_counter + 1
            to_write_dg = tile_to_roi(pre_arr[pred_counter])
            f = open(out_path + 'dg_rois\\roi' + str(i+1) + '.txt', "w")
            f.writelines(to_write_dg)

            to_write_gt = tile_to_roi(gt_arr[pred_counter])
            f = open(out_path + 'gt_rois\\roi' + str(i+1) + '.txt', "w")
            f.writelines(to_write_dg)

            print(pred_counter)
    #print(i)


'''
start_gzpt = start_frame_gzpt[games.index(game)] - int(start_frame_rec[games.index(game)]/frames_to_jump*2)
for i in range(last_frame_gzpt[games.index(game)] - start_gzpt + 1):
    to_write_od = load_obj(i+start_gzpt)
    for j in range(int(frames_to_jump/2)):
        file_count = int(frames_to_jump/2 * i+j)
        if file_count > last_frame_rec[games.index(game)]:
            break
        else:
            f = open(out_path + 'od_rois\\roi' + str(file_count+1) + '.txt', "w")
            f.writelines(to_write_od)
            f.close()
print(file_count-1)
'''
