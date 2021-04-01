from __future__ import print_function
import numpy as np
import pickle
import math
import os
from shapely.geometry import Polygon
import torch
from torch import nn
from torch.autograd import Variable
from torch.utils.data import TensorDataset, DataLoader
import math

import sys, os
sys.path.append(os.path.abspath(os.path.join('..')))
import utils
import time
import datetime


class RNN(nn.Module):
		def __init__(self):
				super(RNN, self).__init__()

				self.lstm_x = nn.LSTM(				 							# if use nn.RNN(), it hardly learns
						input_size=INPUT_SIZE,
						hidden_size=128,				 							# rnn hidden unit
						num_layers=1,				  			 				# number of rnn layer
						batch_first=True,		  								# input & output will has batch size as 1s dimension. e.g. (batch, time_step, input_size)
				)

				self.fc_x = nn.Sequential(nn.Linear(128, 64), nn.LeakyReLU(), nn.BatchNorm1d(64), nn.Linear(64,32), nn.LeakyReLU(), nn.Dropout(0.4), nn.Linear(32,8))

				self.lstm_y = nn.LSTM(
						input_size=INPUT_SIZE,
						hidden_size=128,
						num_layers=1,
						batch_first=True,
				)

				self.fc_y = nn.Sequential(nn.Linear(128, 64), nn.LeakyReLU(), nn.BatchNorm1d(64), nn.Linear(64,32), nn.LeakyReLU(), nn.Dropout(0.4), nn.Linear(32,8))

		def forward(self, x):
				# x shape (batch, time_step, input_size)
				# r_out shape (batch, time_step, output_size)
				# h_n shape (n_layers, batch, hidden_size)
				# h_c shape (n_layers, batch, hidden_size)
				r_out_x, (h_n_x, h_c_x) = self.lstm_x(x[:,:,0,:])
				r_out_y, (h_n_y, h_c_y) = self.lstm_y(x[:,:,1,:])
				fc_out_x = self.fc_x(r_out_x[:, -1, :])
				fc_out_y = self.fc_y(r_out_y[:, -1, :])
				return fc_out_x, fc_out_y

model_path = 'C:\\Users\\omossad\\Desktop\\dataset\\model_data\\nhl\\checkpoints\\checkpoint_19.pt'
## FI:FA ##
model_path = 'D:\\Encoding\\encoding_files\\nhl\\model\\checkpoints\\checkpoint_86.pt'

## CS:GO ##
model_path = 'D:\\Encoding\\encoding_files\\nhl\\model\\checkpoints\\checkpoint_99.pt'


## NH:L ##
#model_path = 'D:\\Encoding\\encoding_files\\nhl\\model\\checkpoints\\checkpoint_91.pt'

## NB:A ##
#model_path = 'D:\\Encoding\\encoding_files\\nhl\\model\\checkpoints\\checkpoint_94.pt'

#model = torch.load(model_path)
#model.eval()


game ='nhl'
obj_classes = 80
obj_classes = 3

#objects_folder = 'C:\\Users\\omossad\\Desktop\\dataset\\model_data\\' + game + '\\tiled_objects\\'
#labels_folder  = 'C:\\Users\\omossad\\Desktop\\dataset\\model_data\\' + game + '\\tiled_labels\\'
objects_folder = 'D:\\Encoding\\encoding_files\\nhl\\model\\tiled_objects\\'
labels_folder = 'D:\\Encoding\\encoding_files\\nhl\\model\\tiled_labels\\'


[W,H] = [2560,1440]
num_tiles = utils.get_num_tiles()
[ts, t_overlap, fut] = utils.get_model_conf()


### READ NUMBER OF FILES and NAMES ###
#num_files = 1
test_dat  = []
test_lbl  = []


dat = torch.load(objects_folder + 'nhl.pt').numpy()
lbl = torch.load(labels_folder  + 'nhl.pt').numpy()


test_dat.extend(dat)
test_lbl.extend(lbl)

test_dat = np.asarray(test_dat)
test_lbl = np.asarray(test_lbl)

print('________________')
print('Testing  data size ' + str(test_dat.shape) + ' , and labels ' + str(test_lbl.shape))
print('________________')


test_dat = np.reshape(test_dat, (test_dat.shape[0], ts, 2, num_tiles*obj_classes))

test_dat = torch.Tensor(test_dat)
temp_test_lbl = []

for i in range(len(test_lbl)):
	temp_test_lbl.append(np.logical_or(test_lbl[i][0],test_lbl[i][1]))
test_lbl = np.asarray(temp_test_lbl)

print('Testing  data size ' + str(test_dat.shape) + ' , and labels ' + str(test_lbl.shape))
print('________________')



torch.manual_seed(1)								# reproducible
TIME_STEP = 2				  						# rnn time step
INPUT_SIZE = num_tiles * obj_classes				# rnn input size


#rnn = model.cuda()

def map_tile(x_cor, y_cor):
	return int(x_cor*num_tiles + y_cor)

def test_accuracy(pre_x, pre_y, gt):
	gt_x = gt[:,0,:]
	gt_y = gt[:,1,:]
	#print(pre_x[0])

	pre_x = (pre_x >= 0.5).int().cpu().data.numpy().squeeze()
	pre_y = (pre_y >= 0.5).int().cpu().data.numpy().squeeze()
	#pre_x = [[0,1,1,1,0,0,0,0] for i in range(len(gt_x))]
	#pre_y = [[0,1,1,1,0,0,0,0] for i in range(len(gt_y))]
	#np.savetxt('..\\visualize\\predicted.txt', pre)
	#np.savetxt('..\\visualize\\labels.txt', gt)
	#for i in range(len(pre_x)):
	#	print(sum(pre_x[i]))
	#	print(sum(pre_y[i]))

	tp = 0
	tn = 0
	fp = 0
	fn  = 0
	precision_ = 0
	recall_ = 0
	precision_1 = 0
	recall_1 = 0
	precision_2 = 0
	recall_2 = 0
	hits = 0
	predicted_area = 0
	gt_area = 0
	miss = 0
	total_intersection = 0
	total_overshoot = 0
	total_undershoot =  0
	total_inter_norm = 0
	total_over_norm = 0
	total_under_norm =  0
	pre_arrs = []
	gt_arrs = []
	for i in range(len(gt)):
		tp_ = 0
		tn_ = 0
		fp_ = 0
		fn_  = 0
		tp_1 = 0
		tn_1 = 0
		fp_1 = 0
		fn_1  = 0
		tp_2 = 0
		tn_2 = 0
		fp_2 = 0
		fn_2  = 0

		inter_norm = 0
		over_norm = 0
		under_norm =  0
		pre_arr =  [0 for p in range(num_tiles*num_tiles)]
		gt_arr = [0 for p in range(num_tiles*num_tiles)]
		pre_arr_1 =  [0 for p in range(num_tiles*num_tiles)]
		pre_arr_2 =  [0 for p in range(num_tiles*num_tiles)]

		#print(gt_x[i])
		#print(pre_x[i])
		#print(gt_y[i])
		#print(pre_y[i])
		for j in range(num_tiles):
			for k in range(num_tiles):
				if pre_x[i][j] == 1 and pre_y[i][k] == 1:
					pre_arr[map_tile(j,k)] = 1

				if gt_x[i][j] == 1 and gt_y[i][k] == 1:
					gt_arr[map_tile(j,k)] = 1
		pre_arrs.append(pre_arr)
		gt_arrs.append(gt_arr)
		for j in range(len(pre_arr)):
			if pre_arr[j] == 1:
				pre_arr_1[j] = pre_arr[j]
				if j > 0:
					pre_arr_1[j-1] = 1
				if j+1 < len(pre_arr):
					pre_arr_1[j+1] = 1
				if j-num_tiles > 0:
					pre_arr_1[j-num_tiles] = 1
				if j+num_tiles < len(pre_arr):
					pre_arr_1[j+num_tiles] = 1
		#print(pre_arr)

		for j in range(len(pre_arr_1)):
			if pre_arr_1[j] == 1:
				pre_arr_2[j] = pre_arr_1[j]
				if j > 0:
					pre_arr_2[j-1] = 1
				if j+1 < len(pre_arr_1):
					pre_arr_2[j+1] = 1
				if j-num_tiles > 0:
					pre_arr_2[j-num_tiles] = 1
				if j+num_tiles < len(pre_arr_1):
					pre_arr_2[j+num_tiles] = 1

		#print(pre_arr)
		#print(gt_arr)
		for j in range(len(pre_arr)):
			if pre_arr[j] == 1:
				if gt_arr[j] == 1:
					tp += 1
					tp_+= 1
				else:
					fp += 1
					fp_+= 1
			else:
				if gt_arr[j] == 1:
					fn += 1
					fn_+= 1
				else:
					tn += 1
					tn_+= 1
		for j in range(len(pre_arr_1)):
			if pre_arr_1[j] == 1:
				if gt_arr[j] == 1:
					tp_1 += 1
				else:
					fp_1 += 1
			else:
				if gt_arr[j] == 1:
					fn_1 += 1
				else:
					tn_1 += 1
		for j in range(len(pre_arr_2)):
			if pre_arr_2[j] == 1:
				if gt_arr[j] == 1:
					tp_2 += 1
				else:
					fp_2 += 1
			else:
				if gt_arr[j] == 1:
					fn_2 += 1
				else:
					tn_2 += 1

		inter_norm = tp/(tp+fn)
		over_norm = fp/(tp+fn)
		under_norm = fn/(tp+fn)
		total_intersection += tp
		total_overshoot += fp
		total_undershoot += fn
		total_inter_norm += inter_norm
		total_over_norm += over_norm
		total_under_norm += under_norm
		if tp_ > 0:
			hits += 1
		else:
			miss += 1
		if tp_ > 0 or fp_ > 0:
			precision_ += tp_/(tp_+fp_)
		if tp_ > 0 or fn_ > 0:
			recall_ += tp_/(tp_+fn_)

		if tp_1 > 0 or fp_1 > 0:
			precision_1 += tp_1/(tp_1+fp_1)
		if tp_1 > 0 or fn_1 > 0:
			recall_1 += tp_1/(tp_1+fn_1)

		if tp_2 > 0 or fp_2 > 0:
			precision_2 += tp_2/(tp_2+fp_2)
		if tp_2 > 0 or fn_2 > 0:
			recall_2 += tp_2/(tp_2+fn_2)

		predicted_area += sum(pre_arr)/(num_tiles*num_tiles)
		gt_area += sum(gt_arr)/(num_tiles*num_tiles)
	num_samples = len(gt)
	print("number of samples: " + str(num_samples))


	print("Hit  Ratio: " + str(hits/num_samples))
	#print("Predicted Area: " + str(predicted_area))
	print("Predicted Area: " + str(predicted_area/num_samples))
	print("GT Area: " + str(gt_area/num_samples))

	#print("Miss Ratio: " + str(miss/num_samples))
	print('________________')
	np.savetxt('..\\visualize\\'+ game +'_predicted.txt', pre_arrs)
	np.savetxt('..\\visualize\\'+ game + '_labels.txt', gt_arrs)


for i in range(100):
	model_path = 'D:\\Encoding\\encoding_files\\nhl\\model\\checkpoints\\checkpoint_' + str(i) + '.pt'
	model = torch.load(model_path)
	model.eval()
	rnn = model.cuda()
	#prev_time = time.time()
	test_output_x, test_output_y = rnn(Variable(test_dat).cuda())							# (samples, time_step, input_size)
	#current_time = time.time()
	#inference_time = datetime.timedelta(seconds=current_time - prev_time)
	#prev_time = current_time
	#print("--- %s seconds ---" % (inference_time))
	print(i)
	test_accuracy(test_output_x, test_output_y, test_lbl)
