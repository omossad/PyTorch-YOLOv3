

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

########################################
###  THE DL Model for prediction		 ###
########################################
## VARIABLES ###
# input folder is where the selected data is located #
game ='fifa'
obj_classes = 3

objects_folder = 'C:\\Users\\omossad\\Desktop\\dataset\\model_data\\' + game + '\\tiled_objects\\'
labels_folder  = 'C:\\Users\\omossad\\Desktop\\dataset\\model_data\\' + game + '\\tiled_labels\\'

[W,H] = utils.get_img_dim()
num_tiles = utils.get_num_tiles()
[ts, t_overlap, fut] = utils.get_model_conf()

test_ratio = 0.3

### READ NUMBER OF FILES and NAMES ###
num_files = utils.get_no_files(game)
#num_files = 1
file_names = utils.get_files_list(num_files, game)

train_dat = []
test_dat  = []
train_lbl = []
test_lbl  = []


for i in range(num_files):
	dat = torch.load(objects_folder + file_names[i] + '.pt').numpy()
	lbl = torch.load(labels_folder + file_names[i] + '.pt').numpy()

	#no_test = int(test_ratio*len(dat))
	#no_train = len(dat) - no_test
	#if file_names[i].startswith('se_'):

	#else:
	train_dat.extend(dat)
	train_lbl.extend(lbl)

dat = torch.load('C:\\Users\\omossad\\Desktop\\recorded_samples\\fifa\\model_data\\tiled_objects\\fifa.pt').numpy()
lbl = torch.load('C:\\Users\\omossad\\Desktop\\recorded_samples\\fifa\\model_data\\tiles\\fifa.pt').numpy()
test_dat.extend(dat)
test_lbl.extend(lbl)

train_dat = np.asarray(train_dat)

test_dat = np.asarray(test_dat)
train_lbl = np.asarray(train_lbl)
test_lbl = np.asarray(test_lbl)
print('Training data size ' + str(train_dat.shape) + ' , and labels ' + str(train_lbl.shape))
print('Testing  data size ' + str(test_dat.shape) + ' , and labels ' + str(test_lbl.shape))


train_dat = np.reshape(train_dat, (train_dat.shape[0], ts, 2, num_tiles*obj_classes))
test_dat = np.reshape(test_dat, (test_dat.shape[0], ts, 2, num_tiles*obj_classes))

test_dat = torch.Tensor(test_dat)
print(train_lbl[3])
temp_train_lbl = []
temp_test_lbl = []
print('summation')
print(np.logical_or(train_lbl[3][0], train_lbl[3][1]))
print(train_lbl[3][0] + train_lbl[3][1])
print('separator')
for i in range(len(train_lbl)):
	temp_train_lbl.append(np.logical_or(train_lbl[i][0],train_lbl[i][1]))
for i in range(len(test_lbl)):
	temp_test_lbl.append(np.logical_or(test_lbl[i][0],test_lbl[i][1]))
train_lbl = np.asarray(temp_train_lbl)
test_lbl = np.asarray(temp_test_lbl)
print(train_lbl[3])
#train_lbl = train_lbl[:,-1,:,:]
print('separator')
#print(train_lbl[3])
#test_lbl = test_lbl[:,-1,:,:]
print('Training data size ' + str(train_dat.shape) + ' , and labels ' + str(train_lbl.shape))
print('Testing  data size ' + str(test_dat.shape) + ' , and labels ' + str(test_lbl.shape))



torch.manual_seed(1)								# reproducible
EPOCH = 2
BATCH_SIZE = 10
TIME_STEP = 10				  						# rnn time step
INPUT_SIZE = num_tiles * obj_classes				# rnn input size
LR = 0.001					  						# learning rate


tensor_x = torch.Tensor(train_dat) # transform to torch tensor
tensor_y = torch.Tensor(train_lbl)

my_dataset = TensorDataset(tensor_x,tensor_y) # create your datset
train_loader = DataLoader(my_dataset, batch_size=BATCH_SIZE, shuffle=False) # create your dataloader


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

rnn = RNN().cuda()
print(rnn)
optimizer = torch.optim.Adam(rnn.parameters(), lr=LR)   # optimize all rnn parameters
pos_weight = torch.ones([8]).cuda()
alpha = 7
beta  = 6
loss_func_x = nn.BCEWithLogitsLoss(pos_weight = alpha*pos_weight)
loss_func_y = nn.BCEWithLogitsLoss(pos_weight = beta*pos_weight)

def custom_loss(output_x, output_y, target):
	target_x = target[:,0,:]
	target_y = target[:,1,:]
	pre_x = (output_x >= 0.5).int()
	pre_y = (output_y >= 0.5).int()
	#print(target_x)
	#print(pre_x)
	num_samples = len(target)
	total_loss = 0
	for i in range(num_samples):
		sample_loss = 0
		for j in range(num_tiles):
			if target_x[i][j] == 1 and output_x[i][j] < 0.5:
				#print(max(output_x[i]))
				sample_loss += -1 * math.log(output_x[i][j])
			if target_y[i][j] == 1 and output_y[i][j] < 0.5:
				sample_loss += -1 * math.log(output_y[i][j])
		total_loss += sample_loss
	#total_loss = total_loss/num_samples
	loss =  Variable(torch.Tensor([total_loss]), requires_grad = True).cuda()
	return loss

def my_loss(output_x, output_y, target):

		target_x = target[:,0,:]
		target_y = target[:,1,:]
		total_loss = 0
		for i in range(len(output_x)):
			total_loss += loss_func_x(output_x[i], target_x[i])
			total_loss += loss_func_y(output_y[i], target_y[i])
			#total_loss += item_loss
		#total_loss = total_loss/len(target)
		return total_loss

def map_tile(x_cor, y_cor):
	return int(x_cor*num_tiles + y_cor)

def test_accuracy(pre_x, pre_y, gt):
	gt_x = gt[:,0,:]
	gt_y = gt[:,1,:]
	#print(pre_x[0])
	pre_x = (pre_x >= 0.5).int().cpu().data.numpy().squeeze()
	pre_y = (pre_y >= 0.5).int().cpu().data.numpy().squeeze()

	np.savetxt('..\\visualize\\predicted_x.txt', pre_x)
	np.savetxt('..\\visualize\\labels_x.txt', gt_x)
	np.savetxt('..\\visualize\\predicted_y.txt', pre_y)
	np.savetxt('..\\visualize\\labels_y.txt', gt_y)

	tp = 0
	tn = 0
	fp = 0
	fn  = 0
	precision_ = 0
	recall_ = 0
	hits = 0
	miss = 0
	total_intersection = total_overshoot = total_undershoot =  0
	total_inter_norm = total_over_norm = total_under_norm =  0
	for i in range(len(gt)):
		tp_ = 0
		tn_ = 0
		fp_ = 0
		fn_  = 0
		#print(pre_x[i])
		#[val_x, idx_x] = torch.topk(pre_x[i],k=2)
		#[val_y, idx_y] = torch.topk(pre_x[i],k=2)
		#print(idx_x)
		#print(idx_y)
		#print(pre_x[i])
		#print(prk_0)
		#intersection = overshoot = undershoot = 0
		inter_norm = over_norm = under_norm =  0
		pre_arr =  [0 for p in range(num_tiles*num_tiles)]
		gt_arr = [0 for p in range(num_tiles*num_tiles)]
		#print(gt_x[i])
		#print(pre_x[i])
		#print(gt_y[i])
		#print(pre_y[i])
		for j in range(num_tiles):
			for k in range(num_tiles):
				if pre_x[i][j] == 1 and pre_y[i][k] == 1:
					pre_arr[map_tile(j,k)] = 1
					#print('here')
				if gt_x[i][j] == 1 and gt_y[i][k] == 1:
					gt_arr[map_tile(j,k)] = 1
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
	num_samples = len(gt)
	print("number of samples: " + str(num_samples))
	#print("intersection : " + str(total_intersection) + "-------" + str(total_intersection/num_samples))
	#print("overshoot :" + str(total_overshoot) + "-------" + str(total_overshoot/num_samples))
	#print("undershoot :" + str(total_undershoot) + "-------" + str(total_undershoot/num_samples))
	print("intersection normalized :" + str(total_inter_norm) + "-------" + str(total_inter_norm/num_samples))
	print("overshoot normalized :" + str(total_over_norm) + "-------" + str(total_over_norm/num_samples))
	print("undershoot normalized :" + str(total_under_norm) + "-------" + str(total_under_norm/num_samples))
	print("TP: " + str(tp))
	print("TN: " + str(tn))
	print("FP: " + str(fp))
	print("FN: " + str(fn))
	print("Precision: " + str(tp/(tp+fp)))
	print("Recall: " + str(tp/(tp+fn)))
	print("Accuracy: " + str((tp+tn)/(tp+tn+fp+fn)))
	print("B Acc: " + str(((tp/(tp+fn))+(tn/(tn+fp)))/2))
	print("Norm Precision: " + str(precision_/num_samples))
	print("Norm Recall: " + str(recall_/num_samples))
	print("Hit  Ratio: " + str(hits/num_samples))
	print("Miss Ratio: " + str(miss/num_samples))

# training and testing
for epoch in range(EPOCH):
		for step, (x, y) in enumerate(train_loader):
				x = Variable(x.view(-1, ts, 2, INPUT_SIZE)).cuda()						# reshape x to (batch, time_step, input_size)
				y = Variable(y).cuda()											# batch y
				rnn.zero_grad()
				optimizer.zero_grad()
				output_x, output_y = rnn(x)													# rnn output
				loss = my_loss(output_x, output_y, y)
				#loss = custom_loss(output_x, output_y, y)
				#print(loss.data)
				#print(loss.grad)
				optimizer.zero_grad()											# clear gradients for this training step
				loss.backward()													# backpropagation, compute gradients
				optimizer.step()												# apply gradients
				#print(loss.grad)

				#if step % 50 == 0:
		prev_time = time.time()
		test_output_x, test_output_y = rnn(Variable(test_dat).cuda())							# (samples, time_step, input_size)
		current_time = time.time()
		inference_time = datetime.timedelta(seconds=current_time - prev_time)
		prev_time = current_time
		print("--- %s seconds ---" % (inference_time))
		prev_time = time.time()
		test_output_x, test_output_y = rnn(Variable(test_dat).cuda())							# (samples, time_step, input_size)
		current_time = time.time()
		inference_time = datetime.timedelta(seconds=current_time - prev_time)
		prev_time = current_time
		print("--- %s seconds ---" % (inference_time))
		prev_time = time.time()
		test_output_x, test_output_y = rnn(Variable(test_dat).cuda())							# (samples, time_step, input_size)
		current_time = time.time()
		inference_time = datetime.timedelta(seconds=current_time - prev_time)
		prev_time = current_time
		print("--- %s seconds ---" % (inference_time))
		print('Epoch: ', epoch, '| train loss: %.4f' % loss.item())
		test_accuracy(test_output_x, test_output_y, test_lbl)
