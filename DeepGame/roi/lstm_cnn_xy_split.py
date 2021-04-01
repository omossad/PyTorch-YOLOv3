

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

########################################
###  THE DL Model for prediction		 ###
########################################
## VARIABLES ###
# input folder is where the selected data is located #
objects_folder = 'C:\\Users\\omossad\\Desktop\\dataset\\model_data\\tiled_objects\\'
labels_folder = 'C:\\Users\\omossad\\Desktop\\dataset\\model_data\\tiled_labels\\'

[W,H] = utils.get_img_dim()
num_tiles = utils.get_num_tiles()
[ts, t_overlap, fut] = utils.get_model_conf()

test_ratio = 0.3

### READ NUMBER OF FILES and NAMES ###
num_files = utils.get_no_files()
#num_files = 1
file_names = utils.get_files_list(num_files)

train_dat = []
test_dat  = []
train_lbl = []
test_lbl  = []


for i in range(num_files):
	dat = torch.load(objects_folder + file_names[i] + '.pt').numpy()
	lbl = torch.load(labels_folder + file_names[i] + '.pt').numpy()
	#dat = torch.load(objects_folder + file_names[i] + '_x.pt').numpy()
	#lbl = np.loadtxt(labels_folder + file_names[i] + '_x.txt')
	#dat_x = torch.load(objects_folder + file_names[i] + '_x.pt').numpy()
	#lbl_x = np.loadtxt(labels_folder + file_names[i] + '_x.txt')
	#dat_y = torch.load(objects_folder + file_names[i] + '_y.pt').numpy()
	#lbl_y = np.loadtxt(labels_folder + file_names[i] + '_y.txt')
	no_test = int(test_ratio*len(dat))
	no_train = len(dat) - no_test
	#if file_names[i].startswith('se_'):
	test_dat.extend(dat[no_train:])
	test_lbl.extend(lbl[no_train:])
	#else:
	train_dat.extend(dat[:no_train])
	train_lbl.extend(lbl[:no_train])
	#train_dat.extend(dat)
	#train_lbl.extend(lbl)
	#if i < 8:
	#	train_dat.extend(dat)
	#	train_lbl.extend(lbl)
	#else:
	#	test_dat.extend(dat)
	#	test_lbl.extend(lbl)

train_dat = np.asarray(train_dat)

test_dat = np.asarray(test_dat)
train_lbl = np.asarray(train_lbl)
test_lbl = np.asarray(test_lbl)
print('Training data size ' + str(train_dat.shape) + ' , and labels ' + str(train_lbl.shape))
print('Testing  data size ' + str(test_dat.shape) + ' , and labels ' + str(test_lbl.shape))


train_dat = np.reshape(train_dat, (train_dat.shape[0], ts, 2, 24))
test_dat = np.reshape(test_dat, (test_dat.shape[0], ts, 2, 24))

test_dat = torch.Tensor(test_dat)
train_lbl = train_lbl[:,-1,:,:]
test_lbl = test_lbl[:,-1,:,:]
print('Training data size ' + str(train_dat.shape) + ' , and labels ' + str(train_lbl.shape))
print('Testing  data size ' + str(test_dat.shape) + ' , and labels ' + str(test_lbl.shape))



torch.manual_seed(1)		# reproducible
EPOCH = 30
BATCH_SIZE = 8
TIME_STEP = 10				  # rnn time step
INPUT_SIZE = 24				  # rnn input size
LR = 0.001					  # learning rate


tensor_x = torch.Tensor(train_dat) # transform to torch tensor
tensor_y = torch.Tensor(train_lbl)

my_dataset = TensorDataset(tensor_x,tensor_y) # create your datset
train_loader = DataLoader(my_dataset, batch_size=BATCH_SIZE, shuffle=False) # create your dataloader


class RNN(nn.Module):
		def __init__(self):
				super(RNN, self).__init__()

				self.lstm = nn.LSTM(				 							# if use nn.RNN(), it hardly learns
						input_size=24,
						hidden_size=64,				 							# rnn hidden unit
						num_layers=1,				  			 				# number of rnn layer
						batch_first=True,		  								# input & output will has batch size as 1s dimension. e.g. (batch, time_step, input_size)
				)

				self.fc = nn.Sequential(nn.Linear(64, 32), nn.ReLU(), nn.BatchNorm1d(32), nn.Linear(32,32), nn.ReLU(), nn.Dropout(0.4), nn.Linear(32,8))

		def forward(self, x):
				# x shape (batch, time_step, input_size)
				# r_out shape (batch, time_step, output_size)
				# h_n shape (n_layers, batch, hidden_size)
				# h_c shape (n_layers, batch, hidden_size)
				r_out, (h_n, h_c) = self.lstm(x)
				fc_out = self.fc(r_out[:, -1, :])
				return fc_out

rnn_x = RNN().cuda()
print(rnn_x)
rnn_y = RNN().cuda()
print(rnn_y)

optimizer_x = torch.optim.Adam(rnn_x.parameters(), lr=LR)   # optimize all rnn parameters
optimizer_y = torch.optim.Adam(rnn_y.parameters(), lr=LR)   # optimize all rnn parameters

pos_weight_x = 5*torch.ones([8]).cuda()
pos_weight_y = 5*torch.ones([8]).cuda()
loss_func_x = nn.BCEWithLogitsLoss(pos_weight = pos_weight_x)										   # the target label is not one-hotted
loss_func_y = nn.BCEWithLogitsLoss(pos_weight = pos_weight_y)										   # the target label is not one-hotted


def my_loss_x(output, target):
		target_x = target[:,0,:]
		total_loss = 0
		for i in range(len(output)):
			total_loss += loss_func_x(output[i], target_x[i])
			#total_loss += item_loss
		#total_loss = total_loss/len(target)
		return total_loss

def my_loss_x_y(output, target):
		target_y = target[:,1,:]
		total_loss = 0
		for i in range(len(output)):
			total_loss += loss_func_y(output[i], target_y[i])
			#total_loss += item_loss
		#total_loss = total_loss/len(target)
		return total_loss

def map_tile(x_cor, y_cor):
	return int(x_cor*num_tiles + y_cor)

def test_accuracy(pre_x, pre_y, gt):
	gt_x = gt[:,0,:]
	gt_y = gt[:,1,:]
	#print(pre_x[0])
	#print(torch.topk(pre_x,2))
	pre_x = (pre_x >= 0.1).int().cpu().data.numpy().squeeze()
	pre_y = (pre_y >= 0.1).int().cpu().data.numpy().squeeze()
	#np.savetxt('..\\visualize\\predicted.txt', pre)
	#np.savetxt('..\\visualize\\labels.txt', gt)
	tp = 0
	tn = 0
	fp = 0
	fn  = 0
	total_intersection = total_overshoot = total_undershoot =  0
	total_inter_norm = total_over_norm = total_under_norm =  0
	for i in range(len(gt)):
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
				if gt_x[i][j] == 1 and gt_y[i][k] == 1:
					gt_arr[map_tile(j,k)] = 1
		#print(pre_arr)
		#print(gt_arr)
		for j in range(len(pre_arr)):
			if pre_arr[j] == 1:
				if gt_arr[j] == 1:
					tp += 1
				else:
					fp += 1
			else:
				if gt_arr[j] == 1:
					fn += 1
				else:
					tn += 1

		inter_norm = tp/(tp+fn)
		over_norm = fp/(tp+fn)
		under_norm = fn/(tp+fn)
		total_intersection += tp
		total_overshoot += fp
		total_undershoot += fn
		total_inter_norm += inter_norm
		total_over_norm += over_norm
		total_under_norm += under_norm
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

# training and testing
for epoch in range(EPOCH):
		for step, (x, y) in enumerate(train_loader):
				x_x = x[:,:,0,:]
				x_y = x[:,:,1,:]

				x_x = Variable(x_x.view(-1, ts, 24)).cuda()						# reshape x to (batch, time_step, input_size)
				x_y = Variable(x_y.view(-1, ts, 24)).cuda()

				y = Variable(y).cuda()											# batch y

				output_x = rnn_x(x_x)											# rnn output
				loss_x = my_loss_x(output_x, y)
				optimizer_x.zero_grad()											# clear gradients for this training step
				loss_x.backward()													# backpropagation, compute gradients
				optimizer_x.step()												# apply gradients

				output_y = rnn_x(x_y)											# rnn output
				loss_y = my_loss_x(output_y, y)
				optimizer_y.zero_grad()											# clear gradients for this training step
				loss_y.backward()													# backpropagation, compute gradients
				optimizer_y.step()
				#print(loss.grad)

				#if step % 50 == 0:

		test_output_x = rnn_x(Variable(test_dat[:,:,0,:]).cuda())
		test_output_y = rnn_y(Variable(test_dat[:,:,1,:]).cuda())							# (samples, time_step, input_size)
		print('Epoch: ', epoch, '| train loss x: %.4f' % loss_x.item())
		print('Epoch: ', epoch, '| train loss y: %.4f' % loss_y.item())
		test_accuracy(test_output_x, test_output_y, test_lbl)
