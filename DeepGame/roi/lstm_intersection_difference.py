

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
test_dat = test_dat[:,:,0,:]
test_lbl = test_lbl[:,0,0,:]

print('Training data size ' + str(train_dat.shape) + ' , and labels ' + str(train_lbl.shape))
print('Testing  data size ' + str(test_dat.shape) + ' , and labels ' + str(test_lbl.shape))



torch.manual_seed(1)		# reproducible
EPOCH = 30
BATCH_SIZE = 4
TIME_STEP = 10				  # rnn time step
INPUT_SIZE = 24				  # rnn input size
LR = 0.01					  # learning rate


tensor_x = torch.Tensor(train_dat) # transform to torch tensor
tensor_y = torch.Tensor(train_lbl)

my_dataset = TensorDataset(tensor_x,tensor_y) # create your datset
train_loader = DataLoader(my_dataset, batch_size=BATCH_SIZE, shuffle=True) # create your dataloader


class RNN(nn.Module):
		def __init__(self):
				super(RNN, self).__init__()

				self.rnn = nn.LSTM(				 # if use nn.RNN(), it hardly learns
						input_size=24,
						hidden_size=64,				 # rnn hidden unit
						num_layers=1,				   # number of rnn layer
						batch_first=True,		   # input & output will has batch size as 1s dimension. e.g. (batch, time_step, input_size)
				)
				self.out = nn.Sequential(nn.Linear(64, 64), nn.ReLU(), nn.Linear(64,8), nn.Sigmoid())


		def forward(self, x):
				# x shape (batch, time_step, input_size)
				# r_out shape (batch, time_step, output_size)
				# h_n shape (n_layers, batch, hidden_size)
				# h_c shape (n_layers, batch, hidden_size)
				r_out, (h_n, h_c) = self.rnn(x, None)   # None represents zero initial hidden state

				# choose r_out at the last time step
				out = self.out(r_out[:, -1, :])
				return out

				#return out
rnn = RNN().cuda()
print(rnn)
optimizer = torch.optim.Adam(rnn.parameters(), lr=LR)   # optimize all cnn parameters
loss_func = nn.BCELoss()										   # the target label is not one-hotted
def my_loss(output, target):
		temp_out = (output > 0.5).float()
		factor = torch.sum(torch.abs(temp_out - target))
		loss = torch.mean((output - target)**2) + factor
		return loss
# training and testing
for epoch in range(EPOCH):
		for step, (x, y) in enumerate(train_loader):
				#print(x[:,:,0,:].shape)
				#print(y[:,0,0,:].shape)
				x = x[:,:,0,:]
				y = y[:,0,0,:]
				b_x = Variable(x.view(-1, ts, 24)).cuda()						# reshape x to (batch, time_step, input_size)
				b_y = Variable(y).cuda()															# batch y

				output = rnn(b_x)														   # rnn output
				#print(output)
				#print(b_y)
				temp = (output > 0.5).float()
				#print(temp)
				#factor = torch.sum(torch.abs(temp - b_y))
				#print(factor)
				#print(output)
				#print(b_y)
				#loss = loss_func(output, b_y) 					# cross entropy loss
				loss = my_loss(output, b_y)
				optimizer.zero_grad()												   # clear gradients for this training step
				loss.backward()																 # backpropagation, compute gradients
				optimizer.step()																# apply gradients

				#if step % 50 == 0:

		test_output = rnn(Variable(test_dat).cuda())								   # (samples, time_step, input_size)
		#print(test_output)
		out = (test_output > 0.5).int()
		#print(out)
		pred_y = out.cpu().data.numpy().squeeze()
		np.savetxt('..\\visualize\\predicted.txt', pred_y)
		np.savetxt('..\\visualize\\labels.txt', test_lbl)
		#print(pred_y)
		#print(sum(sum(pred_y == test_lbl)))
		iou = 0
		corr_pre = 0
		fals_pre = 0
		for pre in range(len(pred_y)):
			intersection = 0
			union = 0
			print('predicted: ' + str(pred_y[pre]))
			print('true: ' + str(test_lbl[pre]))
			for it in range(num_tiles):
				if pred_y[pre][it] == 1 or test_lbl[pre][it] == 1:
					if pred_y[pre][it] == test_lbl[pre][it]:
						intersection += 1
					union += 1
			iou_entry = intersection/union
			if iou_entry >= 0.5:
				corr_pre +=1
			else:
				fals_pre +=1
			iou += iou_entry
		print('total_corr: ', corr_pre)
		print('total_fals: ', fals_pre)
		print('total samp: ', len(pred_y))
		accuracy = iou/len(pred_y)
		print('Epoch: ', epoch, '| train loss: %.4f' % loss.item(), '| test accuracy: %.2f' % accuracy)
