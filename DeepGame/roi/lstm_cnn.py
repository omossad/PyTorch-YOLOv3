

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
train_dat = train_dat[:,:,0,:]
train_lbl = train_lbl[:,-1,0,:]
test_dat = test_dat[:,:,0,:]
test_lbl = test_lbl[:,-1,0,:]
print('Training data size ' + str(train_dat.shape) + ' , and labels ' + str(train_lbl.shape))
print('Testing  data size ' + str(test_dat.shape) + ' , and labels ' + str(test_lbl.shape))



torch.manual_seed(1)		# reproducible
EPOCH = 30
BATCH_SIZE = 8
TIME_STEP = 10				  # rnn time step
INPUT_SIZE = 24				  # rnn input size
LR = 0.01					  # learning rate


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
				#self.out = nn.Sequential(nn.Linear(64, 64), nn.ReLU(), nn.Linear(64,8), nn.Sigmoid())

				self.fc = nn.Sequential(nn.Linear(64, 32), nn.ReLU(), nn.BatchNorm1d(32), nn.Linear(32,32), nn.ReLU(), nn.Dropout(0.4), nn.Linear(32,8), nn.Sigmoid())
				#for i in range(num_tiles):
				#	self.out.append(nn.Sequential(nn.Linear(32, 8), nn.ReLU(), nn.Linear(8,1), nn.Softmax()))
				#self.out_0 = nn.Sequential(nn.Linear(8, 8), nn.ReLU(), nn.Linear(8,1))
				#self.out_1 = nn.Sequential(nn.Linear(8, 8), nn.ReLU(), nn.Linear(8,1))
				#self.out_2 = nn.Sequential(nn.Linear(8, 8), nn.ReLU(), nn.Linear(8,1))
				#self.out_3 = nn.Sequential(nn.Linear(8, 8), nn.ReLU(), nn.Linear(8,1))
				#self.out_4 = nn.Sequential(nn.Linear(8, 8), nn.ReLU(), nn.Linear(8,1))
				#self.out_5 = nn.Sequential(nn.Linear(8, 8), nn.ReLU(), nn.Linear(8,1))
				#self.out_6 = nn.Sequential(nn.Linear(8, 8), nn.ReLU(), nn.Linear(8,1))
				#self.out_7 = nn.Sequential(nn.Linear(8, 8), nn.ReLU(), nn.Linear(8,1))
				#for i in range(num_tiles):
				#	self.out_arr[i] =
				#self.out = nn.Sequential(nn.Linear(64, 64), nn.ReLU(), nn.Linear(64,8), nn.Sigmoid())
				#return out


		def forward(self, x):
				# x shape (batch, time_step, input_size)
				# r_out shape (batch, time_step, output_size)
				# h_n shape (n_layers, batch, hidden_size)
				# h_c shape (n_layers, batch, hidden_size)
				rnn_out, (h_n, h_c) = self.lstm(x)   # None represents zero initial hidden state
				#print(rnn_out[:, -1, :])
				#print(rnn_out)
				fc_out = self.fc(rnn_out[:, -1, :])
				#print(fc_out)
				#out = torch.zeros([x.shape[0], num_tiles]).cuda()
				#out[:,0] = torch.transpose(self.out_0(fc_out)[:,-1], 0, 1)
				#out[:,1] = torch.transpose(self.out_1(fc_out)[:,-1], 0, 1)
				#out[:,2] = torch.transpose(self.out_2(fc_out)[:,-1], 0, 1)
				#out[:,3] = torch.transpose(self.out_3(fc_out)[:,-1], 0, 1)
				#out[:,4] = torch.transpose(self.out_4(fc_out)[:,-1], 0, 1)
				#out[:,5] = torch.transpose(self.out_5(fc_out)[:,-1], 0, 1)
				#out[:,6] = torch.transpose(self.out_6(fc_out)[:,-1], 0, 1)
				#out[:,7] = torch.transpose(self.out_7(fc_out)[:,-1], 0, 1)
				# choose r_out at the last time step

				#out = self.out(rnn_out[:, -1, :])
				#return out
				return fc_out

				#return out
rnn = RNN().cuda()
print(rnn)
optimizer = torch.optim.Adam(rnn.parameters(), lr=LR)   # optimize all cnn parameters
loss_func = nn.BCELoss()										   # the target label is not one-hotted
e = math.e

def custom_loss(output, target):
	output = (output >= 0.5).int()
	alpha = 1
	beta  = 1
	gamma = 1
	total_loss = 0
	for i in range(len(output)):
		intersection = 0
		overshoot = 0
		undershoot = 0
		inter_norm = 0
		if(sum(target[i]) == 0):
			target[i][3] = 1
			output[i][3] = 1
		for j in range(num_tiles):
			if output[i][j] == 1:
				if target[i][j] == 1:
					intersection += 1
				else:
					overshoot +=1
			else:
				if target[i][j] == 1:
					undershoot += 1
		inter_norm = intersection/(intersection+undershoot)
		over_norm  = overshoot/(intersection+overshoot)
		under_norm = undershoot/(intersection+undershoot)
		#print(str(inter_norm) + ' ' + str(over_norm) + ' ' + str(under_norm))
		#sample_loss = alpha*math.exp(-1+inter_norm) + beta*math.exp(-over_norm) + gamma*math.exp(-under_norm)
		sample_loss = alpha*(-1+inter_norm) + beta*over_norm + gamma*under_norm
		#sample_loss = under_norm
		total_loss += sample_loss
	total_loss = total_loss/len(output)
	#print(total_loss)
	total_loss = torch.Tensor([total_loss])
	#print(total_loss)
	loss =  Variable(total_loss, requires_grad = True).cuda()
	return loss

def my_loss(output, target):
		'''

		### OLD LOSS

		total_loss = 0
		for i in range(len(output)):
			sample_loss = 0
			for j in range(num_tiles):
				sample_loss += loss_func(output[i][j], target[i][j])
			total_loss += sample_loss
		total_loss = total_loss/len(output)
		return total_loss
		'''
		total_loss = 0
		#print(output)
		#print(target)
		for i in range(len(output)):
			item_loss = loss_func(output[i], target[i])

			#print(item_loss)
			total_loss += item_loss
		total_loss = total_loss/len(output)
		return total_loss

def test_accuracy(pre, gt):
	#print(pre)
	#print(gt)
	pre = (pre >= 0.5).int()
	pre = pre.cpu().data.numpy().squeeze()
	np.savetxt('..\\visualize\\predicted.txt', pre)
	np.savetxt('..\\visualize\\labels.txt', gt)
	tp = 0
	tn = 0
	fp = 0
	fn = 0
	total_intersection = 0
	total_overshoot = 0
	total_undershoot = 0
	total_inter_norm = 0
	total_over_norm = 0
	total_under_norm = 0
	for i in range(len(pre)):
		intersection = 0
		overshoot = 0
		undershoot = 0
		inter_norm = 0
		over_norm = 0
		under_norm = 0
		#print(i)
		#print(pre[i])
		#print(gt[i])
		if(sum(gt[i]) == 0):
			gt[i][3] = 1
			pre[i][3] = 1
		for j in range(num_tiles):
			if pre[i][j] == 1:
				if gt[i][j] == 1:
					intersection += 1
					tp += 1
				else:
					overshoot +=1
					fp += 1
			else:
				if gt[i][j] == 1:
					undershoot += 1
					fn += 1
				else:
					tn += 1
		#print(intersection)
		#print(overshoot)
		#print(undershoot)
		#if intersection == 0:
		#	inter_norm = 0
		#else:
		#print(sum(gt[i]))
		inter_norm = intersection/(intersection+undershoot)
		over_norm = overshoot/(intersection+undershoot)
		under_norm = undershoot/(intersection+undershoot)
		total_intersection += intersection
		total_overshoot += overshoot
		total_undershoot += undershoot
		total_inter_norm += inter_norm
		total_over_norm += over_norm
		total_under_norm += under_norm

	print("number of samples: " + str(len(pre)))
	print("intersection : " + str(total_intersection) + "-------" + str(total_intersection/len(pre)))
	print("overshoot :" + str(total_overshoot) + "-------" + str(total_overshoot/len(pre)))
	print("undershoot :" + str(total_undershoot) + "-------" + str(total_undershoot/len(pre)))
	print("intersection normalized :" + str(total_inter_norm) + "-------" + str(total_inter_norm/len(pre)))
	print("overshoot normalized :" + str(total_over_norm) + "-------" + str(total_over_norm/len(pre)))
	print("undershoot normalized :" + str(total_under_norm) + "-------" + str(total_under_norm/len(pre)))
	print("TP: " + str(tp))
	print("TN: " + str(tn))
	print("FP: " + str(fp))
	print("FN: " + str(fn))


# training and testing
for epoch in range(EPOCH):
		for step, (x, y) in enumerate(train_loader):
				#print(x[:,:,0,:].shape)
				#print(y[:,0,0,:].shape)
				#x = x[:,:,0,:]
				#y = y[:,0,0,:]
				b_x = Variable(x.view(-1, ts, 24)).cuda()						# reshape x to (batch, time_step, input_size)
				b_y = Variable(y).cuda()										# batch y
				rnn.zero_grad()
				optimizer.zero_grad()

				output = rnn(b_x)												# rnn output
				loss = my_loss(output, b_y)
				#loss = custom_loss(output, b_y)
				#print(loss.data)
				#print(loss.grad)
				optimizer.zero_grad()											# clear gradients for this training step
				loss.backward()													# backpropagation, compute gradients
				optimizer.step()												# apply gradients
				#print(loss.grad)

				#if step % 50 == 0:

		test_output = rnn(Variable(test_dat).cuda())							# (samples, time_step, input_size)
		#print(out)

		#print(pred_y)
		#print(sum(sum(pred_y == test_lbl)))
		print('Epoch: ', epoch, '| train loss: %.4f' % loss.item())
		test_accuracy(test_output, test_lbl)
