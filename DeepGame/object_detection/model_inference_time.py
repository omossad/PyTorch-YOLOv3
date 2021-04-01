from __future__ import print_function
from __future__ import division

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

import sys
import time
import datetime

from models import *
from yolo_utils import *
from datasets import *


from PIL import Image




game ='fifa'

[W,H] = [2560,1440]
num_tiles = 8
[ts, t_overlap, fut] = [10,2,2]

obj_classes = 80
obj_classes = 3
INPUT_SIZE = num_tiles * obj_classes				# rnn input size
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


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




# Set up model
model_od = Darknet("base_model.cfg", img_size=416).to(device)
model_od.load_state_dict(torch.load("fifa.pth"))
model_od.eval()  # Set in evaluation mode

dataloader = DataLoader(
	ImageFolder("temp_input\\", img_size=416),
	batch_size=1,
	shuffle=False,
)

classes = load_classes("classes.names")  # Extracts class labels from file

Tensor = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor

imgs = []  # Stores image paths
img_detections = []  # Stores detections for each image index

print("\nPerforming object detection:")
#prev_time = time.time()
mean_syn = 0
std_syn = 0

for batch_i, (img_paths, input_imgs) in enumerate(dataloader):
	# Configure input
	input_imgs = Variable(input_imgs.type(Tensor))
	starter, ender = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
	repetitions = 300
	timings=np.zeros((repetitions,1))
	for _ in range(10):
		detections = model_od(input_imgs)
		detections = non_max_suppression(detections, 0.05, 0.1)
	#with torch.no_grad():
	for rep in range(repetitions):
		starter.record()
		with torch.no_grad():
			detections = model_od(input_imgs)
			detections = non_max_suppression(detections, 0.05, 0.1)
        #_ = pretrained_model(X)
		ender.record()
        # WAIT FOR GPU SYNC
		torch.cuda.synchronize()
		curr_time = starter.elapsed_time(ender)
		timings[rep] = curr_time
	mean_syn += np.sum(timings) / repetitions
	std_syn += np.std(timings)




	#current_time = time.time()
	#inference_time = datetime.timedelta(seconds=current_time - prev_time)
	#prev_time = current_time
	#print("\t+ Batch %d, Inference Time: %s" % (batch_i, inference_time))
	imgs.extend(img_paths)
	img_detections.extend(detections)

print("Object Detection and NON max suppression average running time: ", mean_syn/10)
print("Object Detection and NON max suppression running time std: ",std_syn/10)

# Iterate through images and save plot of detections
for img_i, (path, detections) in enumerate(zip(imgs, img_detections)):
	if detections is not None:
		# Rescale boxes to original image
		detections = rescale_boxes(detections, 416, (1080,1920))
	#print(detections)



#model_path = 'C:\\Users\\omossad\\Desktop\\dataset\\model_data\\nhl\\checkpoints\\checkpoint_19.pt'
## FI:FA ##
model_path = 'D:\\Encoding\\encoding_files\\fifa\\model\\checkpoints\\checkpoint_86.pt'

## CS:GO ##
#model_path = 'D:\\Encoding\\encoding_files\\csgo\\model\\checkpoints\\checkpoint_99.pt'


## NH:L ##
#model_path = 'D:\\Encoding\\encoding_files\\nhl\\model\\checkpoints\\checkpoint_91.pt'

## NB:A ##
#model_path = 'D:\\Encoding\\encoding_files\\nba\\model\\checkpoints\\checkpoint_94.pt'

model = torch.load(model_path)
model.eval()




#objects_folder = 'C:\\Users\\omossad\\Desktop\\dataset\\model_data\\' + game + '\\tiled_objects\\'
#labels_folder  = 'C:\\Users\\omossad\\Desktop\\dataset\\model_data\\' + game + '\\tiled_labels\\'
objects_folder = 'D:\\Encoding\\encoding_files\\fifa\\model\\tiled_objects\\'



### READ NUMBER OF FILES and NAMES ###
#num_files = 1
test_dat  = []
test_lbl  = []


dat = torch.load(objects_folder + 'fifa.pt').numpy()


test_dat.extend(dat[:10])

test_dat = np.asarray(test_dat)

print('________________')
print('Testing  data size ' + str(test_dat.shape))
print('________________')


test_dat = np.reshape(test_dat, (test_dat.shape[0], ts, 2, num_tiles*obj_classes))

test_dat = torch.Tensor(test_dat)
temp_test_lbl = []


print('Testing  data size ' + str(test_dat.shape))
print('________________')



torch.manual_seed(1)								# reproducible
TIME_STEP = 2				  						# rnn time step


rnn = model.cuda()

starter, ender = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
repetitions = 300
timings=np.zeros((repetitions,1))
for _ in range(10):
	test_output_x, test_output_y = rnn(Variable(test_dat).cuda())							# (samples, time_step, input_size)
for rep in range(repetitions):
	starter.record()
	test_output_x, test_output_y = rnn(Variable(test_dat).cuda())							# (samples, time_step, input_size)
	ender.record()
	# WAIT FOR GPU SYNC
	torch.cuda.synchronize()
	curr_time = starter.elapsed_time(ender)
	timings[rep] = curr_time
mean_syn = np.sum(timings) / repetitions
std_syn = np.std(timings)
print("Model average running time: ", mean_syn)
print("Model running time std: ",std_syn)
