from __future__ import division

from models import *
from yolo_utils import *
from datasets import *

import sys
import time
import datetime

import torch
from torch.autograd import Variable

import numpy as np
import cv2
import ctypes

###### PARAMS ######
    # PARAMETERS

game = 'fifa'
#game = 'csgo'

if game == 'fifa':
    weights_path = "fifa.pth"
    model_def = "base_model.cfg"
    num_objects = 3
    class_path= "classes.names"
    rnn_model_path = 'D:\\Encoding\\encoding_files\\fifa\\model\\checkpoints\\checkpoint_80.pt'
else:
    weights_path = "yolov3-tiny.weights"
    model_def = "yolov3-tiny.cfg"
    class_path = "coco.names"
    weights_path = "fifa.pth"
    model_def = "base_model.cfg"
    class_path= "classes.names"
    num_objects = 80
    rnn_model_path = 'D:\\Encoding\\encoding_files\\csgo\\model\\checkpoints\\checkpoint_80.pt'

img_size = 416
model_def = "base_model.cfg"
conf_thres = 0.05
nms_thres = 0.1
image_folder = "C:\\Users\\omossad\\Desktop\\ga_shared\\"
out_folder = "C:\\Users\\omossad\\Desktop\\ga_shared\\"
num_tiles = 8
[ts, t_overlap, fut] = [10, 2, 2]

fps = 30
dp_fps = 10

obj_det_interval = fps // dp_fps
predict_interval = fut * (fps // dp_fps) 

user32 = ctypes.windll.user32
[W, H] = [user32.GetSystemMetrics(0), user32.GetSystemMetrics(1)]
print(W, H)
#[W, H] = [1600,900]
[W, H] = [1920, 1080]


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

def arr_to_roi(arr_x, arr_y):
    to_write = []
    start_x = 0
    end_x = num_tiles - 1
    start_y = 0
    end_y = num_tiles - 1
    for i in range(len(arr_x)):
        if arr_x[i] == 1 and start_x == 0:
            start_x = i
        if arr_x[i] == 1 and end_x < num_tiles:
            end_x = i
    for i in range(len(arr_y)):
        if arr_y[i] == 1 and start_y == 0:
            start_y = i
        if arr_y[i] == 1 and end_y < num_tiles:
            end_y = i
    to_write = '0 ' + str(start_x/num_tiles+ (1/num_tiles*(1/2))) + ' ' + str(start_y/num_tiles + (1/num_tiles*(1/2))) + ' ' + str((end_x - start_x + 1)/num_tiles) + ' ' + str((end_y - start_y + 1)/num_tiles) + '\n'
    if game == "fifa":
        to_write += '1 0.135 0.118 0.165 0.077 \n'
        to_write += '1 0.49 0.872 0.1375 0.144 \n'
        #to_write += '1 0.053 0.08 0.165 0.077 \n'
        #to_write += '1 0.43 0.8 0.1375 0.144 \n'
    else:
        to_write += '1 0.009 0.077 0.134 0.23 \n' 
    return [to_write]

'''
    to_write = []
    for i in range(len(arr_x)):
        for j in range(len(arr_y)):
            if arr_x[i] == 1 and arr_y[j] == 1:
                # NEED IMPROVEMENT
                roi_txt = '0 ' + str(i/num_tiles) + ' ' + str(j/num_tiles) + ' ' + str(1/num_tiles) + ' ' + str(1/num_tiles) + '\n'
                to_write.append(roi_txt)
    return to_write
'''



#### LSTM MODEL ####
class RNN(nn.Module):
    def __init__(self):
        super(RNN, self).__init__()
        self.lstm_x = nn.LSTM(
            input_size=INPUT_SIZE,
            hidden_size=128,
            num_layers=1,
            batch_first=True,
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
        r_out_x, (h_n_x, h_c_x) = self.lstm_x(x[:,:,0,:])
        r_out_y, (h_n_y, h_c_y) = self.lstm_y(x[:,:,1,:])
        fc_out_x = self.fc_x(r_out_x[:, -1, :])
        fc_out_y = self.fc_y(r_out_y[:, -1, :])
        return fc_out_x, fc_out_y



if __name__ == "__main__":


    # Set up object detection model
    model = Darknet(model_def, img_size).to(device)

    if weights_path.endswith(".weights"):
        # Load darknet weights
        model.load_darknet_weights(weights_path)
    else:
        # Load checkpoint weights
        model.load_state_dict(torch.load(weights_path))

    model.eval()  # Set in evaluation mode
    classes = load_classes(class_path)  # Extracts class labels from file
    

    # Set up LSTM model
    

    rnn_model = torch.load(rnn_model_path)
    rnn_model.eval()
    rnn = rnn_model.to(device)
	
    	







    yuv_filename = 'C:\\Users\\omossad\\Desktop\\capture.yuv'
    f = open(yuv_filename, 'rb')

    frame_no = 0
    img = np.zeros((1, 3, img_size, img_size))

    objects_arr = torch.zeros([2, num_tiles, num_objects], dtype=torch.float)
    tmp_inpt_arr = torch.zeros([1, ts, 2, num_tiles, num_objects], dtype=torch.float)
    inpt_arr = torch.zeros([1, ts, 2, num_tiles, num_objects], dtype=torch.float)
    yuv_frame_size = W*H*3//2

    inpt_counter = 0
    total_time = 0
    od_time = 0
    od_count = 0
    lstm_time = 0
    lstm_count = 0
    while True:
        if frame_no % obj_det_interval > 0:
            frame_no += 1
            continue

        
        
        
        try:
            f.seek(yuv_frame_size*frame_no)    
            yuv = np.frombuffer(f.read(yuv_frame_size), dtype=np.uint8).reshape((H*3//2, W))
            #time_elapsed = time.time()
            #print(time_elapsed)
        except:
            print("waiting for frame")
            #print(time.time())
            #if time.time() - time_elapsed > 10:
            #time_elapsed += 1
            #if time_elapsed > 10:
            #    quit()
            continue

        bgr = cv2.cvtColor(yuv, cv2.COLOR_YUV2BGR_I420)
        #cv2.imshow('rgb', bgr)
        #cv2.waitKey(500)
        resized = cv2.resize(bgr, (img_size,img_size))
        img[0] = np.transpose(resized, (2, 0, 1))
        input_tensor = torch.from_numpy(img).to(device).float()
        od_start = time.time()
        detections = model(input_tensor)
        detections = non_max_suppression(detections, conf_thres, nms_thres)
        detections = detections[0]
        for obj in range(len(detections)):
            x1 = detections[obj][0]/img_size
            y1 = detections[obj][1]/img_size
            x2 = detections[obj][2]/img_size
            y2 = detections[obj][3]/img_size
            X = int(min(num_tiles - 1, (x1+x2)/2.0 * num_tiles))
            Y = int(min(num_tiles - 1, (y1+y2)/2.0 * num_tiles))
            objects_arr[0][X][int(detections[obj][6])] += float(detections[obj][4])
            objects_arr[1][Y][int(detections[obj][6])] += float(detections[obj][4])
        if inpt_counter < 10:
            tmp_inpt_arr[0][inpt_counter] = objects_arr
            inpt_counter += 1
            frame_no = frame_no + 1
            inpt_arr[0][:] = tmp_inpt_arr[0][:]
            continue
        inpt_arr[0][:-1] = tmp_inpt_arr[0][1:]
        inpt_arr[0][-1] = objects_arr
        tmp_inpt_arr[0][:] = inpt_arr[0][:]


        od_time += time.time() - od_start
        od_count = od_count + 1

        if frame_no % predict_interval == 0:
            lstm_start = time.time()
            test_output_x, test_output_y = rnn(Variable(torch.reshape(inpt_arr, (1, ts, 2, num_tiles * num_objects))).to(device))
            lstm_time += time.time() - lstm_start
            lstm_count += 1
            #print(test_output_x, test_output_y)
            pre_x = (test_output_x >= 0.5).int().cpu().data.numpy().squeeze()
            pre_y = (test_output_y >= 0.5).int().cpu().data.numpy().squeeze()
            #print(pre_x, pre_y)
            print(frame_no)
            print("Updating ROIS")
            out_f = open(out_folder + 'roi0.txt', 'w')
            cnt = 0
            for l in arr_to_roi(pre_x, pre_y):
                if cnt < 3:
                    out_f.write(l)
                    cnt += 1
            out_f.close()
        frame_no = frame_no + 1
        out_info = open(out_folder + 'info.txt', 'w')
        out_info.write(str(od_count) + ' ' + str(od_time/od_count) +'\n')
        out_info.write(str(lstm_count) + ' ' + str(lstm_time/lstm_count) +'\n')
        out_info.close()
    
    
