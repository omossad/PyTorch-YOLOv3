#from numpy import loadtxt
#from keras.models import Sequential
#from keras.layers import Dense
import torch
import numpy as np
#data = torch.load('/home/omossad/scratch/Gaming-Dataset/features/fifa/yolov3-tiny/ha_0/frame_00251.pt',map_location=lambda storage, loc: storage)
#print(data)
W = 1920
H = 1080
h_tiles = 4
v_tiles = 4
num_obj = 3
tile_w = W/h_tiles
tile_h = H/v_tiles

def process_frame(frame_name):
    frame_features = np.zeros((h_tiles,num_obj))
    print(frame_features)
    frame_data = torch.load(frame_name, map_location=lambda storage, loc: storage)
    for i in frame_data:
        x1 = int(i[0]/tile_w)
        x2 = int(i[2]/tile_w)
        obj = int(i[6])
        print(x1)
        print(x2)
        print(obj)
        frame_features[x1][obj] += 1
        if x1 != x2:
            frame_features[x2][obj] += 1
    print(frame_features)

process_frame('/home/omossad/scratch/Gaming-Dataset/features/fifa/yolov3-tiny/ha_0/frame_00251.pt')
# load the dataset
'''
dataset = loadtxt('pima-indians-diabetes.csv', delimiter=',')
# split into input (X) and output (y) variables
X = dataset[:,0:8]
y = dataset[:,8]
# define the keras model
model = Sequential()
model.add(Dense(12, input_dim=8, activation='relu'))
model.add(Dense(8, activation='relu'))
model.add(Dense(1, activation='sigmoid'))
# compile the keras model
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
# fit the keras model on the dataset
model.fit(X, y, epochs=150, batch_size=10, verbose=0)
# make class predictions with the model
predictions = model.predict_classes(X)
# summarize the first 5 cases
for i in range(5):
	print('%s => %d (expected %d)' % (X[i].tolist(), predictions[i], y[i]))
'''
