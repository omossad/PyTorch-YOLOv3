#from numpy import loadtxt
#from keras.models import Sequential
#from keras.layers import Dense
import torch
#import tensorflow as tf
#from tensorflow import keras

import numpy as np
#data = torch.load('/home/omossad/scratch/Gaming-Dataset/features/fifa/yolov3-tiny/ha_0/frame_00251.pt',map_location=lambda storage, loc: storage)
#print(data)
W = 1920
H = 1080
h_tiles = 8
v_tiles = 8
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
    return frame_features




max_files = 10
def read_info():
    file_names = []
    num_files = 0
    with open('frames_info') as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        for row in csv_reader:
            if num_files == 0:
                num_files += 1
            elif num_files < max_files+1:
                file_names.append(row[0])
                num_files += 1
            else:
                break
    print("Total number of files is:", num_files)
    return file_names

all_filenames = read_info()
print(all_filenames)

#data_dir  = '/home/omossad/projects/def-hefeeda/omossad/roi_detection/temporary_data/data/resnet/'
#data_dir  = '/home/omossad/scratch/Gaming-Dataset/features/fifa/mobilenetV2/'
data_dir  = '/home/omossad/scratch/Gaming-Dataset/features/fifa/yolov3-tiny/'
#data_dir  = '/home/omossad/scratch/Gaming-Dataset/features/fifa/resnet152/'
label_dir = '/home/omossad/scratch/Gaming-Dataset/frame_labels/fifa/'

train_list = []
test_list = []
train_label = []
test_label = []

for f in all_filenames:
    images = sorted(glob.glob(data_dir + f + '/*'))
    labels = sorted(glob.glob(label_dir + f + '/*'))
    if f.startswith('ha_8'):
        test_list.extend(images)
        test_label.extend(labels)
    elif f.startswith('ha_9'):
        test_list.extend(images)
        test_label.extend(labels)
    else:
        train_list.extend(images)
        train_label.extend(labels)

train_list = np.asarray(train_list)
test_list  = np.asarray(test_list)
train_label = np.asarray(train_label)
test_label = np.asarray(test_label)

print(train_list.shape)
print(train_label.shape)
print(test_list.shape)
print(test_label.shape)
'''
train_images = []
train_labels = []
test_images = []
test_labels = []

for i in train_list:
    train_images.append()


process_frame('/home/omossad/scratch/Gaming-Dataset/features/fifa/yolov3-tiny/ha_0/frame_00251.pt')
# load the dataset
model = keras.Sequential([
    keras.layers.Flatten(input_shape=(h_tiles,num_obj)),
    keras.layers.Dense(128, activation='relu'),
    keras.layers.Dense(h_tiles)
])
model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])
model.fit(train_images, train_labels, epochs=10)
test_loss, test_acc = model.evaluate(test_images,  test_labels, verbose=2)

print('\nTest accuracy:', test_acc)


probability_model = tf.keras.Sequential([model,
                                         tf.keras.layers.Softmax()])
predictions = probability_model.predict(test_images)
print(predictions[0])
print(np.argmax(predictions[0]))


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
