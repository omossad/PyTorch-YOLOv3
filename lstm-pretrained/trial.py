import glob
import numpy as np
from sklearn.model_selection import train_test_split

foldername = "/home/omossad/scratch/temp/roi/images/"
print(foldername)
print(foldername.split("/")[0:-1])

ha_0_images = sorted(glob.glob("/home/omossad/scratch/temp/roi/images/ha_0_*"))
ha_0_labels = sorted(glob.glob("/home/omossad/scratch/temp/roi/labels/ha_0_*"))
print(len(ha_0_images))
print(len(ha_0_labels))

filename = '/home/omossad/projects/def-hefeeda/omossad/roi_detection/temporary_data/ha_0_labels/frame_00647.txt'
with open(filename, 'rb+') as f:
    a = f.read()
    print(f.seek(2))
    print(f.read(1))
    #f.seek(f.tell()-1,2)    # f.seek(0,2) is legal for last char in both python 2 and 3 though
    #print(f.read())

time_steps = 4


def process_data(images):
    num_images = len(images)
    image_indices = np.arange(0,num_images)
    indices = np.array([ image_indices[i:i+time_steps] for i in range(num_images-time_steps) ])
    images=np.asarray(images)
    print(images[indices].shape)
    return images[indices]

def process_labels(labels):
    num_labels = len(labels)
    indices = np.arange(time_steps-1,num_labels-1)
    labels=np.asarray(labels)
    print(labels[indices].shape)
    return labels[indices]


a = process_data(ha_0_images)
b = process_labels(ha_0_labels)
train_list, test_list, train_label, test_label = train_test_split(a, b, test_size=0.25, random_state=42)

print(len(train_list))
print(len(test_list))
print(train_list[10])
print(test_list[5])
print(train_label[10])
print(test_label[5])
