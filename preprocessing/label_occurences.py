import numpy as np
import pickle
import os

#######################
### Check number    ###
### of labels 	    ###
### in file   	    ###
#######################

dir = '../data/8x8/'
nClasses = 64
list = os.listdir(dir)
number_files = len(list)
print('Reading a total of ' + str(number_files) + ' files')
print('Total Classes', nClasses)

indices = [i for i in range(number_files)]

#labels = []


def process_data(index, labels):
    pickle_in = open(dir+ 'data_' + str(index) + ".pickle","rb")
    data = pickle.load(pickle_in)
    #print('Data #' + str(index) + 'Loaded Successfully')
    for key in data:
        labels.append(data[key][0])

for i in indices:
    labels = []
    process_data(i, labels)
    num_samples = len(labels)
    #nClasses = np.max(labels) + 1
    if i == 0:
        print('filename', end = '')
        for j in range(nClasses):
            print(', ' + str(j), end = '')
        print('\n', end = '')
    print('file_' + str(i), end = '')
    for j in range(nClasses):
        print(', ' + str(labels.count(j)), end = '')
    print('\n', end = '')


print('\n' + 'AGGREGATION' + '\n')
labels = []
for i in indices:
    process_data(i, labels)
print('Number of labels is: ' + str(len(labels)))
for i in range(nClasses):
    print('Class' + str(i) + '  has '+ str(labels.count(i)) + ' occurences')
