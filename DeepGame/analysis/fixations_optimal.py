import numpy as np
import sys, os
sys.path.append(os.path.abspath(os.path.join('..')))
import utils

### READ NUMBER OF FILES and NAMES ###
num_files = utils.get_no_files()
### READ NUMBER OF FRAMES in each FILE ###
file_names = utils.get_files_list(num_files)

fixations_folder = 'C:\\Users\\omossad\\Desktop\\dataset\\model_data\\labels\\text_labels\\'

for i in range(num_files):
    tiled_labels = []
    labels_file = fixations_folder + file_names[i]
    labels_file = labels_file + '.txt'
    labels = np.loadtxt(labels_file)
    print(file_names[i])
    for l in labels:
        tiled_labels.append(utils.fixation_to_tile(l[0],l[1], 8))
    consecutives = []
    counter = 0
    for j in range(len(tiled_labels)-1):
        if tiled_labels[j] == tiled_labels[j+1]:
            counter = counter + 1
        else:
            consecutives.append(counter)
            if counter > 50:
                print("trigger")
            counter = 0
    print(consecutives)
