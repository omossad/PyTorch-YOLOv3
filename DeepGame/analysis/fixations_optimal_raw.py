import numpy as np
import sys, os
sys.path.append(os.path.abspath(os.path.join('..')))
import utils

### READ NUMBER OF FILES and NAMES ###
num_files = utils.get_no_files()
### READ NUMBER OF FRAMES in each FILE ###
file_names = utils.get_files_list(num_files)
outlier_threshold = 50
num_tiles = utils.get_num_tiles()
fixations_folder = 'C:\\Users\\omossad\\Desktop\\dataset\\model_data\\labels\\text_labels\\'

cons_tot = []
for i in range(num_files):
    tiled_labels = []
    labels_file = fixations_folder + file_names[i]
    labels_file = labels_file + '.txt'
    labels = np.loadtxt(labels_file)
    print(file_names[i])
    #for l in labels:
    #    tiled_labels.append(utils.fixation_to_tile(l[0],l[1]))
    consecutives = []
    counter = 0
    for j in range(len(labels)-1):
        if labels[j][0] == labels[j+1][0] and labels[j][1] == labels[j+1][1]:
            counter = counter + 1
        else:
            consecutives.append(counter)
            if counter > outlier_threshold:
                print("Same fixation for more thant 50 frames")
            counter = 0
    print(consecutives)
    cons_tot.extend(consecutives)
np.savetxt('consecutive_fixations.txt', np.asarray(cons_tot))
