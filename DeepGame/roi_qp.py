import numpy as np
import math

## DUPLICATE ROI ##
# This function duplicates the ROI region
# accross multiple frames
top_x = 0.48
top_y = 0.463
w = 0.0417
h = 0.074
'''
float e = 1/(K*ROI[block_ind]);
weights[block_ind] = exp(-1.0*e);


float dist = max(1.0f, (float)sqrt(pow(xpixIndex - yMid, 2.0) + pow(ypixIndex - xMid, 2.0)));
					float e = (K*dist/ diagonal) / (importance[ROIs[r].category]) ;
					sumDist = sumDist + exp(-1.0*e);

'''
def duplicate_ROI(output_folder, no_frames):

    for i in range(no_frames):
        f = open(opt_folder + "roi"+str(i)+".txt", "w")
        f.write("0 " + str(top_x) + " " + str(top_y) + " " + str(w) + " " + str(h))
        f.close()

    return

def create_ROI_arr(top_x, top_y, w, h):
    arr = np.zeros((120,68))
    start_x = math.floor(top_x*1920/16)
    start_y = math.floor(top_y*1080/16)
    print(arr)
    return arr

def create_weights_arr(ROI_arr):
    arr = np.zeros((120,68))
    print(arr)

    return arr
'''
opt_folder = 'C:\\Users\\omossad\\Desktop\\dataset\\encoding\\ga15\\'
num_frames = 670
duplicate_ROI(opt_folder, num_frames)
'''
roi_arr = create_ROI_arr(top_x, top_y, w, h)
weights_arr = create_weights_arr(roi_arr)
