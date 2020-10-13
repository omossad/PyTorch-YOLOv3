import numpy as np
import math
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import mean_squared_error
## DUPLICATE ROI ##
# This function duplicates the ROI region
# accross multiple frames
top_x_gt = 0.48
top_y_gt = 0.463
w = 0.0417
h = 0.074
W = 1920.0
H = 1080.0
CU_size = 16
K = 3
diagonal = math.sqrt(W**2 + H**2);

'''
float e = 1/(K*ROI[block_ind]);
weights[block_ind] = exp(-1.0*e);


float dist = max(1.0f, (float)sqrt(pow(xpixIndex - yMid, 2.0) + pow(ypixIndex - xMid, 2.0)));
					float e = (K*dist/ diagonal) / (importance[ROIs[r].category]) ;
					sumDist = sumDist + exp(-1.0*e);

unsigned int xTop=ROIs[r].x;
			unsigned int yTop=ROIs[r].y;
			unsigned int xBottom=xTop + ROIs[r].width;
			unsigned int yBottom=yTop + ROIs[r].height;
			xTop = xTop / CU_SIZE;
			yTop = yTop / CU_SIZE;
			xBottom = xBottom / CU_SIZE;
			yBottom = yBottom / CU_SIZE;
			for (unsigned int j = yTop; j <= yBottom; j++)
			{
				for (unsigned int k = xTop; k <= xBottom; k++)
				{
					if(ROIs[r].category==PLAYER)
						ROI[k+j*widthDelta]=0.45;
					else if(ROIs[r].category==ENEMY)
						ROI[k+j*widthDelta]=0.3;
					else if (ROIs[r].category == INFORMATION_MAP)
						ROI[k + j * widthDelta] = 0.15;
				}

'''
def duplicate_ROI(output_folder, no_frames):
	for i in range(no_frames):
		f = open(opt_folder + "roi"+str(i)+".txt", "w")
		f.write("0 " + str(top_x) + " " + str(top_y) + " " + str(w) + " " + str(h))
		#f.write("\n")
		#f.write("3 " + str(top_x+0.2) + " " + str(top_y-0.3) + " " + str(w) + " " + str(h))
		#f.close()
	return

def create_ROI_arr(top_x, top_y, w, h):
	arr = np.zeros((68, 120))
	start_x = math.floor(top_x*W/CU_size)
	start_y = math.floor(top_y*H/CU_size)
	end_x = math.ceil(start_x + w*W/CU_size)
	end_y = math.ceil(start_y + h*H/CU_size)
	arr[start_y:end_y,start_x:end_x] = 1
	#print(start_x)
	#print(start_y)
	#print(arr[np.where(arr == 1)])
	return arr

def create_weights_arr(ROI_arr, top_x, top_y):
	hd = ROI_arr.shape[0]
	wd = ROI_arr.shape[1]
	arr = np.zeros((hd,wd))
	mid_x_pix = (top_x + w/2)*W
	mid_y_pix = (top_y + h/2)*H
	for i in range(hd):
		for j in range(wd):
			if ROI_arr[i][j] > 0:
				e = 1/(K*ROI_arr[i][j])
				arr[i][j] =  math.exp(-1.0*e)
			else:
				y_pix = i*CU_size + CU_size/2
				x_pix = j*CU_size + CU_size/2

				dist = max(1.0, math.sqrt((mid_y_pix-y_pix)**2 + (mid_x_pix-x_pix)**2));
				#dist = math.sqrt((mid_x_pix-x_pix)**2 + (mid_y_pix-y_pix)**2);
				e = (K*dist/ diagonal);
				arr[i][j] = math.exp(-1.0*e);
	#print(arr)
	#arr = arr / arr.mean()
	arr = arr / arr.sum()
	#print(arr)
	#print(arr.sum())
	return arr
'''
opt_folder = 'C:\\Users\\omossad\\Desktop\\dataset\\encoding\\ga15\\'
num_frames = 670
duplicate_ROI(opt_folder, num_frames)
'''
roi_arr = create_ROI_arr(top_x_gt, top_y_gt, w, h)
weights_arr = create_weights_arr(roi_arr, top_x_gt, top_y_gt)
#print(weights_arr)
#new_top_x = top_x - w
#new_top_y = top_y - h


for i in range(10):
	new_top_x = top_x_gt + i*w
	#print(new_top_x+w)
	new_top_y = top_y_gt + i*h
	#print(new_top_y+h)
	new_top_y = top_y_gt
	#new_top_x = top_x
	#new_top_x = -0.2
	#new_top_y = -0.2
	roi_arr_pre = create_ROI_arr(new_top_x, new_top_y, w, h)
	weights_arr_pre = create_weights_arr(roi_arr_pre, new_top_x, new_top_y)
	temp = weights_arr_pre - weights_arr

	data_range=weights_arr.max() - weights_arr.min()
	ssim_noise = ssim(weights_arr, weights_arr_pre, data_range=data_range)
	mse_const = mean_squared_error(weights_arr, weights_arr_pre)
	#print(mse_const)
	#print(ssim_noise)
	#print(temp)
	print("for " + str(i) + " The sum of absolute difference is : " + str(np.abs(temp).sum()))
	print("for " + str(i) + " The mean squared error is : " + str(mse_const))
	print("for " + str(i) + " The ssim is : " + str(ssim_noise))
