import numpy as np
import math
from skimage.metrics import structural_similarity as ssim_sk
from skimage.metrics import mean_squared_error
import sys, os
sys.path.append(os.path.abspath(os.path.join('..')))
import utils
from pytorch_msssim import ssim, ms_ssim, SSIM, MS_SSIM

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
diagonal = math.sqrt(W**2 + H**2)
'''
ALPHA=3.200
BETA=-1.367
realWidth=1920.0
realHeight=1080.0
MIN_LAMBDA = 0.1
MAX_LAMBDA = 10000.0
C1=4.2005
C2=13.7112
GOP=90
wd=120
hd=68
QP_BASE=22
pixelsPerBlock = CU_size*CU_size
'''
'''
float e = 1/(K*ROI[block_ind])
weights[block_ind] = exp(-1.0*e)


float dist = max(1.0f, (float)sqrt(pow(xpixIndex - yMid, 2.0) + pow(ypixIndex - xMid, 2.0)))
					float e = (K*dist/ diagonal) / (importance[ROIs[r].category])
					sumDist = sumDist + exp(-1.0*e)

unsigned int xTop=ROIs[r].x
			unsigned int yTop=ROIs[r].y
			unsigned int xBottom=xTop + ROIs[r].width
			unsigned int yBottom=yTop + ROIs[r].height
			xTop = xTop / CU_SIZE
			yTop = yTop / CU_SIZE
			xBottom = xBottom / CU_SIZE
			yBottom = yBottom / CU_SIZE
			for (unsigned int j = yTop j <= yBottom j++)
			{
				for (unsigned int k = xTop k <= xBottom k++)
				{
					if(ROIs[r].category==PLAYER)
						ROI[k+j*widthDelta]=0.45
					else if(ROIs[r].category==ENEMY)
						ROI[k+j*widthDelta]=0.3
					else if (ROIs[r].category == INFORMATION_MAP)
						ROI[k + j * widthDelta] = 0.15
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
'''
def create_ROI_arr(rois, w, h):
	arr = np.zeros((68, 120))
	for roi in rois:
		start_x = math.floor(roi[0]*W/CU_size)
		start_y = math.floor(roi[1]*H/CU_size)
		end_x = math.ceil(start_x + w*W/CU_size)
		end_y = math.ceil(start_y + h*H/CU_size)
		arr[start_y:end_y,start_x:end_x] = 1
	#print(start_x)
	#print(start_y)
	#print(arr[np.where(arr == 1)])
	return arr

def create_weights_arr(ROI_arr, rois):
	hd = ROI_arr.shape[0]
	wd = ROI_arr.shape[1]
	arr = np.zeros((hd,wd))

	for i in range(hd):
		for j in range(wd):
			if ROI_arr[i][j] > 0:
				e = 1/(K*ROI_arr[i][j])
				arr[i][j] =  math.exp(-1.0*e)
			else:
				y_pix = i*CU_size + CU_size/2
				x_pix = j*CU_size + CU_size/2
				sumDist = 0
				for roi in rois:
					mid_x_pix = (roi[0] + w/2)*W
					mid_y_pix = (roi[1] + h/2)*H
					dist = max(1.0, math.sqrt((mid_y_pix-y_pix)**2 + (mid_x_pix-x_pix)**2))
					#dist = math.sqrt((mid_x_pix-x_pix)**2 + (mid_y_pix-y_pix)**2)
					e = (K*dist/ diagonal)
					sumDist = sumDist +  math.exp(-1.0*e)
				arr[i][j] = sumDist / len(rois)
	#print(arr)
	#arr = arr / arr.mean()
	arr = arr / arr.sum()
	#print(arr)
	#print(arr.sum())
	return arr

opt_folder = 'C:\\Users\\omossad\\Desktop\\dataset\\encoding\\ga15\\'
num_frames = 670
duplicate_ROI(opt_folder, num_frames)
'''

#print(weights_arr)
#new_top_x = top_x - w
#new_top_y = top_y - h

def CLIP(min_, max_, value):
	return max(min_, min(max_, value))

def clipLambda(lambda_):
	if math.isnan(lambda_):
		lambda_ = MAX_LAMBDA
	else:
		lambda_ = CLIP(MIN_LAMBDA, MAX_LAMBDA, lambda_)

def roundRC(d):
	return int(d + 0.5)

def LAMBDA_TO_QP(lambda_):
	return int(CLIP(0.0, 51.0, roundRC(C1 * math.log(lambda_) + C2)))

def QPToBits(QP):
	lambda_ = math.exp((QP - C2) / C1)
	bpp = ((lambda_/ALPHA)*1.0) ** (1.0 / BETA)
	return bpp * (64.0**2.0)
'''
def calcQPLambda(weights, roi_array, bitrate, fps):
	roi_array = roi_array.flatten()
	weights = weights.flatten()
	bitsPerBlock = np.zeros((hd*wd))
	QP = np.zeros((hd*wd))
	avg_bits_per_pic = bitrate/fps
	targetbppFrame = avg_bits_per_pic / (realWidth * realHeight)
	FRAME_LAMBDA = ALPHA * (targetbppFrame**BETA)
	#print(FRAME_LAMBDA)
	FRAME_LAMBDA = clipLambda(FRAME_LAMBDA)
	bitsAlloc = avg_bits_per_pic
	#if frames % GOP == 0:
	#	bitsAlloc = bitsAlloc
	qp_frame = LAMBDA_TO_QP(ALPHA*(bitsAlloc / (realWidth*realHeight))**BETA)
	qp_delta = 0.0
	extra = 0.0
	sumSaliencyImportantAreas = 0.0
	sumSaliency = 0.0
	reassigned = 0
	avgLambda = 0.0
	block_ind=0
	avg_qp = 0
	sum_non_roi = 0
	#for i in range(hd):
	#	for j in range(wd):
	for i in range(block_ind, hd*wd):
		sumSaliency = sumSaliency + weights[block_ind]
		assigned = max(1.0, 1.0 / (1.0*wd*hd) * bitsAlloc)
		assignedW = weights[block_ind] * bitsAlloc
		targetbpp=0.0
		targetbpp = assignedW / pixelsPerBlock
		if roi_array[block_ind] == 0:
			sum_non_roi += 1
		lambdaConst = ALPHA * (targetbpp**BETA)
		avgLambda = avgLambda + math.log(lambdaConst)
		temp = LAMBDA_TO_QP(lambdaConst)
		qp_delta = qp_delta + temp - qp_frame
		QP[block_ind] = temp -QP_BASE
		avg_qp = avg_qp + QP[block_ind]+QP_BASE
		bitsPerBlock[block_ind] = min(QPToBits(int(QP[block_ind])),assigned)
		block_ind += 1

	if qp_delta != 0:
		while qp_delta < 0:
			for block_ind in range(0, hd*wd and qp_delta<0):
				if ROI[block_ind] == 0:
					QP[block_ind] = min(51-QP_BASE, QP[block_ind] + 1)
					qp_delta = qp_delta + 1
	avgLambda = math.exp(avgLambda / (hd*wd))
	old_avg_bits_per_pic = avg_bits_per_pic
	return QP
'''


rois_list = [[top_x_gt, top_y_gt], [top_x_gt+w, top_y_gt+h]]
roi_arr = utils.create_ROI_arr(rois_list, 'gt')
weights_arr = utils.create_weights_arr(roi_arr, rois_list)
#qp_gt = calcQPLambda(weights_arr, roi_arr, 1000000, 10)
qp_gt = utils.calcQPLambda(weights_arr, roi_arr)

for i in range(20):
	new_top_x = top_x_gt + i*w
	#print(new_top_x+w)
	new_top_y = top_y_gt + i*h
	#print(new_top_y+h)
	#new_top_y = top_y_gt
	#new_top_x = top_x
	#new_top_x = -0.2
	#new_top_y = -0.2
	rois_pre_list = [[new_top_x, new_top_y], [new_top_x+w, new_top_y+h]]
	roi_arr_pre = utils.create_ROI_arr(rois_pre_list, 'pre')
	weights_arr_pre = utils.create_weights_arr(roi_arr_pre, rois_pre_list)
	#qp_pre = calcQPLambda(weights_arr_pre, roi_arr_pre, 1000000, 10)
	qp_pre = utils.calcQPLambda(weights_arr_pre, roi_arr_pre)

	weights_arr_pre = qp_pre
	weights_arr = qp_gt
	temp = weights_arr_pre - weights_arr
	print(qp_pre)
	print(qp_gt)
	data_range=weights_arr.max() - weights_arr.min()
	ssim_noise = ssim_sk(weights_arr, weights_arr_pre, data_range=data_range)
	#ssim_torch = ssim(weights_arr, weights_arr_pre, data_range=data_range, size_average=True, channel=1)
	mse_const = mean_squared_error(weights_arr, weights_arr_pre)
	#print(mse_const)
	#print(ssim_noise)
	#print(temp)
	print("for " + str(i) + " The sum of absolute difference is : " + str(np.abs(temp).sum()))
	print("for " + str(i) + " The mean squared error is : " + str(mse_const))
	print("for " + str(i) + " The ssim using sklearn is : " + str(ssim_noise))
	#print("for " + str(i) + " The ssim using torch is : " + str(ssim_torch))
