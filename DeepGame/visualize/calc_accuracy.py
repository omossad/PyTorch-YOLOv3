import numpy as np
import math
from skimage.metrics import structural_similarity as ssim_sk
from skimage.metrics import mean_squared_error
import sys, os
sys.path.append(os.path.abspath(os.path.join('..')))
import utils
from pytorch_msssim import ssim, ms_ssim, SSIM, MS_SSIM


ground_truth_labels = np.loadtxt('labels.txt')
predicted_labels = np.loadtxt('predicted.txt')
print(len(ground_truth_labels))
print(len(predicted_labels))

tw = 1/8
th = 1/8
fixed_y = 0.5

sum_err = 0
err_list = []
for i in range(len(predicted_labels)):
	rois_pr_list = []
	rois_gt_list = []
	for j in range(8):
		if predicted_labels[i][j] > 0:
			rois_pr_list.append([j*tw, fixed_y])
		if ground_truth_labels[i][j] > 0:
			rois_gt_list.append([j*tw, fixed_y])
	roi_arr_gt = utils.create_ROI_arr(rois_gt_list, 'gt')
	weights_arr_gt = utils.create_weights_arr(roi_arr_gt, rois_gt_list)
	qp_gt = utils.calcQPLambda(weights_arr_gt, roi_arr_gt)

	roi_arr_pr = utils.create_ROI_arr(rois_pr_list, 'pre')
	weights_arr_pr = utils.create_weights_arr(roi_arr_pr, rois_pr_list)
	qp_pr = utils.calcQPLambda(weights_arr_pr, roi_arr_pr)
	temp = qp_pr - qp_gt
	temp_err = np.abs(temp).sum()
	err_list.append(temp_err)
	sum_err = sum_err + temp_err
	print("for " + str(i) + " The sum of absolute difference is : " + str(np.abs(temp).sum()))
print('Total  error = ')
print(sum_err/len(ground_truth_labels))
np.savetxt('err_list.txt', np.asarray(err_list))
# TOTAL ERROR = 6875.487860551982
# MIN ERROR = 381.0
# MAX ERROR = 29000.0
