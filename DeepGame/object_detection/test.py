#while(True):
#    f = open("C:\\Users\\omossad\\Desktop\\temp_frames\\test.txt", "r")
#    print(f.read())

import cv2
import numpy as np
import os
import subprocess as sp

yuv_filename = 'C:\\Users\\omossad\\Desktop\\capture.yuv'
#flow=[]

width, height = 1600, 900

file_size = os.path.getsize(yuv_filename)
n_frames = file_size // (width*height*3 // 2)
f = open(yuv_filename, 'rb')


old_yuv = np.frombuffer(f.read(width*height*3//2), dtype=np.uint8).reshape((height*3//2, width))
cv2.imshow('frame',old_yuv)
cv2.waitKey(3000)

# Convert YUV420 to Grayscale
old_gray = cv2.cvtColor(old_yuv, cv2.COLOR_YUV2GRAY_I420)
cv2.imshow('frame_gs',old_gray)
cv2.waitKey(3000)