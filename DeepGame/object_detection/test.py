#while(True):
#    f = open("C:\\Users\\omossad\\Desktop\\temp_frames\\test.txt", "r")
#    print(f.read())
'''
import numpy as np 
x = np.arange(16).reshape((2, 2, 4))
print(x.shape)
x = np.transpose(x, (2, 0, 1))
print(x.shape)
'''
import cv2
import numpy as np
import io
yuv_filename = 'C:\\Users\\omossad\\Desktop\\capture.yuv'
f = open(yuv_filename, 'rb')
height = 900
width = 1600
for i in range(5):
    # Read Y, U and V color channels and reshape to height*1.5 x width numpy array
    yuv = np.frombuffer(f.read(width*height*3//2), dtype=np.uint8).reshape((height*3//2, width))

    # Convert YUV420 to BGR (for testing), applies BT.601 "Limited Range" conversion.
    bgr = cv2.cvtColor(yuv, cv2.COLOR_YUV2BGR_I420)
    resized = cv2.resize(bgr, (416,416))
    print(resized.shape)

    # Convert YUV420 to Grayscale
    #gray = cv2.cvtColor(yuv, cv2.COLOR_YUV2GRAY_I420)

    #Show RGB image and Grayscale image for testing
    cv2.imshow('rgb', bgr)
    cv2.waitKey(500)  # Wait a 0.5 second (for testing)
    #cv2.imshow('gray', gray)
    #cv2.waitKey(500)  # Wait a 0.5 second (for testing)

f.close()

cv2.destroyAllWindows()

'''
# Building the input:
###############################################################################
img = cv2.imread('GrandKingdom.jpg')

#yuv = cv2.cvtColor(img, cv2.COLOR_BGR2YUV)
#y, u, v = cv2.split(yuv)

# Convert BGR to YCrCb (YCrCb apply YCrCb JPEG (or YCC), "full range", 
# where Y range is [0, 255], and U, V range is [0, 255] (this is the default JPEG format color space format).
yvu = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)
y, v, u = cv2.split(yvu)

# Downsample U and V (apply 420 format).
u = cv2.resize(u, (u.shape[1]//2, u.shape[0]//2))
v = cv2.resize(v, (v.shape[1]//2, v.shape[0]//2))

# Open In-memory bytes streams (instead of using fifo)
f = io.BytesIO()

# Write Y, U and V to the "streams".
f.write(y.tobytes())
f.write(u.tobytes())
f.write(v.tobytes())

f.seek(0)
###############################################################################

# Read YUV420 (I420 planar format) and convert to BGR
###############################################################################
data = f.read(y.size*3//2)  # Read one frame (number of bytes is width*height*1.5).

# Reshape data to numpy array with height*1.5 rows
yuv_data = np.frombuffer(data, np.uint8).reshape(y.shape[0]*3//2, y.shape[1])

# Convert YUV to BGR
bgr = cv2.cvtColor(yuv_data, cv2.COLOR_YUV2BGR_I420);


# How to How should I be placing the u and v channel information in all_yuv_data?
# -------------------------------------------------------------------------------
# Example: place the channels one after the other (for a single frame)
f.seek(0)
y0 = f.read(y.size)
u0 = f.read(y.size//4)
v0 = f.read(y.size//4)
yuv_data = y0 + u0 + v0
yuv_data = np.frombuffer(yuv_data, np.uint8).reshape(y.shape[0]*3//2, y.shape[1])
bgr = cv2.cvtColor(yuv_data, cv2.COLOR_YUV2BGR_I420);
###############################################################################

# Display result:
cv2.imshow("bgr incorrect colors", bgr)


###############################################################################
f.seek(0)
y = np.frombuffer(f.read(y.size), dtype=np.uint8).reshape((y.shape[0], y.shape[1]))  # Read Y color channel and reshape to height x width numpy array
u = np.frombuffer(f.read(y.size//4), dtype=np.uint8).reshape((y.shape[0]//2, y.shape[1]//2))  # Read U color channel and reshape to height x width numpy array
v = np.frombuffer(f.read(y.size//4), dtype=np.uint8).reshape((y.shape[0]//2, y.shape[1]//2))  # Read V color channel and reshape to height x width numpy array

# Resize u and v color channels to be the same size as y
u = cv2.resize(u, (y.shape[1], y.shape[0]))
v = cv2.resize(v, (y.shape[1], y.shape[0]))
yvu = cv2.merge((y, v, u)) # Stack planes to 3D matrix (use Y,V,U ordering)

bgr = cv2.cvtColor(yvu, cv2.COLOR_YCrCb2BGR)
###############################################################################


# Display result:
cv2.imshow("bgr", bgr)
cv2.waitKey(0)
cv2.destroyAllWindows()
'''

'''
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
'''