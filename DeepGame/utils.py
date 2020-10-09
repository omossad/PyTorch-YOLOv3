import csv
import numpy as np
import configparser
import math
from shapely.geometry import Polygon


config = configparser.ConfigParser()
config.read(['C:\\Users\\omossad\\Desktop\\codes\\ROI-PyTorch\\DeepGame\\config.ini'])

def get_visual_pixels():
    return int(config.get("preprocessing", "radius"))

def get_num_tiles():
    return int(config.get("preprocessing", "num_tiles"))

def get_img_dim():
    W = float(config.get("data", "W"))
    H = float(config.get("data", "H"))
    return [W,H]

def get_fps():
    return int(config.get("data", "fps"))

def get_model_conf():
    ts = int(config.get("model", "input_frames"))
    t_overlap = int(config.get("model", "sample_overlap"))
    fut = int(config.get("model", "pred_frames"))
    return [ts, t_overlap, fut]


def get_no_files():
    num_files = 0
    with open('..\\frames_info.csv', 'r') as f:
        for line in f:
            num_files += 1
        # number of files is the number of files to be processed #
        num_files = num_files - 1
        print("Total number of files is:", num_files)
    return num_files

def get_files_list(num_files):
    frame_time = np.zeros((num_files,1))
    file_names = []
    with open('..\\frames_info.csv') as csv_file:
            csv_reader = csv.reader(csv_file, delimiter=',')
            line_count = 0
            for row in csv_reader:
                if line_count == 0:
                    line_count += 1
                elif line_count < num_files+1:
                    file_names.append(row[0])
                    frame_time[line_count-1] = int(row[5])
                    line_count += 1
                else:
                    break
            print('Files read in order are')
            print(file_names)
    return file_names

def circleRectangleIntersectionArea(r, xcenter, ycenter, xleft, xright, ybottom, ytop):
#find the signed (negative out) normalized distance from the circle center to each of the infinitely extended rectangle edge lines,
    d = [0, 0, 0, 0]
    d[0]=(xcenter-xleft)/r
    d[1]=(ycenter-ybottom)/r
    d[2]=(xright-xcenter)/r
    d[3]=(ytop-ycenter)/r
    #for convenience order 0,1,2,3 around the edge.

    # To begin, area is full circle
    area = math.pi*r*r

    # Check if circle is completely outside rectangle, or a full circle
    full = True
    for d_i in d:
        if d_i <= -1:   #Corresponds to a circle completely out of bounds
            return 0
        if d_i < 1:     #Corresponds to the circular segment out of bounds
            full = False

    if full:
        return area

    # this leave only one remaining fully outside case: circle center in an external quadrant, and distance to corner greater than circle radius:
    #for each adjacent i,j
    adj_quads = [1,2,3,0]
    for i in [0,1,2,3]:
        j=adj_quads[i]
        if d[i] <= 0 and d[j] <= 0 and d[i]*d[i]+d[j]*d[j] > 1:
            return 0

    # now begin with full circle area  and subtract any areas in the four external half planes
    a = [0, 0, 0, 0]
    for d_i in d:
        if d_i > -1 and d_i < 1:
            a[i] = math.asin( d_i )  #save a_i for next step
            area -= 0.5*r*r*(math.pi - 2*a[i] - math.sin(2*a[i]))

    # At this point note we have double counted areas in the four external quadrants, so add back in:
    #for each adjacent i,j

    for i in [0,1,2,3]:
        j=adj_quads[i]
        if  d[i] < 1 and d[j] < 1 and d[i]*d[i]+d[j]*d[j] < 1 :
            # The formula for the area of a circle contained in a plane quadrant is readily derived as the sum of a circular segment, two right triangles and a rectangle.
            area += 0.25*r*r*(math.pi - 2*a[i] - 2*a[j] - math.sin(2*a[i]) - math.sin(2*a[j]) + 4*math.sin(a[i])*math.sin(a[j]))

    return area

def fixation_to_tile(x,y):
	n_tiles = get_num_tiles()
	#X = x*W
	#Y = y*H
	#tile_width  = W/num_tiles
	#tile_height = H/num_tiles
	X = min(n_tiles - 1, x * n_tiles)
	Y = min(n_tiles - 1, y * n_tiles)
	return [int(X), int(Y)]


def object_to_tile_intersection(x1,y1,x2,y2):
    #print(x1)
    #print(y1)
    #print(x2)
    #print(y2)
    [W,H] = get_img_dim()
    n_tiles = get_num_tiles()
    arr_x = np.zeros((n_tiles))
    arr_y = np.zeros((n_tiles))
    tile_w = W/n_tiles
    tile_h = H/n_tiles
    object_poly = Polygon([(x1, y1), (x2, y1), (x2, y2), (x1, y2)])
    for i in range(n_tiles):
        #print(i*tile_w)
        #print(i*tile_h)
        #print((i+1)*tile_w)
        #print((i+1)*tile_h)
        #tile_poly = Polygon([(i*tile_w, i*tile_h), ((i+1)*tile_w, i*tile_h), ((i+1)*tile_w, (i+1)*tile_h), (i*tile_w, (i+1)*tile_h)])
        tile_poly_x = Polygon([(i*tile_w, y1), ((i+1)*tile_w, y1), ((i+1)*tile_w, y2), (i*tile_w, y2)])
        tile_poly_y = Polygon([(x1, i*tile_h), (x2, i*tile_h), (x2, (i+1)*tile_h), (x1, (i+1)*tile_h)])
        intersection = object_poly.intersection(tile_poly_x)
        arr_x[i] = intersection.area/(W)
        intersection = object_poly.intersection(tile_poly_y)
        arr_y[i] = intersection.area/(H)
        #print(intersection.area/(W))
    return [arr_x, arr_y]
    #for i in range(n_tiles):


def fixation_to_tile_intersection(x,y):
	n_tiles = get_num_tiles()
	radius = get_visual_pixels()
	#X = x*W
	#Y = y*H
	#tile_width  = W/num_tiles
	#tile_height = H/num_tiles
	X = min(n_tiles - 1, x * n_tiles)
	Y = min(n_tiles - 1, y * n_tiles)
	return [int(X), int(Y)]
