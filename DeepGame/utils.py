import csv
import numpy as np
import configparser

config = configparser.ConfigParser()
config.read(['C:\\Users\\omossad\\Desktop\\codes\\ROI-PyTorch\\DeepGame\\config.ini'])

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

def fixation_to_tile(x,y):
	n_tiles = get_num_tiles()
	#X = x*W
	#Y = y*H
	#tile_width  = W/num_tiles
	#tile_height = H/num_tiles
	X = min(n_tiles - 1, x * n_tiles)
	Y = min(n_tiles - 1, y * n_tiles)
	return [int(X), int(Y)]
