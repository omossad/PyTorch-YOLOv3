import csv
import numpy as np

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
