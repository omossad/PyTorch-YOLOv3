import numpy as np
import csv
print(float(0.355))
#data = np.loadtxt('/home/omossad/projects/def-hefeeda/omossad/roi_detection/data/fifa/1/fixation1.txt')
#data = np.loadtxt('/home/omossad/scratch/Gaming-Dataset/processed/puria/fifa/0000-scrn.avi/label.txt')
file = '/home/omossad/scratch/Gaming-Dataset/processed/puria/fifa/0007-scrn.avi/label.txt'
objects = []
with open(file, newline='') as csvfile:
	reader = csv.reader(csvfile, delimiter=',')
	for row in reader:
		objects.append(int(row[1]))
print(np.unique(objects))
