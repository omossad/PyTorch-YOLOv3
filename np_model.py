import pprint, pickle
import numpy as np

data_path = '/home/omossad/scratch/temp/numpy/'
pkl_file = open(data_path + 'data_array.pkl', 'rb')

data = pickle.load(pkl_file)
targets = np.loadtxt(data_path + 'trgt_array.dat')
#pprint.pprint(data1)
print(targets)
pkl_file.close()
