import pprint, pickle

data_path = '/home/omossad/scratch/temp/numpy/'
pkl_file = open(data_path + 'data_array.pkl', 'rb')

data1 = pickle.load(pkl_file)
pprint.pprint(data1)

pkl_file.close()
