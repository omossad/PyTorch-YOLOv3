import glob
'''
files = glob.glob("C:\\Users\\omossad\\Desktop\\dataset\\encoding\\ma_2\\cutscenes\\*.txt")
for f in files:
    f = open(f, "w")
    f.write("0 0.5 0.5 0.1 0.1")
    f.close()

'''
import os
files = glob.glob("C:\\Users\\omossad\\Desktop\\dataset\\encoding\\ga0\\roi*.txt")
print(len(files))
for f in files:
    if os.stat(f).st_size == 0:
        print('problem')
    #f = open(f, "w")
    #f.write("0 0.01 0.01 0.95 0.95")
    #f.close()
