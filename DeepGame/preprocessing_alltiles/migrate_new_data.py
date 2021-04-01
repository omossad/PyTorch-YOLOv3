import glob, os
import csv
import numpy as np
path = 'C:\\Users\\omossad\\Desktop\\dataset\\model_data\\fifa\\labels\\text_labels\\'

#path = 'D:\\Encoding\\encoding_files\\nhl\\gazepoint\\'
#path = 'D:\\Encoding\\encoding_files\\nba\\gazepoint\\'


os.chdir(path)
for fil in glob.glob("labels.csv"):
    output_file = path + fil.replace('.csv', '.txt')
    print(fil)
    print(output_file)
    with open(fil) as f:
        reader = csv.reader(f)
        next(reader) # skip header
        l_data = [r[1:3] for r in reader]

        #print(l_data)
    #np.savetxt(path+'tico.txt', l_data)
    to_write = []
    for j in l_data:
        to_write.append([float(j[0]),float(j[1])])
    np.savetxt(output_file, to_write)
    #    labels.append(fixations_val[idx])
    #    np.savetxt(output_folder+file_names[i]+'.txt', labels)
