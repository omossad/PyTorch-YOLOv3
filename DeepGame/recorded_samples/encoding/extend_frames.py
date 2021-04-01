import os
import re

inpt_data_path = 'C:\\Users\\omossad\\Desktop\\recorded_samples\\fifa\\encoding\\deepGame\\'
outp_data_path = 'C:\\Users\\omossad\\Desktop\\recorded_samples\\fifa\\encoding\\deepGame-e\\'

path, dirs, files = next(os.walk(inpt_data_path))
files.sort(key=lambda var:[int(x) if x.isdigit() else x for x in re.findall(r'[^0-9]|[0-9]+', var)])
file_count = len(files)
for i in range(file_count):
    f = open(inpt_data_path + 'roi' + str(i) + '.txt', 'r')
    l = f.readline()
    print(l)
    f.close()
    for j in range(6):
        f = open(outp_data_path + 'roi' + str(i*6 + j) + '.txt', 'w')
        f.write(l)
        f.close()
