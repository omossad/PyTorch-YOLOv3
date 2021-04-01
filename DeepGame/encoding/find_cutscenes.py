import glob

full_data_path = 'C:\\Users\\omossad\\Desktop\\dataset\\raw_frames\\amin\\ma_2\\'
filt_data_path = 'C:\\Users\\omossad\\Desktop\\dataset\\model_data\\selected_frames\\ma_2\\'
outp_data_path = 'C:\\Users\\omossad\\Desktop\\dataset\\encoding\\ma_2\\cutscenes\\'

full_list = glob.glob(full_data_path + "*.png")
filt_list = glob.glob(filt_data_path + "*.png")

print('length of the full number of frames is: ' + str(len(full_list)))
print('length of the filt number of frames is: ' + str(len(filt_list)))
print('Number of cutscenes: ' + str(len(full_list) - len(filt_list)))


for i in range(len(full_list)):
    full_list[i]= full_list[i].split('\\')[-1]
for i in range(len(filt_list)):
    filt_list[i]= filt_list[i].split('\\')[-1]

outp_list = list(set(full_list) - set(filt_list))

for i in range(len(outp_list)):
    filename = int(outp_list[i].split('_')[1].replace('.png',''))
    #print(filename)
    f= open(outp_data_path + "roi" + str(filename) + ".txt","w")
    f.close()
