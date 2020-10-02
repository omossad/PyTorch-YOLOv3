import glob
import sys, os
sys.path.append(os.path.abspath(os.path.join('..')))
print(sys.path)
import utils
import shutil


input_path = 'C:\\Users\\omossad\\Desktop\\dataset\\model_data\\selected_frames\\'
output_path = 'C:\\Users\\omossad\\Desktop\\dataset\\model_data\\selected_frames_new\\'

def find_cut(list_no):
    cuts = []
    for i in range(len(list_no)-1):
        if list_no[i] == list_no[i+1] - 1:
            continue
        else:
            cuts.append(list_no[i+1])
    return cuts

### READ NUMBER OF FILES and NAMES ###
num_files = utils.get_no_files()
### READ NUMBER OF FRAMES in each FILE ###
file_names = utils.get_files_list(num_files)
print(file_names)

for i in range(num_files):
    user_path = input_path + file_names[i] + '\\'
    frame_names = sorted(glob.glob(user_path +'*'))
    frame_nos = []
    for j in range(len(frame_names)):
        frame_names[j] = frame_names[j].replace(user_path , '')
        frame_names[j] = frame_names[j].replace('frame_', '')
        frame_names[j] = frame_names[j].replace('.png', '')
        frame_nos.append(int(frame_names[j]))
    cuts = find_cut(frame_nos)
    print(file_names[i] + " " + str(cuts))
    for c in range(len(cuts)):
        out_user_path = output_path + file_names[i] + '_' + str(c) + '\\'
        try:
            os.mkdir(out_user_path)
        except:
            print ("Folder exists")
        indices = frame_nos.index(cuts[c])
        to_copy_list = frame_names[:indices]
        frame_nos   = frame_nos[indices:]
        frame_names = frame_names[indices:]
        #print(to_copy_list)
        for fr in to_copy_list:
            fr_name = "frame_" + fr + ".png"
            frame_src = user_path + fr_name
            frame_tgt = out_user_path + fr_name
            print(frame_src + ' ' + frame_tgt)
            shutil.copyfile(frame_src, frame_tgt)
    out_user_path = output_path + file_names[i] + '_' + str(len(cuts)) + '\\'
    try:
        os.mkdir(out_user_path)
    except:
        print ("Folder exists")
    for fr in frame_names:
        fr_name = "frame_" + fr + ".png"
        frame_src = user_path + fr_name
        frame_tgt = out_user_path + fr_name
        print(frame_src + ' ' + frame_tgt)
        shutil.copyfile(frame_src, frame_tgt)
