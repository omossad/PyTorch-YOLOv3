import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import torchvision.transforms as transforms
from torch.utils.data import TensorDataset, DataLoader
import torchvision
import glob
from torch.utils import data
from PIL import Image
import pickle
import argparse
from torchsummary import summary

parser = argparse.ArgumentParser()
parser.add_argument("--input_dir", type=str, default='/home/omossad/projects/def-hefeeda/omossad/roi_detection/codes/ROI-PyTorch/data/samples/ha*', help="input directory")
opt = parser.parse_args()
# set path
#data_path = "./jpegs_256/"    # define UCF-101 RGB data path
#action_name_path = './UCF101actions.pkl'
#save_model_path = "./ResNetCRNN_ckpt/"

# EncoderCNN architecture

res_size = 224        # ResNet image size
batch_size = 1

class Dataset_CRNN(data.Dataset):
    "Characterizes a dataset for PyTorch"
    def __init__(self, images, transform=None):
        "Initialization"
        self.images = images
        self.transform = transform

    def __len__(self):
        "Denotes the total number of samples"
        return len(self.images)

    def read_images(self, selected_image, use_transform):
        X = []
        image = Image.open(selected_image)
        if use_transform is not None:
            image = use_transform(image)
        X.append(image)
        X = torch.stack(X, dim=0)
        return X, selected_image


    def __getitem__(self, index):
        "Generates one sample of data"

        X = self.read_images(self.images[index], self.transform)     # (input) spatial images

        # print(X.shape)
        return X


class ResNet(nn.Module):
    def __init__(self):
        """Load the pretrained ResNet-152 and replace top fc layer."""
        super(ResNet, self).__init__()
        resnet = models.mobilenet_v2(pretrained=True)
        print(list(resnet.children()))
        print('----------')
        print(list(resnet.children())[0])
        #print(resnet)
        #modules = list(resnet.children())[:-1]      # delete the last fc layer.
        #self.resnet = nn.Sequential(*modules)
        self.resnet = nn.Sequential(*list(resnet.children())[0])
        self.resnet = self.resnet.to(device)

        summary(self.resnet, (3, 244, 224))

    def forward(self, x_3d):
        for t in range(x_3d.size(1)):
            with torch.no_grad():
                x = self.resnet(x_3d[:, t, :, :, :])  # ResNet
                x = x.view(x.size(0), -1)                  # flatten output of conv
        return x


########## INSERTED CODE ########
input_directory = opt.input_dir
list_images = sorted(glob.glob(input_directory))
#output_dir = input_directory.replace('frames', 'resnet')
#output_dir = output_dir.replace('*', '')
#ha_0_labels = sorted(glob.glob("/home/omossad/projects/def-hefeeda/omossad/roi_detection/temporary_data/ha_0_images/f*"))
print(len(list_images))
#print(len(ha_0_labels))



def process_data(images):
    num_images = len(images)
    image_indices = np.arange(0,num_images)
    images=np.asarray(images)
    return images[image_indices]

train_list = process_data(list_images)
print(train_list)






#################################

# Detect devices
use_cuda = torch.cuda.is_available()                   # check if GPU exists
device = torch.device("cuda" if use_cuda else "cpu")   # use CPU or GPU

# Data loading parameters
params = {'batch_size': batch_size, 'shuffle': False, 'num_workers': 4, 'pin_memory': True} if use_cuda else {}


# load UCF101 actions names
#with open(action_name_path, 'rb') as f:
#    action_names = pickle.load(f)

# convert labels -> category
#le = LabelEncoder()
#le.fit(action_names)

# show how many classes there are
#list(le.classes_)

# convert category -> 1-hot
#action_category = le.transform(action_names).reshape(-1, 1)
#enc = OneHotEncoder()
#enc.fit(action_category)

# # example
# y = ['HorseRace', 'YoYo', 'WalkingWithDog']
# y_onehot = labels2onehot(enc, le, y)
# y2 = onehot2labels(le, y_onehot)

#actions = []
#fnames = os.listdir(data_path)

#all_names = []
#for f in fnames:
#    loc1 = f.find('v_')
#    loc2 = f.find('_g')
#    actions.append(f[(loc1 + 2): loc2])

#    all_names.append(f)


# list all data files
#all_X_list = all_names                  # all video file names
#all_y_list = labels2cat(le, actions)    # all video labels

# train, test split
#train_list, test_list, train_label, test_label = train_test_split(all_X_list, all_y_list, test_size=0.25, random_state=42)

transform = transforms.Compose([transforms.Resize([res_size, res_size]),
                                transforms.ToTensor(),
                                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

#selected_frames = np.arange(begin_frame, end_frame, skip_frame).tolist()


### INSERTED CODE ####


train_set = Dataset_CRNN(train_list, transform=transform)
##########################

train_loader = DataLoader(train_set, **params)


# Create model
resnet_model = ResNet().to(device)

# Parallelize model to multiple GPUs
if torch.cuda.device_count() > 1:
    print("Using", torch.cuda.device_count(), "GPUs!")
    resnet_model = nn.DataParallel(resnet_model)

    # Combine all EncoderCNN + DecoderRNN parameters
    #crnn_params = list(cnn_encoder.module.fc1.parameters()) + list(cnn_encoder.module.bn1.parameters()) + \
    #              list(cnn_encoder.module.fc2.parameters()) + list(cnn_encoder.module.bn2.parameters()) + \
    #              list(cnn_encoder.module.fc3.parameters()) + list(rnn_decoder.parameters())

elif torch.cuda.device_count() == 1:
    print("Using", torch.cuda.device_count(), "GPU!")
    # Combine all EncoderCNN + DecoderRNN parameters
    #crnn_params = list(cnn_encoder.fc1.parameters()) + list(cnn_encoder.bn1.parameters()) + \
    #              list(cnn_encoder.fc2.parameters()) + list(cnn_encoder.bn2.parameters()) + \
    #              list(cnn_encoder.fc3.parameters()) + list(rnn_decoder.parameters())



# start training
for batch_idx, (X, img_name) in enumerate(train_loader):
    # distribute data to device
    dump_name = img_name[0].split("/")[-1]
    dump_name = dump_name.split(".")[0]
    #output_file = open(output_dir + dump_name + '.pkl', 'wb')
    X = X.to(device)
    output = resnet_model(X)
    print(output.shape)
    output = output.view(-1)
    #pickle.dump(output, output_file)
    #output_file.close()
