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
parser = argparse.ArgumentParser()
parser.add_argument("--input_dir", type=str, default='/home/omossad/projects/def-hefeeda/omossad/roi_detection/codes/ROI-PyTorch/data/samples/single_sample/*', help="input directory")
parser.add_argument("--base_model", type=str, default='resnet152', help="base network")

opt = parser.parse_args()

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

        return X


class ResNet(nn.Module):
    def __init__(self):
        """Load the pretrained ResNet-152 and replace top fc layer."""
        super(ResNet, self).__init__()
        if opt.base_model == 'resnet152':
            resnet = models.resnet152(pretrained=True)
        elif opt.base_model == 'resnet18':
            resnet = models.resnet18(pretrained=True)
        modules = list(resnet.children())[:-1]      # delete the last fc layer.
        self.resnet = nn.Sequential(*modules)

    def forward(self, x_3d):
        for t in range(x_3d.size(1)):
            with torch.no_grad():
                x = self.resnet(x_3d[:, t, :, :, :])  # ResNet
                x = x.view(x.size(0), -1)                  # flatten output of conv
        return x

class MobileNet(nn.Module):
    def __init__(self):
        """Load the pretrained MobileNet_V2 and replace top fc layer."""
        super(MobileNet, self).__init__()
        mobilenet = models.mobilenet_v2(pretrained=True)
        modules = list(mobilenet.children())[:-1]      # delete the last fc layer.
        self.mobilenet = nn.Sequential(*modules)

    def forward(self, x_3d):
        for t in range(x_3d.size(1)):
            with torch.no_grad():
                x = self.mobilenet(x_3d[:, t, :, :, :])  # MobileNet
                x = nn.functional.adaptive_avg_pool2d(x, 1).reshape(x.shape[0], -1)
                x = x.view(x.size(0), -1)                  # flatten output of conv
        return x


########## INSERTED CODE ########
input_directory = opt.input_dir
list_images = sorted(glob.glob(input_directory))
output_dir = input_directory.replace('selected_frames', 'features')
if opt.base_model == 'resnet152':
    output_dir = output_dir.replace('fifa', 'fifa/resnet152')
elif opt.base_model == 'resnet18':
    output_dir = output_dir.replace('fifa', 'fifa/resnet18')
elif opt.base_model == 'mobilenet':
    output_dir = output_dir.replace('fifa', 'fifa/mobilenetV2')

output_dir = output_dir.replace('*', '')
print(output_dir)

def process_data(images):
    num_images = len(images)
    image_indices = np.arange(0,num_images)
    images=np.asarray(images)
    return images[image_indices]

train_list = process_data(list_images)






#################################

# Detect devices
use_cuda = torch.cuda.is_available()                   # check if GPU exists
device = torch.device("cuda" if use_cuda else "cpu")   # use CPU or GPU

# Data loading parameters
params = {'batch_size': batch_size, 'shuffle': False, 'num_workers': 4, 'pin_memory': True} if use_cuda else {}


transform = transforms.Compose([transforms.Resize([res_size, res_size]),
                                transforms.ToTensor(),
                                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])



### INSERTED CODE ####


train_set = Dataset_CRNN(train_list, transform=transform)
##########################

train_loader = DataLoader(train_set, **params)


# Create model
if opt.base_model == 'resnet152':
    pretrained_model = ResNet().to(device)
elif opt.base_model == 'resnet18':
    pretrained_model = ResNet().to(device)
elif opt.base_model == 'mobilenet':
    pretrained_model = MobileNet().to(device)


# Parallelize model to multiple GPUs
if torch.cuda.device_count() > 1:
    print("Using", torch.cuda.device_count(), "GPUs!")
    pretrained_model = nn.DataParallel(pretrained_model)

elif torch.cuda.device_count() == 1:
    print("Using", torch.cuda.device_count(), "GPU!")


# start training
for batch_idx, (X, img_name) in enumerate(train_loader):
    dump_name = img_name[0].split("/")[-1]
    dump_name = dump_name.split(".")[0]
    dump_name = output_dir + dump_name + '.pt'
    X = X.to(device)
    starter, ender = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    repetitions = 300
    timings=np.zeros((repetitions,1))
    for _ in range(10):
        _ = pretrained_model(X)
    #with torch.no_grad():
    for rep in range(repetitions):
        starter.record()
        _ = pretrained_model(X)
        ender.record()
        # WAIT FOR GPU SYNC
        torch.cuda.synchronize()
        curr_time = starter.elapsed_time(ender)
        print(curr_time)
        print(curr_time*1000)
        timings[rep] = curr_timemean_syn = np.sum(timings) / repetitions
    std_syn = np.std(timings)
    print(curr_timemean_syn)
    print('--------------')
    print(timings)

    #print(batch_idx)
    # distribute data to device




    #output = output.view(-1)
    #torch.save(output, dump_name)
