import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from torch.utils.data import TensorDataset, DataLoader
import torchvision
from torch.autograd import Variable
import matplotlib.pyplot as plt
from functions import *
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn.metrics import accuracy_score
import glob
import pickle
import csv

# EncoderCNN architecture
CNN_fc_hidden1, CNN_fc_hidden2 = 1024, 768
CNN_embed_dim = 256   # latent dim extracted by 2D CNN

res_size = 224        # ResNet image size
dropout_p = 0.0       # dropout probability

# DecoderRNN architecture
RNN_hidden_layers = 3
RNN_hidden_nodes = 512
RNN_FC_dim = 256

# training parameters
k = 8            # number of target category
epochs = 120        # training epochs
batch_size = 20
learning_rate = 1e-3
wd = 0
log_interval = 1   # interval for displaying training info


def train(log_interval, model, device, train_loader, optimizer, epoch, coordinate):
    # set model as training mode
    cnn_encoder, rnn_decoder = model
    cnn_encoder.train()
    rnn_decoder.train()

    losses = []
    scores = []
    N_count = 0   # counting total trained sample in one epoch
    for batch_idx, (X, y_x, y_y) in enumerate(train_loader):
        if coordinate == 'x':
            y = y_x.to(device).view(-1, )
        else:
            y = y_y.to(device).view(-1, )
        # distribute data to device
        X = X.to(device)
        #X, y = X.to(device), y.to(device).view(-1, )

        N_count += X.size(0)

        optimizer.zero_grad()
        output = rnn_decoder(cnn_encoder(X))   # output has dim = (batch, number of classes)

        loss = F.cross_entropy(output, y)
        losses.append(loss.item())

        # to compute accuracy
        y_pred = torch.max(output, 1)[1]  # y_pred != output
        step_score = accuracy_score(y.cpu().data.squeeze().numpy(), y_pred.cpu().data.squeeze().numpy())
        scores.append(step_score)         # computed on CPU

        loss.backward()
        optimizer.step()

        # show information
        #if (batch_idx + 1) % log_interval == 0:
        #    print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}, Accu: {:.2f}%'.format(
        #        epoch + 1, N_count, len(train_loader.dataset), 100. * (batch_idx + 1) / len(train_loader), loss.item(), 100 * step_score))

    return losses, scores


def validation(model, device, optimizer, test_loader, coordinate):
    # set model as testing mode
    cnn_encoder, rnn_decoder = model
    cnn_encoder.eval()
    rnn_decoder.eval()

    test_loss = 0
    all_y = []
    all_y_pred = []
    with torch.no_grad():
        for X, y_x, y_y in test_loader:
            # distribute data to device
            if coordinate == 'x':
                y = y_x.to(device).view(-1, )
            else:
                y = y_y.to(device).view(-1, )
            # distribute data to device
            X = X.to(device)
            #X, y = X.to(device), y.to(device).view(-1, )

            output = rnn_decoder(cnn_encoder(X))

            loss = F.cross_entropy(output, y, reduction='sum')
            test_loss += loss.item()                 # sum up batch loss
            y_pred = output.max(1, keepdim=True)[1]  # (y_pred != output) get the index of the max log-probability
            #print(y_pred)
            # collect all y and y_pred in all batches
            all_y.extend(y)
            all_y_pred.extend(y_pred)

    test_loss /= len(test_loader.dataset)

    # compute accuracy
    all_y = torch.stack(all_y, dim=0)
    all_y_pred = torch.stack(all_y_pred, dim=0)
    test_score = accuracy_score(all_y.cpu().data.squeeze().numpy(), all_y_pred.cpu().data.squeeze().numpy())

    # show information
    #print('\nTest set ({:d} samples): Average loss: {:.4f}, Accuracy: {:.2f}%\n'.format(len(all_y), test_loss, 100* test_score))

    # save Pytorch models of best record

    #torch.save(cnn_encoder.state_dict(), os.path.join(save_model_path, 'cnn_encoder_epoch{}.pth'.format(epoch + 1)))  # save spatial_encoder
    #torch.save(rnn_decoder.state_dict(), os.path.join(save_model_path, 'rnn_decoder_epoch{}.pth'.format(epoch + 1)))  # save motion_encoder
    #torch.save(optimizer.state_dict(), os.path.join(save_model_path, 'optimizer_epoch{}.pth'.format(epoch + 1)))      # save optimizer
    #print("Epoch {} model saved!".format(epoch + 1))

    return test_loss, test_score, all_y.cpu().data.squeeze().numpy(), all_y_pred.cpu().data.squeeze().numpy()


########## INSERTED CODE ########



time_steps = 4


def process_data(images):
    num_images = len(images)
    image_indices = np.arange(0,num_images)
    indices = np.array([ image_indices[i:i+time_steps] for i in range(num_images-time_steps+1) ])
    images=np.asarray(images)
    return images[indices]

def process_labels(labels):
    num_labels = len(labels)
    indices = np.arange(time_steps-1,num_labels)
    labels=np.asarray(labels)
    return labels[indices]

max_files = 7
def read_info():
    file_names = []
    num_files = 0
    with open('frames_info') as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        for row in csv_reader:
            if num_files == 0:
                num_files += 1
            elif num_files < max_files+1:
                file_names.append(row[0])
                num_files += 1
            else:
                break
    print("Total number of files is:", num_files)
    return file_names

all_filenames = read_info()
print(all_filenames)

#data_dir  = '/home/omossad/projects/def-hefeeda/omossad/roi_detection/temporary_data/data/resnet/'
#data_dir  = '/home/omossad/scratch/Gaming-Dataset/features/fifa/mobilenetV2/'
data_dir  = '/home/omossad/scratch/Gaming-Dataset/features/fifa/resnet18/'
label_dir = '/home/omossad/scratch/Gaming-Dataset/frame_labels/fifa/'

train_list = []
test_list = []
train_label = []
test_label = []

for f in all_filenames:
    images = sorted(glob.glob(data_dir + f + '/*'))
    labels = sorted(glob.glob(label_dir + f + '/*'))
    processed_images = process_data(images)
    processed_labels = process_labels(labels)
    if f.startswith('ha_5'):
        test_list.extend(processed_images)
        test_label.extend(processed_labels)
    #if f.startswith('ha_8'):
    #    test_list.extend(processed_images)
    #    test_label.extend(processed_labels)
    #if f.startswith('ha_9'):
    #    test_list.extend(processed_images)
    #    test_label.extend(processed_labels)
    else:
        train_list.extend(processed_images)
        train_label.extend(processed_labels)

train_list = np.asarray(train_list)
test_list  = np.asarray(test_list)
train_label = np.asarray(train_label)
test_label = np.asarray(test_label)

print(train_list.shape)
print(train_label.shape)
print(test_list.shape)
print(test_label.shape)





#################################





# Detect devices
use_cuda = torch.cuda.is_available()                   # check if GPU exists
device = torch.device("cuda" if use_cuda else "cpu")   # use CPU or GPU

# Data loading parameters
params = {'batch_size': batch_size, 'shuffle': True, 'num_workers': 4, 'pin_memory': True} if use_cuda else {}


### INSERTED CODE ####


train_set, valid_set = Dataset_CRNN(train_list, train_label), \
                       Dataset_CRNN(test_list, test_label)
##########################

train_loader = DataLoader(train_set, **params)
valid_loader = DataLoader(valid_set, **params)


# Create model
cnn_encoder_x = ResCNNEncoder(num_tiles=k,fc_hidden1=CNN_fc_hidden1, fc_hidden2=CNN_fc_hidden2, drop_p=dropout_p, CNN_embed_dim=CNN_embed_dim).to(device)
rnn_decoder_x = DecoderRNN(CNN_embed_dim=CNN_embed_dim, h_RNN_layers=RNN_hidden_layers, h_RNN=RNN_hidden_nodes,
                         h_FC_dim=RNN_FC_dim, drop_p=dropout_p, num_classes=k).to(device)

cnn_encoder_y = ResCNNEncoder(num_tiles=k,fc_hidden1=CNN_fc_hidden1, fc_hidden2=CNN_fc_hidden2, drop_p=dropout_p, CNN_embed_dim=CNN_embed_dim).to(device)
rnn_decoder_y = DecoderRNN(CNN_embed_dim=CNN_embed_dim, h_RNN_layers=RNN_hidden_layers, h_RNN=RNN_hidden_nodes,
                         h_FC_dim=RNN_FC_dim, drop_p=dropout_p, num_classes=k).to(device)

# Parallelize model to multiple GPUs
if torch.cuda.device_count() > 1:
    print("Using", torch.cuda.device_count(), "GPUs!")
    cnn_encoder_x = nn.DataParallel(cnn_encoder_x)
    rnn_decoder_x = nn.DataParallel(rnn_decoder_x)

    cnn_encoder_y = nn.DataParallel(cnn_encoder_y)
    rnn_decoder_y = nn.DataParallel(rnn_decoder_y)

    # Combine all EncoderCNN + DecoderRNN parameters
    crnn_params_x = list(cnn_encoder_x.module.fc1.parameters()) + list(cnn_encoder_x.module.bn1.parameters()) + \
                  list(cnn_encoder_x.module.fc2.parameters()) + list(cnn_encoder_x.module.bn2.parameters()) + \
                  list(cnn_encoder_x.module.fc3.parameters()) + list(rnn_decoder_x.parameters())

    crnn_params_y = list(cnn_encoder_y.module.fc1.parameters()) + list(cnn_encoder_y.module.bn1.parameters()) + \
                  list(cnn_encoder_y.module.fc2.parameters()) + list(cnn_encoder_y.module.bn2.parameters()) + \
                  list(cnn_encoder_y.module.fc3.parameters()) + list(rnn_decoder_y.parameters())

elif torch.cuda.device_count() == 1:
    print("Using", torch.cuda.device_count(), "GPU!")
    # Combine all EncoderCNN + DecoderRNN parameters
    crnn_params_x = list(cnn_encoder_x.fc1.parameters()) + list(cnn_encoder_x.bn1.parameters()) + \
                  list(cnn_encoder_x.fc2.parameters()) + list(cnn_encoder_x.bn2.parameters()) + \
                  list(cnn_encoder_x.fc3.parameters()) + list(rnn_decoder_x.parameters())

    crnn_params_y = list(cnn_encoder_y.fc1.parameters()) + list(cnn_encoder_y.bn1.parameters()) + \
                  list(cnn_encoder_y.fc2.parameters()) + list(cnn_encoder_y.bn2.parameters()) + \
                  list(cnn_encoder_y.fc3.parameters()) + list(rnn_decoder_y.parameters())


optimizer_x = torch.optim.Adam(crnn_params_x, lr=learning_rate, weight_decay=wd)
optimizer_y = torch.optim.Adam(crnn_params_y, lr=learning_rate, weight_decay=wd)


# start training
for epoch in range(epochs):
    # train, test model
    train_losses, train_scores = train(log_interval, [cnn_encoder_x, rnn_decoder_x], device, train_loader, optimizer_x, epoch, 'x')
    epoch_test_loss, epoch_test_score, true_x, pred_x = validation([cnn_encoder_x, rnn_decoder_x], device, optimizer_x, valid_loader, 'x')
    print('\nEpoch ({:d}): Train X Accuracy: {:.2f}%\n'.format(epoch, 100* np.average(np.asarray(train_scores))))
    print('\nEpoch ({:d}): Test  X Accuracy: {:.2f}%\n'.format(epoch, 100* epoch_test_score))
    train_losses, train_scores = train(log_interval, [cnn_encoder_y, rnn_decoder_y], device, train_loader, optimizer_y, epoch, 'y')
    epoch_test_loss, epoch_test_score, true_y, pred_y = validation([cnn_encoder_y, rnn_decoder_y], device, optimizer_y, valid_loader, 'y')
    print('\nEpoch ({:d}): Train Y Accuracy: {:.2f}%\n'.format(epoch, 100* np.average(np.asarray(train_scores))))
    print('\nEpoch ({:d}): Test  Y Accuracy: {:.2f}%\n'.format(epoch, 100* epoch_test_score))
    x_score = (true_x == pred_x)
    y_score = (true_y == pred_y)
    test_score = np.average(x_score*y_score)

    # show information
    print('\nEpoch ({:d}): Accuracy: {:.2f}%\n'.format(epoch, 100* test_score))
