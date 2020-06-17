import pickle
import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
import csv

max_files = 99
num_tiles = 8
if num_tiles == 64:
    data_path = '/home/omossad/scratch/Gaming-Dataset/processed/lstm_input/input_64/fifa/'
    labels_path = '/home/omossad/scratch/Gaming-Dataset/processed/lstm_labels/labels_64/fifa/'
else:
    data_path = '/home/omossad/scratch/Gaming-Dataset/processed/lstm_input/input_8x8/fifa/'
    labels_path = '/home/omossad/scratch/Gaming-Dataset/processed/lstm_labels/labels_8x8/fifa/'

num_classes = 3
time_steps = 8
batch_size = 16
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def closestNumber(n, m) :
    q = int(n / m)
    return int(q*m)


def read_info():
    file_names = []
    num_files = 0
    with open('preprocessing/frames_info') as csv_file:
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



def read_file(filename):
    if num_tiles == 64:
        pkl_file = open(data_path + filename + '.pkl', 'rb')
    else:
        pkl_file = open(data_path + filename + '_x.pkl', 'rb')
    data = pickle.load(pkl_file)
    pkl_file.close()
    return data



def process_data(data):

    num_images = len(data)
    image_indices = np.arange(0,num_images)
    indices = np.array([ image_indices[i:i+time_steps] for i in range(num_images-time_steps) ])
    data = np.asarray(data)
    data = np.reshape(data, (num_images,-1))
    img_data = []
    selected_indices = closestNumber(len(indices), batch_size)
    for i in range(selected_indices):
        img_data.append(data[indices[i]])
    data = np.asarray(img_data)
    data = np.reshape(data, (len(data)//batch_size, batch_size, time_steps, -1))
    data = np.transpose(data, (0, 2, 1, 3))

    return data

def read_labels(filename):
    if num_tiles == 64:
        targets = np.loadtxt(labels_path + filename + '.dat', dtype=np.dtype('uint8'))
    else:
        targets = np.loadtxt(labels_path + filename + '_x.dat', dtype=np.dtype('uint8'))
    targets = np.asarray(targets)
    return targets

def process_labels(targets):
    targets = targets[time_steps:]
    selected_indices = closestNumber(len(targets), batch_size)
    targets = targets[:selected_indices]
    targets = np.reshape(targets, (len(targets)//batch_size, batch_size, -1))
    #targets = np.transpose(targets, (0, 2, 1, 3))
    return targets

def lstm_model():
    ### MODEL PARAMS ####


    in_size = num_tiles * num_classes
    classes_no = num_tiles

    model = nn.LSTM(in_size, classes_no, 2)
    out_model = nn.Sequential(
        nn.Linear(classes_no, 32),
        nn.ReLU(inplace=False),
        nn.Linear(32, 32),
        nn.ReLU(inplace=False),
        #nn.Dropout(0.2),
        nn.Linear(32, classes_no),
        nn.ReLU(inplace=False)
    )
    out_model.to(device)
    model.to(device)
    return model


def test(model, test_data, test_labels):
    score_val = 0
    test_size = len(test_data)
    for d in range(test_size):
        input_seq = Variable(torch.from_numpy(test_data[d]).float().to(device))
        output_seq, _ = model(input_seq)
        last_output = output_seq[-1]
        #last_output = out_model(last_output)
        #target = Variable(torch.tensor([test_labels[d]]).to(device))
        target = Variable(torch.from_numpy(test_labels[d]).long().view(-1).to(device))
        _, pred_x = torch.max(last_output, 1)
        score = torch.eq(pred_x, target).float()
        score_val += score.mean().item()
    print(' ---- test acc: ' + str(score_val/test_size) + '\n')

def train(train_data, test_data, train_labels, test_labels, model):
    epochs = 200
    learning_rate = 0.0001
    weight_decay = 0
    loss = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

    train_size = len(train_data)
    for e in range(epochs):
        loss_val = 0
        score_val = 0
        for d in range(train_size):
            input_seq = Variable(torch.from_numpy(train_data[d]).float().to(device))
            output_seq, _ = model(input_seq)
            last_output = output_seq[-1]
            #print(train_labels[d])
            #target = Variable(torch.tensor([train_labels[d]]).to(device))
            #print(target)
            target = Variable(torch.from_numpy(train_labels[d]).long().view(-1).to(device))
            _, pred_x = torch.max(last_output, 1)
            score = torch.eq(pred_x, target).float()
            factor = torch.abs(pred_x-target).float().mean()
            err = loss(last_output, target)*factor
            optimizer.zero_grad()
            err.backward()
            optimizer.step()
            loss_val += err.item()
            score_val += score.mean().item()
        print('Epoch ' + str(e) + ' --- tr loss: ' + str(loss_val/train_size) + ' ---- tr acc: ' + str(score_val/train_size))
        test(model, test_data, test_labels)






def main():
    filenames = read_info()
    train_file_count = 0
    test_file_count = 0
    for f in filenames:
        print(f)
        data = read_file(f)
        labels = read_labels(f)
        if f.startswith('se'):
            if test_file_count == 0:
                test_data = np.asarray(process_data(data))
                test_labels = np.asarray(process_labels(labels))
            else:
                test_data = np.vstack((test_data, process_data(data)))
                test_labels = np.vstack((test_labels , process_labels(labels)))
            test_file_count = test_file_count + 1
        else:
            if train_file_count  == 0:
                train_data = np.asarray(process_data(data))
                train_labels = np.asarray(process_labels(labels))
            else:
                train_data = np.vstack((train_data, process_data(data)))
                train_labels = np.vstack((train_labels , process_labels(labels)))
            train_file_count = train_file_count + 1
            print(train_data.shape)
            print(train_labels.shape)
            #print(train_data.shape)
            #print(train_labels.shape)
            #train_data = np.vstack(train_data, process_data(data))
            #train_labels = np.vstack(train_data , process_labels(labels))
            #print(np.asarray(train_data).shape)
            #print(np.asarray(train_labels).shape)

    model = lstm_model()
    #train_data = np.asarray(train_data)
    #train_labels = np.asarray(train_labels)
    print(train_data.shape)
    print(train_labels.shape)
    train(train_data, test_data, train_labels, test_labels, model)
    #train(model)


if __name__ == '__main__':
    main()
