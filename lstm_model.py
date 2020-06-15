import pickle
import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
import csv

max_files = 2
data_path = '/home/omossad/scratch/Gaming-Dataset/processed/lstm_input/input_8x8/fifa/'
num_tiles = 8
num_classes = 3
time_steps = 4
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


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
    pkl_file = open(data_path + filename + '_x.pkl', 'rb')
    data = pickle.load(pkl_file)
    pkl_file.close()
    return data



def process_data(data):
    num_images = len(data)
    image_indices = np.arange(0,num_images)
    print(image_indices)
    indices = np.array([ image_indices[i:i+time_steps] for i in range(len(image_indices)-time_steps+1) ])
    data = np.asarray(data)
    data = np.reshape(data, (num_images,-1))
    img_data = []
    for i in range(len(indices)-adjust):
        img_data.append(data[indices[i]])
    data = np.asarray(img_data)
    data = np.reshape(data, (len(data)//batch_size, batch_size, time_steps, -1))
    data = np.transpose(data, (0, 2, 1, 3))
    return data


def lstm_model():
    ### MODEL PARAMS ####
    batch_size = 4
    epochs = 200
    learning_rate = 0.0001
    weight_decay = 0

    test_size = 104
    adjust = 10
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
    loss = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)


def test(test_data, model, test_labels):
    score_val = 0
    test_size = len(test_data)
    for d in range(test_size):
        input_seq = Variable(torch.from_numpy(test_data[d]).float().to(device))
        output_seq, _ = model(input_seq)
        last_output = output_seq[-1]
        #last_output = out_model(last_output)
        target = Variable(torch.from_numpy(test_labels[d]).long().view(-1).to(device))
        _, pred_x = torch.max(last_output, 1)
        score = torch.eq(pred_x, target).float()
        score_val += score.mean().item()
    print(' ---- test acc: ' + str(score_val/test_size) + '\n')

def train():
    train_size = len(train_data)
    for e in range(epochs):
        loss_val = 0
        score_val = 0
        for d in range(train_size):
            input_seq = Variable(torch.from_numpy(train_data[d]).float().to(device))
            output_seq, _ = model(input_seq)
            last_output = output_seq[-1]
            target = Variable(torch.from_numpy(train_labels[d][-1]).long().view(-1).to(device))
            _, pred_x = torch.max(last_output, 1)
            score = torch.eq(pred_x, target).float()
            factor = torch.abs(pred_x-target).float().mean()
            err = loss(last_output, target)*factor
            optimizer.zero_grad()
            err.backward()
            optimizer.step()
            loss_val += err.item()
            score_val += score.mean().item()
        print('Epoch ' + str(e) + ' --- tr loss: ' + str(loss_val/(len(data)-test_size)) + ' ---- tr acc: ' + str(score_val/(len(data)-test_size)))
        test(test_data, model, test_labels)






def main():
    filenames = read_info()
    for f in filenames:
        data = read_file(f)
        data = process_data(data)
    model = lstm_model()
    #train(model)


if __name__ == '__main__':
    main()
