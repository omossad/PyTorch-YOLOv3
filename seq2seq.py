import pickle
import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
import csv

max_files = 99
data_path = '/home/omossad/scratch/Gaming-Dataset/processed/lstm_input/input_8x8/fifa/'
labels_path = '/home/omossad/scratch/Gaming-Dataset/processed/lstm_labels/labels_8x8/fifa/'
num_tiles = 8
num_classes = 3
time_steps = 4
batch_size = 8
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
    counter = 0
    for f in filenames:
        if counter > 9:
            break
        print(f)
        data = read_file(f)
        labels = read_labels(f)
        if f.startswith('ha_9'):
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
        counter = counter + 1
    model = lstm_model()
    #train_data = np.asarray(train_data)
    #train_labels = np.asarray(train_labels)
    print(train_data.shape)
    print(train_labels.shape)
    train(train_data, test_data, train_labels, test_labels, model)
    #train(model)


class ResCNNEncoder(nn.Module):
    def __init__(self, fc_hidden1=512, fc_hidden2=512, drop_p=0.3, CNN_embed_dim=300):
        """Load the pretrained ResNet-152 and replace top fc layer."""
        super(ResCNNEncoder, self).__init__()
        self.fc_hidden1, self.fc_hidden2 = fc_hidden1, fc_hidden2
        self.drop_p = drop_p
        #resnet = models.resnet152(pretrained=True)
        #modules = list(resnet.children())[:-1]      # delete the last fc layer.
        ##self.resnet = nn.Sequential(*modules)
        self.fc1 = nn.Linear(resnet.fc.in_features, fc_hidden1)
        self.bn1 = nn.BatchNorm1d(fc_hidden1, momentum=0.01)
        self.fc2 = nn.Linear(fc_hidden1, fc_hidden2)
        self.bn2 = nn.BatchNorm1d(fc_hidden2, momentum=0.01)
        self.fc3 = nn.Linear(fc_hidden2, CNN_embed_dim)

    def forward(self, x_3d):
        cnn_embed_seq = []
        for t in range(x_3d.size(1)):
            # ResNet CNN
            with torch.no_grad():
                x = self.resnet(x_3d[:, t, :, :, :])  # ResNet
                x = x.view(x.size(0), -1)             # flatten output of conv

            # FC layers
            x = self.bn1(self.fc1(x))
            x = F.relu(x)
            x = self.bn2(self.fc2(x))
            x = F.relu(x)
            x = F.dropout(x, p=self.drop_p, training=self.training)
            x = self.fc3(x)

            cnn_embed_seq.append(x)

        # swap time and sample dim such that (sample dim, time dim, CNN latent dim)
        cnn_embed_seq = torch.stack(cnn_embed_seq, dim=0).transpose_(0, 1)
        # cnn_embed_seq: shape=(batch, time_step, input_size)

        return cnn_embed_seq


class DecoderRNN(nn.Module):
    def __init__(self, CNN_embed_dim=300, h_RNN_layers=3, h_RNN=256, h_FC_dim=128, drop_p=0.3, num_classes=50):
        super(DecoderRNN, self).__init__()

        self.RNN_input_size = CNN_embed_dim
        self.h_RNN_layers = h_RNN_layers   # RNN hidden layers
        self.h_RNN = h_RNN                 # RNN hidden nodes
        self.h_FC_dim = h_FC_dim
        self.drop_p = drop_p
        self.num_classes = num_classes

        self.LSTM = nn.LSTM(
            input_size=self.RNN_input_size,
            hidden_size=self.h_RNN,
            num_layers=h_RNN_layers,
            batch_first=True,       # input & output will has batch size as 1s dimension. e.g. (batch, time_step, input_size)
        )

        self.fc1 = nn.Linear(self.h_RNN, self.h_FC_dim)
        self.fc2 = nn.Linear(self.h_FC_dim, self.num_classes)

    def forward(self, x_RNN):

        self.LSTM.flatten_parameters()
        RNN_out, (h_n, h_c) = self.LSTM(x_RNN, None)
        """ h_n shape (n_layers, batch, hidden_size), h_c shape (n_layers, batch, hidden_size) """
        """ None represents zero initial hidden state. RNN_out has shape=(batch, time_step, output_size) """

        # FC layers
        x = self.fc1(RNN_out[:, -1, :])   # choose RNN_out at the last time step
        x = F.relu(x)
        x = F.dropout(x, p=self.drop_p, training=self.training)
        x = self.fc2(x)

        return x

def train(log_interval, model, device, train_loader, optimizer, epoch):
    # set model as training mode
    cnn_encoder, rnn_decoder = model
    cnn_encoder.train()
    rnn_decoder.train()

    losses = []
    scores = []
    N_count = 0   # counting total trained sample in one epoch
    for batch_idx, (X, y) in enumerate(train_loader):
        # distribute data to device
        X, y = X.to(device), y.to(device).view(-1, )

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
        if (batch_idx + 1) % log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}, Accu: {:.2f}%'.format(
                epoch + 1, N_count, len(train_loader.dataset), 100. * (batch_idx + 1) / len(train_loader), loss.item(), 100 * step_score))

    return losses, scores

main
cnn_encoder = ResCNNEncoder(fc_hidden1=CNN_fc_hidden1, fc_hidden2=CNN_fc_hidden2, drop_p=dropout_p, CNN_embed_dim=CNN_embed_dim).to(device)
rnn_decoder = DecoderRNN(CNN_embed_dim=CNN_embed_dim, h_RNN_layers=RNN_hidden_layers, h_RNN=RNN_hidden_nodes,
                         h_FC_dim=RNN_FC_dim, drop_p=dropout_p, num_classes=k).to(device)
crnn_params = list(cnn_encoder.fc1.parameters()) + list(cnn_encoder.bn1.parameters()) + \
                  list(cnn_encoder.fc2.parameters()) + list(cnn_encoder.bn2.parameters()) + \
                  list(cnn_encoder.fc3.parameters()) + list(rnn_decoder.parameters())

optimizer = torch.optim.Adam(crnn_params, lr=learning_rate)
for epoch in range(epochs):
    # train, test model
    train_losses, train_scores = train(log_interval, [cnn_encoder, rnn_decoder], device, train_loader, optimizer, epoch)
    epoch_test_loss, epoch_test_score = validation([cnn_encoder, rnn_decoder], device, optimizer, valid_loader)


if __name__ == '__main__':
    main()
