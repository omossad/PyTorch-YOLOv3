import pickle
import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable



data_path = '/home/omossad/scratch/temp/numpy/'
pkl_file = open(data_path + 'data_array.pkl', 'rb')
data = pickle.load(pkl_file)
#data = data[:640]
pkl_file.close()

### PARAMS ####
num_images = len(data)
num_tiles = 4
num_classes = 3
time_steps = 4
batch_size = 8
epochs = 200
learning_rate = 0.0007
weight_decay = 0
test_size = 36
adjust = 2
in_size = num_tiles * num_tiles * num_classes
classes_no = num_tiles * num_tiles
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

print(num_images)
image_indices = np.arange(0,num_images)
indices = np.array([ image_indices[i:i+time_steps] for i in range(len(image_indices)-time_steps+1) ])
print(image_indices)
print(indices)

#### DATA MANIPULATION ####
data = np.asarray(data)
data = np.reshape(data, (num_images,-1))
print(data.shape)
img_data = []
for i in range(len(indices)-adjust):
    img_data.append(data[indices[i]])
data = np.asarray(img_data)
print(data.shape)

#data = np.reshape(data, (num_images//time_steps, time_steps, -1))
data = np.reshape(data, (len(data)//batch_size, batch_size, time_steps, -1))
#data = np.reshape(data, (num_images//(time_steps*batch_size), batch_size, time_steps, -1))
data = np.transpose(data, (0, 2, 1, 3))
print(data.shape)

#### TARGET MANIPULATION ####
targets = np.loadtxt(data_path + 'trgt_array.dat')
targets = np.asarray(targets)
print(targets.shape)

targets_data = []
for i in range(len(indices)-adjust):
    targets_data.append(targets[indices[i]])
targets = np.asarray(targets_data)
print(targets.shape)

#targets = [i for i in range(num_images)]
#targets = np.array([ targets[i:i+time_steps-1] for i in range(len(targets)-time_steps-2) ])
#targets = np.reshape(targets, (num_images//(time_steps*batch_size), batch_size, time_steps, -1))
targets = np.reshape(targets, (len(targets)//batch_size, batch_size, time_steps, -1))
print(targets.shape)
targets = np.transpose(targets, (0, 2, 1, 3))

#print(targets.shape)
#print(targets)


# time_steps are frames
# batch size is the num_samples in a single batch
# in_size features
# class_no  is the number of tiles

model = nn.LSTM(in_size, classes_no, 2)
model.to(device)
loss = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
#print(len(data))
#input_seq = Variable(torch.randn(time_steps, batch_size, in_size))
for e in range(epochs):
    loss_val = 0
    score_val = 0
    for d in range(len(data)-test_size):
        input_seq = Variable(torch.from_numpy(data[d]).float().to(device))
        #print(input_seq.shape)
        #print(input_seq)
        #input_seq = Variable(torch.randn(time_steps, batch_size, in_size))
        #print(input_seq.shape)
        #print(input_seq)
        output_seq, _ = model(input_seq)
        #print(output_seq.shape)

        last_output = output_seq[-1]
        #print(last_output.shape)
        #target = Variable(torch.LongTensor(batch_size).random_(0, classes_no-1))
        target = Variable(torch.from_numpy(targets[d][-1]).long().view(-1).to(device))
        #print(target)
        #print(last_output)
        _, pred_x = torch.max(last_output, 1)
        #print(pred_x)
        score = torch.eq(pred_x, target).float()
        #print(score)
        #print(score.mean())
        #print(target)
        #print(target.shape)
        #print(pred_x)
        #print(target)
        #print(torch.abs(pred_x-target))
        factor = torch.abs(pred_x-target).float().mean()
        #print(factor)
        err = loss(last_output, target)*factor
        optimizer.zero_grad()
        err.backward()
        optimizer.step()
        #print(err.item())
        loss_val += err.item()
        score_val += score.mean().item()
    print('Epoch ' + str(e) + ' --- tr loss: ' + str(loss_val/(len(data)-test_size)) + ' ---- tr acc: ' + str(score_val/(len(data)-test_size)))
    score_val = 0
    for d in range(test_size):
        input_seq = Variable(torch.from_numpy(data[len(data)-test_size+d]).float().to(device))
        output_seq, _ = model(input_seq)
        last_output = output_seq[-1]
        target = Variable(torch.from_numpy(targets[len(data)-test_size+d][-1]).long().view(-1).to(device))
        _, pred_x = torch.max(last_output, 1)
        score = torch.eq(pred_x, target).float()
        score_val += score.mean().item()
    print(' ---- test acc: ' + str(score_val/test_size) + '\n')

'''

# Here we define our model as a class
class LSTM(nn.Module):

    def __init__(self, input_dim, hidden_dim, batch_size, output_dim=1,
                    num_layers=2):
        super(LSTM, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.batch_size = batch_size
        self.num_layers = num_layers

        # Define the LSTM layer
        self.lstm = nn.LSTM(self.input_dim, self.hidden_dim, self.num_layers)

        # Define the output layer
        self.linear = nn.Linear(self.hidden_dim, output_dim)

    def init_hidden(self):
        # This is what we'll initialise our hidden state as
        return (torch.zeros(self.num_layers, self.batch_size, self.hidden_dim),
                torch.zeros(self.num_layers, self.batch_size, self.hidden_dim))

    def forward(self, input):
        # Forward pass through LSTM layer
        # shape of lstm_out: [input_size, batch_size, hidden_dim]
        # shape of self.hidden: (a, b), where a and b both
        # have shape (num_layers, batch_size, hidden_dim).
        lstm_out, self.hidden = self.lstm(input.view(len(input), self.batch_size, -1))

        # Only take the output from the final timetep
        # Can pass on the entirety of lstm_out to the next layer if it is a seq2seq prediction
        y_pred = self.linear(lstm_out[-1].view(self.batch_size, -1))
        return y_pred.view(-1)

output_dim = 4
lstm_input_size = 48
num_train = 2
h1 = 128
num_layers = 4
learning_rate = 0.001
num_epochs = 10
X_train = torch.from_numpy(data).float().to(device)
X_train = X_train.view(2,-1)
y_train = torch.from_numpy(targets).float().to(device)

print(X_train.shape)
print(y_train.shape)

model = LSTM(lstm_input_size, h1, batch_size=num_train, output_dim=output_dim, num_layers=num_layers)
model.to(device)
loss_fn = torch.nn.MSELoss(size_average=False)

optimiser = torch.optim.Adam(model.parameters(), lr=learning_rate)

#####################
# Train model
#####################

hist = np.zeros(num_epochs)

for t in range(num_epochs):
    # Clear stored gradient
    model.zero_grad()

    # Initialise hidden state
    # Don't do this if you want your LSTM to be stateful
    model.hidden = model.init_hidden()

    # Forward pass
    y_pred = model(X_train)

    loss = loss_fn(y_pred, y_train)
    if t % 100 == 0:
        print("Epoch ", t, "MSE: ", loss.item())
    hist[t] = loss.item()

    # Zero out gradient, else they will accumulate between epochs
    optimiser.zero_grad()

    # Backward pass
    loss.backward()

    # Update parameters
    optimiser.step()
'''
