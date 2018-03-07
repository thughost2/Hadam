import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data.dataloader as dataloader
import torch.optim as optim

import torchvision
from torch.utils.data import TensorDataset
from torch.autograd import Variable
from torchvision import transforms as transforms
from torchvision.datasets import MNIST
    
import os
import argparse
 
    
parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')
parser.add_argument('--lr', default=0.1, type=float, help='learning rate')
parser.add_argument('--resume', '-r', action='store_true', help='resume from checkpoint')
parser.add_argument('--method', '-m', help='optimization method')
parser.add_argument('--fraction', default=0.5, type=float, help='fraction for HMC')
parser.add_argument('--eta', default=0.1, type=float, help='eta (gradient) for HMC')
parser.add_argument('--gamma', default=10000, type=float, help='temperature for HMC')
parser.add_argument('--Nepoch', default=100, type=int, help='number of epoch')
parser.add_argument('--batch', '-b', default=60, type=int, help='mini-batch size (1 = pure stochastic) Default: 60')
parser.add_argument('--bias_correction', action='store_true', default=False, help='whether to use bias correction Default: False')

args = parser.parse_args()    
    
train = MNIST('./data', train=True, download=True, transform=transforms.Compose([
    transforms.ToTensor(), # ToTensor does min-max normalization. 
]), )

test = MNIST('./data', train=False, download=True, transform=transforms.Compose([
    transforms.ToTensor(), # ToTensor does min-max normalization. 
]), )
# train.train_data = train.train_data[1:500,:,:]
# Create DataLoader

dataloader_args = dict(shuffle=True, batch_size=args.batch, num_workers=1)
train_loader = dataloader.DataLoader(train, **dataloader_args)
test_loader = dataloader.DataLoader(test, **dataloader_args)
train_data = train.train_data
  

class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()

        self.encoder = nn.Sequential(
            nn.Linear(28*28, 512),
            nn.Softplus(),
            nn.Linear(512, 256),
            nn.Softplus(),
            nn.Linear(256, 128),
            nn.Softplus(),
            nn.Linear(128, 32),   # compress to 3 features which can be visualized in plt
        )
        self.decoder = nn.Sequential(
            nn.Linear(32, 128),
            nn.Softplus(),
            nn.Linear(128, 256),
            nn.Softplus(),
            nn.Linear(256, 512),
            nn.Softplus(),
            nn.Linear(512, 28*28),
            nn.Sigmoid(),       # compress to a range (0, 1)
        )
 
    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded    
    
    
    
Ntrain_batch = len(train_loader.dataset) / args.batch
Ntest_batch = len(test_loader.dataset) / args.batch
losses = []
train_losses = []
test_losses = []
start_epoch = 0

if args.resume:
    # Load checkpoint.
    print('==> Resuming from checkpoint..')
    assert os.path.isdir('checkpoint'), 'Error: no checkpoint directory found!'
    checkpoint = torch.load('./checkpoint/autoencoder_mnist_'+args.method)
    model = checkpoint['model']
    start_epoch = checkpoint['epoch']
    train_losses = checkpoint['train_losses']
    test_losses = checkpoint['test_losses']
else:
    print('==> Building model..')
 
 
model = Model().cuda()

if args.method == 'sgd':
    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum= 0.9)
elif args.method == 'adam':
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
elif args.method == 'hadam':
    import Hadam
    optimizer = Hadam.Hadam(model.parameters(), lr=args.lr, fraction = args.fraction, eta = args.eta, gamma = args.gamma, bias_correction = args.bias_correction)
elif args.method == 'adagrad':
    optimizer = optim.Adagrad(model.parameters(), lr=args.lr)

loss_func = nn.MSELoss()   
    
model.train()


for epoch in range(start_epoch+1, args.Nepoch+1):
    for batch_idx, (data, target) in enumerate(train_loader):
        # Get Samples
        data = Variable(data.view(-1,28*28)).cuda()
        # Init
        optimizer.zero_grad()

        # Predict
        y_pred = model(data) 
        
        # Calculate loss
        loss = loss_func(y_pred, data)
        losses.append(loss.data[0])
        
        # Backpropagation
        loss.backward()
        optimizer.step()
        
        # Display
        if batch_idx % (Ntrain_batch / 10) == 1:
            print('\r Train Epoch: {} [{}/{} ({:.0f}%)]\t Loss: {:.6f}'.format(
                epoch, 
                batch_idx * len(data), 
                len(train_loader.dataset),
                100. * batch_idx / len(train_loader), 
                loss.data[0]))             
    
    #Full train/test loss
    train_loss = 0
    for batch_idx, (data, target) in enumerate(train_loader):
        data = Variable(data.view(-1,28*28)).cuda()
        y_pred = model(data) 
        loss = loss_func(y_pred, data)
        train_loss += loss.data[0]
    train_loss = train_loss / Ntrain_batch
    train_losses.append(train_loss)
    print('\r Epoch: {} \t Full Train Loss: {:.6f}'.format(epoch, train_loss))
        
    test_loss = 0
    for batch_idx, (data, target) in enumerate(test_loader):
        data = Variable(data.view(-1,28*28)).cuda()
        y_pred = model(data) 
        loss = loss_func(y_pred, data)
        test_loss += loss.data[0]
    test_loss = test_loss / Ntest_batch
    test_losses.append(test_loss)
    print('\r Epoch: {} \t Full Test Loss: {:.6f}'.format(epoch, test_loss))
 
    if not os.path.isdir('checkpoint'):
        os.mkdir('checkpoint')
    state = {
            'model': model,
            'epoch': epoch,
            'train_losses':train_losses,
            'test_losses':test_losses
            }
    torch.save(state, './checkpoint/autoencoder_mnist_' + args.method)

    
import json
f_train = open('mnist_autoencoder_trainloss_'+ args.method + '.txt', 'w')
json.dump(train_losses, f_train)
f_train.close()
f_test = open('mnist_autoencoder_testloss_'+ args.method + '.txt', 'w')
json.dump(test_losses, f_test)
f_test.close()
 
