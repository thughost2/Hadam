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
 
import os
import argparse
 
    
parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')
parser.add_argument('--lr', default=0.1, type=float, help='learning rate')
parser.add_argument('--resume', '-r', action='store_true', help='resume from checkpoint')
parser.add_argument('--method', '-m', help='optimization method')
parser.add_argument('--fraction', default=0.5, type=float, help='fraction for HMC')
parser.add_argument('--eta', default=0.1, type=float, help='eta (gradient) for HMC')
parser.add_argument('--gamma', default=100, type=float, help='temperature for HMC')
parser.add_argument('--Nepoch', default=100, type=int, help='number of epoch')
parser.add_argument('--batch', '-b', default=100, type=int, help='mini-batch size (1 = pure stochastic) Default: 100')
parser.add_argument('--bias_correction', action='store_true', default=False, help='whether to use bias correction Default: False')
                            
args = parser.parse_args()
    
dataloader_args = dict(shuffle=True, batch_size=args.batch, num_workers=1)

transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])
transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

train = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)
train_loader = torch.utils.data.DataLoader(train, **dataloader_args)
test = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)
test_loader = torch.utils.data.DataLoader(test, **dataloader_args)

train_data = train.train_data
 
classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')


 
Ntrain_batch = len(train_loader.dataset) / args.batch
Ntest_batch = len(test_loader.dataset) / args.batch
losses = []
train_errs = []
test_errs = []
start_epoch = 0

if args.resume:
    # Load checkpoint.
    print('==> Resuming from checkpoint..')
    assert os.path.isdir('checkpoint'), 'Error: no checkpoint directory found!'
    checkpoint = torch.load('./checkpoint/cnn_cifar10_'+args.method)
    model = checkpoint['model']
    start_epoch = checkpoint['epoch']
    train_errs = checkpoint['train_errs']
    test_errs = checkpoint['test_errs']
else:
    print('==> Building model..')

from models import resnet
model = resnet.ResNet18().cuda()

if args.method == 'sgd':
    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum= 0.9)
elif args.method == 'adam':
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
elif args.method == 'hadam':
    import Hadam
    optimizer = Hadam.Hadam(model.parameters(), lr=args.lr, fraction = args.fraction, eta = args.eta, gamma = args.gamma, bias_correction = args.bias_correction)
elif args.method == 'adagrad':
    optimizer = optim.Adagrad(model.parameters(), lr=args.lr)

    
model.train()



for epoch in range(start_epoch+1, args.Nepoch+1):
    for batch_idx, (data, target) in enumerate(train_loader):
        # Get Samples
        data, target = Variable(data).cuda(), Variable(target).cuda()
        # Init
        optimizer.zero_grad()

        # Predict
        y_pred = model(data) 
        
        # Calculate loss
        loss = F.cross_entropy(y_pred, target)
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
    train_err = 1
    for batch_idx, (data, target) in enumerate(train_loader):
        evaluate_x = Variable(data).cuda() 
        evaluate_y = Variable(target).cuda()
        y_pred = model(evaluate_x)
        pred = y_pred.data.max(1)[1]
        train_err -= pred.eq(evaluate_y.data).sum()/len(train_loader.dataset)
    train_errs.append(train_err)
    print('\r Epoch: {} \t Full Train Error: {:.6f}'.format(epoch, train_err))
        
    test_err = 1
    for batch_idx, (data, target) in enumerate(test_loader):
        evaluate_x = Variable(data).cuda() 
        evaluate_y = Variable(target).cuda()
        y_pred = model(evaluate_x)
        pred = y_pred.data.max(1)[1]
        test_err -= pred.eq(evaluate_y.data).sum()/len(test_loader.dataset)
    test_errs.append(test_err)
    print('\r Epoch: {} \t Full Test Error: {:.6f}'.format(epoch, test_err))
    
    if not os.path.isdir('checkpoint'):
        os.mkdir('checkpoint')
    state = {
            'model': model,
            'epoch': epoch,
            'train_errs':train_errs,
            'test_errs':test_errs
            }
    torch.save(state, './checkpoint/cnn_cifar10_' + args.method)
         

    
import simplejson
f_train = open('cifar10_cnn_trainerr_'+ args.method + '.txt', 'w')
simplejson.dump(train_errs, f_train)
f_train.close()
f_test = open('cifar10_cnn_testerr_'+ args.method + '.txt', 'w')
simplejson.dump(test_errs, f_test)
f_test.close()
 
