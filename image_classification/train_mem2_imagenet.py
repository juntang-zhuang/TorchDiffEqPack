import torch
import time
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.nn.init as init
import torchvision
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import argparse
import logging
import numpy as np
from tensorboardX import SummaryWriter
import math
import sys
import os
sys.path.append('../')
import shutil
from DiffEqPack.odesolver_mem import odesolve_adjoint_sym12 as odesolve


parser = argparse.ArgumentParser()
parser.add_argument('--network', type=str, choices=['resnet', 'sqnxt'], default='resnet')
parser.add_argument('--method', type=str, choices=['Euler', 'RK2', 'RK4', 'RK23', 'RK12', 'Dopri5',
                                                   'Sym12Async','ODE23s', 'ADALF','FixedStep_ADALF','FixedStep_Sym12Async'],
                    default='Sym12Async')
parser.add_argument('--num_epochs', type=int, default=90)
parser.add_argument('--run', type=int, default=0)
parser.add_argument('--start_epoch', type=int, default=0)
# Checkpoints
parser.add_argument('-c', '--checkpoint', default='./checkpoint_mem', type=str, metavar='PATH',
                    help='path to save checkpoint (default: checkpoint)')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('--datadir', default=None, type=str, metavar='PATH',
                    help='path to data')
parser.add_argument('--lr', type=float, default=0.1)
parser.add_argument('--h', type=float, default=0.5, help='Initial Stepsize')
parser.add_argument('--t0', type=float, default=0.0, help='Initial time')
parser.add_argument('--t1', type=float, default=1.0, help='End time')
parser.add_argument('--rtol', type=float, default=1e-1, help='Releative tolerance')
parser.add_argument('--atol', type=float, default=1e-2, help='Absolute tolerance')
parser.add_argument('--print_neval', type=bool, default=False, help='Print number of evaluation or not')
parser.add_argument('--neval_max', type=int, default=5000000, help='Maximum number of evaluation in integration')

parser.add_argument('--batch_size', type=int, default=258)
parser.add_argument('--workers', type=int, default=16)

args = parser.parse_args()

def lr_schedule(lr, epoch, total_epochs = args.num_epochs):
    optim_factor = 0
    if epoch > total_epochs//3*2:
        optim_factor = 2
    elif epoch > total_epochs//3:
        optim_factor = 1

    return lr / math.pow(10, (optim_factor))

if args.network == 'sqnxt':
    from models.sqnxt import SqNxt_23_1x

    writer = SummaryWriter(
        'sqnxt/' + args.network + '_mem_' + args.method + '_lr_' + str(args.lr) + '_h_' + str(args.h) + '/')
elif args.network == 'resnet':
    from models.resnet import ResNet18

    writer = SummaryWriter(
        'resnet/' + args.network + '_mem_' + args.method + '_lr_' + str(args.lr) + '_h_' + str(args.h) + '/')

num_epochs = int(args.num_epochs)
lr = float(args.lr)
start_epoch = int(args.start_epoch)
batch_size = int(args.batch_size)

is_use_cuda = torch.cuda.is_available()
device = torch.device("cuda:0" if is_use_cuda else "cpu")


class ODEBlock(nn.Module):

    def __init__(self, odefunc):
        super(ODEBlock, self).__init__()
        self.odefunc = odefunc
        self.options = {}
        self.options.update({'method': args.method})
        self.options.update({'h': args.h})
        self.options.update({'t0': args.t0})
        self.options.update({'t1': args.t1})
        self.options.update({'rtol': args.rtol})
        self.options.update({'atol': args.atol})
        self.options.update({'print_neval': args.print_neval})
        self.options.update({'neval_max': args.neval_max})
        self.options.update({'print_time': False})
        print(self.options)

    def forward(self, x):
        out = odesolve(self.odefunc, x, self.options)
        return out

    @property
    def nfe(self):
        return self.odefunc.nfe

    @nfe.setter
    def nfe(self, value):
        self.odefunc.nfe = value


def conv_init(m):
    class_name = m.__class__.__name__
    if class_name.find('Conv') != -1 and m.bias is not None:
        init.xavier_uniform_(m.weight, gain=np.sqrt(2))
        init.constant_(m.bias, 0)
    elif class_name.find('BatchNorm') != -1:
        init.constant_(m.weight, 1)
        init.constant_(m.bias, 0)

normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

traindir = os.path.join(args.datadir,'train')
valdir = os.path.join(args.datadir, 'val')
train_dataset = datasets.ImageFolder(
        traindir,
        transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
]))

train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True,
        num_workers=args.workers, pin_memory=True)

test_dataset = datasets.ImageFolder(valdir, transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            normalize,
        ]))
test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=args.batch_size, shuffle=False,
num_workers=args.workers, pin_memory=True)

if args.network == 'sqnxt':
    net = SqNxt_23_1x(10, ODEBlock)
elif args.network == 'resnet':
    net = ResNet18(ODEBlock, num_classes=1000)

net.apply(conv_init)
print(net)
if is_use_cuda:
    net.cuda()  # to(device)
    net = nn.DataParallel(net)
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=args.lr, momentum=0.9, weight_decay=1e-4)


def train(epoch):
    net.train()
    train_loss = 0
    correct = 0
    total = 0

    print('Training Epoch: #%d, LR: %.4f' % (epoch, lr_schedule(lr, epoch, args.num_epochs)))
    for idx, (inputs, labels) in enumerate(train_loader):
        if is_use_cuda:
            inputs, labels = inputs.cuda(), labels.cuda()
        optimizer.zero_grad()
        with torch.autograd.set_detect_anomaly(True):
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()

        optimizer.step()
        writer.add_scalar('Train/Loss', loss.item(), epoch * 50000 + batch_size * (idx + 1))
        train_loss += loss.item()
        _, predict = torch.max(outputs, 1)
        total += labels.size(0)
        correct += predict.eq(labels).cpu().sum().double()

        sys.stdout.write('\r')
        sys.stdout.write('[%s] Training Epoch [%d/%d] Iter[%d/%d]\t\tLoss: %.4f Acc@1: %.3f'
                         % (time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time())),
                            epoch, num_epochs, idx, len(train_dataset) // batch_size,
                            train_loss / (batch_size * (idx + 1)), correct / total))
        sys.stdout.flush()
    writer.add_scalar('Train/Accuracy', correct / total, epoch)


def test(epoch):
    net.eval()
    test_loss = 0
    correct = 0
    total = 0
    for idx, (inputs, labels) in enumerate(test_loader):
        if is_use_cuda:
            inputs, labels = inputs.cuda(), labels.cuda()

        with torch.no_grad():
            outputs = net(inputs)

        loss = criterion(outputs, labels)

        test_loss += loss.item()
        _, predict = torch.max(outputs, 1)
        total += labels.size(0)
        correct += predict.eq(labels).cpu().sum().double()
        writer.add_scalar('Test/Loss', loss.item(), epoch * 50000 + test_loader.batch_size * (idx + 1))

        sys.stdout.write('\r')
        sys.stdout.write('[%s] Testing Epoch [%d/%d] Iter[%d/%d]\t\tLoss: %.4f Acc@1: %.3f'
                         % (time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time())),
                            epoch, num_epochs, idx, len(test_dataset) // test_loader.batch_size,
                            test_loss / (100 * (idx + 1)), correct / total))
        sys.stdout.flush()

    acc = correct / total
    writer.add_scalar('Test/Accuracy', acc, epoch)
    return acc


def adjust_learning_rate(optimizer, lr):
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def save_checkpoint(state, is_best, checkpoint='checkpoint', filename='checkpoint.pth.tar'):
    if not os.path.exists(checkpoint):
        os.makedirs(checkpoint)
    filepath = os.path.join(checkpoint, filename)
    torch.save(state, filepath)
    if is_best:
        shutil.copyfile(filepath, os.path.join(checkpoint, 'model_best.pth.tar'))


best_acc = 0.0

if args.resume:
    # Load checkpoint.
    print('==> Resuming from checkpoint..')
    assert os.path.isfile(args.resume), 'Error: no checkpoint directory found!'
    args.checkpoint = os.path.dirname(args.resume)
    checkpoint = torch.load(args.resume)
    best_acc = checkpoint['best_acc']
    start_epoch = checkpoint['epoch']
    net.load_state_dict(checkpoint['state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer'])

for _epoch in range(start_epoch, start_epoch + num_epochs):
    start_time = time.time()

    _lr = lr_schedule(args.lr, _epoch, args.num_epochs)
    adjust_learning_rate(optimizer, _lr)

    train(_epoch)
    print()
    test_acc = test(_epoch)
    print()
    print()
    end_time = time.time()
    print('Epoch #%d Cost %ds' % (_epoch, end_time - start_time))

    # save model
    is_best = test_acc > best_acc
    best_acc = max(test_acc, best_acc)
    save_checkpoint({
        'epoch': _epoch + 1,
        'state_dict': net.state_dict(),
        'acc': test_acc,
        'best_acc': best_acc,
        'optimizer': optimizer.state_dict(),
    }, is_best, checkpoint=args.checkpoint + '_' + args.method + '_' + args.network + '_' + str(args.run))

print('Best Acc@1: %.4f' % (best_acc * 100))
writer.close()
