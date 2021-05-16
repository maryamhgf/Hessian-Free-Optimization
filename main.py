'''Train CIFAR10/CIFAR100 with PyTorch.'''
import sys
from hf import HessianFreeOptimizer
from vgg import VGG
import vgg
import argparse
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import MultiStepLR
from tqdm import tqdm
from tensorboardX import SummaryWriter
from utils.network_utils import get_network
from utils.data_utils import get_dataloader
from torchsummary import summary
from backpack import backpack, extend
from backpack.utils.conv import unfold_func
from backpack import backpack, extend, extensions
import math
import time
import copy
import matplotlib.pylab as plt

from torch import einsum, matmul, eye
from torch.linalg import inv
import numpy as np
# for REPRODUCIBILITY
# torch.manual_seed(0)

# fetch args
parser = argparse.ArgumentParser()

parser.add_argument('--network', default='vgg16', type=str)
parser.add_argument('--depth', default=19, type=int)
parser.add_argument('--dataset', default='cifar10', type=str)

# densenet
parser.add_argument('--growthRate', default=12, type=int)
parser.add_argument('--compressionRate', default=2, type=int)

# wrn, densenet
parser.add_argument('--widen_factor', default=1, type=int)
parser.add_argument('--dropRate', default=0.0, type=float)
parser.add_argument('--base_width', default=24, type=int)
parser.add_argument('--cardinality', default=32, type=int)


parser.add_argument('--device', default='cuda', type=str)
parser.add_argument('--resume', '-r', action='store_true')
parser.add_argument('--load_path', default='', type=str)
parser.add_argument('--log_dir', default='runs/pretrain', type=str)


parser.add_argument('--optimizer', default='hf', type=str)
parser.add_argument('--batch_size', default=64, type=int)
parser.add_argument('--epoch', default=100, type=int)
parser.add_argument('--milestone', default=None, type=str)
parser.add_argument('--learning_rate', default=0.01, type=float)
parser.add_argument('--learning_rate_decay', default=0.1, type=float)
parser.add_argument('--momentum', default=0.9, type=float)
parser.add_argument('--stat_decay', default=0.95, type=float)
parser.add_argument('--damping', default=1e-3, type=float)
parser.add_argument('--kl_clip', default=1e-2, type=float)
parser.add_argument('--weight_decay', default=3e-3, type=float)
parser.add_argument('--TCov', default=20, type=int)
parser.add_argument('--TScal', default=20, type=int)
parser.add_argument('--TInv', default=100, type=int)

parser.add_argument('--freq', default=100, type=int)
parser.add_argument('--low_rank', default='false', type=str)
parser.add_argument('--gamma', default=0.9, type=float)
parser.add_argument('--batchnorm', default='false', type=str)
parser.add_argument('--step_info', default='false', type=str)
parser.add_argument('--memory_efficient', default='false', type=str)
parser.add_argument('--trial', default='true', type=str)
parser.add_argument('--super_opt', default='false', type=str)

# for adam optimizer
parser.add_argument('--epsilon', default=1e-8, type=float)

# for hf optimizer
parser.add_argument('--maxIter', default=100, type=int)
parser.add_argument('--tol', default=1e-1, type=float)
parser.add_argument('--atol', default=1e-8, type=float)

parser.add_argument('--debug_mem', default='false', type=str)
parser.add_argument('--prefix', default=None, type=str)

args = parser.parse_args()
# init model
nc = {
    'cifar10': 10,
    'cifar100': 100,
    'mnist':10,
    'fashion-mnist': 10
}
num_classes = nc[args.dataset]

net = get_network(args.network,
                  depth=args.depth,
                  num_classes=num_classes,
                  growthRate=args.growthRate,
                  compressionRate=args.compressionRate,
                  widen_factor=args.widen_factor,
                  dropRate=args.dropRate,
                  base_width=args.base_width,
                  cardinality=args.cardinality)
print(net)
optim_name = args.optimizer.lower()

net = net.to(args.device)
net = extend(net)

module_names = ''
if hasattr(net, 'features'): 
    module_names = 'features'
elif hasattr(net, 'children'):
    module_names = 'children'
else:
    print('unknown net modules...')


if args.dataset == 'mnist':
    summary(net, ( 1, 28, 28))
elif args.dataset == 'cifar10':
    summary(net, ( 3, 32, 32))
elif args.dataset == 'cifar100':
    summary(net, ( 3, 32, 32))
elif args.dataset == 'fashion-mnist':
    summary(net, ( 1, 28, 28))
# init dataloader
trainloader, testloader = get_dataloader(dataset=args.dataset,
                                         train_batch_size=args.batch_size,
                                         test_batch_size=256)

# init optimizer and lr scheduler

tag = optim_name

if optim_name == 'hf':
    print("Hessian Free Optimizer selected.")
 
    optimizer = HessianFreeOptimizer(net.parameters(),
                    extensions.GGNMP(),
                    lr=args.learning_rate,
                    damping=args.damping,
                    maxIter=args.maxIter,
                    tol=args.tol,
                    atol=args.atol,
                    weight_decay=args.weight_decay)
    print("FLAG")
                    
else:
    raise NotImplementedError

if args.milestone is None:
    lr_scheduler = MultiStepLR(optimizer, milestones=[int(args.epoch*0.5), int(args.epoch*0.75)], gamma=args.learning_rate_decay)
else:
    milestone = [int(_) for _ in args.milestone.split(',')]
    lr_scheduler = MultiStepLR(optimizer, milestones=milestone, gamma=args.learning_rate_decay)

# init criterion
criterion = nn.CrossEntropyLoss()
criterion_none = nn.CrossEntropyLoss(reduction='none')

damping = args.damping
start_epoch = 0
best_acc = 0
if args.resume:
    print('==> Resuming from checkpoint..')
    assert os.path.isfile(args.load_path), 'Error: no checkpoint directory found!'
    checkpoint = torch.load(args.load_path)
    net.load_state_dict(checkpoint['net'])
    best_acc = checkpoint['acc']
    start_epoch = checkpoint['epoch']
    print('==> Loaded checkpoint at epoch: %d, acc: %.2f%%' % (start_epoch, best_acc))

# init summary writter

log_dir = os.path.join(args.log_dir, args.dataset, args.network, args.optimizer,
                       'lr%.3f_wd%.4f_damping%.4f' %
                       (args.learning_rate, args.weight_decay, args.damping))
if not os.path.isdir(log_dir):
    os.makedirs(log_dir)
writer = SummaryWriter(log_dir)

TRAIN_INFO  = {}
TRAIN_INFO['train_loss'] = []
TRAIN_INFO['test_loss'] = []
TRAIN_INFO['train_acc'] = []
TRAIN_INFO['test_acc'] = []
TRAIN_INFO['total_time'] = []
TRAIN_INFO['epoch_time'] = []

if args.debug_mem == 'true':
  TRAIN_INFO['memory'] = []
  
def store_io_(Flag=True):
    if module_names == 'children':
        all_modules = net.children()
    elif module_names == 'features':
        all_modules = net.features.children()
    for m in all_modules:
        if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d):
            m.training = Flag


def train(epoch):
    print('\nEpoch: %d' % epoch)
    net.train()
    train_loss = 0
    correct = 0
    total = 0
    step_st_time = time.time()
    epoch_time = 0
    # 
    desc = ('[%s][LR=%s] Loss: %.3f | Acc: %.3f%% (%d/%d)' %
            (tag, lr_scheduler.get_last_lr()[0], 0, 0, correct, total))

    writer.add_scalar('train/lr', lr_scheduler.get_last_lr()[0], epoch)

    prog_bar = tqdm(enumerate(trainloader), total=len(trainloader), desc=desc, leave=True)
    criterion = torch.nn.CrossEntropyLoss().to(args.device)
    criterion = extend(criterion)
    for batch_idx, (inputs, targets) in prog_bar:
        if optim_name in ['hf']:
            print("OPTIMIZER: HF")
            optimizer.zero_grad()
            inputs, targets = inputs.to(args.device), targets.to(args.device)
            outputs = net(inputs)
            loss = criterion(outputs, targets)
            with backpack(extensions.GGNMP()):
                loss.backward()    
            optimizer.step()    
            
        if optimizer in ['hf']:
            train_loss += loss.detach().item()
            predicted = outputs.argmax(dim=1, keepdim=True).view_as(targets)
        else:
            train_loss += loss.item()
            _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()
        desc = ('[%s][LR=%s] Loss: %.3f | Acc: %.3f%% (%d/%d)' %
                (tag, lr_scheduler.get_last_lr()[0], train_loss / (batch_idx + 1), 100. * correct / total, correct, total))
        prog_bar.set_description(desc, refresh=True)
        if args.step_info == 'true' and (batch_idx % 50 == 0 or batch_idx == len(prog_bar) - 1):
            step_saved_time = time.time() - step_st_time
            epoch_time += step_saved_time
            test_acc, test_loss = test(epoch)
            TRAIN_INFO['train_acc'].append(float("{:.4f}".format(100. * correct / total)))
            TRAIN_INFO['test_acc'].append(float("{:.4f}".format(test_acc)))
            TRAIN_INFO['train_loss'].append(float("{:.4f}".format(train_loss/(batch_idx + 1))))
            TRAIN_INFO['test_loss'].append(float("{:.4f}".format(test_loss)))
            TRAIN_INFO['total_time'].append(float("{:.4f}".format(step_saved_time)))
            if args.debug_mem == 'true':
                TRAIN_INFO['memory'].append(torch.cuda.memory_reserved())
            step_st_time = time.time()
            net.train()
        
        if args.maxIter is not None and batch_idx > args.maxIter:
            break
    writer.add_scalar('train/loss', train_loss/(batch_idx + 1), epoch)
    writer.add_scalar('train/acc', 100. * correct / total, epoch)
    acc = 100. * correct / total
    train_loss = train_loss/(batch_idx + 1)
    if args.step_info == 'true':
        TRAIN_INFO['epoch_time'].append(float("{:.4f}".format(epoch_time)))

    return acc, train_loss


def test(epoch):
    global best_acc
    net.eval()
    test_loss = 0
    correct = 0
    total = 0
    desc = ('[%s][LR=%s] Loss: %.3f | Acc: %.3f%% (%d/%d)'
            % (tag,lr_scheduler.get_lr()[0], test_loss/(0+1), 0, correct, total))

    prog_bar = tqdm(enumerate(testloader), total=len(testloader), desc=desc, position=0, leave=True)
    with torch.no_grad():
        for batch_idx, (inputs, targets) in prog_bar:
            inputs, targets = inputs.to(args.device), targets.to(args.device)
            outputs = net(inputs)
            loss = criterion(outputs, targets)

            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

            desc = ('[%s][LR=%s] Loss: %.3f | Acc: %.3f%% (%d/%d)'
                    % (tag, lr_scheduler.get_lr()[0], test_loss / (batch_idx + 1), 100. * correct / total, correct, total))
            prog_bar.set_description(desc, refresh=True)

    # Save checkpoint.
    acc = 100.*correct/total
    writer.add_scalar('test/loss', test_loss / (batch_idx + 1), epoch)
    writer.add_scalar('test/acc', 100. * correct / total, epoch)

    if acc > best_acc:
        print('Saving..')
        state = {
            'net': net.state_dict(),
            'acc': acc,
            'epoch': epoch,
            'loss': test_loss,
            'args': args
        }

        torch.save(state, '%s/%s_%s_%s%s_best.t7' % (log_dir,
                                                     args.optimizer,
                                                     args.dataset,
                                                     args.network,
                                                     args.depth))
        best_acc = acc

    test_loss = test_loss/(batch_idx + 1)
    return acc, test_loss

def optimal_JJT(outputs, targets, batch_size, damping=1.0, alpha=0.95, low_rank='false', gamma=0.95, memory_efficient='false'):
    jac_list = 0
    vjp = 0
    update_list = {}
    with backpack(FisherBlock(damping, alpha, low_rank, gamma, memory_efficient)):
        loss = criterion(outputs, targets)
        loss.backward(retain_graph=True)
    for name, param in net.named_parameters():
        fisher_vals = param.fisher_block
        update_list[name] = fisher_vals[2]
    return update_list, loss
    
def optimal_JJT_v2(outputs, targets, batch_size, damping=1.0, alpha=0.95, low_rank='false', gamma=0.95, memory_efficient='false', super_opt='false'):
    jac_list = 0
    vjp = 0
    update_list = {}
    with backpack(FisherBlockEff(damping, alpha, low_rank, gamma, memory_efficient, super_opt)):
        loss = criterion(outputs, targets)
        loss.backward()
    for name, param in net.named_parameters():
        if hasattr(param, "fisher_block"):
            update_list[name] = param.fisher_block
        else:
            update_list[name] = param.grad.data
        
    return update_list, loss

def main():
    print("MAIN:::::::::::::::::::::::::::::")
    train_acc, train_loss = get_accuracy(trainloader)
    test_acc, test_loss = get_accuracy(testloader)
    TRAIN_INFO['train_acc'].append(float("{:.4f}".format(train_acc)))
    TRAIN_INFO['test_acc'].append(float("{:.4f}".format(test_acc)))
    TRAIN_INFO['train_loss'].append(float("{:.4f}".format(train_loss)))
    TRAIN_INFO['test_loss'].append(float("{:.4f}".format(test_loss)))
    TRAIN_INFO['total_time'].append(0.)
    if args.debug_mem == 'true':
      TRAIN_INFO['memory'].append(torch.cuda.memory_reserved())
    st_time = time.time()
    for epoch in range(start_epoch, args.epoch):
        print("EPOCH:::::::::::::::::", epoch)
        ep_st_time = time.time()
        train_acc, train_loss = train(epoch)
        print("AFTER TRAIN:::::::::::::")
        if args.step_info == "false":
            TRAIN_INFO['train_acc'].append(float("{:.4f}".format(train_acc)))
            TRAIN_INFO['train_loss'].append(float("{:.4f}".format(train_loss)))
            TRAIN_INFO['total_time'].append(float("{:.4f}".format(time.time() - st_time)))
            TRAIN_INFO['epoch_time'].append(float("{:.4f}".format(time.time() - ep_st_time)))
        print("TEST THE MODEL:::::::::::::::::::")
        test_acc, test_loss = test(epoch)
        if args.step_info == "false":
            TRAIN_INFO['test_loss'].append(float("{:.4f}".format(test_loss)))
            TRAIN_INFO['test_acc'].append(float("{:.4f}".format(test_acc)))
        
        lr_scheduler.step()

    if args.step_info == "true":
        a = TRAIN_INFO['total_time']
        a = np.cumsum(a)
        TRAIN_INFO['total_time'] = a

    # save the train info to file:
    fname = "lr_" + str(args.learning_rate) + "_b_" + str(args.batch_size)
    fname = fname + str(np.random.rand()) 
    path = "./" + args.dataset + "/" + args.network + "/" + args.optimizer
    if not os.path.exists(path):
        try:
            os.makedirs(path)
        except OSError:
            print ("Creation of the directory %s failed" % path)
        else:
            print ("Successfully created the directory %s " % path)
    
    f = open( path + "/" + fname + ".csv", 'w')
    if args.debug_mem == 'true':
      f.write('time(s), train_loss, test_loss, train_acc, test_acc, mem(b), epoch_time(s)\n')     
    else:
      f.write('time(s), train_loss, test_loss, train_acc, test_acc, epoch_time(s)\n')
    for i in range(len(TRAIN_INFO['total_time'])):
        t1 = TRAIN_INFO['total_time'][i]
        t2 = TRAIN_INFO['train_loss'][i]
        t3 = TRAIN_INFO['test_loss'][i]
        t4 = TRAIN_INFO['train_acc'][i]
        t5 = TRAIN_INFO['test_acc'][i]

        line = str(t1) + ", " + str(t2) + ", " + str(t3) + ", " + str(t4) + ", " + str(t5) 
        if args.debug_mem == 'true':
            line = line + ", " + str(TRAIN_INFO['memory'][i])
        if i < len(TRAIN_INFO['epoch_time']):
            line = line + ", " + str(TRAIN_INFO['epoch_time'][i]) + "\n"
        else:
            line = line + "\n"
        f.write(line) 
    f.close()
    return best_acc


def get_accuracy(data):
    net.eval()
    total_loss = 0
    correct = 0
    total = 0

    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(data):
            inputs, targets = inputs.to(args.device), targets.to(args.device)
            outputs = net(inputs)
            loss = criterion(outputs, targets)
            total_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()


    acc = 100.*correct/total
    loss = total_loss / (batch_idx + 1)
    
    ### cleaning memory
    if module_names == 'children':
        all_modules = net.children()
    elif module_names == 'features':
        all_modules = net.features.children()
    for m in all_modules:
        memory_cleanup(m)
    return acc, loss

def memory_cleanup(module):
    """Remove I/O stored by backpack during the forward pass.

    Deletes the attributes created by `hook_store_io` and `hook_store_shapes`.
    """
    # if self.mem_clean_up:
    if hasattr(module, "output"):
        delattr(module, "output")
    if hasattr(module, "output_shape"):
        delattr(module, "output_shape")
    i = 0
    while hasattr(module, "input{}".format(i)):
        delattr(module, "input{}".format(i))
        i += 1
    i = 0
    while hasattr(module, "input{}_shape".format(i)):
        delattr(module, "input{}_shape".format(i))
        i += 1

if __name__ == '__main__':
    main()
