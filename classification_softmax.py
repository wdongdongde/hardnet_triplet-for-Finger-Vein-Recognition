# coding=utf-8
import torch
import torchvision

from torchvision import transforms
import torch
from torch.optim import lr_scheduler
import torch.optim as optim
from trainer import fit
from networks import EmbeddingNet, ClassificationNet,HardNet
from metrics import AccumulatedAccuracyMetric
import numpy as np

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
cuda = torch.cuda.is_available()
# mean, std = 0., 1.
n_classes = 3852

# Set up data loaders
# 要 分析是否需要加预处理
train_dataset = torchvision.datasets.ImageFolder('/home/wdxia/Finger_ROI_Database/GRG_3852_train',
                                                  transform=transforms.Compose([transforms.Resize((32, 32)), transforms.Grayscale(),
                                                      transforms.ToTensor()
                                                  ]))
test_dataset = torchvision.datasets.ImageFolder('/home/wdxia/Finger_ROI_Database/GRG_3852_split_valid',
                                                  transform=transforms.Compose([transforms.Resize((32, 32)), transforms.Grayscale(),
                                                      transforms.ToTensor()
                                                  ]))
batch_size = 32
kwargs = {'num_workers': 1, 'pin_memory': True} if cuda else {}
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, **kwargs)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False, **kwargs)

# Set up the network and training parameters

embedding_net = HardNet()
model = ClassificationNet(embedding_net, n_classes=n_classes)
print model
if cuda:
    model.cuda()
loss_fn = torch.nn.NLLLoss()  # 输入由LOG_SOFTMAX计算出来的属于每一类的概率，输出损失
lr = 1e-2
optimizer = optim.Adam(model.parameters(), lr=lr)

''' optimizer (Optimizer): Wrapped optimizer.
        step_size (int): Period of learning rate decay.
        gamma (float): Multiplicative factor of learning rate decay.
            Default: 0.1.
        last_epoch (int): The index of last epoch. Default: -1.'''
scheduler = lr_scheduler.StepLR(optimizer, 8, gamma=0.1, last_epoch=-1)
n_epochs = 100
log_interval = len(train_loader)//batch_size   # 隔多少个batch记录一次

fit(train_loader, test_loader, model, loss_fn, optimizer, scheduler, n_epochs, cuda, log_interval, metrics=[AccumulatedAccuracyMetric()],log_file_path = 'hardnet_softmax_log.txt')