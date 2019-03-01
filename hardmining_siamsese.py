# coding=utf-8
import torchvision
from datasets import myBalancedBatchSampler
from torchvision import transforms
import torch
from torch.optim import lr_scheduler
import torch.optim as optim
from trainer import fit

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
cuda = torch.cuda.is_available()

import numpy as np
n_classes = 3852

# target_transform是fit的时候再做的？


# Set up data loaders
# 得到的数据对象中的classes是数据原始标签，即每个文件夹的名字；classes_to_idx 前面的标签与数字类别的对应关系；imgs则是包含图像路径和对应数字标签的列表
train_dataset = torchvision.datasets.ImageFolder('/home/wdxia/Finger_ROI_Database/GRG_3853_50greater_oversample',
                                                  transform=transforms.Compose([transforms.Resize((32, 32)), transforms.Grayscale(),
                                                      transforms.ToTensor()
                                                  ])
                                            )
test_dataset = torchvision.datasets.ImageFolder('/home/wdxia/Finger_ROI_Database/GRG_3852_split_valid',
                                                  transform=transforms.Compose([transforms.Resize((32, 32)), transforms.Grayscale(),
                                                      transforms.ToTensor()
                                                  ])
                                            )

# 初步理解为每个batch
# We'll create mini batches by sampling labels that will be present in the mini batch and number of examples from each class
train_batch_sampler = myBalancedBatchSampler(train_dataset, n_classes=10, n_samples=4)

test_batch_sampler = myBalancedBatchSampler(test_dataset, n_classes=10, n_samples=4)

kwargs = {'num_workers': 1, 'pin_memory': True} if cuda else {}
online_train_loader = torch.utils.data.DataLoader(train_dataset, batch_sampler=train_batch_sampler, **kwargs)
online_test_loader = torch.utils.data.DataLoader(test_dataset, batch_sampler=test_batch_sampler, **kwargs)

# Set up the network and training parameters
from networks import EmbeddingNet,HardNet
from losses import OnlineContrastiveLoss
from utils import AllPositivePairSelector, HardNegativePairSelector # Strategies for selecting pairs within a minibatch

margin = 1.
embedding_net = HardNet()
model = embedding_net
if cuda:
    model.cuda()
loss_fn = OnlineContrastiveLoss(margin, HardNegativePairSelector())
lr = 1e-3
optimizer = optim.Adam(model.parameters(), lr=lr)
scheduler = lr_scheduler.StepLR(optimizer, 8, gamma=0.1, last_epoch=-1)
n_epochs = 100
log_interval = 10

fit(online_train_loader, online_test_loader, model, loss_fn, optimizer, scheduler, n_epochs, cuda, log_interval)