# coding=utf-8
import torchvision
from datasets import myBalancedBatchSampler
from torchvision import transforms
import torch
from torch.optim import lr_scheduler
import torch.optim as optim
from trainer import fit
from metrics import AverageNonzeroTripletsMetric

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
cuda = torch.cuda.is_available()

# #########################################################  导入数据  ####################################################

train_dataset = torchvision.datasets.ImageFolder('/home/wdxia/Finger_ROI_Database/GRG_3853_50greater_oversample',
                                                  transform = transforms.Compose([
                                                      transforms.Grayscale(),
                                                      transforms.Resize((64, 64)),
                                                      transforms.RandomCrop(size=(60, 60)),
                                                      transforms.Resize((32, 32)),
                                                      transforms.ToTensor()
                                                  ]))  # 数据变换以提高多样性，顺序不能变

test_dataset = torchvision.datasets.ImageFolder('/home/wdxia/Finger_ROI_Database/GRG_3852_split_valid',
                                                  transform = transforms.Compose([
                                                      transforms.Resize((32, 32)),
                                                      transforms.Grayscale(),
                                                      transforms.ToTensor()
                                                  ]))

# 创建minibatch的数据选择方式，通过每个batch中样本类别数目（n_classes）和每个类别样本数量（n_samples）决定
train_batch_sampler = myBalancedBatchSampler(train_dataset, n_classes=16, n_samples=4)
test_batch_sampler = myBalancedBatchSampler(test_dataset, n_classes=16, n_samples=4)

kwargs = {'num_workers': 1, 'pin_memory': True} if cuda else {}
online_train_loader = torch.utils.data.DataLoader(train_dataset, batch_sampler=train_batch_sampler, **kwargs)
online_test_loader = torch.utils.data.DataLoader(test_dataset, batch_sampler=test_batch_sampler, **kwargs)

# 使用的网络和损失函数，以及数据选择的类
from networks import HardNet
from losses import OnlineTripletLoss
from utils import AllTripletSelector,HardestNegativeTripletSelector, RandomNegativeTripletSelector  # 多种样本选择方式

# 参数配置
margin = 1.
embedding_net = HardNet()
model = embedding_net
if cuda:
    model.cuda()
loss_fn = OnlineTripletLoss(margin, HardestNegativeTripletSelector(margin))
lr = 1e-3
optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
scheduler = lr_scheduler.StepLR(optimizer, 8, gamma=0.1, last_epoch=-1)
n_epochs = 100
log_interval = 10

# 训练
fit(online_train_loader, online_test_loader, model, loss_fn, optimizer, scheduler, n_epochs, cuda, log_interval,
    metrics=[AverageNonzeroTripletsMetric()],
    log_file_path='onlinetriplet_hardnet_log.txt')