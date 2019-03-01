# coding=utf-8
from datasets import MyDatasetSiamese, SiameseDataset
from torchvision import transforms
import torch
from torch.optim import lr_scheduler
import torch.optim as optim
from trainer import fit
from metrics import Sample_distance_metric
import os
import torchvision

os.environ["CUDA_VISIBLE_DEVICES"] = "1"
cuda = torch.cuda.is_available()
LOAD_FROM_TXT = False
LOAD_FROM_FOLDER = True
# mean, std = 0, 1
if LOAD_FROM_TXT:
    # 载入数据
    siamese_train_dataset = MyDatasetSiamese(txt='/home/wdxia/Finger_ROI_Database/GRG_3852_train.txt',
                                             transform=transforms.Compose(
                                                 [transforms.Resize((32, 32)), transforms.ToTensor()]), trainornot=True,
                                             count=0)

    siamese_test_dataset = MyDatasetSiamese(txt='/home/wdxia/Finger_ROI_Database/GRG_3852_split_valid.txt',
                                            transform=transforms.Compose(
                                                [transforms.Resize((32, 32)), transforms.ToTensor()]), trainornot=False,
                                            count=0)

if LOAD_FROM_FOLDER:
    # /home/wdxia/Finger_ROI_Database/GRG_3852_train
    train_dataset = torchvision.datasets.ImageFolder('/home/wdxia/Finger_ROI_Database/GRG_3852_train')
    test_dataset = torchvision.datasets.ImageFolder('/home/wdxia/Finger_ROI_Database/GRG_3852_split_valid')
    siamese_train_dataset = SiameseDataset(train_dataset, transform=transforms.Compose([transforms.Resize((32, 32)),
                                                                                        transforms.ToTensor(),

                                                                                        ]), trainornot=True,
                                           count=0)  # Returns pairs of images and target same/different
    siamese_test_dataset = SiameseDataset(test_dataset, transform=transforms.Compose([transforms.Resize((32, 32)),
                                                                                      transforms.ToTensor(),

                                                                                      ]), trainornot=False, count=0)

batch_size = 32
kwargs = {'num_workers': 1, 'pin_memory': True} if cuda else {}
siamese_train_loader = torch.utils.data.DataLoader(siamese_train_dataset, batch_size=batch_size, shuffle=True, **kwargs)
siamese_test_loader = torch.utils.data.DataLoader(siamese_test_dataset, batch_size=batch_size, shuffle=False, **kwargs)

# Set up the network and training parameters
from networks import EmbeddingNet, SiameseNet, HardNet
from losses import ContrastiveLoss

margin = 1.
embedding_net = HardNet()
model = SiameseNet(embedding_net)
if cuda:
    model.cuda()
loss_fn = ContrastiveLoss(margin)
lr = 1e-3
optimizer = optim.Adam(model.parameters(), lr=lr)
scheduler = lr_scheduler.StepLR(optimizer, 8, gamma=0.1, last_epoch=-1)
n_epochs = 100
log_interval = 1

# 训练
fit(siamese_train_loader, siamese_test_loader, model, loss_fn, optimizer, scheduler, n_epochs, cuda, log_interval,
    metrics=[Sample_distance_metric()], log_file_path='siamese_hardnet_log.txt')
