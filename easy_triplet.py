# coding=utf-8
import torchvision
from datasets import MyDatasetTriplet,TripletDataset
from torchvision import transforms
import torch
from torch.optim import lr_scheduler
import torch.optim as optim
from trainer import fit
from metrics import AccumulatedAccuracyMetric
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
cuda = torch.cuda.is_available()
LOAD_FROM_TXT = True
LOAD_FROM_FOLDER = False

if LOAD_FROM_TXT:
    # 载入数据
    triplet_train_dataset = MyDatasetTriplet(txt='/home/wdxia/Finger_ROI_Database/GRG_3852_train.txt',  transform=transforms.Compose(
        [transforms.Resize((32, 32)), transforms.ToTensor()]), trainornot=True)

    triplet_test_dataset = MyDatasetTriplet(txt='/home/wdxia/Finger_ROI_Database/GRG_3852_split_valid.txt',  transform=transforms.Compose(
        [transforms.Resize((32,32)), transforms.ToTensor()]), trainornot=False)

if LOAD_FROM_FOLDER:
    # /home/wdxia/Finger_ROI_Database/GRG_3852_train
    train_dataset = torchvision.datasets.ImageFolder('/home/wdxia/Finger_ROI_Database/GRG_3852_train')
    test_dataset = torchvision.datasets.ImageFolder('/home/wdxia/Finger_ROI_Database/GRG_3852_split_valid')
    siamese_train_dataset = TripletDataset(train_dataset, transform=transforms.Compose([transforms.Resize((32, 32)),
                                                                                        transforms.ToTensor(),

                                                                                        ]), trainornot=True,
                                           count=0)  # Returns pairs of images and target same/different
    siamese_test_dataset = TripletDataset(test_dataset, transform=transforms.Compose([transforms.Resize((32, 32)),
                                                                                      transforms.ToTensor(),

                                                                                      ]), trainornot=False, count=0)
# if LOAD_FROM_FOLDER:
batch_size = 32
kwargs = {'num_workers': 1, 'pin_memory': True} if cuda else {}
triplet_train_loader = torch.utils.data.DataLoader(triplet_train_dataset, batch_size=batch_size, shuffle=True, **kwargs)
triplet_test_loader = torch.utils.data.DataLoader(triplet_test_dataset, batch_size=batch_size, shuffle=False, **kwargs)

# Set up the network and training parameters
from networks import EmbeddingNet, TripletNet,HardNet
from losses import TripletLoss

margin = 1.
embedding_net = HardNet()
model = TripletNet(embedding_net)
if cuda:
    model.cuda()
loss_fn = TripletLoss(margin)
lr = 1e-3
optimizer = optim.Adam(model.parameters(), lr=lr)
scheduler = lr_scheduler.StepLR(optimizer, 8, gamma=0.1, last_epoch=-1)
n_epochs = 100
log_interval = 10

fit(triplet_train_loader, triplet_test_loader, model, loss_fn, optimizer, scheduler, n_epochs, cuda, log_interval,log_file_path = 'triplet_hardnet_log.txt')

# torch.save(model.state_dict(), 'triplet_params.pkl')
