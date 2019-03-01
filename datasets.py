# coding=utf-8
import numpy as np
from PIL import Image

from torch.utils.data import Dataset
from torch.utils.data.sampler import BatchSampler


# 定义一个torch.utils.data.Dataset的子类是导入自定义数据的方法
class SiameseMNIST(Dataset):
    """
    对于训练集，对每个样本随机找一个组队，形成正负样本
    对于测试集，一半的样本取对形成正样本，一半形成负样本
    Train: For each sample creates randomly a positive or a negative pair
    Test: Creates fixed pairs for testing
    """

    def __init__(self, mnist_dataset):
        self.mnist_dataset = mnist_dataset

        self.train = self.mnist_dataset.train
        self.transform = self.mnist_dataset.transform

        if self.train:
            self.train_labels = self.mnist_dataset.train_labels
            self.train_data = self.mnist_dataset.train_data
            self.labels_set = set(self.train_labels.numpy())  # 标签集合
            self.label_to_indices = {label: np.where(self.train_labels.numpy() == label)[0]
                                     for label in self.labels_set}
        else:
            # generate fixed pairs for testing
            self.test_labels = self.mnist_dataset.test_labels
            self.test_data = self.mnist_dataset.test_data
            self.labels_set = set(self.test_labels.numpy())
            self.label_to_indices = {label: np.where(self.test_labels.numpy() == label)[0]
                                     for label in self.labels_set}  # dict label对应与其相同的所在的index

            random_state = np.random.RandomState(29)  # 产生伪随机数种子
            # 在每一个对应的标签里中找正样本对
            positive_pairs = [[i,
                               random_state.choice(self.label_to_indices[self.test_labels[i].item()]),
                               1]
                              for i in range(0, len(self.test_data), 2)]

            negative_pairs = [[i,
                               random_state.choice(self.label_to_indices[
                                                       np.random.choice(
                                                           list(self.labels_set - set([self.test_labels[i].item()]))
                                                       )
                                                   ]),
                               0]
                              for i in range(1, len(self.test_data), 2)]
            self.test_pairs = positive_pairs + negative_pairs     # 这些pairs只对应着图像的index和标签

    def __getitem__(self, index):   # 实现数据集的下表索引
        if self.train:
            target = np.random.randint(0, 2)
            img1, label1 = self.train_data[index], self.train_labels[index].item()   # train_labels也是dict
            if target == 1:
                siamese_index = index
                while siamese_index == index:
                    siamese_index = np.random.choice(self.label_to_indices[label1])
            else:
                siamese_label = np.random.choice(list(self.labels_set - set([label1])))
                siamese_index = np.random.choice(self.label_to_indices[siamese_label])
            img2 = self.train_data[siamese_index]
        else:
            img1 = self.test_data[self.test_pairs[index][0]]
            img2 = self.test_data[self.test_pairs[index][1]]
            target = self.test_pairs[index][2]

        img1 = Image.fromarray(img1.numpy(), mode='L')
        img2 = Image.fromarray(img2.numpy(), mode='L')
        if self.transform is not None:
            img1 = self.transform(img1)
            img2 = self.transform(img2)
        return (img1, img2), target  # 返回一对图像和对应标签

    def __len__(self):
        return len(self.mnist_dataset)


class TripletMNIST(Dataset):
    """
    Train: For each sample (anchor) randomly chooses a positive and negative samples
    Test: Creates fixed triplets for testing
    """

    def __init__(self, mnist_dataset):
        self.mnist_dataset = mnist_dataset
        self.train = self.mnist_dataset.train
        self.transform = self.mnist_dataset.transform

        if self.train:
            self.train_labels = self.mnist_dataset.train_labels
            self.train_data = self.mnist_dataset.train_data
            self.labels_set = set(self.train_labels.numpy())
            self.label_to_indices = {label: np.where(self.train_labels.numpy() == label)[0]
                                     for label in self.labels_set}

        else:
            self.test_labels = self.mnist_dataset.test_labels
            self.test_data = self.mnist_dataset.test_data
            # generate fixed triplets for testing
            self.labels_set = set(self.test_labels.numpy())
            self.label_to_indices = {label: np.where(self.test_labels.numpy() == label)[0]
                                     for label in self.labels_set}

            random_state = np.random.RandomState(29)

            triplets = [[i,
                         random_state.choice(self.label_to_indices[self.test_labels[i].item()]),
                         random_state.choice(self.label_to_indices[
                                                 np.random.choice(
                                                     list(self.labels_set - set([self.test_labels[i].item()]))
                                                 )
                                             ])
                         ]
                        for i in range(len(self.test_data))]
            self.test_triplets = triplets

    def __getitem__(self, index):
        if self.train:
            img1, label1 = self.train_data[index], self.train_labels[index].item()
            positive_index = index
            while positive_index == index:
                positive_index = np.random.choice(self.label_to_indices[label1])
            negative_label = np.random.choice(list(self.labels_set - set([label1])))
            negative_index = np.random.choice(self.label_to_indices[negative_label])
            img2 = self.train_data[positive_index]
            img3 = self.train_data[negative_index]
        else:
            img1 = self.test_data[self.test_triplets[index][0]]
            img2 = self.test_data[self.test_triplets[index][1]]
            img3 = self.test_data[self.test_triplets[index][2]]

        img1 = Image.fromarray(img1.numpy(), mode='L')
        img2 = Image.fromarray(img2.numpy(), mode='L')
        img3 = Image.fromarray(img3.numpy(), mode='L')
        if self.transform is not None:
            img1 = self.transform(img1)
            img2 = self.transform(img2)
            img3 = self.transform(img3)
        return (img1, img2, img3), []

    def __len__(self):
        return len(self.mnist_dataset)


class BalancedBatchSampler(BatchSampler):
    """
    BatchSampler - from a MNIST-like dataset, samples n_classes and within these classes samples n_samples.
    Returns batches of size n_classes * n_samples
    """

    def __init__(self, dataset, n_classes, n_samples):
        if dataset.train:
            self.labels = dataset.train_labels
        else:
            self.labels = dataset.test_labels
        self.labels_set = list(set(self.labels.numpy()))
        self.label_to_indices = {label: np.where(self.labels.numpy() == label)[0]
                                 for label in self.labels_set}
        for l in self.labels_set:
            np.random.shuffle(self.label_to_indices[l])
        self.used_label_indices_count = {label: 0 for label in self.labels_set}  # 用来标志哪些label已经用过
        self.count = 0
        self.n_classes = n_classes
        self.n_samples = n_samples
        self.dataset = dataset
        self.batch_size = self.n_samples * self.n_classes

    def __iter__(self):
        self.count = 0
        while self.count + self.batch_size < len(self.dataset):
            classes = np.random.choice(self.labels_set, self.n_classes, replace=False)  # 在所有类别的集合中选出n_classes个，岂不是全选了？
            indices = []
            for class_ in classes:
                indices.extend(self.label_to_indices[class_][
                               self.used_label_indices_count[class_]:self.used_label_indices_count[
                                                                         class_] + self.n_samples])
                self.used_label_indices_count[class_] += self.n_samples
                if self.used_label_indices_count[class_] + self.n_samples > len(self.label_to_indices[class_]):
                    np.random.shuffle(self.label_to_indices[class_])
                    self.used_label_indices_count[class_] = 0
            yield indices
            self.count += self.n_classes * self.n_samples

    def __len__(self):
        return len(self.dataset) // self.batch_size

# ----------------------------------------------------------------------------------------------------------------- #
# 将自己的数据转成siamese格式，需要自己的数据是一个类，具有train_labels等成员
# 挑选准则：对每一个样本，根据随机生成的数，来决定选择是和相同手指还是不同手指组队，而标签就是看二者标签相不相等
import random
import linecache
import torch
from PIL import Image


class MyDatasetSiamese(Dataset):
    def __init__(self, txt, transform=None, target_transform=None, trainornot=True,count=0):
        # 从txt文件中读取得到图像的路径矩阵，作为 train_data 和test_data
        # 读取标签
        # 标签索引
        self.transform = transform
        self.target_transform = target_transform
        self.txt = txt  # 之前生成的train.txt
        self.train = trainornot
        self.count = count

        if self.train:
            # self.train_labels = dict()
            self.train_labels = []
            self.train_data = []
            for i in range(self.__len__()):
                line_num = i + 1
                line = linecache.getline(self.txt, line_num)
                line.strip('\n')
                train_img = line.split()[0]
                train_label = line.split()[1]
                # self.train_labels[i]=train_label  # 字典，index与标签
                self.train_labels.append(train_label)
                self.train_data.append(train_img)  # list
            self.labels_set = set(self.train_labels)
            self.label_to_indices = {label: np.where(np.array(self.train_labels) == label) for label in self.labels_set}
        else:
            self.test_labels = []
            self.test_data = []
            for i in range(self.__len__()):
                line_num = i + 1
                line = linecache.getline(self.txt, line_num)
                line.strip('\n')
                test_img = line.split()[0]
                test_label = line.split()[1]
                # self.train_labels[i]=train_label  # 字典，index与标签
                self.test_labels.append(test_label)
                self.test_data.append(test_img)  # list
            self.labels_set = set(self.test_labels)
            self.label_to_indices = {label: np.where(np.array(self.test_labels) == label) for label in self.labels_set}

            random_state = np.random.RandomState(29)  # 产生伪随机数种子
            # 在每一个对应的标签里中找正样本对
            positive_pairs = [[i,
                               random_state.choice(np.squeeze(self.label_to_indices[self.test_labels[i]])),
                               1]
                              for i in range(0, len(self.test_data), 2)]
            # tt =set(np.squeeze([self.test_labels[1]]))
            negative_pairs = [[i,
                               random_state.choice(np.squeeze(self.label_to_indices[
                                                       np.random.choice(
                                                           list(self.labels_set - set([self.test_labels[i]]))
                                                       )
                                                   ])),
                               0]
                              for i in range(1, len(self.test_data), 2)]
            self.test_pairs = positive_pairs + negative_pairs  # 这些pairs只对应着图像的index和标签

    def __getitem__(self, index):  # 实现数据集的下表索引
        self.count +=1
        print self.count,"\n"
        pair_combine_method = open('pairs.txt','a')
        if self.train:
            target = np.random.randint(0, 2)
            img1, label1 = self.train_data[index], self.train_labels[index]
            if target == 1:
                siamese_index = index
                while siamese_index == index:
                    # print "find the right one..."
                    # print self.label_to_indices[label1]
                    siamese_index = np.random.choice(np.squeeze(self.label_to_indices[label1]))  # 这里没陷入死循环？？
            else:

                siamese_label = np.random.choice(list(self.labels_set - set([label1])))
                # test = self.label_to_indices[siamese_label]
                siamese_index = np.random.choice(np.squeeze(self.label_to_indices[siamese_label]))
            img2 = self.train_data[siamese_index]
            print "训练样本对：",img1," ",img2,"标签：",target
            pair_combine_method.write("训练样本对："+img1+" "+img2+"标签："+str(target)+"\n")
        else:
            img1 = self.test_data[self.test_pairs[index][0]]
            img2 = self.test_data[self.test_pairs[index][1]]
            target = self.test_pairs[index][2]
            print "测试样本对：", img1, " ", img2, "标签：", target
            pair_combine_method.write("测试样本对："+img1+ " "+img2+"标签："+str(target)+"\n")
        # 前面处理的都是图像的路径
        img1 = Image.open(img1)
        img1 = img1.convert("L")
        img2 = Image.open(img2)
        img2 = img2.convert("L")
        if self.transform is not None:
            img1 = self.transform(img1)
            img2 = self.transform(img2)
        return (img1, img2), target  # 返回一对图像和对应标签

    def __len__(self):  # 数据总长
        fh = open(self.txt, 'r')
        num = len(fh.readlines())
        fh.close()
        return num


class MyDatasetTriplet(Dataset):
    def __init__(self, txt, transform=None, target_transform=None, trainornot=True):
        # 从txt文件中读取得到图像的路径矩阵，作为 train_data 和test_data
        # 读取标签
        # 标签索引
        self.transform = transform
        self.target_transform = target_transform
        self.txt = txt  # 之前生成的train.txt
        self.train = trainornot

        if self.train:
            # self.train_labels = dict()
            self.train_labels = []
            self.train_data = []
            for i in range(self.__len__()):
                line_num = i + 1
                line = linecache.getline(self.txt, line_num)
                line.strip('\n')
                train_img = line.split()[0]
                train_label = line.split()[1]
                # self.train_labels[i]=train_label  # 字典，index与标签
                self.train_labels.append(train_label)
                self.train_data.append(train_img)  # list
            self.labels_set = set(self.train_labels)
            self.label_to_indices = {label: np.where(np.array(self.train_labels) == label) for label in self.labels_set}
        else:
            self.test_labels = []
            self.test_data = []
            for i in range(self.__len__()):
                line_num = i + 1
                line = linecache.getline(self.txt, line_num)
                line.strip('\n')
                test_img = line.split()[0]
                test_label = line.split()[1]
                # self.train_labels[i]=train_label  # 字典，index与标签
                self.test_labels.append(test_label)
                self.test_data.append(test_img)  # list
            self.labels_set = set(self.test_labels)
            self.label_to_indices = {label: np.where(np.array(self.test_labels) == label) for label in self.labels_set}

            random_state = np.random.RandomState(29)  # 产生伪随机数种子
            triplets = [[i,
                         random_state.choice(np.squeeze(self.label_to_indices[self.test_labels[i]])),
                         random_state.choice(np.squeeze(self.label_to_indices[
                                                 np.random.choice(
                                                     np.squeeze(list(self.labels_set - set([self.test_labels[i]])))
                                                 )
                                             ]))
                         ]
                        for i in range(len(self.test_data))]
            self.test_triplets = triplets

    def __getitem__(self, index):  # 实现数据集的下表索引
         if self.train:
             img1, label1 = self.train_data[index], self.train_labels[index]
             positive_index = index
             while positive_index == index:
                 positive_index = np.random.choice(np.squeeze(self.label_to_indices[label1]))
             negative_label = np.random.choice(np.squeeze(list(self.labels_set - set([label1]))))
             negative_index = np.random.choice(np.squeeze((self.label_to_indices[negative_label])))
             img2 = self.train_data[positive_index]
             img3 = self.train_data[negative_index]
         else:
             img1 = self.test_data[self.test_triplets[index][0]]
             img2 = self.test_data[self.test_triplets[index][1]]
             img3 = self.test_data[self.test_triplets[index][2]]
         img1 = Image.open(img1)
         img1 = img1.convert("L")
         img2 = Image.open(img2)
         img2 = img2.convert("L")
         img3 = Image.open(img3)
         img3 = img3.convert("L")
         if self.transform is not None:
             img1 = self.transform(img1)
             img2 = self.transform(img2)
             img3 = self.transform(img3)
         return (img1, img2,img3), []  # 返回一对图像和对应标签

    def __len__(self):
        fh = open(self.txt, 'r')
        num = len(fh.readlines())
        fh.close()
        return num


class myBalancedBatchSampler(BatchSampler):
    """
    BatchSampler - from a MNIST-like dataset, samples n_classes and within these classes samples n_samples.
    Returns batches of size n_classes * n_samples
    """

    # n_samples是什么 在每一个min_batch中每一类有多少张图片
    # dataset是Folder读来的对象
    def __init__(self, dataset, n_classes, n_samples):

        self.labels = [img[1] for img in dataset.samples]

        self.labels_set = list(set(self.labels))
        self.label_to_indices = {label: np.where(np.array(self.labels) == label)[0] for label in self.labels_set}
        for l in self.labels_set:
            np.random.shuffle(self.label_to_indices[l])
        self.used_label_indices_count = {label: 0 for label in self.labels_set}
        self.count = 0
        self.n_classes = n_classes
        self.n_samples = n_samples
        self.dataset = dataset
        self.batch_size = self.n_samples * self.n_classes
        # print "myBalancedBatchSampler __init__ finished!"

    def __iter__(self):
        self.count = 0
        while self.count + self.batch_size < len(self.dataset):
            classes = np.random.choice(self.labels_set, self.n_classes, replace=False)  # 选出来的类别
            indices = []  # 存放选出来的那几个类对应的其他放在这个batch中的几张图
            for class_ in classes:
                # tt = np.squeeze(self.label_to_indices[class_])
                # #self.label_to_indices[class_] = np.array(np.squeeze(self.label_to_indices[class_]))  # 去掉不要的维度
                # self.label_to_indices[class_] = np.array(self.label_to_indices[class_])  # 去掉不要的维度
                # test_ = self.label_to_indices[class_]  # 与该类形相同的其他图像的索引 array(2)这种形式的一个不能算矩阵
                # # lene = len(test_)
                # test_1 = self.used_label_indices_count[class_]  # 该类已经取出过的类别的数量
                # 只有一个相同的情况，去掉维度后是一个数，数不能用下标
                # if len(self.label_to_indices[class_]) == 1:
                #     indices.extend(self.label_to_indices[class_])
                #     self.used_label_indices_count[class_] += 1  # 选完了的数量
                # else:
                indices.extend(self.label_to_indices[class_][
                           self.used_label_indices_count[class_]:self.used_label_indices_count[
                                                                     class_] + self.n_samples])  # 从当前选到的类的其他图片中选出n_samples张
                self.used_label_indices_count[class_] += self.n_samples  # 选完了的数量
                # 如果只有一个，就不用管了
                if self.used_label_indices_count[class_] + self.n_samples > len(self.label_to_indices[class_]): #剩下的不足要选取的样本个数的情况
                    np.random.shuffle(self.label_to_indices[class_])  # 为什么要打乱，
                    self.used_label_indices_count[class_] = 0   # 这有什么作用？！下次再抽到这个类的时候可以选到其他值
            # print len(indices)
            yield indices  # 下次调用 __iter__时接着上次的来
            self.count += self.n_classes * self.n_samples
            # print "myBalancedBatchSampler __iter__ finished!"

    def __len__(self):
        return len(self.dataset) // self.batch_size   # batch的数量
        # print "myBalancedBatchSampler __len__ finished!", len(self.dataset) // self.batch_size


class SiameseDataset(Dataset):
    """
    对于训练集，对每个样本随机找一个组队，形成正负样本
    对于测试集，一半的样本取对形成正样本，一半形成负样本
    Train: For each sample creates randomly a positive or a negative pair
    Test: Creates fixed pairs for testing
    """

    def __init__(self, dataset, transform, trainornot=False,count = 0):
        self.dataset = dataset
        self.train = trainornot
        self.transform = transform
        self.count = count

        if self.train:
            #self.train_labels = self.dataset.train_labels
            self.train_labels = [img[1] for img in dataset.samples]
            self.train_data = [img[0] for img in dataset.samples]
            self.labels_set = set(self.train_labels)
            self.label_to_indices = {label: np.where(np.array(self.train_labels) == label)[0] for label in self.labels_set}
        else:
            # generate fixed pairs for testing
            self.test_labels = [img[1] for img in dataset.samples]
            self.test_data = [img[0] for img in dataset.samples]
            self.labels_set = set(self.test_labels)
            self.label_to_indices = {label: np.where(np.array(self.test_labels) == label)[0] for label in self.labels_set}

            random_state = np.random.RandomState(29)  # 产生伪随机数种子
            # 在每一个对应的标签里中找正样本对
            positive_pairs = [[i,
                               random_state.choice(self.label_to_indices[self.test_labels[i]]),
                               1]
                              for i in range(0, len(self.test_data), 2)]
            # tt =set(np.squeeze([self.test_labels[1]]))
            negative_pairs = [[i,
                               random_state.choice(self.label_to_indices[
                                                       np.random.choice(
                                                           list(self.labels_set - set([self.test_labels[i]]))
                                                       )
                                                   ]),
                               0]
                              for i in range(1, len(self.test_data), 2)]
            self.test_pairs = positive_pairs + negative_pairs  # 这些pairs只对应着图像的index和标签

    def __getitem__(self, index):  # 实现数据集的下表索引
        self.count+=1
        print self.count,"\n"
        pair_combine_method1 = open('random_pairs.txt', 'a')
        if self.train:
            target = np.random.randint(0, 2)
            print "create target:",target,"\n"
            img1, label1 = self.train_data[index], self.train_labels[index]  # 当前要组队的图像和标签
            print "the sample is:",img1,label1,"\n"
            if target == 1:
                siamese_index = index
                while siamese_index == index:
                    print "choose the right one..."
                    if len(self.label_to_indices[label1])==1:  # 如果本来图像就只有一张,就直接自己组成对
                        break
                    siamese_index = np.random.choice(self.label_to_indices[label1])  # 在相同标签中找不是本图像的图像，若只有本图像一张，则永远找不到，死循环
            else:
                print "choose the error one..."
                siamese_label = np.random.choice(list(self.labels_set - set([label1])))  # 取不同标签里面的
                # test = self.label_to_indices[siamese_label]
                siamese_index = np.random.choice(self.label_to_indices[siamese_label])
            img2 = self.train_data[siamese_index]
            print ("训练样本对：" + img1 + " " + img2 + "标签：" + str(target) + "\n")
            pair_combine_method1.write("训练样本对："+str(self.count)+ " "+ img1 + " " + img2 + "标签：" + str(target) + "\n")
        else:
            img1 = self.test_data[self.test_pairs[index][0]]
            img2 = self.test_data[self.test_pairs[index][1]]
            target = self.test_pairs[index][2]
            print ("测试样本对：" + img1 + " " + img2 + "标签：" + str(target) + "\n")
            pair_combine_method1.write("测试样本对：" + str(self.count) + " "+img1 + " " + img2 + "标签：" + str(target) + "\n")
        # 前面处理的都是图像的路径
        img1 = Image.open(img1)
        img1 = img1.convert("L")
        img2 = Image.open(img2)
        img2 = img2.convert("L")
        if self.transform is not None:
            img1 = self.transform(img1)
            img2 = self.transform(img2)
        return (img1, img2), target  # 返回一对图像和对应标签

    def __len__(self):  # 数据总长
        num = len(self.dataset)
        return num


class TripletDataset(Dataset):
    """
    对于训练集，对每个样本随机找一个组队，形成正负样本
    对于测试集，一半的样本取对形成正样本，一半形成负样本
    Train: For each sample creates randomly a positive or a negative pair
    Test: Creates fixed pairs for testing
    """

    def __init__(self, dataset, transform, trainornot=False, count=0):
        self.dataset = dataset
        self.train = trainornot
        self.transform = transform
        self.count = count

        if self.train:
            # self.train_labels = self.dataset.train_labels
            self.train_labels = [img[1] for img in dataset.samples]
            self.train_data = [img[0] for img in dataset.samples]
            self.labels_set = set(self.train_labels)
            self.label_to_indices = {label: np.where(np.array(self.train_labels) == label)[0] for label in
                                     self.labels_set}
        else:
            # generate fixed pairs for testing
            self.test_labels = [img[1] for img in dataset.samples]
            self.test_data = [img[0] for img in dataset.samples]
            self.labels_set = set(self.test_labels)
            self.label_to_indices = {label: np.where(np.array(self.test_labels) == label)[0] for label in
                                     self.labels_set}

            random_state = np.random.RandomState(29)  # 产生伪随机数种子

            triplets = [[i,
                         random_state.choice(self.label_to_indices[self.test_labels[i]]),
                         random_state.choice(self.label_to_indices[
                                                 np.random.choice(
                                                     list(self.labels_set - set([self.test_labels[i]]))
                                                 )
                                             ])
                         ]
                        for i in range(len(self.test_data))]
            self.test_triplets = triplets

    def __getitem__(self, index):  # 实现数据集的下表索引
        self.count += 1
        print self.count, "\n"
        if self.train:
            img1, label1 = self.train_data[index], self.train_labels[index]
            positive_index = index
            while positive_index == index:
                if len(self.label_to_indices[label1]) == 1:  # 如果本来图像就只有一张,就直接自己组成对
                    break
                positive_index = np.random.choice(self.label_to_indices[label1])

            negative_label = np.random.choice(list(self.labels_set - set([label1])))
            negative_index = np.random.choice(self.label_to_indices[negative_label])
            img2 = self.train_data[positive_index]
            img3 = self.train_data[negative_index]

        else:
            img1 = self.test_data[self.test_triplets[index][0]]
            img2 = self.test_data[self.test_triplets[index][1]]
            img3 = self.test_data[self.test_triplets[index][2]]
    # 前面处理的都是图像的路径
        img1 = Image.open(img1)
        img1 = img1.convert("L")
        img2 = Image.open(img2)
        img2 = img2.convert("L")
        img3 = Image.open(img3)
        img3 = img3.convert("L")
        if self.transform is not None:
            img1 = self.transform(img1)
            img2 = self.transform(img2)
            img3 = self.transform(img3)
        return (img1, img2, img3), []

    def __len__(self):  # 数据总长
        num = len(self.dataset)
        return num