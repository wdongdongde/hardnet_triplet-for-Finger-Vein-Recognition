# coding=utf-8
import numpy as np
from sklearn.metrics import roc_curve, auc

class Metric:
    def __init__(self):
        pass

    def __call__(self, outputs, target, loss):
        raise NotImplementedError

    def reset(self):
        raise NotImplementedError

    def value(self):
        raise NotImplementedError

    def name(self):
        raise NotImplementedError


class AccumulatedAccuracyMetric(Metric):
    """
    Works with classification model
    """

    def __init__(self):
        self.correct = 0
        self.total = 0

    def __call__(self, outputs, target, loss):
        pred = outputs[0].data.max(1, keepdim=True)[1]  # 预测得到的标签是输出向量中值最大的所在的Index,
        self.correct += pred.eq(target[0].data.view_as(pred)).cpu().sum()   # 标签相等表示预测正确
        self.total += target[0].size(0)
        # print self.correct, self.total
        # return self.value()

    def reset(self):
        self.correct = 0
        self.total = 0

    def value(self):
        return 100 * float(self.correct) / self.total   # 一个batch分类正确的数量/总数量

    def name(self):
        return 'Accuracy'


class AverageNonzeroTripletsMetric(Metric):
    '''
    Counts average number of nonzero triplets found in minibatches
    '''

    def __init__(self):
        self.values = []

    def __call__(self, outputs, target, loss):
        self.values.append(loss[1])
        return self.value()

    def reset(self):
        self.values = []

    def value(self):
        return np.mean(self.values)

    def name(self):
        return 'Average nonzero triplets'


class Sample_distance_metric(Metric):
    # 记录
    def __init__(self):
        self.pos_avg_distance = 0
        self.neg_avg_distance = 0
        self.total = 0
        self.total_pos=0
        self.total_neg=0

    def __call__(self, outputs, target, loss):
        # 计算两个特征的L2距离
        # l2_distance=np.sqrt(np.sum(np.square(outputs[0] - outputs[1])))
        l2_distance = (outputs[0] - outputs[1]).pow(2).sum(1).sqrt() # 张量的做法
        # l2_distance = np.sqrt(np.sum(np.square(feat_1 - feat_2)))  # 一个矩阵这么计算的话得到的什么？希望是一个batch_sizex1矩阵
        batch_size = target[0].size(0)
        self.total += target[0].size(0)

        for i in range(batch_size):
            if target[0][i]==1:
                self.pos_avg_distance += l2_distance[i]
                self.total_pos +=1
            else:
                self.neg_avg_distance += l2_distance[i]
                self.total_neg += 1
        self.pos_avg_distance = float(self.pos_avg_distance)/self.total_pos
        self.neg_avg_distance = float(self.neg_avg_distance)/self.total_neg

    def reset(self):
        self.pos_avg_distance = 0
        self.neg_avg_distance = 0
        self.total = 0
        self.total_pos = 0
        self.total_neg = 0

    def value(self):
        return self.pos_avg_distance / self.neg_avg_distance  # 一个batch分类正确的数量/总数量

    def name(self):
        return 'Pos_neg_distance_ratio'

