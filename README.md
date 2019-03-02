# hardnet_triplet
分析siamese,triplet和不同样本选择方式在指静脉识别上的效果，结合hardnet文中的l2网络得到最终一个效果比较好的hardnet_triplet
## 主要文件

1. 可执行文件（实现某种算法）
   - classification_softmax.py 直接使用softmax
   - easy_siamese.py 使用简单的siamese
   - hardming_siamese.py 难分样本挖掘的siamese
   - easy_triplet.py
   - hardest_triplet_hardnet.py 

2. 功能性文件
   - works.py  实验用的一些网络结构，每个网络用一个类来定
   -  trainer.py 训练网络的函数 
   - losses.py 定义了损失函数，siamese和triplet loss，以及进行样本选择的版本
   - datasets.py  定义了数据集类型
   - metrics.py 一些度量模型性能的类
   - utils.py 样本组合和选择的类

测试对branch的更改
