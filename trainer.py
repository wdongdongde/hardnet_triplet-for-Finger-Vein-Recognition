# coding=utf-8
import torch
import numpy as np
import copy


def fit(train_loader, val_loader, model, loss_fn, optimizer, scheduler, n_epochs, cuda, log_interval, metrics=[],
        start_epoch=0, log_file_path='log.txt', model_save_path=None, loss_name=None ):
    """
    Loaders, model, loss function and metrics should work together for a given task,
    i.e. The model should be able to process data output of loaders,
    loss function should process target output of loaders and outputs from the model

    Examples: Classification: batch loader, classification model, NLL loss, accuracy metric
    Siamese network: Siamese loader, siamese model, contrastive loss
    Online triplet learning: batch loader, embedding model, online triplet loss
    """

    # 为保存最好的模型做准备
    smallest_val_loss_model_wts = copy.deepcopy(model.state_dict())
    smallest_val_loss = 10
    the_val_epoch = start_epoch

    log_file = open(log_file_path, 'w')
    for epoch in range(0, start_epoch):
        scheduler.step()

    for epoch in range(start_epoch, n_epochs):
        scheduler.step()

        # Train stage
        train_loss, metrics = train_epoch(train_loader, model, loss_fn, optimizer, cuda, log_interval, metrics,log_file)
        # train_loss是这一次训练中，所有batch的损失的均值，而每个batch的损失也是对每个样本而言的（即均值）
        message = 'Epoch: {}/{}. Train set: Average loss: {:.4f}'.format(epoch + 1, n_epochs, train_loss)
        for metric in metrics:
            message += '\t{}: {}'.format(metric.name(), metric.value())

        val_loss, metrics = test_epoch(val_loader, model, loss_fn, cuda, metrics)
        val_loss /= len(val_loader)

        message += '\nEpoch: {}/{}. Validation set: Average loss: {:.4f}'.format(epoch + 1, n_epochs,
                                                                                 val_loss)
        for metric in metrics:
            message += '\t{}: {}'.format(metric.name(), metric.value())

        print(message)
        log_file.write(message)
        log_file.write('\n')

        # 在训练过程中不断更新最好的模型
        if model_save_path != None:
            if val_loss < smallest_val_loss:
                smallest_val_loss = val_loss
                the_val_epoch = epoch
                smallest_val_loss_model_wts = copy.deepcopy(model.state_dict())
    # 保存模型
    if model_save_path != None:
        if loss_name !=None:
            smallest_val_loss_model_wts_path = model_save_path+'{}_smallest_val_loss_{}_{}.pth'.format(the_val_epoch, loss_name, smallest_val_loss)
        else:
            smallest_val_loss_model_wts_path = model_save_path + '{}_smallest_val_loss_{}.pth'.format(the_val_epoch,
                                                                                                      smallest_val_loss)
    torch.save(smallest_val_loss_model_wts, smallest_val_loss_model_wts_path)


def train_epoch(train_loader, model, loss_fn, optimizer, cuda, log_interval, metrics, log_file):
    for metric in metrics:
        metric.reset()

    model.train()
    losses = []
    total_loss = 0  # 每次batch图像的loss的和

    for batch_idx, (data, target) in enumerate(train_loader):
        # data是训练的数据 target是标签  一个batch中的，每个图像是一个张量
        # print "batch_idx:",batch_idx,"target:",target,"\n"
        target = target if len(target) > 0 else None
        if not type(data) in (tuple, list):
            data = (data,)
        if cuda:
            data = tuple(d.cuda() for d in data)
            if target is not None:
                target = target.cuda()  # 用GPU运行，不用variable来包装？
        # 测试每个batch是否真的有32对图像
        # print data[0].size(),"\n"
        # print data[1].size(),"\n"
        optimizer.zero_grad()
        #print "训练数据",data
        outputs = model(*data)    # *是什么意思  outputs为batch_size* class_n

        if type(outputs) not in (tuple, list):
            outputs = (outputs,)

        loss_inputs = outputs    # 输出是预测的类别
        if target is not None:
            target = (target,)
            loss_inputs += target   # 预测标签与真实标签相加,作为分开的两个元素

        loss_outputs = loss_fn(*loss_inputs)  # 将预测标签和真实标签的和传给损失函数，计算输出一个值
        loss = loss_outputs[0] if type(loss_outputs) in (tuple, list) else loss_outputs
        losses.append(loss.item())
        total_loss += loss.item()  # 浮点型
        loss.backward()
        optimizer.step()

        for metric in metrics:
            metric(outputs, target, loss_outputs)  # 输入是预测的类别的向量，实际标签 ，损失值

        if batch_idx % log_interval == 0:
            message = 'Train: [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                batch_idx * len(data[0]), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), np.mean(losses))  # 打印的是那个batch里面的损失的均值
            for metric in metrics:
                message += '\t{}: {}'.format(metric.name(), metric.value())
            print(message)
            # log_file.write(message)
            losses = []   # log_interval内所有的losses的list

    total_loss /= (batch_idx + 1)
    return total_loss, metrics


def test_epoch(val_loader, model, loss_fn, cuda, metrics):
    with torch.no_grad():
        for metric in metrics:
            metric.reset()
        model.eval()
        val_loss = 0
        for batch_idx, (data, target) in enumerate(val_loader):
            target = target if len(target) > 0 else None
            if not type(data) in (tuple, list):
                data = (data,)
            if cuda:
                data = tuple(d.cuda() for d in data)
                if target is not None:
                    target = target.cuda()

            outputs = model(*data)

            if type(outputs) not in (tuple, list):
                outputs = (outputs,)
            loss_inputs = outputs
            if target is not None:
                target = (target,)
                loss_inputs += target

            loss_outputs = loss_fn(*loss_inputs)
            loss = loss_outputs[0] if type(loss_outputs) in (tuple, list) else loss_outputs
            val_loss += loss.item()

            for metric in metrics:
                metric(outputs, target, loss_outputs)

    return val_loss, metrics
