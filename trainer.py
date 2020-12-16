#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @author: Wesley
# @time: 2020-12-11 10:47

import os
import time
import torch
from torch import nn
from models.dinknet34 import DinkNet34
from loss import dice_bce_loss
from models.unet import UNet
from dataset import MyDataset
from torch.utils.data import DataLoader

img_path = r'E:\PyCharmProject\datasets\5k\train_set\JPEGImages'
mask_path = r'E:\PyCharmProject\datasets\5k\train_set\SegmentationClass'
val_img_path = r'E:\PyCharmProject\datasets\5k\validate_set\JPEGImages'
val_mask_path = r'E:\PyCharmProject\datasets\5k\validate_set\SegmentationClass'
log = './dinknet.txt'

class Trainer:
    def __init__(self, batch_size_per_card, epoch_limit, weight):
        super().__init__()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.batch_size_per_card = batch_size_per_card
        batch_size = batch_size_per_card * torch.cuda.device_count()
        epoch_limit = epoch_limit
        net = DinkNet34().to(self.device)
        self.net = torch.nn.DataParallel(net, device_ids=range(torch.cuda.device_count()))

        self.weight = weight
        if os.path.exists(weight):
            net.load_state_dict(torch.load(weight))

train_dataset = MyDataset(img_path, mask_path)
val_dataset = MyDataset(val_img_path, val_mask_path)
train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_dataloader = DataLoader(val_dataset, batch_size=batch_size)

adam = torch.optim.Adam(net.parameters(), lr=2e-4)
sgd = torch.optim.SGD(net.parameters(), lr=0.01, momentum=0.9)

loss_fun = dice_bce_loss()

if __name__ == '__main__':

    epoch = 1
    log = open(log, 'w', encoding='utf-8')
    log.write('epoch' + '\t' + 'loss' + '\t' + 'pa' + '\t' + 'iou' + '\t' + 'precision' + '\n')
    log.flush()
    while epoch < 300:
        s_time = time.time()
        print('epoch - {} - training'.format(epoch))
        net.train()
        TP = FP = TN = FN = 0
        pa = 0
        iou = 0
        stop = 0
        flag = 0
        train_loss = 0
        batch = len(train_dataloader)
        for i, (img, mask) in enumerate(train_dataloader):
            img = img.to(device)
            mask = mask.to(device)
            out = net(img)
            loss = loss_fun(mask, out)

            adam.zero_grad()
            loss.backward()
            adam.step()

            if i % 10 == 0:
                print('{}: {}/{} - loss: {}'.format(epoch, i, batch, loss.item()))
                # torch.save(net.state_dict(), weight)
                # print('save success')
            train_loss += loss.item()
        epoch_loss = train_loss / len(train_dataloader)

        e_time = time.time()
        print('epoch - {} - epoch_loss: {}'.format(epoch, epoch_loss))
        print('total-time: ', e_time - s_time)
        print('epoch - {} - evaluating'.format(epoch))

        net.eval()
        for img, mask in val_dataloader:
            img = img.to(device)
            mask = mask.to(device)
            with torch.no_grad():
                pred = net(img)
            pred[pred >= 0.5] = 1
            pred[pred < 0.5] = 0

            TP += ((pred == 1) & (mask == 1)).cpu().sum().item()
            TN += ((pred == 0) & (mask == 0)).cpu().sum().item()
            FN += ((pred == 0) & (mask == 1)).cpu().sum().item()
            FP += ((pred == 1) & (mask == 0)).cpu().sum().item()

        pa = (TP + TN) / (TP + TN + FP + FN)
        precision = TP / (TP + FN)
        iou = TP / (TP + FP + FN)

        print('pa: ', pa)
        print('iou: ', iou)
        print('precision', precision)
        log.write(
            str(epoch) + '\t' + str(epoch_loss) + '\t' + str(pa) + '\t' + str(iou) + '\t' + str(precision) + '\n')
        log.flush()

        if iou > stop:
            stop = iou
            torch.save(net.state_dict(), weight)
            print("save success，iou updated to: {}".format(iou))
            flag = 0
        else:
            flag += 1
            print("pa为{},没有提升，参数未更新,iou为{},第{}次未更新".format(iou, stop, flag))
            if flag >= epoch_limit:
                print("early stop at epoch {}, finally iou: {}".format(epoch, stop))
                break
        epoch += 1
    log.close()
