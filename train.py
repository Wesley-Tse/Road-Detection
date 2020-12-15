#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @author: Wesley
# @time: 2020-12-11 10:47

import os
import cv2
import torch
from math import ceil
from torch import nn
from models.dinknet34 import DinkNet34
from loss import dice_bce_loss
from models.unet import UNet
from dataset import MyDataset
from torch.utils.data import DataLoader

# img_path = r'E:\PyCharmProject\datasets\patch\image'
# mask_path = r'E:\PyCharmProject\datasets\patch\mask'
img_path = r'E:\PyCharmProject\datasets\5k\train_set\JPEGImages'
mask_path = r'E:\PyCharmProject\datasets\5k\train_set\SegmentationClass'

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
batch_size = 1
input_size = (128, 128)
epoch_limit = 10
net = DinkNet34().to(device)
# net = DinkNet34().to(device)
weight = r'E:\PyCharmProject\Road-Detection\weights\weight1.pt'
if os.path.exists(weight):
    net.load_state_dict(torch.load(weight))

dataset = MyDataset(img_path, mask_path)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

adam = torch.optim.Adam(net.parameters())
sgd = torch.optim.SGD(net.parameters(), lr=0.01, momentum=0.9)
# loss_fun = nn.BCEWithLogitsLoss()
loss_fun = dice_bce_loss()

if __name__ == '__main__':

    epoch = 1

    while epoch < 10:
        print('Epoch - ', epoch)
        net.train()
        TP = FP = TN = FN = 0
        PA = 0
        IOU = 0

        loss_plot = []
        for i, (img, mask) in enumerate(dataloader):
            img = img.to(device)
            mask = mask.to(device)
            out = net(img)
            loss = loss_fun(mask, out)
            adam.zero_grad()
            loss.backward()
            adam.step()

            if i % 10 == 0:
                print('Loss: ', loss.item())
                torch.save(net.state_dict(), weight)
        print('save success')
        epoch += 1

        # net.eval()
        # with torch.no_grad():
        #     for i, (img, mask) in enumerate(dataloader):
        #         img = img.to(device)
        #         mask = mask.to(device)
        #         pred = net(img)
        #
        #         pred[pred >= 0.5] = 1
        #         pred[pred < 0.5] = 0
        #         TP += ((pred == 1) & (mask == 1)).cpu().sum()
        #         TN += ((pred == 0) & (mask == 0)).cpu().sum()
        #         FN += ((pred == 0) & (mask == 1)).cpu().sum()
        #         FP += ((pred == 1) & (mask == 0)).cpu().sum()
        #         PA += (TP + TN) / (TP + TN + FP + FN)
        #         IOU += TP / (TP + FP + FN)
        #         # print(TP)
        #         # print(FN)
        #         # print(FP)
        #         # print(TN)
        #         # print(acc)
        #     MPA = PA / ceil(len(dataloader) // batch_size)
        #     MIOU = IOU / ceil(len(dataloader) // batch_size)
        #     if MIOU > acc_end:
        #         torch.save(net.state_dict(), weight)
        #         acc_end = MIOU
        #         print("save success，iou更新为{}".format(IOU))
        #         round = 0
        #     else:
        #         round += 1
        #         print("精确度为{},没有提升，参数未更新,iou仍为{},第{}次未更新".format(MIOU, acc_end, round))
        #         if round >= epoch_limit:
        #             print("最终iou为{}".format(acc_end))
        #             break