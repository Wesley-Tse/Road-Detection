#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @author: Wesley
# @time: 2020-12-11 10:47

import os
import cv2
import torch
from models.unet import UNet
import numpy as np

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
net = UNet(1, 1).to(device)

weight = r'E:\PyCharmProject\Road-Detection\weights\weight.pt'
if os.path.exists(weight):
    net.load_state_dict(torch.load(weight))
img_path = 'data/1_sat.jpg'
mask_path = 'data/1_mask.png'

if __name__ == '__main__':

    img0 = cv2.resize(cv2.imread(img_path, 1), (576, 576))
    cv2.imshow('origin', img0)
    img = cv2.resize(cv2.imread(img_path, 0), (576, 576))
    mask = cv2.resize(cv2.imread(mask_path, 0), (576, 576))
    x = torch.tensor(img).float()
    x = x.view(-1, 1, 576, 576).to(device)

    # net.eval()
    out = net(x)
    print(out.shape)
    print(torch.maximum(out))
    exit()
    pred = torch.sigmoid(out).cpu().detach().numpy().reshape(576, 576)
    print(pred)
    pred[pred >= 0.5] = 1
    pred[pred < 0.5] = 0
    print(pred)
    TP = ((pred == 1) & (mask == 1)).sum()
    TN = ((pred == 0) & (mask == 0)).sum()
    FN = ((pred == 0) & (mask == 1)).sum()
    FP = ((pred == 1) & (mask == 0)).sum()
    acc = (TP + TN) / (TP + TN + FP + FN)
    print(TP)
    print(FN)
    print(FP)
    print(TN)
    print(acc)

    out_mask = out.cpu().detach().numpy().reshape(576, 576)
    cv2.imshow('out', pred)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

