#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @author: Wesley
# @time: 2020-12-11 10:47

import os
import cv2
import torch
from models.unet import UNet
from torchvision import transforms
import numpy as np

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
net = UNet(1, 1).to(device)

weight = r'E:\PyCharmProject\Road-Detection\weights\weight.pt'
if os.path.exists(weight):
    net.load_state_dict(torch.load(weight))
img_path = 'data/1_sat.jpg'
mask_path = 'data/1_mask.png'

if __name__ == '__main__':

    origin = cv2.imread(img_path, 1)
    cv2.imshow('origin', origin)
    tr = transforms.Compose([transforms.ToTensor()])
    img = tr(origin).unsqueeze(0).to(device)
    mask = tr(cv2.imread(mask_path, 0))

    net.eval()
    with torch.no_grad():
        pred = net(img)

    pred[pred >= 0.5] = 1
    pred[pred < 0.5] = 0

    TP = ((pred == 1) & (mask == 1)).sum()
    TN = ((pred == 0) & (mask == 0)).sum()
    FN = ((pred == 0) & (mask == 1)).sum()
    FP = ((pred == 1) & (mask == 0)).sum()
    pa = (TP + TN) / (TP + TN + FP + FN)
    iou = TP / (TP + FP + FN)
    print('pa: ', pa)
    print('iou', iou)

    cv2.imshow('origin_out', np.hstack([img, pred]))
    cv2.waitKey(0)
    cv2.destroyAllWindows()

