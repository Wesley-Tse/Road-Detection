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

    out = net(x)
    out_mask = out.cpu().detach().numpy().reshape(576, 576)
    cv2.imshow('out', np.hstack([mask, out_mask]))
    cv2.waitKey(0)
    cv2.destroyAllWindows()

