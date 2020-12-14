#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @author: Wesley
# @time: 2020-12-11 10:47

import os
import torch
from torch import nn
from models.
from dataset import MyDataset
from torch.utils.data import DataLoader

img_path = r'E:\PyCharmProject\datasets\patch\image'
mask_path = r'E:\PyCharmProject\datasets\patch\mask'

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
batch_size = 16

net = UNet(1, 1).to(device)
weight = r'E:\PyCharmProject\Road-Detection\weights\weight.pt'
if os.path.exists(weight):
    net.load_state_dict(torch.load(weight))

dataset = MyDataset(img_path, mask_path)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

adam = torch.optim.Adam(net.parameters())
sgd = torch.optim.SGD(net.parameters(), lr=0.01, momentum=0.9)
loss_fun = nn.BCEWithLogitsLoss()

if __name__ == '__main__':
    epoch = 0
    while True:
        print('Epoch - ', epoch)
        net.train()

        for i, (img, mask) in enumerate(dataloader):
            img = img.to(device)
            mask = mask.to(device)

            output = net(img)

            loss = loss_fun(output, mask)

            adam.zero_grad()
            loss.backward()
            adam.step()

            if i % 10 == 0:

                print('Loss: ', loss.item())
                torch.save(net.state_dict(), weight)

        print('save success')
        epoch += 1
