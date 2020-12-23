#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @author: Wesley
# @time: 2020-12-23 17:46


import os
from matplotlib import pyplot as plt
import numpy as np

# path = './src/log'
path = r'E:\PyCharmProject\Road-Detection\src\log\U_Net24k.txt'

log = []
with open(path, 'r') as f:
    next(f)
    lines = f.readlines()
    for line in lines:
        log.append(list(map(float, line.split('\t'))))
log = np.stack(log)

epoch = log[..., :1].astype(int)
loss = log[..., 1:2]
pa = log[..., 2:3]
iou = log[..., 3:4]
precision = log[..., 4:]

x = [i for item in epoch for i in item]
y1 = [i for item in log[..., 1:2] for i in item]
y2 = [i for item in log[..., 2:3] for i in item]
y3 = [i for item in log[..., 3:4] for i in item]
y4 = [i for item in log[..., 4:] for i in item]

plt.figure()
plt.plot(x, y1)
plt.plot(x, y2)
plt.plot(x, y3)
plt.plot(x, y4)
plt.title('net')
plt.xlabel('epoch')
plt.ylabel('loss')
plt.plot(x, y1, label='loss')
plt.plot(x, y2, label='pa')
plt.plot(x, y3, label='iou')
plt.plot(x, y4, label='precision')
plt.legend(loc="upper left")
plt.show()
