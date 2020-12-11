#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @author: Wesley
# @time: 2020-12-11 10:48

import os
import cv2
from torchvision import transforms
from torch.utils.data import Dataset

data_dir = r'E:\PyCharmProject\datasets\4'
label_dir = './datasets/label'


class MyDataset(Dataset):

    def __init__(self):
        super().__init__()

        self.filename = []
        for name in os.listdir(data_dir):
            self.filename.append(name)

    def __getitem__(self, index):
        transform = transforms.Compose([
            transforms.ToTensor()
        ])
        image = transform(cv2.imread(os.path.join(data_dir, self.filename[index]), 0))
        mask = transform(cv2.imread(os.path.join(label_dir, self.filename[index]), 0))

        return image, mask

    def __len__(self):
        return len(self.filename)


if __name__ == '__main__':
    dataset = MyDataset()
    for img, mask in dataset:

        print('img', img.shape)
        print('mask', mask.shape)
