#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @author: Wesley
# @time: 2020-12-11 10:48

import os
import cv2
from PIL import Image
from torchvision import transforms
from torch.utils.data import Dataset

class MyDataset(Dataset):

    def __init__(self, img_path, mask_path):
        super().__init__()
        self.img_path = img_path
        self.mask_path = mask_path
        self.filename = []
        for name in os.listdir(img_path):
            self.filename.append(name.split('.')[0])
            # self.filename.append(name.split('_')[0])

    def __getitem__(self, index):
        transform = transforms.Compose([
            transforms.ToTensor()
        ])
        image = transform(cv2.imread(os.path.join(self.img_path, self.filename[index] + '.jpg'), 1))
        mask = transform(cv2.imread(os.path.join(self.mask_path, self.filename[index] + '.png'), 0))
        # image = transform(cv2.imread(os.path.join(self.img_path, self.filename[index] + '_sat.jpg'), 1))
        # mask = transform(cv2.imread(os.path.join(self.mask_path, self.filename[index] + '_mask.png'), 0))

        return image, mask

    def __len__(self):
        return len(self.filename)


if __name__ == '__main__':
    # img_path = r'E:\PyCharmProject\datasets\patch\image'
    # mask_path = r'E:\PyCharmProject\datasets\patch\mask'
    img_path = r'E:\PyCharmProject\datasets\5k\train_set\JPEGImages'
    mask_path = r'E:\PyCharmProject\datasets\5k\train_set\SegmentationClass'
    dataset = MyDataset(img_path, mask_path)
    for img, mask in dataset:

        print('img', img.shape)
        print('mask', mask.shape)
