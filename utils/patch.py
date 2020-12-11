#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @author: Wesley
# @time: 2020-12-11 12:04

import os
import cv2
import random
import numpy as np

img_path = r'E:\PyCharmProject\datasets\image'
mask_path = r'E:\PyCharmProject\datasets\mask'
img_save = r'E:\PyCharmProject\datasets\patch\image'
mask_save = r'E:\PyCharmProject\datasets\patch\mask'
patch_size = 128
patch_num = 10

if __name__ == '__main__':
    if not os.path.exists(img_path):
        print('{} does not exist'.format(img_path))
    if not os.path.exists(img_path):
        print('{} does not exist'.format(mask_path))

    files = os.listdir(img_path)
    num = len(files)
    counter = 1
    for i, file in enumerate(files):
        print('{} / {}'.format(i, num))
        img = cv2.imread(os.path.join(img_path, file), 0)
        mask = cv2.imread(os.path.join(mask_path, file.split('_')[0] + '_mask.png'), 0)
        h, w = img.shape
        for j in range(1, patch_num + 1):
            print('{} / {}'.format(j, patch_num))
            x = random.randint(patch_size, w - patch_size)
            y = random.randint(patch_size, h - patch_size)
            s = patch_size // 2

            x1 = x - s
            y1 = y - s
            x2 = x + s
            y2 = y + s

            crop_mask = mask[x1:x2, y1:y2]
            rate = crop_mask[np.where(crop_mask == 255)].size / patch_size ** 2
            if rate > 0.02:
                crop_img = img[x1:x2, y1:y2]
                cv2.imwrite(os.path.join(img_save, str(counter) + '.jpg'), crop_img)
                cv2.imwrite(os.path.join(mask_save, str(counter) + '.png'), crop_mask)
                counter += 1
    print('All done!')
