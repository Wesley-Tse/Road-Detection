#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @author: Wesley
# @time: 2020-12-11 10:56

import cv2
import os

src = r'E:\PyCharmProject\datasets\3'
img_path = r'E:\PyCharmProject\datasets\image'
mask_path = r'E:\PyCharmProject\datasets\mask'

if __name__ == '__main__':
    if not os.path.exists(src):
        print('path not exists!')

    files = os.listdir(src)
    count = len(files)

    i = j = 1
    for n, filename in enumerate(files):
        print('{} / {} ------{}'.format(n, count, filename))
        info = filename.split('_')[1]
        img = cv2.imread(os.path.join(src, filename), 1)
        if info == '1_sat.jpg':
            cv2.imwrite(os.path.join(img_path, str(i) + '_' + info), img)
            i += 1
        if info == '1_mask.png':
            cv2.imwrite(os.path.join(mask_path, str(j) + '_' + info), img)
            j += 1
    print('done!')

