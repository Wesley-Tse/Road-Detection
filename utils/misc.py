#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @author: Wesley
# @time: 2020-12-22 14:24

import warnings

__all__ = ['AverageMeter', 'EncodingDeprecationWarning']


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        # self.val = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        # self.val = val
        self.sum += val * n
        self.count += n

    @property
    def avg(self):
        avg = 0 if self.count == 0 else self.sum / self.count
        return avg


class EncodingDeprecationWarning(DeprecationWarning):
    pass


warnings.simplefilter('once', EncodingDeprecationWarning)
