#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @author: Wesley
# @time: 2020-12-11 10:36

import torch
import torch.nn as nn
from torch.nn import functional


class DoubleConv(nn.Module):

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)


class Down(nn.Module):

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)


class Up(nn.Module):

    def __init__(self, in_channels, out_channels):
        super().__init__()
        # self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):
        # x1 = self.up(x1)

        x1 = functional.interpolate(x1, scale_factor=2, mode='bilinear', align_corners=True)

        padY = x2.shape[2] - x1.shape[2]
        padX = x2.shape[3] - x1.shape[3]

        dim = (padX // 2, padX - padX // 2, padY // 2, padY - padY // 2)

        x1 = functional.pad(x1, dim, 'constant', 0)

        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)


class UNet(nn.Module):
    def __init__(self, in_channels, num_cls):
        super(UNet, self).__init__()
        self.in_channels = in_channels
        self.num_cls = num_cls
        # 输入
        self.inconv = DoubleConv(self.in_channels, 64)
        # 下采样
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        self.down4 = Down(512, 1024)
        # 上采样
        self.up1 = Up(1024 + 512, 512)
        self.up2 = Up(512 + 256, 256)
        self.up3 = Up(256 + 128, 128)
        self.up4 = Up(128 + 64, 64)
        # 输出
        self.outconv = OutConv(64, self.num_cls)

    def forward(self, x):
        y1 = self.inconv(x)

        y2 = self.down1(y1)
        y3 = self.down2(y2)
        y4 = self.down3(y3)
        y5 = self.down4(y4)

        y6 = self.up1(y5, y4)
        y7 = self.up2(y6, y3)
        y8 = self.up3(y7, y2)
        y9 = self.up4(y8, y1)

        output = self.outconv(y9)

        return output


if __name__ == '__main__':
    net = UNet(1, 1).cuda()
    img = torch.ones([1, 1, 512, 512]).cuda()
    out = net(img)
    print(out.shape)
