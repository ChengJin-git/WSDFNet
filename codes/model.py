# ------------------------------------------------------------------
# Copyright (c) 2021, Zi-Rong Jin, Tian-Jing Zhang, Cheng Jin, and 
# Liang-Jian Deng, All rights reserved.
#
# This work is licensed under GNU Affero General Public License
# v3.0 International To view a copy of this license, see the
# LICENSE file.
#
# This file is running on WorldView-3 dataset. For other dataset
# (i.e., QuickBird and GaoFen-2), please change the corresponding
# inputs.
# ------------------------------------------------------------------

import torch
from torch import nn


class Conv_Block(nn.Module):
    def __init__(self, in_planes):
        super(Conv_Block, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, in_planes, 3, 1, 1)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(in_planes, in_planes, 3, 1, 1)

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu(x)
        x = self.conv2(x)
        return x


class ASW(nn.Module):
    def __init__(self, in_planes, b_num):
        super(ASW, self).__init__()
        self.b_num = b_num
        self.asw = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_planes, b_num, 1, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(b_num, b_num, 1, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(b_num, b_num, 1, 1),
            nn.Softmax(dim=1)
        )

    def forward(self, x):
        w = self.asw(x)  # b,b_num,1,1
        w = torch.chunk(w, self.b_num, 1)  # [b_num](b,1,1,1)
        y = []
        for wi in w:
            y.append(x*wi.expand_as(x))
        return y


class WSDFNet_Maintrunk(nn.Module):
    def __init__(self):
        super(WSDFNet_Maintrunk, self).__init__()
        self.cb1 = Conv_Block(32)
        self.cb2 = Conv_Block(32)
        self.cb3 = Conv_Block(32)
        self.cb4 = Conv_Block(32)

        self.asw1 = ASW(32, 4)
        self.asw2 = ASW(32, 3)
        self.asw3 = ASW(32, 2)

    def forward(self, x):
        w1 = self.asw1(x)  # 4,b,c,h,w
        x1 = self.cb1(x)+w1[0]

        w2 = self.asw2(x1)  # 3,b,c,h,w
        x2 = self.cb2(x1)+w1[1]+w2[0]

        w3 = self.asw3(x2)  # 2,b,c,h,w
        x3 = self.cb2(x2)+w1[2]+w2[1]+w3[0]

        x4 = self.cb4(x3)+w1[3]+w2[2]+w3[1]+x3

        return x4


class WSDFNet(nn.Module):
    def __init__(self):
        super(WSDFNet, self).__init__()
        self.head_conv = nn.Sequential(
            nn.Conv2d(9, 32, 3, 1, 1),
            nn.ReLU(inplace=True)
        )
        self.mainbody = WSDFNet_Maintrunk()
        self.tail_conv = nn.Conv2d(32, 8, 3, 1, 1)

    def forward(self, pan, lms):
        x = torch.cat([pan, lms], 1)
        x = self.head_conv(x)
        x = self.mainbody(x)
        x = self.tail_conv(x)
        sr = x + lms

        return sr


if __name__ == '__main__':
    from torchsummary import summary
    summary(WSDFNet(), [(1, 64, 64), (8, 64, 64)], device='cpu')
