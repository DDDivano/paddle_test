#!/bin/env python
# -*- coding: utf-8 -*-
# @author DDDivano
# encoding=utf-8 vi:ts=4:sw=4:expandtab:ft=python


import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from composition_operator.utils.generate import rand_array


class ConvPoolNet(nn.Module):
    def __init__(self, dtype):
        super(ConvPoolNet, self).__init__()
        # 定义卷积层
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1)

        # 定义池化层
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        # 定义全连接层
        self.fc1 = nn.Linear(in_features=64 * 8 * 8, out_features=512)
        self.fc2 = nn.Linear(in_features=512, out_features=10)

    def forward(self, x):
        x = self.conv1(x)
        x = nn.functional.relu(x)
        x = self.pool(x)
        x = self.conv2(x)
        x = nn.functional.relu(x)
        x = self.pool(x)
        x = torch.flatten(x, start_dim=1, end_dim=-1)
        x = self.fc1(x)
        x = nn.functional.relu(x)
        x = self.fc2(x)
        return x