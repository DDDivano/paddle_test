#!/bin/env python
# -*- coding: utf-8 -*-
# @author DDDivano
# encoding=utf-8 vi:ts=4:sw=4:expandtab:ft=python

import paddle
import paddle.nn as nn
import numpy as np
import torch
import paddle.nn.functional as F
from composition_operator.utils.generate import rand_array


class ConvPoolNet(nn.Layer):
    def __init__(self):
        super(ConvPoolNet, self).__init__()
        # 定义卷积层
        self.conv1 = nn.Conv2D(in_channels=3, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2D(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1)

        # 定义池化层
        self.pool = nn.MaxPool2D(kernel_size=2, stride=2)

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
        x = paddle.flatten(x, start_axis=1, stop_axis=-1)
        x = self.fc1(x)
        x = nn.functional.relu(x)
        x = self.fc2(x)
        return x


class SimpleNet(nn.Layer):
    def __init__(self):
        super(SimpleNet, self).__init__()

        self.conv1 = nn.Conv2D(in_channels=3, out_channels=16, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2D(num_features=16)
        self.pool1 = nn.MaxPool2D(kernel_size=2, stride=2)

        self.conv2 = nn.Conv2D(in_channels=16, out_channels=32, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2D(num_features=32)
        self.pool2 = nn.MaxPool2D(kernel_size=2, stride=2)

        self.fc = nn.Linear(in_features=196 * 8, out_features=10)

    def forward(self, x):
        x = self.pool1(self.bn1(paddle.nn.functional.relu(self.conv1(x))))
        x = self.pool2(self.bn2(paddle.nn.functional.relu(self.conv2(x))))
        x = paddle.flatten(x, start_axis=1)
        x = self.fc(x)
        return x
