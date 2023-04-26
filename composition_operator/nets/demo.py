#!/bin/env python
# -*- coding: utf-8 -*-
# @author DDDivano
# encoding=utf-8 vi:ts=4:sw=4:expandtab:ft=python



import paddle
import paddle.nn as nn
import numpy as np

np.random.seed(33)
paddle.seed(33)
paddle.set_default_dtype(np.float64)

class ConvPoolNet(nn.Layer):
    def __init__(self, num_classes):
        super(ConvPoolNet, self).__init__()

        # 定义卷积层
        self.conv1 = nn.Conv2D(in_channels=3, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2D(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1)

        # 定义池化层
        self.pool = nn.MaxPool2D(kernel_size=2, stride=2)

        # 定义全连接层
        self.fc1 = nn.Linear(in_features=64 * 8 * 8, out_features=512)
        self.fc2 = nn.Linear(in_features=512, out_features=num_classes)

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


if __name__ == '__main__':
    input_data = np.random.rand(1, 3, 32, 32).astype('float64')
    tensor = paddle.to_tensor(input_data)
    layer = ConvPoolNet(num_classes=10)
    layer = paddle.jit.to_static(layer)
    res = layer(tensor)
    print(res)