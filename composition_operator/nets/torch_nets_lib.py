#!/bin/env python
# -*- coding: utf-8 -*-
# @author DDDivano
# encoding=utf-8 vi:ts=4:sw=4:expandtab:ft=python


import torch
import torch.nn as tnn
import torch.nn.functional as F
import numpy as np
from framework.composition_operator.utils.generate import rand_array


class ConvPoolNet(tnn.Module):
    def __init__(self, dtype):
        super(ConvPoolNet, self).__init__()
        # 定义池化层
        self.pool = tnn.MaxPool2d(kernel_size=2, stride=2)
        np.random.seed(33)
        self.weight_np = rand_array(size=[32, 3, 3, 3], dtype=dtype)
        self.bias_np = rand_array(size=[32], dtype=dtype)
        self.weight = rand_array(size=[7200, 10], dtype=dtype)
        self.bias = rand_array(size=[1], dtype=dtype)

    def forward(self, x):
        x = F.conv2d(x, weight=torch.tensor(self.weight_np), bias=torch.tensor(self.bias_np))
        x = tnn.functional.relu(x)
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        x = F.linear(x, weight=torch.tensor(self.weight.transpose((1, 0))), bias=torch.tensor(self.bias))
        return x