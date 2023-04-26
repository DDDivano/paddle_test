#!/bin/env python
# -*- coding: utf-8 -*-
# @author DDDivano
# encoding=utf-8 vi:ts=4:sw=4:expandtab:ft=python

import paddle
import paddle.nn as nn
import numpy as np
import torch
import paddle.nn.functional as F
from framework.composition_operator.utils.generate import rand_array


class ConvPoolNet(nn.Layer):
    def __init__(self, dtype):
        super(ConvPoolNet, self).__init__()
        np.random.seed(33)
        self.pool = nn.MaxPool2D(kernel_size=2, stride=2)
        self.weight_np = rand_array(size=[32, 3, 3, 3], dtype=dtype)
        self.bias_np = rand_array(size=[32], dtype=dtype)
        self.weight = rand_array(size=[7200, 10], dtype=dtype)
        self.bias = rand_array(size=[1], dtype=dtype)

    def forward(self, x):
        x = F.conv2d(x, weight=paddle.to_tensor(self.weight_np), bias=paddle.to_tensor(self.bias_np))
        x = nn.functional.relu(x)
        x = self.pool(x)
        x = paddle.flatten(x, start_axis=1, stop_axis=-1)
        x = F.linear(x, weight=paddle.to_tensor(self.weight), bias=paddle.to_tensor(self.bias))
        return x