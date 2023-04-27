#!/bin/env python
# -*- coding: utf-8 -*-
# @author DDDivano
# encoding=utf-8 vi:ts=4:sw=4:expandtab:ft=python



import paddle
import paddle.nn as nn
import numpy as np
import torch
import paddle.nn.functional as F

np.random.seed(33)
paddle.seed(33)
paddle.set_default_dtype(np.float64)
torch.set_default_tensor_type(torch.DoubleTensor)

weight_np = np.random.random(size=[32, 3, 3, 3])
bias_np = np.random.random(size=[32])
weight=np.random.random(size=[7200, 10])
bias=np.random.random(size=[1])
class ConvPoolNet(nn.Layer):
    def __init__(self):
        super(ConvPoolNet, self).__init__()

        self.pool = nn.MaxPool2D(kernel_size=2, stride=2)
    def forward(self, x):
        x = F.conv2d(x, weight=paddle.to_tensor(weight_np), bias=paddle.to_tensor(bias_np))
        x = nn.functional.relu(x)
        x = self.pool(x)
        x = paddle.flatten(x, start_axis=1, stop_axis=-1)
        x = F.linear(x, weight=paddle.to_tensor(weight), bias=paddle.to_tensor(bias))
        return x


import torch.nn as tnn
import torch.nn.functional as TF
class ConvPoolNet_Torch(tnn.Module):
    def __init__(self):
        super(ConvPoolNet_Torch, self).__init__()
        # 定义池化层
        self.pool = tnn.MaxPool2d(kernel_size=2, stride=2)
        self.weight = tnn.Parameter(torch.tensor(weight_np, requires_grad=True))
        self.bias1 = tnn.Parameter(torch.tensor(bias_np, requires_grad=True))

    def forward(self, x):
        x = TF.conv2d(x, weight=self.weight, bias=self.bias1)
        x = tnn.functional.relu(x)
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        return x



if __name__ == '__main__':
    input_data = np.random.rand(1, 3, 32, 32).astype('float64')
    tensor = paddle.to_tensor(input_data)
    tensor.stop_gradient = False
    layer = ConvPoolNet()
    res = layer(tensor)
    print(res.numpy().sum())
    grad1 = paddle.grad(res, tensor)

    for i in range(len(grad1)):
        grad1[i] = grad1[i].numpy()


    tensor_t = torch.tensor(input_data, requires_grad=True)
    layer = ConvPoolNet_Torch()
    res = layer(tensor_t)
    print(res.detach().numpy().sum())
    grad2 = torch.autograd.grad(res.sum(), tensor_t)
    grad2 = list(grad2)
    for i in range(len(grad2)):
        grad2[i] = grad2[i].numpy()


    for name, param in layer.named_parameters():
        print(name, param.grad)

    from framework.composition_operator.compare import Compare
    Compare(np.array(grad1), np.array(grad2))

