#!/bin/env python
# -*- coding: utf-8 -*-
# @author DDDivano
# encoding=utf-8 vi:ts=4:sw=4:expandtab:ft=python

"""
网络主框架
"""
import numpy as np
from inspect import isclass
from torch import tensor
import torch
import framework.composition_operator.nets.torch_nets_lib as torch_net
import framework.composition_operator.nets.paddle_nets_lib as paddle_net

class Torch_Net(object):
    def __init__(self, layer, inputs, dtype="float64"):
        if dtype == "float64":
            torch.set_default_tensor_type(torch.DoubleTensor)
            self.dtype = np.float64
        elif dtype == "float32":
            torch.set_default_tensor_type(torch.FloatTensor)
            self.dtype = np.float32
        elif dtype == "float16":
            torch.set_default_tensor_type(torch.FloatTensor)
            self.dtype = np.float16
        else:
            raise TypeError("输入数据类型错误")
        self.layer = layer
        self.inputs = inputs
        self._set_param()


    def run_forward(self):
        layer = eval("torch_net." + self.layer + "(np." + self.dtype.__name__ + ")")
        self.result = layer(self.inputs)
        return self.result.sum().detach().numpy()

    def run_backward(self):
        grad = torch.autograd.grad(self.result.sum(), self.inputs)
        grad = list(grad)
        for i in range(len(grad)):
            grad[i] = grad[i].numpy()
        return np.array(grad)

    def _set_param(self):
        """
        设置paddle 输入参数
        """
        self.inputs = tensor(self.inputs.astype(self.dtype), requires_grad=True)
