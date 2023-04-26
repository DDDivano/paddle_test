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

class Torch_Api(object):
    def __init__(self, layer, inputs, params, dtype="float64"):
        if dtype == "float64":
            torch.set_default_tensor_type(torch.DoubleTensor)
            self.dtype=np.float64
        elif dtype == "float32":
            torch.set_default_tensor_type(torch.FloatTensor)
            self.dtype = np.float32
        else:
            raise TypeError("输入数据类型错误")
        self.layer = layer
        self.inputs = inputs
        self.params = params
        self._set_param()
        self.types = {0: "func", 1: "class", 2: "reload"}

    def _layertypes(self, func):
        """
        define layertypes
        """
        if isclass(func):
            return self.types[1]
        else:
            return self.types[0]

    def run_forward(self):
        if self._layertypes(eval(self.layer)) == self.types[1]:
            obj = eval(self.layer)(**self.params)
            self.result = obj(**self.inputs)
        else:
            self.result = eval(self.layer)(**dict(self.inputs, **self.params))
        return self.result.sum().detach().numpy()

    def run_backward(self):
        grad = torch.autograd.grad(self.result.sum(), list(self.inputs.values()))
        grad = list(grad)
        for i in range(len(grad)):
            grad[i] = grad[i].numpy()
        return np.array(grad)

    def _set_param(self):
        """
        设置paddle 输入参数
        """
        for key, value in self.inputs.items():
            self.inputs[key] = tensor(value.astype(self.dtype), requires_grad=True)
        for key, value in self.params.items():
            if isinstance(value, (np.generic, np.ndarray)):
                self.params[key] = tensor(value.astype(self.dtype))
            else:
                self.params[key] = value

