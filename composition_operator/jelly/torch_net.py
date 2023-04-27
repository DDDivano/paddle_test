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
import composition_operator.nets.torch_nets_lib as torch_net
import composition_operator.nets.paddle_nets_lib as paddle_net
from copy import deepcopy

class Torch_Net(object):
    def __init__(self, layer, inputs, state_dict, dtype="float64"):
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
        self.state_dict = deepcopy(state_dict)


    def run_forward(self):
        self.layer = eval("torch_net." + self.layer + "(np." + self.dtype.__name__ + ")")
        self.layer.load_state_dict(self.state_dict)
        self.result = self.layer(self.inputs)
        return self.result.sum().detach().numpy()

    def run_backward(self):
        loss = torch.mean(self.result)
        loss.backward()
        result = {}
        for name, param in self.layer.named_parameters():
            result[name] = param.detach().numpy()
            result[name + "@grad"] = param.grad.numpy()
        return result

    def _set_param(self):
        """
        设置paddle 输入参数
        """
        self.inputs = tensor(self.inputs.astype(self.dtype), requires_grad=True)
