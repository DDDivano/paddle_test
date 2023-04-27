#!/bin/env python
# -*- coding: utf-8 -*-
# @author DDDivano
# encoding=utf-8 vi:ts=4:sw=4:expandtab:ft=python

"""
网络主框架
"""
import numpy as np
from inspect import isclass
from paddle import to_tensor
import paddle
import composition_operator.nets.torch_nets_lib as torch_net
import composition_operator.nets.paddle_nets_lib as paddle_net
from copy import deepcopy

class Paddle_Net(object):
    def __init__(self, layer, inputs, dtype="float64"):
        np.random.seed(33)
        paddle.seed(33)
        if dtype == "float64":
            paddle.set_default_dtype(np.float64)
            self.dtype = np.float64
        elif dtype == "float32":
            paddle.set_default_dtype(np.float32)
            self.dtype = np.float32
        elif dtype == "float16":
            paddle.set_default_dtype(np.float16)
            self.dtype = np.float16
        else:
            raise TypeError("输入数据类型错误")
        self.layer = layer
        self.inputs = inputs
        self._set_paddle_param()
        self.state_dict = deepcopy(self._state_dict())


    def _state_dict(self):
        layer = eval("paddle_net." + self.layer + "(np." + self.dtype.__name__ + ")")
        return layer.state_dict()

    def get_state_dict(self):
        return self.state_dict

    def run_forward(self):
        self.layer = eval("paddle_net." + self.layer + "(np." + self.dtype.__name__ + ")")
        self.layer.load_dict(self.state_dict)
        self.result = self.layer(self.inputs)
        return self.result.sum().numpy()[0]

    def run_backward(self):
        loss = paddle.mean(self.result)
        loss.backward()
        result = {}
        for name, param in self.layer.named_parameters():
            result[name] = param.numpy()
            result[name+"@grad"] = param.grad.numpy()
        return result

    def _set_paddle_param(self):
        """
        设置paddle 输入参数
        """
        self.inputs = to_tensor(self.inputs.astype(self.dtype))
        self.inputs.stop_gradient = False