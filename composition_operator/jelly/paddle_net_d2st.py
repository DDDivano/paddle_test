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
import framework.composition_operator.nets.torch_nets_lib as torch_net
import framework.composition_operator.nets.paddle_nets_lib as paddle_net

class Paddle_Net_D2ST(object):
    def __init__(self, layer, inputs, dtype="float64"):
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


    def run_forward(self):
        layer = paddle.jit.to_static(eval("paddle_net." + self.layer + "(np." + self.dtype.__name__ + ")"))
        self.result = layer(self.inputs)
        return self.result.sum().numpy()[0]

    def run_backward(self):
        grad = paddle.grad(self.result, self.inputs)
        for i in range(len(grad)):
            grad[i] = grad[i].numpy()
        return np.array(grad)

    def _set_paddle_param(self):
        """
        设置paddle 输入参数
        """
        self.inputs = to_tensor(self.inputs.astype(self.dtype))
        self.inputs.stop_gradient = False