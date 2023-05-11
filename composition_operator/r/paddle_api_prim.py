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
import os

class Paddle_Api_Prim(object):
    def __init__(self, layer, inputs, params, dtype="float64"):
        if dtype == "float64":
            paddle.set_default_dtype(np.float64)
            self.dtype = np.float64
        elif dtype == "float32":
            paddle.set_default_dtype(np.float32)
            self.dtype = np.float32
        else:
            raise TypeError("输入数据类型错误")
        os.environ["FLAGS_prim_all"] = "true"
        self.layer = layer
        self.inputs = inputs
        self.params = params
        self._set_paddle_param()
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
            self.result = paddle.jit.to_static(obj)(**self.inputs)
        else:
            self.result = paddle.jit.to_static(eval(self.layer))(**dict(self.inputs, **self.params))
        return self.result.sum().numpy()[0]

    def run_backward(self):
        if len(list(self.inputs.values())) == 0:
            return 0
        grad = paddle.grad(self.result, list(self.inputs.values()))
        for i in range(len(grad)):
            grad[i] = grad[i].numpy()
        return grad

    def _set_paddle_param(self):
        """
        设置paddle 输入参数
        """
        for key, value in self.inputs.items():
            self.inputs[key] = to_tensor(value.astype(self.dtype))
            self.inputs[key].stop_gradient = False
        for key, value in self.params.items():
            if isinstance(value, (np.generic, np.ndarray)):
                self.params[key] = to_tensor(value.astype(self.dtype))
            else:
                self.params[key] = value

