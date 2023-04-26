#!/bin/env python
# -*- coding: utf-8 -*-
# @author DDDivano
# encoding=utf-8 vi:ts=4:sw=4:expandtab:ft=python

"""
数据生成器
"""
from copy import deepcopy

class Reader(object):
    def __init__(self, wk):
        self.wk = wk
        if not self.wk.is_model_case():
            self.inputs = self.wk.get_inputs("paddle")
            self.params = self.wk.get_params("paddle")
            self.inputs_torch = deepcopy(self.inputs)
            self.params_torch = deepcopy(self.params)

    def get_paddle_input(self):
        return self.inputs

    def get_torch_input(self):
        mapping = self.wk.get_mapping()
        if mapping is None:
            return self.inputs_torch
        else:
            inputs = dict()
            for k,v in mapping.get("ins").items():
                if self.inputs_torch.get(k) is not None:
                    inputs[v] = self.inputs_torch.get(k)
            return inputs

    def get_paddle_param(self):
        return self.params

    def get_torch_param(self):
        mapping = self.wk.get_mapping()
        if mapping is None:
            return self.params_torch
        else:
            params = dict()
            for k, v in mapping.get("ins").items():
                if self.params_torch.get(k) is not None:
                    params[v] = self.params_torch.get(k)
            return params

    def get_paddle_func(self):
        return self.wk.get_func("paddle")

    def get_torch_func(self):
        return self.wk.get_func("pytorch")

    def get_paddle(self):
        return {
            "layer": self.get_paddle_func(),
            "inputs": self.get_paddle_input(),
            "params": self.get_paddle_param()
        }

    def get_torch(self):
        return {
            "layer": self.get_torch_func(),
            "inputs": self.get_torch_input(),
            "params": self.get_torch_param()
        }

    def get_inputs(self):
        # for models
        # Todo: 后续可能因为多输入要修改
        return self.wk.get_inputs("paddle")["x"]

