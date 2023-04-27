#!/bin/env python
# -*- coding: utf-8 -*-
# @author DDDivano
# encoding=utf-8 vi:ts=4:sw=4:expandtab:ft=python

import numpy as np
import paddle
import torch

def rand_array(size, dtype, framework="torch"):
    value = np.random.random(size).astype(dtype)
    # return value
    if framework == "paddle":
        return paddle.to_tensor(value, stop_gradient=False)
    elif framework == "torch":
        return torch.tensor(value, requires_grad=True)