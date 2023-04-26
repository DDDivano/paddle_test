#!/bin/env python
# -*- coding: utf-8 -*-
# @author DDDivano
# encoding=utf-8 vi:ts=4:sw=4:expandtab:ft=python
import paddle
import numpy as np


# 设置全局默认的数据类型
np.set_printoptions(precision=6, floatmode='maxprec', suppress=True)

# 创建一个新的数组
x = np.array([1.0, 2.0, 3.0])

# 查看数组的数据类型
print(x.dtype)