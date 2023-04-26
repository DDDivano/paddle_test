#!/bin/env python
# -*- coding: utf-8 -*-
# @author DDDivano
# encoding=utf-8 vi:ts=4:sw=4:expandtab:ft=python

import numpy as np

def rand_array(size, dtype):
    return np.random.random(size).astype(dtype)