#!/bin/env python
# -*- coding: utf-8 -*-
# @author DDDivano
# encoding=utf-8 vi:ts=4:sw=4:expandtab:ft=python

from framework.composition_operator.compare import Compare
import os
import datetime


class Checker(object):
    def __init__(self, case_name, logger, atol=0, rtol=0, mode="allclose"):
        self.logger = logger
        self.case_name = case_name
        self.atol = atol
        self.rtol = rtol
        self.mode = mode

    def record(self, case_name, info):
        """记录结果"""
        with open('result_{}_{}.txt'.format(os.getpid(), datetime.date.today()), 'a') as f:
            f.write('{}, {}\n'.format(case_name, info))

    def __call__(self, paddle_forward, torch_forward, paddle_backward, torch_backward):
        try:
            self.logger.info("开始前向校验...")
            Compare(paddle_forward, torch_forward, self.atol, self.rtol, self.mode)
            self.logger.info("前向校验成功...ok")
        except Exception as e:
            self.logger.error(e)
            self.logger.info("前向校验失败...fail")
            self.record(self.case_name, "前向校验失败")
        try:
            self.logger.info("开始反向校验...")
            Compare(paddle_backward, torch_backward, self.atol, self.rtol, self.mode)
            self.logger.info("反向校验成功...ok")
        except Exception as e:
            self.logger.error(e)
            self.logger.info("反向校验失败...fail")
            self.record(self.case_name, "反向校验失败")
