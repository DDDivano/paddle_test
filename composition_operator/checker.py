#!/bin/env python
# -*- coding: utf-8 -*-
# @author DDDivano
# encoding=utf-8 vi:ts=4:sw=4:expandtab:ft=python

from composition_operator.compare import Compare
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

    def compare_dict(self, dict1, dict2, mode="all"):
        """
        对比dict
        """
        if set(dict1.keys()) != set(dict2.keys()):
            self.logger.error("The keys in dict1 and dict2 are different.")
            print(dict1.keys())
            print(dict2.keys())
            raise ValueError("dict 包含的keys不完全相同")
        for k, v in dict1.items():
            if "@grad" in k and mode == "forward": continue
            if "@grad" not in k and mode == "backward": continue
            try:
                self.logger.info("开始校验参数{}...".format(k))
                Compare(dict1[k], dict2[k], self.atol, self.rtol, self.mode)
                self.logger.info("参数{}校验成功...ok".format(k))
            except Exception as e:
                self.logger.error(e)
                self.logger.info("参数{}校验失败...fail".format(k))
                self.record(self.case_name, "参数{}校验失败".format(k))


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
