#!/bin/env python
# -*- coding: utf-8 -*-
# @author DDDivano
# encoding=utf-8 vi:ts=4:sw=4:expandtab:ft=python
"""
主执行逻辑
"""
from wk.logger import Logger
from wk.weaktrans import WeakTrans
from wk.yaml_loader import YamlLoader
from reader import Reader
import numpy as np
import paddle
import torch
from jelly.paddle_net import Paddle_Net
from jelly.paddle_net_d2st import Paddle_Net_D2ST
from jelly.torch_net import Torch_Net
from checker import Checker

class CO_NET(object):
    """
    CO
    """
    def __init__(self, yaml_url, case_name, seed=33, atol=0.0, rtol=0.0, mode="allclose"):
        try:
            self.yaml_loader = YamlLoader(yaml_url)
            self.logger = Logger("co", "channel").get_log()
            self.case_name = case_name
            self.seed = seed
            self.checker = Checker(self.case_name, self.logger, atol, rtol, mode)
        except FileNotFoundError as e:
            self.logger.error(e)
            exit(8)


    def case(self):
        c = self.yaml_loader.get_case_info(self.case_name)
        wk = WeakTrans(case=c, logger=self.logger, seed=self.seed)
        return wk

    def co_net(self, dtype="float64"):
        """
        主执行逻辑
        """
        self.logger.info("【网络测试】 Paddle动态图 vs Torch动态图")
        # 获取case配置
        wk = self.case()
        reader = Reader(wk)
        inputs = reader.get_inputs()

        paddle_net = Paddle_Net(wk.get_nets(), inputs, dtype=dtype)
        paddle_forward = paddle_net.run_forward()
        paddle_backward = paddle_net.run_backward()
        # print(paddle_forward)
        # print(paddle_backward)
        torch_net = Torch_Net(wk.get_nets(), inputs, dtype=dtype)
        torch_forward = torch_net.run_forward()
        torch_backward = torch_net.run_backward()
        # print(torch_forward)
        # print(torch_backward)
        # 对比
        self.checker(paddle_forward, torch_forward, paddle_backward, torch_backward)

    def co_net_d2st(self, dtype="float64"):
        """
        主执行逻辑
        """
        self.logger.info("【网络测试】 Paddle动态图 vs Torch动态图")
        # 获取case配置
        wk = self.case()
        reader = Reader(wk)
        inputs = reader.get_inputs()

        paddle_net = Paddle_Net_D2ST(wk.get_nets(), inputs, dtype=dtype)
        paddle_forward = paddle_net.run_forward()
        paddle_backward = paddle_net.run_backward()
        # print(paddle_forward)
        # print(paddle_backward)
        torch_net = Torch_Net(wk.get_nets(), inputs, dtype=dtype)
        torch_forward = torch_net.run_forward()
        torch_backward = torch_net.run_backward()
        # print(torch_forward)
        # print(torch_backward)
        # 对比
        self.checker(paddle_forward, torch_forward, paddle_backward, torch_backward)

if __name__ == '__main__':
    # for i in range(100):
    co = CO_NET("yaml/nets.yaml", "conv_pool", atol=0, rtol=0)
    co.co_net_d2st(dtype="float64")
