#!/bin/env python
# -*- coding: utf-8 -*-
# @author DDDivano
# encoding=utf-8 vi:ts=4:sw=4:expandtab:ft=python
"""
主执行逻辑
"""
import os
import datetime
from wk.logger import Logger
from wk.weaktrans import WeakTrans
from wk.yaml_loader import YamlLoader
from reader import Reader
from framework.composition_operator.r.paddle_api import Paddle_Api
from framework.composition_operator.r.torch_api import Torch_Api
from framework.composition_operator.r.paddle_api_d2st import Paddle_Api_D2ST
from framework.composition_operator.r.paddle_api_prim import Paddle_Api_Prim
from checker import Checker

class CO_API(object):
    """
    CO
    """
    def __init__(self, yaml_url, case_name, seed=33):
        try:
            self.yaml_loader = YamlLoader(yaml_url)
            self.logger = Logger("co", "channel").get_log()
            self.case_name = case_name
            self.seed = seed
            self.checker = Checker(self.case_name, self.logger)
        except FileNotFoundError as e:
            self.logger.error(e)
            exit(8)


    def case(self):
        c = self.yaml_loader.get_case_info(self.case_name)
        wk = WeakTrans(case=c, logger=self.logger, seed=self.seed)
        return wk


    def co_api(self, dtype="float64"):
        """
        主执行逻辑
        """
        self.logger.info("【常规API测试】 Paddle动态图 vs Torch动态图")
        # 获取case配置
        wk = self.case()

        # 生成对应的input和 param
        reader = Reader(wk)
        # print(reader.get_paddle())
        # print(reader.get_torch())
        # 获得前向结果
        paddle_net = Paddle_Api(**reader.get_paddle(), dtype=dtype)
        paddle_forward = paddle_net.run_forward()
        # print(paddle_forward)

        torch_net = Torch_Api(**reader.get_torch(), dtype=dtype)
        torch_forward = torch_net.run_forward()
        # print(torch_forward)


        # 获取反向结果
        paddle_backward = paddle_net.run_backward()
        # print(paddle_backward)
        torch_backward = torch_net.run_backward()
        # print(torch_backward)
        # 对比
        self.checker(paddle_forward, torch_forward, paddle_backward, torch_backward)


    def co_api_d2st(self, dtype="float64"):
        """
        主执行逻辑
        """
        self.logger.info("【动转静API测试】 Paddle动转静 vs Torch动态图")
        # 获取case配置
        wk = self.case()

        # 生成对应的input和 param
        reader = Reader(wk)
        # print(reader.get_paddle())
        # print(reader.get_torch())
        # 获得前向结果
        paddle_net = Paddle_Api_D2ST(**reader.get_paddle(), dtype=dtype)
        paddle_forward = paddle_net.run_forward()
        # print(paddle_forward)

        torch_net = Torch_Api(**reader.get_torch(), dtype=dtype)
        torch_forward = torch_net.run_forward()
        # print(torch_forward)


        # 获取反向结果
        paddle_backward = paddle_net.run_backward()
        # print(paddle_backward)
        torch_backward = torch_net.run_backward()
        # print(torch_backward)
        # 对比
        self.checker(paddle_forward, torch_forward, paddle_backward, torch_backward)


    def co_api_prim(self, dtype="float64"):
        """
        主执行逻辑
        """
        self.logger.info("【动转静API组合算子测试】 Paddle动转静组合算子 vs Torch动态图")
        # 获取case配置
        wk = self.case()

        # 生成对应的input和 param
        reader = Reader(wk)
        # print(reader.get_paddle())
        # print(reader.get_torch())
        # 获得前向结果
        paddle_net = Paddle_Api_Prim(**reader.get_paddle(), dtype=dtype)
        paddle_forward = paddle_net.run_forward()
        # print(paddle_forward)

        torch_net = Torch_Api(**reader.get_torch(), dtype=dtype)
        torch_forward = torch_net.run_forward()
        # print(torch_forward)
        # 获取反向结果
        paddle_backward = paddle_net.run_backward()
        # print(paddle_backward)
        torch_backward = torch_net.run_backward()
        # print(torch_backward)
        # 对比
        self.checker(paddle_forward, torch_forward, paddle_backward, torch_backward)


if __name__ == '__main__':
    # for i in range(100):
    co = CO_API("yaml/test.yaml", "add")
    co.co_api()
    co.co_api_d2st()
    co.co_api_prim()
    #
    # co = CO("yaml/test.yaml", "conv2d_1")
    # co.co_api()