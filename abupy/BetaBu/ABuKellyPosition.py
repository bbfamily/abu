# -*- encoding:utf-8 -*-
"""
    示例仓位管理：kelly仓位管理模块
"""

from __future__ import division
from __future__ import absolute_import
from __future__ import print_function

from .ABuPositionBase import AbuPositionBase
from . import ABuPositionBase


class AbuKellyPosition(AbuPositionBase):
    """示例kelly仓位管理类"""

    def fit_position(self, factor_object):
        """
        fit_position计算的结果是买入多少个单位（股，手，顿，合约）
        需要factor_object策略因子对象通过历史回测统计胜率，期望收益，期望亏损，
        并设置构造当前factor_object对象，通过kelly公司计算仓位
        :param factor_object: ABuFactorBuyBases子类实例对象
        :return:买入多少个单位（股，手，顿，合约）
        """

        # 检测择时策略因子对象有没有设置设置胜率，期望收益，期望亏损，详情查阅ABuFactorBuyBase
        if not hasattr(factor_object, 'win_rate'):
            raise RuntimeError('AbuKellyPosition need factor_object has  win_rate')
        if not hasattr(factor_object, 'gains_mean'):
            raise RuntimeError('AbuKellyPosition need factor_object has  gains_mean')
        if not hasattr(factor_object, 'losses_mean'):
            raise RuntimeError('AbuKellyPosition need factor_object has  losses_mean')

        # 胜率
        win_rate = factor_object.win_rate
        # 败率
        loss_rate = 1 - win_rate
        # 平均获利期望
        gains_mean = factor_object.gains_mean
        # 平均亏损期望
        losses_mean = factor_object.losses_mean
        # kelly计算出仓位比例
        kelly_pos = win_rate - loss_rate / (gains_mean / losses_mean)
        # 最大仓位限制
        kelly_pos = ABuPositionBase.g_pos_max if kelly_pos > ABuPositionBase.g_pos_max else kelly_pos
        # 结果是买入多少个单位（股，手，顿，合约）
        return self.read_cash * kelly_pos / self.bp * self.deposit_rate
