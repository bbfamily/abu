# -*- encoding:utf-8 -*-
"""示例仓位管理：kelly仓位管理模块"""

from __future__ import division
from __future__ import absolute_import
from __future__ import print_function

from .ABuPositionBase import AbuPositionBase


class AbuKellyPosition(AbuPositionBase):
    """示例kelly仓位管理类"""

    def fit_position(self, factor_object):
        """
        通过kelly公司计算仓位, fit_position计算的结果是买入多少个单位（股，手，顿，合约）
        :param factor_object: ABuFactorBuyBases子类实例对象
        :return:买入多少个单位（股，手，顿，合约）
        """
        # 败率
        loss_rate = 1 - self.win_rate
        # kelly计算出仓位比例
        kelly_pos = self.win_rate - loss_rate / (self.gains_mean / self.losses_mean)
        # 最大仓位限制，依然受上层最大仓位控制限制，eg：如果kelly计算出全仓，依然会减少到75%，如修改需要修改最大仓位值
        kelly_pos = self.pos_max if kelly_pos > self.pos_max else kelly_pos
        # 结果是买入多少个单位（股，手，顿，合约）
        return self.read_cash * kelly_pos / self.bp * self.deposit_rate

    def _init_self(self, **kwargs):
        """kelly仓位控制管理类初始化设置"""

        # 默认kelly仓位胜率0.50
        self.win_rate = kwargs.pop('win_rate', 0.50)
        # 默认平均获利期望0.10
        self.gains_mean = kwargs.pop('gains_mean', 0.10)
        # 默认平均亏损期望0.05
        self.losses_mean = kwargs.pop('losses_mean', 0.05)

        """以默认的设置kelly根据计算0.5 - 0.5 / (0.10 / 0.05) 仓位将是0.25即25%"""
