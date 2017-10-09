# -*- encoding:utf-8 -*-
"""
    示例仓位管理：atr仓位管理模块
"""

from __future__ import division
from __future__ import absolute_import
from __future__ import print_function

from .ABuPositionBase import AbuPositionBase

__author__ = '阿布'
__weixin__ = 'abu_quant'

"""
    默认0.1即10% 外部可通过如：abupy.beta.atr.g_atr_pos_base = 0.01修改仓位基础配比
    需要注意外部其它自定义仓位管理类不要随意使用模块全局变量，AbuAtrPosition特殊因为注册
    在ABuEnvProcess中在多进程启动时拷贝了模块全局设置内存
"""
g_atr_pos_base = 0.1


class AbuAtrPosition(AbuPositionBase):
    """示例atr仓位管理类"""

    s_atr_base_price = 15  # best fit wide: 12-20
    s_std_atr_threshold = 0.5  # best fit wide: 0.3-0.65

    def fit_position(self, factor_object):
        """
        fit_position计算的结果是买入多少个单位（股，手，顿，合约）
        计算：（常数价格 ／ 买入价格）＊ 当天交易日atr21
        :param factor_object: ABuFactorBuyBases实例对象
        :return: 买入多少个单位（股，手，顿，合约）
        """
        std_atr = (self.atr_base_price / self.bp) * self.kl_pd_buy['atr21']

        """
            对atr 进行限制 避免由于股价波动过小，导致
            atr小，产生大量买单，实际上针对这种波动异常（过小，过大）的股票
            需要有其它的筛选过滤策略, 选股的时候取0.5，这样最大取两倍g_atr_pos_base
        """
        atr_wv = self.std_atr_threshold if std_atr < self.std_atr_threshold else std_atr
        # 计算出仓位比例
        atr_pos = self.atr_pos_base / atr_wv
        # 最大仓位限制
        atr_pos = self.pos_max if atr_pos > self.pos_max else atr_pos
        # 结果是买入多少个单位（股，手，顿，合约）
        return self.read_cash * atr_pos / self.bp * self.deposit_rate

    def _init_self(self, **kwargs):
        """atr仓位控制管理类初始化设置"""
        self.atr_base_price = kwargs.pop('atr_base_price', AbuAtrPosition.s_atr_base_price)
        self.std_atr_threshold = kwargs.pop('std_atr_threshold', AbuAtrPosition.s_std_atr_threshold)
        self.atr_pos_base = kwargs.pop('atr_pos_base', g_atr_pos_base)
