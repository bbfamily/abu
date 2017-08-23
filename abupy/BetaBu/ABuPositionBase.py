# -*- encoding:utf-8 -*-
"""
    风险控制仓位管理基础
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from abc import ABCMeta, abstractmethod

from ..CoreBu.ABuFixes import six
from ..MarketBu.ABuMarket import MarketMixin

"""每一笔交易最大仓位比例设置，外部可通过如：abupy.beta.position.g_pos_max = 0.5修改最大每一笔交易最大仓位比例"""
g_pos_max = 0.75
"""
    保证净最小比例，默认1，即不使用融资，不会触发Margin Call
    在期货数据中有每种商品最少保证金比例，可使用设置
    外部可通过如：abupy.beta.position.g_deposit_rate = 0.5
"""
g_deposit_rate = 1


class AbuPositionBase(six.with_metaclass(ABCMeta, MarketMixin)):
    """仓位管理抽象基类"""

    def __init__(self, kl_pd_buy, factor_name, symbol_name, bp, read_cash, deposit_rate):
        """
        :param kl_pd_buy: 交易当日的交易数据
        :param factor_name: 因子名称
        :param symbol_name: symbol代码
        :param bp: 买入价格
        :param read_cash: 初始资金
        :param deposit_rate: 保证金比例
        """
        self.kl_pd_buy = kl_pd_buy
        self.factor_name = factor_name
        self.symbol_name = symbol_name
        self.bp = bp
        self.read_cash = read_cash
        self.deposit_rate = deposit_rate

    def __str__(self):
        """打印对象显示：class name, factor_name, symbol_name, read_cash, deposit_rate"""
        return '{}: factor_name:{}, symbol_name:{}, read_cash:{}, deposit_rate:{}'.format(self.__class__.__name__,
                                                                                          self.factor_name,
                                                                                          self.symbol_name,
                                                                                          self.read_cash,
                                                                                          self.deposit_rate)

    __repr__ = __str__

    @abstractmethod
    def fit_position(self, factor_object):
        """
        fit_position计算的结果是买入多少个单位（股，手，顿，合约）具体计算子类实现
        :param factor_object: ABuFactorBuyBases实例对象
        :return:买入多少个单位（股，手，顿，合约）
        """
        pass
