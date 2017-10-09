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

"""每一笔交易最大仓位比例设置，外部可通过如：abupy.beta.position.g_pos_max = 0.5修改最大每一笔交易最大仓位比例，默认75%"""
g_pos_max = 0.75
"""
    保证金最小比例，默认1，即不使用融资，不会触发Margin Call
    在期货数据中有每种商品最少保证金比例，可使用设置
    外部可通过如：abupy.beta.position.g_deposit_rate = 0.5
"""
g_deposit_rate = 1
"""
    买入因子全局默认仓位管理类，默认None的情况下会使用AbuAtrPosition作为默认仓位管理类.

    和卖出因子，选股因子不同，一个买入因子可以对应多个卖出因子，多个选股因子，但一个买入
    因子只能对应一个仓位管理类，可以是全局仓位管理，也可以是针对买入因子的独有附属仓位管理
    类
"""
g_default_pos_class = None


class AbuPositionBase(six.with_metaclass(ABCMeta, MarketMixin)):
    """仓位管理抽象基类"""

    def __init__(self, kl_pd_buy, factor_name, symbol_name, bp, read_cash, **kwargs):
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

        # 如果有全局最大仓位设置基类负责弹出
        self.pos_max = kwargs.pop('pos_max', g_pos_max)
        # 如果有全局保证金最小比例设置基类负责弹出
        self.deposit_rate = kwargs.pop('deposit_rate', g_deposit_rate)

        # 子类继续完成自有的构造
        self._init_self(**kwargs)

    def __str__(self):
        """打印对象显示：class name, factor_name, symbol_name, read_cash, deposit_rate"""
        return '{}: factor_name:{}, symbol_name:{}, read_cash:{}, deposit_rate:{}'.format(self.__class__.__name__,
                                                                                          self.factor_name,
                                                                                          self.symbol_name,
                                                                                          self.read_cash,
                                                                                          self.deposit_rate)

    __repr__ = __str__

    @abstractmethod
    def _init_self(self, **kwargs):
        """子类仓位管理针对可扩展参数的初始化"""
        pass

    @abstractmethod
    def fit_position(self, factor_object):
        """
        fit_position计算的结果是买入多少个单位（股，手，顿，合约）具体计算子类实现
        :param factor_object: ABuFactorBuyBases实例对象
        :return:买入多少个单位（股，手，顿，合约）
        """
        pass
