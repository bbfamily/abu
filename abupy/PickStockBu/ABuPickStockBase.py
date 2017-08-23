# -*- encoding:utf-8 -*-
"""
    选股因子抽象基类
"""

from __future__ import print_function
from __future__ import absolute_import
from __future__ import division

import functools
from abc import ABCMeta, abstractmethod

from ..CoreBu.ABuFixes import six
from ..CoreBu import ABuEnv
from ..CoreBu.ABuBase import AbuParamBase

__author__ = '阿布'
__weixin__ = 'abu_quant'


def reversed_result(func):
    """对选股结果进行反转的装饰器，装饰在fit_pick上"""
    @functools.wraps(func)
    def wrapper(self, *args, **kwargs):
        result = func(self, *args, **kwargs)
        # 通过对象变量reversed，判断是否反转结果
        result = not result if self.reversed else result
        return result

    return wrapper


class AbuPickStockBase(six.with_metaclass(ABCMeta, AbuParamBase)):
    def __init__(self, capital, benchmark, **kwargs):
        """
        :param capital:资金类AbuCapital实例化对象
        :param benchmark:交易基准对象，AbuBenchmark实例对象
        :param kwargs:其它可扩展参数
        """
        self.capital = capital
        self.benchmark = benchmark

        # 所有自定义参数不使用kwargs.pop('a', default)方式，因为有从配置文件读取等需求，而且后续_init_self可能也还需要
        # 默认反转结果false，通过kwargs参数控制
        self.reversed = False
        if 'reversed' in kwargs:
            self.reversed = kwargs['reversed']

        # 默认选股周期默认一年的交易日
        self.xd = ABuEnv.g_market_trade_year
        if 'xd' in kwargs:
            self.xd = kwargs['xd']

        # 最小选股周期，小于这个将抛弃，即结果投反对票
        self.min_xd = int(self.xd / 2)
        if 'min_xd' in kwargs:
            self.min_xd = kwargs['min_xd']

        # 因子独有的init继续
        self._init_self(**kwargs)

    def __str__(self):
        """打印对象显示：class name, benchmark, reversed, xd, min_xd"""
        return '{}: {}, reversed:{}, xd:{}, min_xd:{}'.format(self.__class__.__name__,
                                                              self.benchmark, self.reversed, self.xd, self.min_xd)

    __repr__ = __str__

    @abstractmethod
    def _init_self(self, **kwargs):
        """子类因子针对可扩展参数的初始化"""
        pass

    @abstractmethod
    def fit_pick(self, *args, **kwargs):
        """选股操作接口，即因子对象针对一个交易目标的投票结果，具体详见示例因子"""
        pass

    @abstractmethod
    def fit_first_choice(self, pick_worker, choice_symbols, *args, **kwargs):
        """因子首选批量选股接口，即因子对象对多个交易目标的投票结果，具体详见示例因子"""
        pass
