# -*- encoding:utf-8 -*-
"""
    卖出择时策略因子基础模块
"""

from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import functools
from enum import Enum
from abc import ABCMeta, abstractmethod

from ..CoreBu.ABuFixes import six
# noinspection PyUnresolvedReferences
from ..CoreBu.ABuFixes import filter
from ..CoreBu.ABuBase import AbuParamBase
from ..SlippageBu.ABuSlippageSellMean import AbuSlippageSellMean
from ..TradeBu.ABuMLFeature import AbuMlFeature
from ..UmpBu.ABuUmpManager import AbuUmpManager

__author__ = '阿布'
__weixin__ = 'abu_quant'


def skip_last_day(func):
    """可选择装饰在fit_day上，过滤选股周期中的最后一个交易日"""
    @functools.wraps(func)
    def wrapper(self, today, orders):
        day_ind = int(today.key)
        if day_ind >= self.kl_pd.shape[0] - 1:
            return

        return func(self, today, orders)

    return wrapper


def filter_sell_order(func):
    """
        择时卖出因子策略可支持正向，反向，或者两个方向都支持，针对order中买入的方向，filter策略
        对仅支持一个方向的一定要在fit_day上加上此装饰，两个方向都支持的话，重视效率情况下可不装饰
    """
    @functools.wraps(func)
    def wrapper(self, today, orders):
        # 根据order支持的方向是否在当前策略支持范围来筛选order
        orders = list(filter(lambda order: order.expect_direction in self.support_direction(), orders))
        return func(self, today, orders)

    return wrapper


class ESupportDirection(Enum):
    """子策略在support_direction中支持的方向数值定义"""
    DIRECTION_CAll = 1.0
    DIRECTION_PUT = -1.0


class AbuFactorSellBase(six.with_metaclass(ABCMeta, AbuParamBase)):
    """
        卖出择时策略因子基类：卖出择时策略基类和买入择时基类不同，买入择时
        必须混入一个方向类，代表买涨还是买跌，且只能有一个方向，，卖出策略
        可以同时支持买涨，也可以只支持一个方向
    """

    def __init__(self, capital, kl_pd, combine_kl_pd, benchmark, **kwargs):
        """
        :param capital: 资金类AbuCapital实例化对象
        :param kl_pd: 择时时段金融时间序列，pd.DataFrame对象
        :param combine_kl_pd:合并了之前一年时间序列的金融时间序列，pd.DataFrame对象
        :param benchmark: 交易基准对象，AbuBenchmark实例对象, 因子可有选择性使用，比如大盘对比等功能
        """

        # 择时金融时间序列走势数据
        self.kl_pd = kl_pd
        # 机器学习特征数据构建需要，详情见make_sell_order_ml_feature中构造特征使用
        self.combine_kl_pd = combine_kl_pd
        # 资金情况数据
        self.capital = capital
        # 交易基准对象，AbuBenchmark实例对象, 因子可有选择性使用，比如大盘对比等功能
        self.benchmark = benchmark

        # 滑点类，默认AbuSlippageSellMean
        self.slippage_class = kwargs.pop('slippage', AbuSlippageSellMean)

        # 构造ump对外的接口对象UmpManager
        self.ump_manger = AbuUmpManager(self)

        # 默认的卖出说明，子类通过_init_self可覆盖更具体的名字
        self.sell_type_extra = '{}'.format(self.__class__.__name__)

        # 子类继续完成自有的构造
        self._init_self(**kwargs)

    def __str__(self):
        """打印对象显示：class name, slippage, kl_pd.info"""
        return '{}: slippage:{}, \nkl:\n{}'.format(self.__class__.__name__, self.slippage_class, self.kl_pd.info())

    __repr__ = __str__

    @abstractmethod
    def _init_self(self, **kwargs):
        """子类因子针对可扩展参数的初始化"""
        pass

    @abstractmethod
    def fit_day(self, today, orders):
        """子类主要需要实现的函数，完成策略因子针对每一个交易日的卖出交易策略"""
        pass

    @abstractmethod
    def support_direction(self):
        """子类需要显视注明自己支持的交易方向"""
        pass

    def make_sell_order(self, order, day_ind):
        """
        根据交易发生的时间索引，依次进行：卖出交易时间序列特征生成，
        决策卖出交易是否拦截，生成特征学习数据，最终返回是否order成交，即订单生效
        :param order: 买入择时策略中生成的订单
        :param day_ind: 卖出交易发生的时间索引，即对应self.kl_pd.key
        :return:
        """

        # 卖出交易时间序列特征生成
        ml_feature_dict = self.make_sell_order_ml_feature(day_ind)
        # 决策卖出交易是否拦截
        block = self.make_ump_block_decision(ml_feature_dict)
        if block:
            return False

        # 如果卖出交易不被拦截，生成特征学习数据
        if order.ml_features is None:
            order.ml_features = ml_feature_dict
        else:
            order.ml_features.update(ml_feature_dict)
        return True

    # noinspection PyUnusedLocal
    def make_ump_block_decision(self, ml_feature_dict):
        """
        输入需要决策的当前卖出交易特征通过ump模块的对外manager对交易进行决策，
        判断是否拦截卖出交易，还是放行卖出交易。子类可复写此方法，即子类策略因子实现
        自己的任意ump组合拦截方式，根据策略的拦截比例需要等等参数确定ump具体策略，
        且对于多种策略并行执行策略本身定制适合自己的拦截策略，提高灵活度
        :param ml_feature_dict: 需要决策的当前卖出时刻交易特征dict
        :return:
        """
        return self.ump_manger.ump_block(ml_feature_dict)

    def make_sell_order_ml_feature(self, day_ind):
        """
         根据卖出交易发生的时间索引构通过AbuMlFeature构建卖出时刻的各个交易特征
         :param day_ind: 交易发生的时间索引，对应self.kl_pd.key
         :return:
         """
        return AbuMlFeature().make_feature_dict(self.kl_pd, self.combine_kl_pd, day_ind, buy_feature=False)

    """TODO: 使用check support方式查询是否支持fit_week，fit_month，上层不再使用hasattr去判断"""
    # def fit_week(self, *args, **kwargs):
    #     pass

    # def fit_month(self, *args, **kwargs):
    #     pass
