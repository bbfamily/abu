# -*- encoding:utf-8 -*-
"""
    买入择时策略因子基础模块
"""

from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

from abc import ABCMeta, abstractmethod

from ..CoreBu.ABuFixes import six
from ..BetaBu.ABuAtrPosition import AbuAtrPosition
from ..TradeBu.ABuOrder import AbuOrder
from ..TradeBu.ABuMLFeature import AbuMlFeature
from ..CoreBu.ABuBase import AbuParamBase
from ..SlippageBu.ABuSlippageBuyMean import AbuSlippageBuyMean
from ..UtilBu.ABuLazyUtil import LazyFunc
from ..UmpBu.ABuUmpManager import AbuUmpManager

__author__ = '阿布'
__weixin__ = 'abu_quant'


class BuyCallMixin(object):
    """
        混入类，混入代表买涨，不完全是期权中buy call的概念，
        只代表看涨正向操作，即期望买入后交易目标价格上涨，上涨带来收益
    """

    @LazyFunc
    def buy_type_str(self):
        """用来区别买入类型unique 值为call"""
        return "call"

    @LazyFunc
    def expect_direction(self):
        """期望收益方向，1.0即正向期望"""
        return 1.0


class BuyPutMixin(object):
    """
        混入类，混入代表买跌，应用场景在于期权，期货策略中，
        不完全是期权中buy put的概念，只代看跌反向操作，
        即期望买入后交易目标价格下跌，下跌带来收益
    """

    @LazyFunc
    def buy_type_str(self):
        """用来区别买入类型unique 值为put"""
        return "put"

    @LazyFunc
    def expect_direction(self):
        """期望收益方向，1.0即反向期望"""
        return -1.0


class AbuFactorBuyBase(six.with_metaclass(ABCMeta, AbuParamBase)):
    """
        买入择时策略因子基类：每一个继承AbuFactorBuyBase的子类必须混入一个方向类，
        且只能混入一个方向类，即具体买入因子必须明确买入方向，且只能有一个买入方向，
        一个因子不能同上又看涨又看跌，详情查阅ABuFactorBuyBreak因子示例
    """

    def __init__(self, capital, kl_pd, combine_kl_pd, benchmark, **kwargs):
        """
        :param capital:资金类AbuCapital实例化对象
        :param kl_pd:择时时段金融时间序列，pd.DataFrame对象
        :param combine_kl_pd:合并了之前一年时间序列的金融时间序列，pd.DataFrame对象
        :param benchmark: 交易基准对象，AbuBenchmark实例对象, 因子可有选择性使用，比如大盘对比等功能
        """
        # 择时金融时间序列走势数据
        self.kl_pd = kl_pd
        # 机器学习特征数据构建需要，详情见make_buy_order_ml_feature中构造特征使用
        self.combine_kl_pd = combine_kl_pd
        # 资金情况数据
        self.capital = capital
        # 交易基准对象，AbuBenchmark实例对象, 因子可有选择性使用，比如大盘对比等功能
        self.benchmark = benchmark

        # 滑点类，默认AbuSlippageBuyMean
        self.slippage_class = kwargs.pop('slippage', AbuSlippageBuyMean)
        # 仓位管理，默认AbuAtrPosition
        self.position_class = kwargs.pop('position', AbuAtrPosition)

        """
            因子可选择根据策略的历史回测设置胜率，期望收益，期望亏损，
            比如使用AbuKellyPosition，必须需要因子的胜率，期望收益，
            期望亏损参数，不要使用kwargs.pop('a', None)设置，因为
            暂时使用hasattr判断是否有设置属性
        """
        if 'win_rate' in kwargs:
            # 策略因子历史胜率
            self.win_rate = kwargs['win_rate']
        if 'gains_mean' in kwargs:
            # 策略因子历史期望收益
            self.gains_mean = kwargs['gains_mean']
        if 'losses_mean' in kwargs:
            # 策略因子历史期望亏损
            self.losses_mean = kwargs['losses_mean']

        # 构造ump对外的接口对象UmpManager
        self.ump_manger = AbuUmpManager(self)

        # 默认的factor_name，子类通过_init_self可覆盖更具体的名字
        self.factor_name = '{}'.format(self.__class__.__name__)

        # 子类继续完成自有的构造
        self._init_self(**kwargs)

    def __str__(self):
        """打印对象显示：class name, slippage, position, kl_pd.info"""
        return '{}: slippage:{}, position:{} \nkl:\n{}'.format(self.__class__.__name__,
                                                               self.slippage_class, self.position_class,
                                                               self.kl_pd.info())

    __repr__ = __str__

    def make_buy_order(self, day_ind):
        """
        根据交易发生的时间索引，依次进行交易订单生成，交易时间序列特征生成，
        决策交易是否拦截，生成特征学习数据，最终返回order，即订单生效
        :param day_ind: 交易发生的时间索引，即对应self.kl_pd.key
        :return:
        """

        order = AbuOrder()
        # AbuOrde对象根据交易发生的时间索引生成交易订单
        order.fit_buy_order(day_ind, self)
        if order.order_deal:
            # 交易时间序列特征生成
            ml_feature_dict = self.make_buy_order_ml_feature(day_ind)
            # 决策交易是否被ump拦截还是可以放行
            block = self.make_ump_block_decision(ml_feature_dict)
            if block:
                return None

            # 如果交易即将成交，将生成的交易特征写入order的特征字段ml_features中，为之后使用机器学习计算学习特征，训练ump
            if order.ml_features is None:
                order.ml_features = ml_feature_dict
            else:
                order.ml_features.update(ml_feature_dict)
        # 返回order，订单生效
        return order

    def make_ump_block_decision(self, ml_feature_dict):
        """
        输入需要决策的当前买入交易特征通过ump模块的对外manager对交易进行决策，
        判断是否拦截买入交易，还是放行买入交易。子类可复写此方法，即子类策略因子实现
        自己的任意ump组合拦截方式，根据策略的拦截比例需要等等参数确定ump具体策略，
        且对于多种策略并行执行策略本身定制适合自己的拦截策略，提高灵活度
        :param ml_feature_dict: 需要决策的当前买入时刻交易特征dict
        :return: bool, 对ml_feature_dict所描述的交易特征是否进行拦截
        """
        return self.ump_manger.ump_block(ml_feature_dict)

    def make_buy_order_ml_feature(self, day_ind):
        """
        根据交易发生的时间索引构通过AbuMlFeature构建买入时刻的各个交易特征
        :param day_ind: 交易发生的时间索引，对应self.kl_pd.key
        :return:
        """
        return AbuMlFeature().make_feature_dict(self.kl_pd, self.combine_kl_pd, day_ind, buy_feature=True)

    @abstractmethod
    def _init_self(self, **kwargs):
        """子类因子针对可扩展参数的初始化"""
        pass

    @abstractmethod
    def fit_day(self, *args, **kwargs):
        """子类主要需要实现的函数，完成策略因子针对每一个交易日的买入交易策略"""
        pass

    """TODO: 使用check support方式查询是否支持fit_week，fit_month，上层不再使用hasattr去判断"""
    # def fit_week(self, *args, **kwargs):
    #     pass

    # def fit_month(self, *args, **kwargs):
    #     pass
