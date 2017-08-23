# -*- encoding:utf-8 -*-
"""
    日内滑点买入价格决策基础模块：暂时迁移简单实现方式，符合回测需求，如迁移实盘模块
    需添加日内择时策略，通过日内分钟k线，实现日内分钟k线择时，更微观的
    实现日内择时滑点功能，不考虑大资金的冲击成本及系统外的大幅滑点
"""

from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

from abc import ABCMeta, abstractmethod
import functools

import numpy as np

from ..CoreBu.ABuFixes import six

__author__ = '阿布'
__weixin__ = 'abu_quant'


class AbuSlippageBuyBase(six.with_metaclass(ABCMeta, object)):
    """非高频日内滑点买入决策抽象基类"""

    def __init__(self, kl_pd_buy, factor_name):
        """
        :param kl_pd_buy: 交易当日的交易数据
        :param factor_name: ABuFactorBuyBases子类实例对象的factor_name
        """
        self.buy_price = np.inf
        self.kl_pd_buy = kl_pd_buy
        self.factor_name = factor_name

    def fit(self):
        """做基础验证比如今天是否停盘后调用fit_price"""
        if self.kl_pd_buy.empty or self.kl_pd_buy.volume == 0:
            # 买入时正无穷为放弃单子
            return np.inf

        return self.fit_price()

    @abstractmethod
    def fit_price(self):
        """
        子类主要需要实现的函数，决策交易当日的最终买入价格
        :return: 最终决策的当前交易买入价格
        """
        pass

"""是否开启涨停板滑点买入价格特殊处理, 默认关闭，外部修改如：abupy.slippage.sbb.g_enable_limit_up = True"""
g_enable_limit_up = False
"""
    初始设定涨停板买入成交概率100%，这里也可以在计算完一次概率后，再使用成交量做二次概率计算，
    外部修改如：abupy.slippage.sbb.g_limit_up_deal_chance = 0.5，即修改为买入概率50%
"""
g_limit_up_deal_chance = 1
"""在集合竞价阶段价格已经达成涨停的情况下买入成功的概率，默认0.2, 即20%成功概率"""
g_pre_limit_up_rate = 0.2


def slippage_limit_up(func):
    """
        针对a股涨停板买入价格决策的装饰器，子类可选择装饰与不装饰在fit_price上
        如果是实盘策略中，使用分钟k线，及日内择时策略，即不需特别处理。
        回测中需要特别处理，处理买入成功概率，根据概率决定是否能买入，
        及涨停下的买入价格决策，涨停下买入价格模型为，越靠近涨停价格
        买入成交概率越大，即在涨停下预期以靠近涨停价格买入，缺点是使用了随机数，
        导致回测结果将出现不一致的情况
    """
    @functools.wraps(func)
    def wrapper(self, *args, **kwargs):
        if g_enable_limit_up and self.kl_pd_buy.p_change >= 10 and self.kl_pd_buy.high == self.kl_pd_buy.close:
            """
                涨停板命中后需要根据涨停板买入成交概率(g_limit_up_deal_chance)来作为
                二项式分布的概率值计算买入成功概率
            """
            if self.kl_pd_buy.high == self.kl_pd_buy.low:
                # 10个点，且最高＝最低，即a股在集合竞价阶段达成涨停，买入成功概率降低到g_limit_up_deal_chance * 0.2
                # TODO 这个概率最好使用成交量当日来计算出来
                limit_up_deal_chance = g_limit_up_deal_chance * g_pre_limit_up_rate
            else:
                limit_up_deal_chance = g_limit_up_deal_chance

            deal = np.random.binomial(1, limit_up_deal_chance)
            if deal:
                if self.kl_pd_buy.high == self.kl_pd_buy.low:
                    return self.kl_pd_buy.high

                # 买入成功后需要进一步决策价位，首选arange出一个从低到涨停价格的序列，间隔0.01
                price_lh = np.arange(self.kl_pd_buy.low, self.kl_pd_buy.high, 0.01)
                # 构造概率序列，可以使用其它比如指数分布等，提高以涨停价格买入的概率，这里只使用最简单方式
                lh_chance = np.linspace(0, 1, len(price_lh))
                """
                    计算出对应的概率, 这里的概率分布并不陡峭，即涨停价格附近权重并不是很高，
                    可以使用如：np.power(price_hl, len(price_hl) / 2) / np.power(price_hl, len(price_hl) / 2).sum()
                    来进一步提升涨跌板附近价格的买入权重
                """
                # noinspection PyUnresolvedReferences
                p = lh_chance / lh_chance.sum()
                # 最后使用随机加权概率抽取，选中一个买入涨停价格
                return np.random.choice(price_lh, 1, p=p)[0]
            # 没能成交返回正无穷
            return np.inf
        else:
            return func(self, *args, **kwargs)
    return wrapper
