# -*- encoding:utf-8 -*-
"""
    日内滑点卖出示例实现：均价卖出
    最简单的回测卖出方式，优点简单，且回测高效，在回测交易
    数量足够多的前提下也能接近实盘
"""

from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

import numpy as np

from .ABuSlippageSellBase import AbuSlippageSellBase, slippage_limit_down

__author__ = '阿布'
__weixin__ = 'abu_quant'


class AbuSlippageSellMean(AbuSlippageSellBase):
    """示例日内滑点均价卖出类"""

    @slippage_limit_down
    def fit_price(self):
        """
        取当天交易日的最高最低均价做为决策价格
        :return: 最终决策的当前交易卖出价格
        """

        self.sell_price = np.mean([self.kl_pd_sell['high'], self.kl_pd_sell['low']])
        return self.sell_price
