# -*- encoding:utf-8 -*-
"""
    卖出择时示例因子：单日最大跌幅n倍atr止损
    做为单边止损因子使用，作为风险控制保护因子
"""

from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

from .ABuFactorSellBase import AbuFactorSellBase, ESupportDirection

__author__ = '阿布'
__weixin__ = 'abu_quant'

"""外部可通过如：abupy.fs.pre.g_default_pre_atr_n = 2.5来修改默认值"""
g_default_pre_atr_n = 1.5


class AbuFactorPreAtrNStop(AbuFactorSellBase):
    """示例单日最大跌幅n倍atr(止损)风险控制因子"""

    def _init_self(self, **kwargs):
        """kwargs中可选参数pre_atr_n: 单日最大跌幅止损的atr倍数"""

        self.pre_atr_n = g_default_pre_atr_n
        if 'pre_atr_n' in kwargs:
            # 设置下跌止损倍数
            self.pre_atr_n = kwargs['pre_atr_n']
            self.sell_type_extra = '{}:pre_atr={}'.format(self.__class__.__name__, self.pre_atr_n)

    def support_direction(self):
        """单日最大跌幅n倍atr(止损)因子支持两个方向"""
        return [ESupportDirection.DIRECTION_CAll.value, ESupportDirection.DIRECTION_PUT.value]

    def fit_day(self, today, orders):
        """
        止损event：今天相比昨天的收益 * 买入时的期望方向 > today.atr21 * pre_atr_n
        :param today: 当前驱动的交易日金融时间序列数据
        :param orders: 买入择时策略中生成的订单序列
        :return:
        """

        for order in orders:
            if (today.pre_close - today.close) * order.expect_direction > today.atr21 * self.pre_atr_n:
                # 只要今天的收盘价格比昨天收盘价格差大于一个差值就止损卖出, 亦可以使用其它计算差值方式
                self.sell_tomorrow(order)
