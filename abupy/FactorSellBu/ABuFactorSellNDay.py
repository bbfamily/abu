# -*- encoding:utf-8 -*-
"""
    卖出择时示例因子：n日卖出策略，不管什么结果，买入后只持有N天
"""

from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

from .ABuFactorSellBase import AbuFactorSellBase, ESupportDirection

__author__ = '阿布'
__weixin__ = 'abu_quant'


class AbuFactorSellNDay(AbuFactorSellBase):
    """n日卖出策略，不管交易现在什么结果，买入后只持有N天"""

    def _init_self(self, **kwargs):
        """kwargs中可以包含: 参数sell_n：代表买入后持有的天数，默认1天"""
        self.sell_n = kwargs.pop('sell_n', 1)
        self.is_sell_today = kwargs.pop('is_sell_today', False)
        self.sell_type_extra = '{}:sell_n={}'.format(self.__class__.__name__, self.sell_n)

    def support_direction(self):
        """因子支持两个方向"""
        return [ESupportDirection.DIRECTION_CAll.value, ESupportDirection.DIRECTION_PUT.value]

    def fit_day(self, today, orders):
        """
        :param today: 当前驱动的交易日金融时间序列数据
        :param orders: 买入择时策略中生成的订单序列
        :return:
        """
        for order in orders:
            # 将单子的持有天数进行增加
            order.keep_days += 1
            if order.keep_days >= self.sell_n:
                # 只要超过self.sell_n即卖出
                self.sell_today(order) if self.is_sell_today else self.sell_tomorrow(order)
