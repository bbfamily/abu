# -*- encoding:utf-8 -*-
"""
    卖出择时示例因子：突破卖出择时因子
"""

from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

from .ABuFactorSellBase import AbuFactorSellBase, filter_sell_order, skip_last_day, ESupportDirection

__author__ = '阿布'
__weixin__ = 'abu_quant'


class AbuFactorSellBreak(AbuFactorSellBase):
    """示例向下突破卖出择时因子"""

    def _init_self(self, **kwargs):
        """kwargs中必须包含: 突破参数xd 比如20，30，40天...突破"""

        # 向下突破参数 xd， 比如20，30，40天...突破
        self.xd = kwargs['xd']
        # 在输出生成的orders_pd中显示的名字
        self.sell_type_extra = '{}:{}'.format(self.__class__.__name__, self.xd)

    def support_direction(self):
        """支持的方向，只支持正向"""
        return [ESupportDirection.DIRECTION_CAll.value]

    @skip_last_day
    @filter_sell_order
    def fit_day(self, today, orders):
        """
        寻找向下突破作为策略卖出驱动event
        :param today: 当前驱动的交易日金融时间序列数据
        :param orders: 买入择时策略中生成的订单序列
        :return:
        """

        day_ind = int(today.key)
        # 今天的收盘价格达到xd天内最低价格则符合条件
        if today.close == self.kl_pd.close[day_ind - self.xd + 1:day_ind + 1].min():
            for order in orders:
                order.fit_sell_order(day_ind, self)
