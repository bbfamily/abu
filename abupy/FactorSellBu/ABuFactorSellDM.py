# -*- encoding:utf-8 -*-
"""
    卖出择时示例因子，双均线策略
"""

from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

from .ABuFactorSellBase import AbuFactorSellXD, ESupportDirection
from ..IndicatorBu.ABuNDMa import calc_ma_from_prices

__author__ = '阿布'
__weixin__ = 'abu_quant'


class AbuDoubleMaSell(AbuFactorSellXD):
    """示例卖出双均线择时因子"""

    def _init_self(self, **kwargs):
        """
            kwargs中可选参数：fast: 均线快线周期，默认不设置，使用5
            kwargs中可选参数：slow: 均线慢线周期，默认不设置，使用60
        """

        # TODO 重构与买入因子重复代码抽取
        # 均线快线周期，默认使用5天均线
        self.ma_fast = kwargs.pop('fast', 5)
        # 均线慢线周期，默认使用60天均线
        self.ma_slow = kwargs.pop('slow', 60)

        if self.ma_fast >= self.ma_slow:
            # 慢线周期必须大于快线
            raise ValueError('ma_fast >= self.ma_slow !')

        # xd周期数据需要比ma_slow大一天，这样计算ma就可以拿到今天和昨天两天的ma，用来判断金叉，死叉
        kwargs['xd'] = self.ma_slow + 1
        # 设置好xd后可以直接使用基类针对xd的初始化
        super(AbuDoubleMaSell, self)._init_self(**kwargs)

    def support_direction(self):
        """支持的方向，因子支持两个方向"""
        return [ESupportDirection.DIRECTION_CAll.value, ESupportDirection.DIRECTION_PUT.value]

    def fit_day(self, today, orders):
        """
            双均线卖出择时因子：
            call方向：快线下穿慢线形成死叉，做为卖出信号
            put方向： 快线上穿慢线做为卖出信号
        """
        # 计算快线
        fast_line = calc_ma_from_prices(self.xd_kl.close, self.ma_fast, min_periods=1)
        # 计算慢线
        slow_line = calc_ma_from_prices(self.xd_kl.close, self.ma_slow, min_periods=1)

        if len(fast_line) >= 2 and len(slow_line) >= 2:
            # 今天的快线值
            fast_today = fast_line[-1]
            # 昨天的快线值
            fast_yesterday = fast_line[-2]
            # 今天的慢线值
            slow_today = slow_line[-1]
            # 昨天的慢线值
            slow_yesterday = slow_line[-2]

            for order in orders:
                if order.expect_direction == 1 \
                        and fast_yesterday >= slow_yesterday and fast_today < slow_today:
                    # call方向：快线下穿慢线线形成死叉，做为卖出信号
                    self.sell_tomorrow(order)
                elif order.expect_direction == -1 \
                        and slow_yesterday >= fast_yesterday and fast_today > slow_today:
                    # put方向：快线上穿慢线做为卖出信号
                    self.sell_tomorrow(order)
