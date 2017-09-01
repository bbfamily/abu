# -*- encoding:utf-8 -*-
"""
    卖出择时示例因子：n倍atr(止盈止损)择时卖出策略
"""

from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

from .ABuFactorSellBase import AbuFactorSellBase, ESupportDirection

__author__ = '阿布'
__weixin__ = 'abu_quant'


class AbuFactorAtrNStop(AbuFactorSellBase):
    """示例n倍atr(止盈止损)因子"""

    def _init_self(self, **kwargs):
        """
            kwargs中可选参数stop_loss_n: 止损的atr倍数
            kwargs中可选参数stop_win_n: 止盈的atr倍数
        """

        if 'stop_loss_n' in kwargs:
            # 设置止损的atr倍数
            self.stop_loss_n = kwargs['stop_loss_n']
            # 在输出生成的orders_pd中及可视化等等显示的名字
            self.sell_type_extra_loss = '{}:stop_loss={}'.format(self.__class__.__name__, self.stop_loss_n)

        if 'stop_win_n' in kwargs:
            # 设置止盈的atr倍数
            self.stop_win_n = kwargs['stop_win_n']
            # 在输出生成的orders_pd中及可视化等等显示的名字
            self.sell_type_extra_win = '{}:stop_win={}'.format(self.__class__.__name__, self.stop_win_n)

    def support_direction(self):
        """n倍atr(止盈止损)因子支持两个方向"""
        return [ESupportDirection.DIRECTION_CAll.value, ESupportDirection.DIRECTION_PUT.value]

    def fit_day(self, today, orders):
        """
        止盈event：截止今天相比买入时的收益 * 买入时的期望方向 > n倍atr
        止损event：截止今天相比买入时的收益 * 买入时的期望方向 < -n倍atr
        :param today: 当前驱动的交易日金融时间序列数据
        :param orders: 买入择时策略中生成的订单序列
        :return:
        """

        for order in orders:
            """
                today.close - order.buy_price：截止今天相比买入时的收益，
                order.expect_direction：买单的方向，收益＊方向＝实际收益
            """
            profit = (today.close - order.buy_price) * order.expect_direction
            # atr常数，示例使用今天的atr21与atr14和作为atr常数，亦可以使用其它组合常量的方式
            stop_base = today.atr21 + today.atr14
            if hasattr(self, 'stop_win_n') and profit > 0 and profit > self.stop_win_n * stop_base:
                # 满足止盈条件卖出股票, 即收益(profit) > n倍atr
                self.sell_type_extra = self.sell_type_extra_win
                # 由于使用了当天的close价格，所以明天才能卖出
                self.sell_tomorrow(order)

            if hasattr(self, 'stop_loss_n') and profit < 0 and profit < -self.stop_loss_n * stop_base:
                # 满足止损条件卖出股票, 即收益(profit) < -n倍atr
                self.sell_type_extra = self.sell_type_extra_loss
                order.fit_sell_order(self.today_ind, self)
                # 由于使用了当天的close价格，所以明天才能卖出
                self.sell_tomorrow(order)
