# -*- encoding:utf-8 -*-
"""
    交易执行代理模块
"""

from __future__ import print_function
from __future__ import absolute_import
from __future__ import division

from contextlib import contextmanager
from functools import total_ordering
from enum import Enum

import numpy as np
import pandas as pd

from . import ABuTradeDrawer
from . import ABuTradeExecute

__author__ = '阿布'
__weixin__ = 'abu_quant'


class EOrderSameRule(Enum):
    """对order_pd中对order判断为是否相同使用的规则"""

    """order有相同的symbol和买入日期就认为是相同"""
    ORDER_SAME_BD = 0
    """order有相同的symbol, 买入日期，和卖出日期，即不考虑价格，只要日期相同就相同"""
    ORDER_SAME_BSD = 1
    """order有相同的symbol, 买入日期，相同的买入价格，即单子买入时刻都相同"""
    ORDER_SAME_BDP = 2
    """order有相同的symbol, 买入日期, 买入价格, 并且相同的卖出日期和价格才认为是相同，即买入卖出时刻都相同"""
    ORDER_SAME_BSPD = 3


@total_ordering
class AbuOrderPdProxy(object):
    """
        包装交易订单构成的pd.DataFrame对象，外部debug因子的交易结果，寻找交易策略的问题使用，
        支持两个orders_pd的并集，交集，差集，类似set的操作，同时支持相等，不等，大于，小于
        的比较操作，eg如下：

            orders_pd1 = AbuOrderPdProxy(orders_pd1)
            with orders_pd1.proxy_work(orders_pd2) as (order1, order2):
                a = order1 | order2 # 两个交易结果的并集
                b = order1 & order2 # 两个交易结果的交集
                c = order1 - order2 # 两个交易结果的差集(在order1中，但不在order2中)
                d = order2 - order1 # 两个交易结果的差集(在order2中，但不在order1中)
                eq = order1 == order2 # 两个交易结果是否相同
                lg = order1 > order2 # order1唯一的交易数量是否大于order2
                lt = order1 < order2 # order1唯一的交易数量是否小于order2
    """

    def __init__(self, orders_pd, same_rule=EOrderSameRule.ORDER_SAME_BSPD):
        """
        初始化函数需要pd.DataFrame对象，暂时未做类型检测
        :param orders_pd: 回测结果生成的交易订单构成的pd.DataFrame对象
        :param same_rule: order判断为是否相同使用的规则, 默认EOrderSameRule.ORDER_SAME_BSPD
                          即：order有相同的symbol和买入日期和相同的卖出日期和价格才认为是相同
        """
        # 需要copy因为会添加orders_pd的列属性等
        self.orders_pd = orders_pd.copy()
        self.same_rule = same_rule
        # 并集, 交集, 差集运算结果存储
        self.op_result = None
        self.last_op_metrics = {}

    @contextmanager
    def proxy_work(self, orders_pd):
        """
        传人需要比较的orders_pd，构造ABuOrderPdProxy对象，返回使用者，
        对op_result进行统一分析
        :param orders_pd: 回测结果生成的交易订单构成的pd.DataFrame对象
        :return:
        """

        # 运算集结果重置
        self.op_result = None
        # 实例化比较的ABuOrderPdProxy对象
        other = AbuOrderPdProxy(orders_pd)
        try:
            yield self, other
        finally:
            if isinstance(self.op_result, pd.DataFrame):
                # 如果有并集, 交集, 差集运算结果存储，
                from ..MetricsBu.ABuMetricsBase import AbuMetricsBase
                metrics = AbuMetricsBase(self.op_result, None, None, None)
                metrics.fit_metrics_order()

                self.last_op_metrics['win_rate'] = metrics.win_rate
                self.last_op_metrics['gains_mean'] = metrics.gains_mean
                self.last_op_metrics['losses_mean'] = metrics.losses_mean
                self.last_op_metrics['sum_profit'] = self.op_result['profit'].sum()
                self.last_op_metrics['sum_profit_cg'] = self.op_result['profit_cg'].sum()

    def __and__(self, other):
        """ & 操作符的重载，计算两个交易集的交集"""
        # self.op = 'intersection(order1 & order2)'
        self.op_result = intersection_in_2orders(self.orders_pd, other.orders_pd, same_rule=self.same_rule)
        return self.op_result

    def __or__(self, other):
        """ | 操作符的重载，计算两个交易集的并集"""
        # self.op = 'union(order1 | order2)'
        self.op_result = union_in_2orders(self.orders_pd, other.orders_pd)
        return self.op_result

    def __sub__(self, other):
        """ - 操作符的重载，计算两个交易集的差集"""
        self.op_result = difference_in_2orders(self.orders_pd, other.orders_pd, same_rule=self.same_rule)
        return self.op_result

    def __eq__(self, other):
        """ == 操作符的重载，计算两个交易集的是否相同"""
        return (self - other).empty and (other - self).empty

    def __gt__(self, other):
        """ > 操作符的重载，计算两个交易集的大小, 类被total_ordering装饰，可以支持lt等操作符"""
        unique_cnt = find_unique_group_symbol(self.orders_pd).shape[0]
        other_unique_cnt = find_unique_group_symbol(other.orders_pd).shape[0]
        return unique_cnt > other_unique_cnt


def union_in_2orders(orders_pd, other_orders_pd):
    """
    并集：分析因子或者参数问题时使用，debug策略问题时筛选出两个orders_pd中所有不同的交易，
    注意这里不认为在相同的交易日买入相同的股票，两笔交易就一样，这里只是两个orders_pd合并
    后使用drop_duplicates做了去除完全一样的order，即结果为并集：
    orders_pd | cmp_orders_pd或orders_pd.union(cmp_orders_pd)
    :param orders_pd: 回测结果生成的交易订单构成的pd.DataFrame对象
    :param other_orders_pd: 回测结果生成的交易订单构成的pd.DataFrame对象
    :return: orders_pd | cmp_orders_pd
    """
    orders_pd = orders_pd.append(other_orders_pd)
    orders_pd = orders_pd.drop_duplicates()
    return orders_pd


def _same_pd(order, other_orders_pd, same_rule):
    """
    根据same_rule的规则从orders_pd和other_orders_pd中返回相同的df

    :param order: orders_pd中的一行order记录数据
    :param other_orders_pd: 回测结果生成的交易订单构成的pd.DataFrame对象
    :param same_rule: order判断为是否相同使用的规则
    :return: 从orders_pd和other_orders_pd中返回相同的df
    """
    symbol = order.symbol
    buy_day = order['buy_date']
    buy_price = order['buy_price']

    sell_day = order['sell_date']
    sell_price = order['sell_price']

    if same_rule == EOrderSameRule.ORDER_SAME_BD:
        # 只根据买入时间和买入symbol确定是否相同，即认为在相同的交易日买入相同的股票，两笔交易就一样，忽略其它所有order中的因素
        same_pd = other_orders_pd[(other_orders_pd['symbol'] == symbol) & (other_orders_pd['buy_date'] == buy_day)]
    elif same_rule == EOrderSameRule.ORDER_SAME_BSD:
        # 根据买入时间，卖出时间和买入symbol确定是否相同
        same_pd = other_orders_pd[(other_orders_pd['symbol'] == symbol) & (other_orders_pd['buy_date'] == buy_day)
                                  & (other_orders_pd['sell_date'] == sell_day)]
    elif same_rule == EOrderSameRule.ORDER_SAME_BDP:
        # 根据买入时间，买入价格和买入symbol确定是否相同
        same_pd = other_orders_pd[(other_orders_pd['symbol'] == symbol) & (other_orders_pd['buy_date'] == buy_day)
                                  & (other_orders_pd['buy_price'] == buy_price)]
    elif same_rule == EOrderSameRule.ORDER_SAME_BSPD:
        # 根据买入时间，卖出时间, 买入价格和卖出价格和买入symbol确定是否相同
        same_pd = other_orders_pd[(other_orders_pd['symbol'] == symbol) & (other_orders_pd['buy_date'] == buy_day)
                                  & (other_orders_pd['sell_date'] == sell_day)
                                  & (other_orders_pd['buy_price'] == buy_price)
                                  & (other_orders_pd['sell_price'] == sell_price)]
    else:
        raise TypeError('same_rule type is {}!!'.format(same_rule))
    return same_pd


def intersection_in_2orders(orders_pd, other_orders_pd, same_rule=EOrderSameRule.ORDER_SAME_BSPD):
    """
    交集: 分析因子或者参数问题时使用，debug策略问题时筛选出两个orders_pd中相同的交易，
    即结果为交集：orders_pd & cmp_orders_pd或orders_pd.intersection(cmp_orders_pd)
    :param orders_pd: 回测结果生成的交易订单构成的pd.DataFrame对象
    :param other_orders_pd: 回测结果生成的交易订单构成的pd.DataFrame对象
    :param same_rule: order判断为是否相同使用的规则, 默认EOrderSameRule.ORDER_SAME_BSPD
                          即：order有相同的symbol和买入日期和相同的卖出日期和价格才认为是相同
    :return: orders_pd & cmp_orders_pd
    """
    def _intersection(order):
        same_pd = _same_pd(order, other_orders_pd, same_rule)
        if same_pd.empty:
            # 如果是空，说明不相交
            return False
        # 相交, intersection=1，是交集
        return True

    orders_pd['intersection'] = orders_pd.apply(_intersection, axis=1)
    return orders_pd[orders_pd['intersection'] == 1]


def difference_in_2orders(orders_pd, other_orders_pd, same_rule=EOrderSameRule.ORDER_SAME_BSPD):
    """
    差集: 分析因子或者参数问题时使用，debug策略问题时筛选出两个orders_pd的不同交易，
    注意返回的结果是存在orders_pd中的交易，但不在cmp_orders_pd中的交易，即结果
    为差集：orders_pd - cmp_orders_pd或orders_pd.difference(cmp_orders_pd)
    :param orders_pd: 回测结果生成的交易订单构成的pd.DataFrame对象
    :param other_orders_pd: 回测结果生成的交易订单构成的pd.DataFrame对象
    :param same_rule: order判断为是否相同使用的规则, 默认EOrderSameRule.ORDER_SAME_BSPD
                      即：order有相同的symbol和买入日期和相同的卖出日期和价格才认为是相同
    :return: orders_pd - cmp_orders_pd
    """

    def _difference(order):
        same_pd = _same_pd(order, other_orders_pd, same_rule)
        if same_pd.empty:
            # 没有相同的说明是差集
            return True
        # 有相同的说明不是差集
        return False

    orders_pd['difference'] = orders_pd.apply(_difference, axis=1)
    return orders_pd[orders_pd['difference'] == 1]


def find_unique_group_symbol(order_pd):
    """
    按照'buy_date', 'symbol'分组后，只筛选组里的第一个same_group.iloc[0]
    :param order_pd:
    :return:
    """

    def _find_unique_group_symbol(same_group):
        # 只筛选组里的第一个, 即同一个交易日，对一个股票的交易只保留一个order
        return same_group.iloc[0]

    # 按照'buy_date', 'symbol'分组后apply same_handle
    order_pds = order_pd.groupby(['buy_date', 'symbol']).apply(_find_unique_group_symbol)
    return order_pds


def find_unique_symbol(order_pd, same_rule=EOrderSameRule.ORDER_SAME_BSPD):
    """
    order_pd中如果一个buy_date对应的一个symbol有多条交易记录，过滤掉，
    注意如果在对应多条记录中保留一个，使用find_unique_group_symbol
    :param order_pd: 回测结果生成的交易订单构成的pd.DataFrame对象
    :param same_rule: order判断为是否相同使用的规则, 默认EOrderSameRule.ORDER_SAME_BSPD
                  即：order有相同的symbol和买入日期和相同的卖出日期和价格才认为是相同
    """

    def _find_unique_symbol(order):
        """根据order的symbol和buy_date在原始order_pd中进行复合条件筛选，结果same_pd如果只有1个就唯一，否则就是重复的"""
        same_pd = _same_pd(order, order_pd, same_rule)
        if same_pd.empty or same_pd.shape[0] == 1:
            return False
        # 同一天一个symbol有多条记录的一个也没留，都过滤
        return True

    same_mark = order_pd.apply(_find_unique_symbol, axis=1)
    return order_pd[same_mark == 0]


def trade_summary(orders, kl_pd, draw=False, show_info=True):
    """
    主要将AbuOrder对象序列转换为pd.DataFrame对象orders_pd，以及将
    交易单子时间序列转换交易行为顺序序列，绘制每笔交易的细节交易图，以及
    简单文字度量输出
    :param orders: AbuOrder对象序列
    :param kl_pd: 金融时间序列，pd.DataFrame对象
    :param draw: 是否可视化交易细节图示
    :param show_info: 是否输出交易文字信息
    """

    # AbuOrder对象序列转换为pd.DataFrame对象orders_pd
    orders_pd = ABuTradeExecute.make_orders_pd(orders, kl_pd)
    # 交易单子时间序列转换交易行为顺序序列
    action_pd = ABuTradeExecute.transform_action(orders_pd)

    summary = ''
    if draw:
        # 绘制每笔交易的细节交易图
        ABuTradeDrawer.plot_his_trade(orders, kl_pd)

    if show_info:
        # simple的意思是没有计算交易费用
        simple_profit = 'simple profit: {} \n'.format(ABuTradeExecute.calc_simple_profit(orders, kl_pd))
        summary += simple_profit

        # 每笔交易收益期望
        mean_win_profit = 'mean win profit {} \n'.format(np.mean(orders_pd[orders_pd.result == 1]['profit']))
        summary += mean_win_profit

        # 每笔交易亏损期望
        mean_loss_profit = 'mean loss profit {} \n'.format(np.mean(orders_pd[orders_pd.result == -1]['profit']))
        summary += mean_loss_profit

        # 盈利笔数
        win_cnt = 0 if len(orders_pd[orders_pd.result == 1].result.value_counts().values) <= 0 else \
            orders_pd[orders_pd.result == 1].result.value_counts().values[0]

        # 亏损笔数
        loss_cnt = 0 if len(orders_pd[orders_pd.result == -1].result.value_counts().values) <= 0 else \
            orders_pd[orders_pd.result == -1].result.value_counts().values[0]

        # 胜率
        win_rate = 'win rate ' + str('*@#')
        if win_cnt + loss_cnt > 0:
            win_rate = 'win rate: {}%'.format(float(win_cnt) / float(float(loss_cnt) + float(win_cnt)))
        summary += win_rate

    return orders_pd, action_pd, summary
