# -*- encoding:utf-8 -*-
"""
    技术指标工具基础模块
"""

from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import logging

import pandas as pd
from enum import Enum

from ..UtilBu import ABuDateUtil

__author__ = '阿布'
__weixin__ = 'abu_quant'


class ECalcType(Enum):
    """
        技术指标技术方式类
    """
    """使用talib透传技术指标计算"""
    E_FROM_TA = 0
    """使用pandas等库实现技术指标计算"""
    E_FROM_PD = 1


# try:
#     # 不强制要求talib，全部局部引用
#     # noinspection PyUnresolvedReferences
#     import talib
#     g_calc_type = ECalcType.E_FROM_TA
# except ImportError:
#     # 没有安装talib，使用E_FROM_PD
#     g_calc_type = ECalcType.E_FROM_PD
"""彻底不用talib，完全都使用自己计算的指标结果"""
g_calc_type = ECalcType.E_FROM_PD


def plot_from_order(plot_nd_func, order, date_ext, **kwargs):
    """
    封装在技术指标上绘制交易order信号通用流程
    :param plot_nd_func: 绘制技术指标的具体实现函数，必须callable
    :param order: AbuOrder对象转换的pd.DataFrame对象or pd.Series对象
    :param date_ext: int对象 eg. 如交易在2015-06-01执行，如date_ext＝120，择start向前推120天，end向后推120天
    :param kwargs: plot_nd_func需要的其它关键字参数，直接透传给plot_nd_func
    """
    if not callable(plot_nd_func):
        # plot_nd_func必须是callable
        raise TypeError('plot_nd_func must callable!!')

    if not isinstance(order, (pd.DataFrame, pd.Series)) and order.shape[0] > 0:
        # order必须是pd.DataFrame对象or pd.Series对象 且 单子数量要 > 0
        raise TypeError('order must DataFrame here!!')

    is_df = isinstance(order, pd.DataFrame)

    if is_df and order.shape[0] == 1:
        # 如果是只有1行pd.DataFrame对象则变成pd.Series
        is_df = False
        # 通过iloc即变成pd.Series对象
        # noinspection PyUnresolvedReferences
        order = order.iloc[0]

    def plot_from_series(p_order):
        """
        根据交易的symbol信息买入，卖出时间，以及date_ext完成通过ABuSymbolPd.make_kl_df获取金融时间序列，
        在成功获取数据后使用plot_nd_func完成买入卖出信号绘制及对应的技术指标绘制
        :param p_order: AbuOrder对象转换的pd.Series对象
        """
        # 确定交易对象
        target_symbol = p_order['symbol']
        # 单子都必须有买入时间
        buy_index = pd.to_datetime(str(p_order['buy_date']))
        sell_index = None

        start = ABuDateUtil.fmt_date(p_order['buy_date'])
        # 通过date_ext确定start，即买人单子向前推date_ext天
        start = ABuDateUtil.begin_date(date_ext, date_str=start, fix=False)
        if p_order['sell_type'] != 'keep':
            sell_index = pd.to_datetime(str(p_order['sell_date']))
            # 如果有卖出，继续通过sell_date，date_ext确定end时间
            end = ABuDateUtil.fmt_date(p_order['sell_date'])
            # -date_ext 向前
            end = ABuDateUtil.begin_date(-date_ext, date_str=end, fix=False)
        else:
            end = None
        from ..MarketBu import ABuSymbolPd
        # 组织好参数，确定了请求范围后开始获取金融时间序列数据
        kl_pd = ABuSymbolPd.make_kl_df(target_symbol, start=start, end=end)
        if kl_pd is None or kl_pd.shape[0] == 0:
            logging.debug(target_symbol + ': has net error in data')
            return
        # 使用plot_nd_func完成买入卖出信号绘制及对应的技术指标绘制
        return plot_nd_func(kl_pd, with_points=buy_index, with_points_ext=sell_index, **kwargs)

    if not is_df:
        return plot_from_series(order)
    else:
        # 多个order, apply迭代执行plot_from_series
        order = order[order['result'] != 0]
        return order.apply(plot_from_series, axis=1)
