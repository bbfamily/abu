# -*- encoding:utf-8 -*-

"""
相对强弱指数（RSI）是通过比较一段时期内的平均收盘涨数和平均收盘跌数来分析市场买沽盘的意向和实力，
从而作出未来市场的走势

计算方法：

具体计算实现可阅读代码中_calc_rsi_from_pd()的实现
1. 根据收盘价格计算价格变动可以使用diff()也可以使用pct_change()
2. 分别筛选gain交易日的价格变动序列gain，和loss交易日的价格变动序列loss
3. 分别计算gain和loss的N日移动平均
4. rs = gain_mean / loss_mean
5. rsi = 100 - 100 / (1 + rs)

"""
from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from .ABuNDBase import plot_from_order, g_calc_type, ECalcType
from ..UtilBu import ABuScalerUtil
from ..CoreBu.ABuPdHelper import pd_rolling_mean

__author__ = '阿布'
__weixin__ = 'abu_quant'

"""_calc_rsi_from_pd计算rs时使用gain，否则使用change"""
g_rsi_gain = True


# noinspection PyUnresolvedReferences
def _calc_rsi_from_ta(prices, time_period=14):
    """
    使用talib计算rsi, 即透传talib.RSI计算结果
    :param prices: 收盘价格序列，pd.Series或者np.array
    :param time_period: rsi的N日参数, 默认14
    """

    import talib
    if isinstance(prices, pd.Series):
        prices = prices.values
    rsi = talib.RSI(prices, timeperiod=time_period)
    return rsi


# noinspection PyTypeChecker
def _calc_rsi_from_pd(prices, time_period=14):
    """
    通过rsi公式手动计算rsi
    :param prices: 收盘价格序列，pd.Series或者np.array
    :param time_period: rsi的N日参数, 默认14
    """

    if not isinstance(prices, pd.Series):
        prices = pd.Series(prices)

    # 根据收盘价格计算价格变动可以使用diff()也可以使用pct_change()
    if g_rsi_gain:
        # 使用前后价格变动gain
        diff_price = prices.diff()
    else:
        # 使用前后价格变动change比例
        diff_price = prices.pct_change()
    diff_price[0] = 0

    # 分别筛选gain交易日的价格变动序列gain，和loss交易日的价格变动序列loss
    gain = np.where(diff_price > 0, diff_price, 0)
    loss = np.where(diff_price < 0, abs(diff_price), 0)
    # 分别计算gain和loss的N日移动平均
    gain_mean = pd_rolling_mean(gain, window=time_period)
    loss_mean = pd_rolling_mean(loss, window=time_period)
    # rsi = 100 - 100 / (1 +  gain_mean / loss_mean)
    rs = gain_mean / loss_mean
    rsi = 100 - 100 / (1 + rs)
    return rsi


"""通过在ABuNDBase中尝试import talib来统一确定指标计算方式, 外部计算只应该使用calc_rsi"""
calc_rsi = _calc_rsi_from_pd if g_calc_type == ECalcType.E_FROM_PD else _calc_rsi_from_ta


def plot_rsi_from_order(order, date_ext=120, **kwargs):
    """
    封装ABuNDBase中的plot_from_order与模块中绘制技术指标的函数，完成技术指标可视化及标注买入卖出点位
    :param order: AbuOrder对象转换的pd.DataFrame对象or pd.Series对象
    :param date_ext: int对象 eg. 如交易在2015-06-01执行，如date_ext＝120，择start向前推120天，end向后推120天
    :param kwargs: 绘制技术指标需要的其它关键字参数，time_period, 最终透传给plot_rsi
    """
    return plot_from_order(plot_rsi_from_klpd, order, date_ext, **kwargs)


def plot_rsi_from_klpd(kl_pd, with_points=None, with_points_ext=None, **kwargs):
    """
    封装plot_rsi，绘制收盘价格，rsi曲线
    :param kl_pd: 金融时间序列，pd.DataFrame对象
    :param with_points: 这里的常规用途是传入买入order, with_points=buy_index=pd.to_datetime(orders['buy_date']))
    :param with_points_ext: 这里的常规用途是传入卖出order, with_points_ext=sell_index=pd.to_datetime(orders['sell_date']))
    :param kwargs: 绘制技术指标需要的其它关键字参数，time_period, 最终透传给plot_rsi
    """

    plot_rsi(kl_pd.close, kl_pd.index, with_points=with_points, with_points_ext=with_points_ext,
             **kwargs)


def plot_rsi(prices, kl_index, with_points=None, with_points_ext=None, with_price=True, time_period=14):
    """
    绘制收盘价格，以及对应的macd曲线，如果有with_points点位标注，使用竖线标注
    :param prices: 收盘价格序列，pd.Series或者np.array
    :param kl_index: pd.Index时间序列
    :param with_points: 这里的常规用途是传入买入order, with_points=buy_index=pd.to_datetime(orders['buy_date']))
    :param with_points_ext: 这里的常规用途是传入卖出order, with_points_ext=sell_index=pd.to_datetime(orders['sell_date']))
    :param with_price: 将价格一起绘制，但两个曲线进行缩放到一个数值级别
    :param time_period: rsi的N日参数, 默认14
    :return:
    """
    # noinspection PyTypeChecker
    rsi = np.array(calc_rsi(prices, time_period))

    plt.figure(figsize=[16, 8])
    plt.axes([0.025, 0.025, 0.95, 0.95])

    if with_price:
        # 绘制在一个画布上, 将两个曲线进行缩放到一个数值级别
        matrix = ABuScalerUtil.scaler_matrix([rsi, prices])
        rsi, prices = matrix[matrix.columns[0]], matrix[matrix.columns[1]]
        plt.plot(kl_index, prices, label='close price')

    plt.plot(kl_index, rsi, label='rsi')

    # with_points和with_points_ext的点位使用竖线标注
    if with_points is not None:
        plt.axvline(with_points, color='green', linestyle='--')

    if with_points_ext is not None:
        plt.axvline(with_points_ext, color='red')

    plt.grid(True)
    plt.legend(loc='best')
    plt.show()
