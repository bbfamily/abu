# -*- encoding:utf-8 -*-

"""
股价的标准差及其信赖区间，从而确定股价的波动范围及未来走势，利用波带显示股价的安全高低价位，因而也被称为布林带。
其上下限范围不固定，随股价的滚动而变化。布林指标和麦克指标MIKE一样同属路径指标，股价波动在上限和下限的区间之内，
这条带状区的宽窄，随着股价波动幅度的大小而变化，股价涨跌幅度加大时，带状区变宽，涨跌幅度狭小盘整时，带状区则变窄

计算公式
中轨线=N日的移动平均线
上轨线=中轨线+nb_dev * N日的移动标准差
下轨线=中轨线－nb_dev * N日的移动标准差
（nb_dev为参数，可根据股票的特性来做相应的调整，一般默为2）
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .ABuNDBase import plot_from_order, g_calc_type, ECalcType
from ..CoreBu.ABuPdHelper import pd_rolling_mean, pd_rolling_std

__author__ = '阿布'
__weixin__ = 'abu_quant'


# noinspection PyUnresolvedReferences
def _calc_boll_from_ta(prices, time_period=20, nb_dev=2):
    """
    使用talib计算boll, 即透传talib.BBANDS计算结果
    :param prices: 收盘价格序列，pd.Series或者np.array
    :param time_period: boll的N值默认值20，int
    :param nb_dev: boll的nb_dev值默认值2，int
    :return: tuple(upper, middle, lower)
    """
    import talib
    if isinstance(prices, pd.Series):
        prices = prices.values

    upper, middle, lower = talib.BBANDS(
        prices,
        timeperiod=time_period,
        nbdevup=nb_dev,
        nbdevdn=nb_dev,
        matype=0)

    return upper, middle, lower


def _calc_boll_from_pd(prices, time_period=20, nb_dev=2):
    """
    通过boll公式手动计算boll
    :param prices: 收盘价格序列，pd.Series或者np.array
    :param time_period: boll的N值默认值20，int
    :param nb_dev: boll的nb_dev值默认值2，int
    :return: tuple(upper, middle, lower)
    """
    # 中轨线 = N日的移动平均线
    middle = pd_rolling_mean(prices, window=time_period, min_periods=time_period)
    # N日的移动标准差
    n_std = pd_rolling_std(prices, window=20, center=False)
    # 上轨线=中轨线+nb_dev * N日的移动标准差
    upper = middle + nb_dev * n_std
    # 下轨线 = 中轨线－nb_dev * N日的移动标准差
    lower = middle - nb_dev * n_std

    # noinspection PyUnresolvedReferences
    return upper.values, middle.values, lower.values


"""通过在ABuNDBase中尝试import talib来统一确定指标计算方式, 外部计算只应该使用calc_boll"""
calc_boll = _calc_boll_from_pd if g_calc_type == ECalcType.E_FROM_PD else _calc_boll_from_ta


def plot_boll_from_klpd(kl_pd, with_points=None, with_points_ext=None, **kwargs):
    """
    封装plot_boll，绘制收盘价格，boll（upper, middle, lower）曲线
    :param kl_pd: 金融时间序列，pd.DataFrame对象
    :param with_points: 这里的常规用途是传入买入order, with_points=buy_index=pd.to_datetime(orders['buy_date']))
    :param with_points_ext: 这里的常规用途是传入卖出order, with_points_ext=sell_index=pd.to_datetime(orders['sell_date']))
    :param kwargs: 绘制技术指标需要的其它关键字参数，ime_period, nb_dev, 最终透传给plot_boll
    """
    plot_boll(kl_pd.close, kl_pd.index, with_points=with_points, with_points_ext=with_points_ext, **kwargs)


def plot_boll_from_order(order, date_ext=120, **kwargs):
    """
    封装ABuNDBase中的plot_from_order与模块中绘制技术指标的函数，完成技术指标可视化及标注买入卖出点位
    :param order: AbuOrder对象转换的pd.DataFrame对象or pd.Series对象
    :param date_ext: int对象 eg. 如交易在2015-06-01执行，如date_ext＝120，择start向前推120天，end向后推120天
    :param kwargs: 绘制技术指标需要的其它关键字参数，ime_period, nb_dev, 最终透传给plot_boll
    """
    return plot_from_order(plot_boll_from_klpd, order, date_ext, **kwargs)


def plot_boll(prices, kl_index, with_points=None, with_points_ext=None, time_period=20, nb_dev=2):
    """
    绘制收盘价格，以及对应的boll曲线，如果有with_points点位标注，使用竖线标注

    :param prices: 收盘价格序列，pd.Series或者np.array
    :param kl_index: pd.Index时间序列
    :param with_points: 这里的常规用途是传入买入order, with_points=buy_index=pd.to_datetime(orders['buy_date']))
    :param with_points_ext: 这里的常规用途是传入卖出order, with_points_ext=sell_index=pd.to_datetime(orders['sell_date']))
    :param time_period: 计算boll使用的n日参数，默认20
    :param nb_dev: 计算boll使用的nb_dev参数，默认2
    :return:
    """
    upper, middle, lower = np.array(calc_boll(prices, time_period, nb_dev))

    plt.figure(figsize=[14, 7])

    plt.plot(kl_index, prices, label='close price')
    plt.plot(kl_index, upper, label='upper')
    plt.plot(kl_index, middle, label='middle')
    plt.plot(kl_index, lower, label='lower')

    # with_points和with_points_ext的点位使用竖线标注
    if with_points is not None:
        plt.axvline(with_points, color='green', linestyle='--')

    if with_points_ext is not None:
        plt.axvline(with_points_ext, color='red')

    plt.grid(True)
    plt.legend(loc='best')
    plt.show()
