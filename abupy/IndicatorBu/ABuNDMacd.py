# -*- encoding:utf-8 -*-

"""
MACD

MACD称为指数平滑异动移动平均线，是从双指数移动平均线发展而来的，由快的加权移动均线（EMA12）减去慢的加权移动均线（EMA26）
得到DIF，再用DIF - (快线-慢线的9日加权移动均线DEA）得到MACD柱。MACD的意义和双移动平均线基本相同，即由快、慢均线的离散、
聚合表征当前的多空状态和股价可能的发展变化趋势，但阅读起来更方便。当MACD从负数转向正数，是买的信号。当MACD从正数转向负数，
是卖的信号。当MACD以大角度变化，表示快的移动平均线和慢的移动平均线的差距非常迅速的拉开，代表了一个市场大趋势的转变。

"""
from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .ABuNDBase import plot_from_order, g_calc_type, ECalcType
from ..UtilBu import ABuScalerUtil
from ..UtilBu.ABuDTUtil import catch_error
from ..CoreBu.ABuPdHelper import pd_ewm_mean

__author__ = '阿布'
__weixin__ = 'abu_quant'


# noinspection PyUnresolvedReferences
def _calc_macd_from_ta(price, fast_period=12, slow_period=26, signal_period=9):
    """
    使用talib计算macd, 即透传talib.MACD计算结果
    :param price: 收盘价格序列，pd.Series或者np.array
    :param fast_period: 快的加权移动均线线, 默认12，即EMA12
    :param slow_period: 慢的加权移动均线, 默认26，即EMA26
    :param signal_period: dif的指数移动平均线，默认9
    """

    import talib
    if isinstance(price, pd.Series):
        price = price.values

    dif, dea, bar = talib.MACD(price,
                               fastperiod=fast_period,
                               slowperiod=slow_period,
                               signalperiod=signal_period)
    return dif, dea, bar


def _calc_macd_from_pd(price, fast_period=12, slow_period=26, signal_period=9):
    """
    通过macd公式手动计算macd
    :param price: 收盘价格序列，pd.Series或者np.array
    :param fast_period: 快的加权移动均线线, 默认12，即EMA12
    :param slow_period: 慢的加权移动均线, 默认26，即EMA26
    :param signal_period: dif的指数移动平均线，默认9
    """

    if isinstance(price, pd.Series):
        price = price.values

    # 快的加权移动均线
    ewma_fast = pd_ewm_mean(price, span=fast_period)
    # 慢的加权移动均线
    ewma_slow = pd_ewm_mean(price, span=slow_period)
    # dif = 快线 - 慢线
    dif = ewma_fast - ewma_slow
    # dea = dif的9日加权移动均线
    dea = pd_ewm_mean(dif, span=signal_period)
    bar = (dif - dea)
    return dif, dea, bar


"""通过在ABuNDBase中尝试import talib来统一确定指标计算方式"""
calc_macd = _calc_macd_from_pd if g_calc_type == ECalcType.E_FROM_PD else _calc_macd_from_ta


def plot_macd_from_klpd(kl_pd, with_points=None, with_points_ext=None, **kwargs):
    """
    封装plot_macd，绘制收盘价格，macd（dif, dea, bar）曲线
    :param kl_pd: 金融时间序列，pd.DataFrame对象
    :param with_points: 这里的常规用途是传入买入order, with_points=buy_index=pd.to_datetime(orders['buy_date']))
    :param with_points_ext: 这里的常规用途是传入卖出order, with_points_ext=sell_index=pd.to_datetime(orders['sell_date']))
    :param kwargs: 绘制技术指标需要的其它关键字参数with_price, fast_period, slow_period, signal_period, 最终透传给plot_macd
    """
    plot_macd(kl_pd.close, kl_pd.index, with_points=with_points, with_points_ext=with_points_ext, **kwargs)


def plot_macd_from_order(order, date_ext=120, **kwargs):
    """
    封装ABuNDBase中的plot_from_order与模块中绘制技术指标的函数，完成技术指标可视化及标注买入卖出点位
    :param order: AbuOrder对象转换的pd.DataFrame对象or pd.Series对象
    :param date_ext: int对象 eg. 如交易在2015-06-01执行，如date_ext＝120，择start向前推120天，end向后推120天
    :param kwargs: 绘制技术指标需要的其它关键字参数with_price, fast_period, slow_period, signal_period, 最终透传给plot_macd
    """
    return plot_from_order(plot_macd_from_klpd, order, date_ext, **kwargs)


def plot_macd(prices, kl_index, with_points=None, with_points_ext=None,
              with_price=True,
              fast_period=12, slow_period=26, signal_period=9):
    """
    绘制收盘价格，以及对应的macd曲线，如果有with_points点位标注，使用竖线标注
    :param prices: 收盘价格序列，pd.Series或者np.array
    :param kl_index: pd.Index时间序列
    :param with_points: 这里的常规用途是传入买入order, with_points=buy_index=pd.to_datetime(orders['buy_date']))
    :param with_points_ext: 这里的常规用途是传入卖出order, with_points_ext=sell_index=pd.to_datetime(orders['sell_date']))
    :param with_price: 将价格一起绘制，但是价格要做数据标准化
    :param fast_period: 快的加权移动均线线, 默认12，即EMA12
    :param slow_period: 慢的加权移动均线, 默认26，即EMA26
    :param signal_period: dif的指数移动平均线，默认9
    :return:
    """
    dif, dea, bar = calc_macd(
        prices, fast_period=fast_period, slow_period=slow_period, signal_period=signal_period)

    plt.figure(figsize=[14, 7])
    plt.plot(kl_index, dif, label='macd dif')
    plt.plot(kl_index, dea, label='signal dea')

    if with_price:
        plt.plot(kl_index, ABuScalerUtil.scaler_std(prices), label='close price')

    # 绘制红绿macd的bar
    # noinspection PyTypeChecker
    bar_red = np.where(bar > 0, bar, 0)
    # noinspection PyTypeChecker
    bar_green = np.where(bar < 0, bar, 0)
    plt.bar(kl_index, bar_red, facecolor='red', label='hist bar')
    plt.bar(kl_index, bar_green, facecolor='green', label='hist bar')

    @catch_error(return_val=None, log=False)
    def plot_with_point(points, co, cc):
        """
        点位使用圆点＋竖线进行标注
        :param points: 点位坐标序列
        :param co: 点颜色 eg. 'go' 'ro'
        :param cc: markeredgecolor和竖线axvline颜色 eg. 'green' 'red'
        """
        v_index_num = kl_index.tolist().index(points)
        plt.plot(points, dif[v_index_num], co, markersize=12, markeredgewidth=1.5,
                 markerfacecolor='None', markeredgecolor=cc)
        plt.axvline(points, color=cc)

    # with_points和with_points_ext的点位使用圆点＋竖线标注，plot_with_point方法
    if with_points is not None:
        plot_with_point(with_points, 'go', 'green')

    if with_points_ext is not None:
        plot_with_point(with_points_ext, 'ro', 'red')

    plt.grid(True)
    plt.legend(loc='best')
    plt.show()
