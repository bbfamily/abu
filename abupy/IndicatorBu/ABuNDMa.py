# -*- encoding:utf-8 -*-
"""
    移动平均线，Moving Average，简称MA，原本的意思是移动平均，由于我们将其制作成线形，所以一般称之为移动平均线，简称均线。
    它是将某一段时间的收盘价之和除以该周期。 比如日线MA5指5天内的收盘价除以5 。
    移动平均线是由著名的美国投资专家Joseph E.Granville（葛兰碧，又译为格兰威尔）于20世纪中期提出来的。
    均线理论是当今应用最普遍的技术指标之一，它帮助交易者确认现有趋势、判断将出现的趋势、发现过度延生即将反转的趋势
"""

from __future__ import division
from __future__ import absolute_import
from __future__ import print_function

from collections import Iterable

import matplotlib.pyplot as plt
import pandas as pd
from enum import Enum

from .ABuNDBase import plot_from_order, g_calc_type, ECalcType
from ..CoreBu.ABuPdHelper import pd_rolling_mean, pd_ewm_mean
from ..CoreBu.ABuFixes import six
from ..UtilBu.ABuDTUtil import catch_error

__author__ = '阿布'
__weixin__ = 'abu_quant'


class EMACalcType(Enum):
    """计算移动移动平均使用的方法"""
    """简单移动平均线"""
    E_MA_MA = 0
    """加权移动平均线"""
    E_MA_EMA = 1


# noinspection PyUnresolvedReferences
def _calc_ma_from_ta(prices, time_period=10, from_calc=EMACalcType.E_MA_MA):
    """
    使用talib计算ma，即透传talib.MA or talib.EMA计算结果
    :param prices: 收盘价格序列，pd.Series或者np.array
    :param time_period: 移动平均的N值，int
    :param from_calc: EMACalcType enum对象，移动移动平均使用的方法
    """

    import talib
    if isinstance(prices, pd.Series):
        prices = prices.values

    if from_calc == EMACalcType.E_MA_MA:
        ma = talib.MA(prices, timeperiod=time_period)
    else:
        ma = talib.EMA(prices, timeperiod=time_period)
    return ma


def _calc_ma_from_pd(prices, time_period=10, from_calc=EMACalcType.E_MA_MA):
    """
    通过pandas计算ma或者ema
    :param prices: 收盘价格序列，pd.Series或者np.array
    :param time_period: 移动平均的N值，int
    :param from_calc: EMACalcType enum对象，移动移动平均使用的方法
    """

    if isinstance(prices, pd.Series):
        prices = prices.values

    if from_calc == EMACalcType.E_MA_MA:
        ma = pd_rolling_mean(prices, window=time_period, min_periods=time_period)
    else:
        ma = pd_ewm_mean(prices, span=time_period, min_periods=time_period)
    return ma


def calc_ma_from_prices(prices, time_period=10, min_periods=None, from_calc=EMACalcType.E_MA_MA):
    """
    通过pandas计算ma或者ema, 添加min_periods参数
    :param prices: 收盘价格序列，pd.Series或者np.array
    :param time_period: 移动平均的N值，int
    :param min_periods: int，默认None则使用time_period
    :param from_calc: EMACalcType enum对象，移动移动平均使用的方法
    """

    if isinstance(prices, pd.Series):
        prices = prices.values

    min_periods = time_period if min_periods is None else min_periods
    if from_calc == EMACalcType.E_MA_MA:
        ma = pd_rolling_mean(prices, window=time_period, min_periods=min_periods)
    else:
        ma = pd_ewm_mean(prices, span=time_period, min_periods=min_periods)
    return ma


"""通过在ABuNDBase中尝试import talib来统一确定指标计算方式, 外部计算只应该使用calc_ma"""
calc_ma = _calc_ma_from_pd if g_calc_type == ECalcType.E_FROM_PD else _calc_ma_from_ta


def plot_ma_from_order(order, date_ext=120, **kwargs):
    """
    封装ABuNDBase中的plot_from_order与模块中绘制技术指标的函数，完成技术指标可视化及标注买入卖出点位
    :param order: AbuOrder对象转换的pd.DataFrame对象or pd.Series对象
    :param date_ext: int对象 eg. 如交易在2015-06-01执行，如date_ext＝120，择start向前推120天，end向后推120天
    :param kwargs: 绘制技术指标需要的其它关键字参数，time_period，from_calc, with_price，最终透传给plot_ma
    """
    return plot_from_order(plot_ma_from_klpd, order, date_ext, **kwargs)


def plot_ma_from_klpd(kl_pd, with_points=None, with_points_ext=None, **kwargs):
    """
    封装plot_ma，绘制收盘价格，以及多条移动均线
    :param kl_pd: 金融时间序列，pd.DataFrame对象
    :param with_points: 这里的常规用途是传入买入order, with_points=buy_index=pd.to_datetime(orders['buy_date']))
    :param with_points_ext: 这里的常规用途是传入卖出order, with_points_ext=sell_index=pd.to_datetime(orders['sell_date']))
    :param kwargs: 绘制技术指标需要的其它关键字参数，time_period，from_calc, with_price，最终透传给plot_ma
    """

    # 如果外部不设置均线，这里pop的default为[30, 60, 90]
    time_period = kwargs.pop('time_period', [30, 60, 90])
    plot_ma(kl_pd.close, kl_pd.index, time_period, with_points=with_points,
            with_points_ext=with_points_ext)


def plot_ma(prices, kl_index, time_period, from_calc=EMACalcType.E_MA_MA,
            with_points=None, with_points_ext=None, with_price=True):
    """
    一个画布上，绘制价格曲线以及多条移动平均线，如果有with_points点位标注，使用竖线标注
    :param prices: 收盘价格序列，pd.Series或者np.array
    :param kl_index: pd.Index时间序列
    :param time_period: 注意是Iterable类型，需要可迭代对象，即使一个元素也要使用如[10]包裹
    :param from_calc: EMACalcType enum对象，默认使用简单移动平均线
    :param with_points: 这里的常规用途是传入买入order, with_points=buy_index=pd.to_datetime(orders['buy_date']))
    :param with_points_ext: 这里的常规用途是传入卖出order, with_points_ext=sell_index=pd.to_datetime(orders['sell_date']))
    :param with_price:  将价格一起绘制
    :return:
    """
    # TODO Iterable和six.string_types的判断抽出来放在一个模块，做为Iterable的判断来使用
    if not isinstance(time_period, Iterable) or isinstance(time_period, six.string_types):
        raise TypeError('MA CALC time_period MUST PASS Iterable!!!')

    calc_type_func = calc_ma
    # 迭代计算多条移动均线，使用from_calc使用的方法计算
    ma_array = [calc_type_func(prices, period, from_calc) for period in time_period]

    plt.figure(figsize=[14, 7])

    for ind, ma in enumerate(ma_array):
        # ind的作用是索引在原始time_period中对应label需要的名称
        # noinspection PyUnresolvedReferences
        plt.plot(kl_index, ma, label='ma {}'.format(time_period[ind]))

    if with_price:
        plt.plot(kl_index, prices, label='prices')

    @catch_error(return_val=None, log=False)
    def plot_with_point(points, co, cc):
        """
        点位使用圆点＋竖线进行标注
        :param points: 点位坐标序列
        :param co: 点颜色 eg. 'go' 'ro'
        :param cc: markeredgecolor和竖线axvline颜色 eg. 'green' 'red'
        """
        v_index_num = kl_index.tolist().index(points)
        # 如果有ma线，y点做目标画在第一根ma线上否则画在价格上面
        y_array = ma_array[0] if len(ma_array) > 0 else prices
        plt.plot(points, y_array[v_index_num], co, markersize=12, markeredgewidth=3.0,
                 markerfacecolor='None', markeredgecolor=cc)
        plt.axvline(points, color=cc)

    # with_points和with_points_ext的点位使用竖线标注
    if with_points is not None:
        # plt.axvline(with_points, color='green', linestyle='--')
        plot_with_point(with_points, 'go', 'green')

    if with_points_ext is not None:
        # plt.axvline(with_points_ext, color='red')
        plot_with_point(with_points_ext, 'ro', 'red')

    plt.grid(True)
    plt.legend(loc='best')
    plt.show()
