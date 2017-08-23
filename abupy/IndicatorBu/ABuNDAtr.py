# -*- encoding:utf-8 -*-
"""
ATR

ATR又称 Average true range平均真实波动范围，简称ATR指标，是由J.Welles Wilder 发明的，ATR指标主要是用来衡量市场波动的强烈度，
即为了显示市场变化率的指标。

计算方法：
1. TR=∣最高价-最低价∣，∣最高价-昨收∣，∣昨收-最低价∣中的最大值
2. 真实波幅（ATR）= MA(TR,N)（TR的N日简单移动平均）
3. 常用参数N设置为14日或者21日

"""
from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from ..CoreBu.ABuPdHelper import pd_ewm_mean
from ..UtilBu import ABuScalerUtil
from .ABuNDBase import plot_from_order, g_calc_type, ECalcType

__author__ = '阿布'
__weixin__ = 'abu_quant'


# noinspection PyUnresolvedReferences
def _calc_atr_from_ta(high, low, close, time_period=14):
    """
    使用talib计算atr，即透传talib.ATR计算结果
    :param high: 最高价格序列，pd.Series或者np.array
    :param low: 最低价格序列，pd.Series或者np.array
    :param close: 收盘价格序列，pd.Series或者np.array
    :param time_period: atr的N值默认值14，int
    :return: atr值序列，np.array对象
    """
    import talib
    if isinstance(high, pd.Series):
        high = high.values
    if isinstance(low, pd.Series):
        low = low.values
    if isinstance(close, pd.Series):
        close = close.values

    atr = talib.ATR(high, low, close, timeperiod=time_period)
    return atr


def _calc_atr_from_pd(high, low, close, time_period=14):
    """
    通过atr公式手动计算atr
    :param high: 最高价格序列，pd.Series或者np.array
    :param low: 最低价格序列，pd.Series或者np.array
    :param close: 收盘价格序列，pd.Series或者np.array
    :param time_period: atr的N值默认值14，int
    :return: atr值序列，np.array对象
    """
    if isinstance(close, pd.Series):
        # shift(1)构成昨天收盘价格序列
        pre_close = close.shift(1).values
    else:
        from scipy.ndimage.interpolation import shift
        # 也可以暂时转换为pd.Series进行shift
        pre_close = shift(close, 1)
    pre_close[0] = pre_close[1]

    if isinstance(high, pd.Series):
        high = high.values
    if isinstance(low, pd.Series):
        low = low.values

    # ∣最高价 - 最低价∣
    tr_hl = np.abs(high - low)
    # ∣最高价 - 昨收∣
    tr_hc = np.abs(high - pre_close)
    # ∣昨收 - 最低价∣
    tr_cl = np.abs(pre_close - low)
    # TR =∣最高价 - 最低价∣，∣最高价 - 昨收∣，∣昨收 - 最低价∣中的最大值
    tr = np.maximum(np.maximum(tr_hl, tr_hc), tr_cl)
    # （ATR）= MA(TR, N)（TR的N日简单移动平均）, 这里没有完全按照标准公式使用简单移动平均，使用了pd_ewm_mean，即加权移动平均
    atr = pd_ewm_mean(pd.Series(tr), span=time_period, min_periods=1)
    # 返回atr值序列，np.array对象
    return atr.values

"""通过在ABuNDBase中尝试import talib来统一确定指标计算方式"""
calc_atr = _calc_atr_from_pd if g_calc_type == ECalcType.E_FROM_PD else _calc_atr_from_ta


def atr14(high, low, close):
    """
    通过high, low, close计算atr14序列值
    :param high: 最高价格序列，pd.Series或者np.array
    :param low: 最低价格序列，pd.Series或者np.array
    :param close: 收盘价格序列，pd.Series或者np.array
    :return: atr值序列，np.array对象
    """
    atr = calc_atr(high, low, close, 14)
    return atr


def atr21(high, low, close):
    """
    通过high, low, close计算atr21序列值
    :param high: 最高价格序列，pd.Series或者np.array
    :param low: 最低价格序列，pd.Series或者np.array
    :param close: 收盘价格序列，pd.Series或者np.array
    :return: atr值序列，np.array对象
    """
    atr = calc_atr(high, low, close, 21)
    return atr


def atr14_min(high, low, close):
    """
    确定常数阀值时使用，通过high, low, close计算atr14序列值，返回计算结果atr14序列中的最小值
    :param high: 最高价格序列，pd.Series或者np.array
    :param low: 最低价格序列，pd.Series或者np.array
    :param close: 收盘价格序列，pd.Series或者np.array
    :return: atr值序列，atr14序列中的最小值，float
    """
    _atr14 = atr14(high, low, close)
    _atr14 = pd.Series(_atr14)
    _atr14.fillna(method='bfill', inplace=True)
    _atr14 = _atr14.min()
    return _atr14


def atr14_max(high, low, close):
    """
    确定常数阀值时使用，通过high, low, close计算atr14序列值，返回计算结果atr14序列中的最大值
    :param high: 最高价格序列，pd.Series或者np.array
    :param low: 最低价格序列，pd.Series或者np.array
    :param close: 收盘价格序列，pd.Series或者np.array
    :return: atr值序列，atr14序列中的最大值，float
    """
    _atr14 = atr14(high, low, close)
    _atr14 = pd.Series(_atr14)
    _atr14.fillna(method='bfill', inplace=True)
    _atr14 = _atr14.max()
    return _atr14


def atr21_min(high, low, close):
    """
    确定常数阀值时使用，通过high, low, close计算atr21序列值，返回计算结果atr21序列中的最小值
    :param high: 最高价格序列，pd.Series或者np.array
    :param low: 最低价格序列，pd.Series或者np.array
    :param close: 收盘价格序列，pd.Series或者np.array
    :return: atr值序列，atr21序列中的最小值，float
    """
    _atr21 = atr21(high, low, close)
    _atr21 = pd.Series(_atr21)
    _atr21.fillna(method='bfill', inplace=True)
    _atr21 = _atr21.min()
    return _atr21


def atr21_max(high, low, close):
    """
    确定常数阀值时使用，通过high, low, close计算atr21序列值，返回计算结果atr21序列中的最大值
    :param high: 最高价格序列，pd.Series或者np.array
    :param low: 最低价格序列，pd.Series或者np.array
    :param close: 收盘价格序列，pd.Series或者np.array
    :return: atr值序列，atr21序列中的最大值，float
    """
    _atr21 = atr21(high, low, close)
    _atr21 = pd.Series(_atr21)
    _atr21.fillna(method='bfill', inplace=True)
    _atr21 = _atr21.max()
    return _atr21


def plot_atr_from_klpd(kl_pd, with_points=None, with_points_ext=None, **kwargs):
    """
    封装plot_atr，绘制收盘价格，atr曲线
    :param kl_pd: 金融时间序列，pd.DataFrame对象
    :param with_points: 这里的常规用途是传入买入order, with_points=buy_index=pd.to_datetime(orders['buy_date']))
    :param with_points_ext: 这里的常规用途是传入卖出order, with_points_ext=sell_index=pd.to_datetime(orders['sell_date']))
    :param kwargs: 绘制技术指标需要的其它关键字参数，time_period, 最终透传给plot_atr
    """
    plot_atr(kl_pd.high, kl_pd.low, kl_pd.close, kl_pd.index,
             with_points=with_points, with_points_ext=with_points_ext, **kwargs)


def plot_atr_from_order(order, date_ext=120, **kwargs):
    """
    封装ABuNDBase中的plot_from_order与模块中绘制技术指标的函数，完成技术指标可视化及标注买入卖出点位
    :param order: AbuOrder对象转换的pd.DataFrame对象or pd.Series对象
    :param date_ext: int对象 eg. 如交易在2015-06-01执行，如date_ext＝120，择start向前推120天，end向后推120天
    :param kwargs: 绘制技术指标需要的其它关键字参数，time_period, 最终透传给plot_atr
    """
    return plot_from_order(plot_atr_from_klpd, order, date_ext, **kwargs)


def plot_atr(high, low, close, kl_index, with_points=None, with_points_ext=None, time_period=14):
    """
    分别在上下两个子画布上绘制收盘价格，以及对应的atr曲线，如果有with_points点位标注，
    则只画在一个画布上，且将两个曲线进行缩放到一个数值级别
    :param high: 最高价格序列，pd.Series或者np.array
    :param low: 最低价格序列，pd.Series或者np.array
    :param close: 收盘价格序列，pd.Series或者np.array
    :param kl_index: pd.Index时间序列
    :param with_points: 这里的常规用途是传入买入order, with_points=buy_index=pd.to_datetime(orders['buy_date']))
    :param with_points_ext: 这里的常规用途是传入卖出order, with_points_ext=sell_index=pd.to_datetime(orders['sell_date']))
    :param time_period: atr的N值默认值14，int
    """
    atr = calc_atr(high, low, close, time_period)

    plt.figure(figsize=(14, 7))
    if with_points is not None or with_points_ext is not None:
        # 如果需要标准买入卖出点，就绘制在一个画布上
        p1 = plt.subplot(111)
        p2 = p1
        # 绘制在一个画布上, 将两个曲线进行缩放到一个数值级别
        matrix = ABuScalerUtil.scaler_matrix([atr, close])
        atr, close = matrix[matrix.columns[0]], matrix[matrix.columns[1]]

        # with_points和with_points_ext的点位使用竖线标注
        if with_points is not None:
            p1.axvline(with_points, color='green', linestyle='--')

        if with_points_ext is not None:
            p1.axvline(with_points_ext, color='red')
    else:
        # 绘制在两个子画布上面
        p1 = plt.subplot(211)
        p2 = plt.subplot(212)

    p1.plot(kl_index, close, "b-", label="close")
    p2.plot(kl_index, atr, "r-.", label="period={} atr".format(time_period), lw=2)
    p1.grid(True)
    p1.legend()
    p2.grid(True)
    p2.legend()

    plt.show()
