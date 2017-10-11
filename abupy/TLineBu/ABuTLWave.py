# -*- encoding:utf-8 -*-
"""
    量化波动程度模块
"""

from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from ..TLineBu.ABuTLine import AbuTLine
from ..CoreBu.ABuPdHelper import pd_rolling_std, pd_ewm_mean, pd_ewm_std, pd_resample
from ..UtilBu import ABuStatsUtil
from ..UtilBu.ABuDTUtil import plt_show


def show_wave_return(kl_pd):
    """
        可视化收益的移动平均std和加权移动平均std
        注意会修改kl_pd，只做测试使用，内部未做copy处理，
        如不能改动，外部自copy操作，再传递进来
        :param kl_pd: 金融时间序列，pd.DataFrame对象
    """

    pre_close = kl_pd['close'].shift(1)
    # noinspection PyTypeChecker
    kl_pd['return'] = np.where(pre_close == 0, 0, np.log(kl_pd['close'] / pre_close))
    kl_pd['mov_std'] = pd_rolling_std(kl_pd['return'], window=20, center=False) * np.sqrt(20)
    kl_pd['std_ewm'] = pd_ewm_std(kl_pd['return'], span=20, min_periods=20, adjust=True) * np.sqrt(20)
    kl_pd[['close', 'mov_std', 'std_ewm', 'return']].plot(subplots=True, figsize=(16, 12), grid=True)
    plt.show()


def calc_wave_std(kl_pd, xd=21, ewm=True, show=True):
    """
    计算收益的移动平均std或者加权移动平均std技术线，使用
    AbuTLine封装技术线实体，不会修改kl_pd，返回AbuTLine对象
    :param kl_pd: 金融时间序列，pd.DataFrame对象
    :param xd: 计算移动平均std或加权移动平均std使用的窗口参数，默认21
    :param ewm: 是否使用加权移动平均std计算
    :param show: 是否可视化，可视化使用AbuTLine.show接口
    :return: 返回AbuTLine对象
    """

    pre_close = kl_pd['close'].shift(1)
    # noinspection PyTypeChecker
    change = np.where(pre_close == 0, 0, np.log(kl_pd['close'] / pre_close))
    if ewm:
        roll_std = pd_ewm_std(change, span=xd, min_periods=1, adjust=True) * np.sqrt(xd)
    else:
        roll_std = pd_rolling_std(change, window=xd, min_periods=1, center=False) * np.sqrt(xd)

    # min_periods=1还是会有两个nan，填了
    roll_std = pd.Series(roll_std).fillna(method='bfill')
    # 主要目的就是通过roll_std构造AbuTLine对象line
    line = AbuTLine(roll_std, 'wave std')
    if show:
        line.show()

    return line


def calc_wave_abs(kl_pd, xd=21, show=True):
    """
    计算金融时间序列kl_pd在的绝对波动，通过参数xd对波动进行重采样
    在默认xd＝21情况下，变成了月震荡幅度，使用ABuStatsUtil.demean对
    月震荡幅度进行去均值操作后得到技术线demean_wave，AbuTLine包裹
    技术线返回
    :param kl_pd: 金融时间序列，pd.DataFrame对象
    :param xd: 对波动进行重采样的周期，单位天，int
    :param show: 是否可视化
    :return: 返回AbuTLine对象
    """
    # 不考虑正负，只考虑波动，np.abs(kl_pd['p_change'])
    abs_pct_change = np.abs(kl_pd['p_change'])
    xd_resample = '%dD' % xd
    # 通过pd_resample重采样，使用how=sum, 即默认xd＝21情况下，变成了月震荡幅度
    change_ratio_sum = pd_resample(abs_pct_change, xd_resample, how='sum')
    """
        eg: change_ratio_sum
            2014-07-24    37.13
            2014-08-14    39.33
            2014-09-04    25.16
            2014-09-25    27.53
            2014-10-16    27.78
                          ...
            2016-04-14    25.17
            2016-05-05    42.07
            2016-05-26    18.93
            2016-06-16    33.25
            2016-07-07    10.79
    """
    # 使用ABuStatsUtil.demean进行去均值操作
    demean_wave = ABuStatsUtil.demean(change_ratio_sum)
    """
        eg: demean_wave
        2014-07-24    -1.6303
        2014-08-14     0.5697
        2014-09-04   -13.6003
        2014-09-25   -11.2303
                       ...
        2016-05-05     3.3097
        2016-05-26   -19.8303
        2016-06-16    -5.5103
        2016-07-07   -27.9703
    """
    line = AbuTLine(demean_wave, 'demean sum change wave')
    if show:
        # 计算pd_resample how='mean'只是为了_show_wave里面显示价格曲线
        xd_mean_close = pd_resample(kl_pd.close, xd_resample, how='mean')
        # 这里不使用AbuTLine.show，因为需要绘制另一个对比line，价格均线xd_mean_close
        _show_wave(demean_wave, line.high, line.mean, line.low, xd_mean_close)
        # TODO AbuTLine中添加多条线的标准对比方法，左右双轴和数据变化方式
    return line


def calc_wave_weight_abs(kl_pd, xd=21, span=3, show=True):
    """
    计算金融时间序列kl_pd的绝对波动，通过参数xd对波动进行重采样
    在默认xd＝21情况下，变成了月震荡幅度，使用ABuStatsUtil.demean对
    月震荡幅度进行去均值操作后得到技术线demean_wave，与calc_wave_abs不同，
    使用squared  * np.sign(demean_wave)放大了wave，即大的愈加大，且
    继续对squared_wave进行时间加权得到技术线形成ewm_wave，AbuTLine包裹技术线返回
    :param kl_pd: 金融时间序列，pd.DataFrame对象
    :param xd: 对波动进行重采样的周期，单位天，int
    :param span: 对squared_wave进行时间加权的窗口参数，int
    :param show: 是否可视化
    :return: 返回AbuTLine对象
    """

    # 不考虑正负，只考虑波动，np.abs(kl_pd['p_change'])
    abs_pct_change = np.abs(kl_pd['p_change'])
    xd_resample = '%dD' % xd
    # 通过pd_resample重采样，使用how=sum, 即默认xd＝21情况下，变成了月震荡幅度
    change_ratio_sum = pd_resample(abs_pct_change, xd_resample, how='sum')
    # 使用ABuStatsUtil.demean进行去均值操作
    demean_wave = ABuStatsUtil.demean(change_ratio_sum)
    # 与calc_wave_abs不同，使用squared  * np.sign(demean_wave)放大了wave，即大的愈加大
    squared_wave = (demean_wave ** 2) * np.sign(demean_wave)
    # ewmd的span最后决定了一切， span默认值之对应xd默认值，xd变动, span也要变
    ewm_wave = pd_ewm_mean(squared_wave, span=span, min_periods=span, adjust=True)
    line = AbuTLine(ewm_wave, 'squared ewm wave')
    if show:
        # 计算pd_resample how='mean'只是为了_show_wave里面显示价格曲线
        xd_mean_close = pd_resample(kl_pd.close, xd_resample, how='mean')
        # 这里不使用AbuTLine.show，因为需要绘制另一个对比line，价格均线xd_mean_close
        _show_wave(ewm_wave, line.high, line.mean, line.low, xd_mean_close)
    return line


def _show_wave(wave, above, wave_mean, below, xd_mean_close):
    """
    calc_wave_abs和calc_wave_weight_abs形成技术线的可视化方法
    不使用AbuTLine.show，因为需要绘制另一个对比line，价格均线xd_mean_close，
    使用双坐标轴的方式进行可视化
    """
    with plt_show():
        fig, ax1 = plt.subplots()
        plt.plot(wave)
        plt.axhline(above, color='c')
        plt.axhline(wave_mean, color='r')
        plt.axhline(below, color='g')
        _ = plt.setp(plt.gca().get_xticklabels(), rotation=30)
        plt.legend(['wave', 'above', 'wave_mean', 'below'],
                   bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)

        # 采用左右两个y轴坐标显示
        # noinspection PyUnusedLocal
        ax2 = ax1.twinx()
        plt.plot(xd_mean_close, c='y')
        plt.plot(xd_mean_close, 'ro')
        plt.legend(['mean close'],
                   bbox_to_anchor=(1.05, 1), loc=3, borderaxespad=0.)
        # 当时间序列太长时使用将时间显示倾斜30度角
        _ = plt.setp(plt.gca().get_xticklabels(), rotation=30)
        plt.title('wave line')
