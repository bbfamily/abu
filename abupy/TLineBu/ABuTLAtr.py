# -*- encoding:utf-8 -*-
"""
    计算线atr模块
"""

from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from ..TLineBu.ABuTLine import AbuTLine
from ..CoreBu.ABuPdHelper import pd_rolling_std, pd_ewm_std

__author__ = '阿布'
__weixin__ = 'abu_quant'


def show_atr_std(kl_pd):
    """
    可视化atr移动平均std和加权移动平均std，
    注意会修改kl_pd，只做测试使用，内部未做copy处理，
    如不能改动，外部自copy操作，再传递进来
    :param kl_pd: 金融时间序列，pd.DataFrame对象
    """

    pre_atr21 = kl_pd['atr21'].shift(1)
    # noinspection PyTypeChecker
    kl_pd['atr_change'] = np.where(pre_atr21 == 0, 0, np.log(kl_pd['atr21'] / pre_atr21))
    kl_pd['mov_std'] = pd_rolling_std(kl_pd['atr_change'], window=20, center=False) * np.sqrt(20)
    kl_pd['std_ewm'] = pd_ewm_std(kl_pd['atr_change'], span=20, min_periods=20, adjust=True) * np.sqrt(20)
    kl_pd[['close', 'atr21', 'mov_std', 'std_ewm', 'atr_change']].plot(subplots=True, figsize=(16, 12), grid=True)
    plt.show()


def calc_atr_std(kl_pd, xd=21, ewm=True, show=True):
    """
    计算atr移动平均std或者加权移动平均std技术线，使用
    AbuTLine封装技术线实体，不会修改kl_pd，返回AbuTLine对象
    :param kl_pd: 金融时间序列，pd.DataFrame对象
    :param xd: 计算移动平均std或加权移动平均std使用的窗口参数，默认21
    :param ewm: 是否使用加权移动平均std计算
    :param show: 是否可视化，可视化使用AbuTLine.show接口
    :return: 返回AbuTLine对象
    """
    pre_atr21 = kl_pd['atr21'].shift(1)
    # noinspection PyTypeChecker
    atr_change = np.where(pre_atr21 == 0, 0, np.log(kl_pd['atr21'] / pre_atr21))

    if ewm:
        atr_roll_std = pd_ewm_std(atr_change, span=xd, min_periods=1, adjust=True) * np.sqrt(xd)
    else:
        atr_roll_std = pd_rolling_std(atr_change, window=xd, min_periods=1, center=False) * np.sqrt(xd)

    # min_periods=1还是会有两个nan，填了
    atr_roll_std = pd.Series(atr_roll_std).fillna(method='bfill')
    # 主要目的就是通过atr_roll_std构造line
    line = AbuTLine(atr_roll_std, 'atr std')
    if show:
        line.show()

    return line
