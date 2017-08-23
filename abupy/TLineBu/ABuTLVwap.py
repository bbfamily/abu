# -*- encoding:utf-8 -*-
"""
    VWAP: Volume Weighted Average Price 成交量加权平均价

    非标准传统意义vwap计算，即非使用高频的分钟k线和量进行计算，只是套用概念计算
    日线级别的vwap
"""
from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

from enum import Enum

import numpy as np

from ..TLineBu.ABuTLine import AbuTLine


# TODO 重构时提到上级模块中
class EDayBarCalcType(Enum):
    """计算金融序列最终price使用的方法"""

    """高开低收的平均值"""
    OHLC_MEAN = 0
    """高开低收的中位数"""
    OHLC_MEDIAN = 1
    """高低的平均值"""
    HL_MEAN = 2
    """使用close价格"""
    CLOSE = 3


def calc_vwap(kl_pd, how=EDayBarCalcType.OHLC_MEAN, show=True):
    """
    非标准传统意义vwap计算，即非使用高频的分钟k线和量进行计算，只是套用概念计算
    日线级别的vwap，使用EDayBarCalcType确定日线级别上的最终点位vwap序列，即确定
    计算vwap使用的价格序列，需要注意成交量的数据准确度问题，且对异常成交量需要进行
    控制筛选
    :param kl_pd: 金融时间序列，pd.DataFrame对象
    :param how: EDayBarCalcType，计算确定日线级别上的最终点位价格序列使用的方法
    :param show: 是否可视化，可视化使用AbuTLine.show接口
    :return: 返回AbuTLine对象
    """
    if how == EDayBarCalcType.OHLC_MEAN:
        # 日k级别上高开低收的平均
        price = np.mean([kl_pd.high, kl_pd.open, kl_pd.close, kl_pd.low], axis=0)
    elif how == EDayBarCalcType.OHLC_MEDIAN:
        # 日k级别上高开低收的中位数
        price = np.median([kl_pd.high, kl_pd.open, kl_pd.close, kl_pd.low], axis=0)
    elif how == EDayBarCalcType.HL_MEAN:
        # 日k级别上高低的平均价格
        price = np.mean([kl_pd.high, kl_pd.low], axis=0)
    elif how == EDayBarCalcType.CLOSE:
        # 日k级别上直接使用收close
        price = kl_pd.close
    else:
        raise TypeError('calc_vwap_std how is error! how= {}'.format(how))

    # 使用得到价格序列做基础计算vwap
    vwap = (kl_pd.volume * price).sum() / kl_pd.volume.sum()
    # 主要目的就是通过vwap构造line, 这里设置了price为line，使用mean=vwap，详阅AbuTLine
    line = AbuTLine(price, 'vwap std', mean=vwap)
    if show:
        line.show()

    return line
