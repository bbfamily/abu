# coding=utf-8
"""
    相关系数具体计算功能实现模块
"""

from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

import numpy as np
import pandas as pd
import scipy.stats as stats

from enum import Enum

from ..UtilBu import ABuDTUtil
from ..CoreBu.ABuFixes import rankdata
from ..CoreBu.ABuPdHelper import pd_rolling_corr
# noinspection PyUnresolvedReferences
from ..CoreBu.ABuFixes import zip


class ECoreCorrType(Enum):
    """
        ECoreCorrType: 相关系数计算方法
    """
    """皮尔逊相关系数计算"""
    E_CORE_TYPE_PEARS = 'pears'
    """斯皮尔曼相关系数计算"""
    E_CORE_TYPE_SPERM = 'sperm'
    """基于PEARS使用序列＋－符号相关系数"""
    E_CORE_TYPE_SIGN = 'sign'
    """基于PEARS使用移动时间加权相关系数"""
    E_CORE_TYPE_ROLLING = 'rolling'

    @classmethod
    def task_cnt(cls):
        """ECoreCorrType暂时支持的相关计算方法个数"""
        return 4


"""加权移动相关系数计算默认使用60d"""
g_rolling_corr_window = 60


def corr_xy(x, y, similar_type=ECoreCorrType.E_CORE_TYPE_PEARS, **kwargs):
    """
    计算两个可迭代序列相关系数对外函数
    :param x: 可迭代序列
    :param y: 可迭代序列
    :param similar_type: ECoreCorrType, 默认值ECoreCorrType.E_CORE_TYPE_PEARS
    :return: x与y的相关系数返回值
    """
    if similar_type == ECoreCorrType.E_CORE_TYPE_SIGN:
        # 序列＋－符号相关系数, 使用np.sign取符号后，再np.corrcoef计算
        x = np.sign(x)
        y = np.sign(y)
        similar_type = ECoreCorrType.E_CORE_TYPE_PEARS

    # noinspection PyTypeChecker
    if np.all(x == x[0]) or np.all(y == y[0]):
        # 如果全序列唯一不能使用相关计算，使用相同的数和与总数的比例
        # noinspection PyUnresolvedReferences
        return (x == y).sum() / x.count()

    if similar_type == ECoreCorrType.E_CORE_TYPE_PEARS:
        # 皮尔逊相关系数计算
        return np.corrcoef(x, y)[0][1]
    elif similar_type == ECoreCorrType.E_CORE_TYPE_SPERM:
        # 斯皮尔曼相关系数计算, 使用自定义spearmanr，不计算p_value
        return spearmanr(x, y)[0][1]
    elif similar_type == ECoreCorrType.E_CORE_TYPE_ROLLING:
        # pop参数window，默认使用g_rolling_corr_window
        window = kwargs.pop('window', g_rolling_corr_window)

        # 加权时间需要可迭代序列是pd.Series
        if not isinstance(x, pd.Series):
            x = pd.Series(x)
        if not isinstance(y, pd.Series):
            y = pd.Series(y)
        return rolling_corr(x, y, window=window)


# noinspection PyUnresolvedReferences
def corr_matrix(df, similar_type=ECoreCorrType.E_CORE_TYPE_PEARS, **kwargs):
    """
    与corr_xy的区别主要是，非两两corr计算，输入参数除类别外，只有一个矩阵的输入，且输入必须为pd.DataFrame对象 or np.array
    :param df: pd.DataFrame or np.array, 之所以叫df，是因为在内部会统一转换为pd.DataFrame
    :param similar_type: ECoreCorrType, 默认值ECoreCorrType.E_CORE_TYPE_PEARS
    :return: pd.DataFrame对象
    """
    if isinstance(df, np.ndarray):
        # 把np.ndarray转DataFrame，便统一处理
        df = pd.DataFrame(df)

    if not isinstance(df, pd.DataFrame):
        raise TypeError('df must pd.DataFrame object!!!')

    # FIXME 这里不应该支持ECoreCorrType.E_CORE_TYPE_PEARS.value，只严格按照ECoreCorrType对象相等
    if similar_type == ECoreCorrType.E_CORE_TYPE_PEARS or similar_type == ECoreCorrType.E_CORE_TYPE_PEARS.value:
        # 皮尔逊相关系数计算
        corr = np.corrcoef(df.T)
    elif similar_type == ECoreCorrType.E_CORE_TYPE_SPERM or similar_type == ECoreCorrType.E_CORE_TYPE_SPERM.value:
        # 斯皮尔曼相关系数计算, 使用自定义spearmanr，不计算p_value
        corr = spearmanr(df)
    elif similar_type == ECoreCorrType.E_CORE_TYPE_SIGN or similar_type == ECoreCorrType.E_CORE_TYPE_SIGN.value:
        # 序列＋－符号相关系数, 使用np.sign取符号后，再np.corrcoef计算
        corr = np.corrcoef(np.sign(df.T))
    elif similar_type == ECoreCorrType.E_CORE_TYPE_ROLLING or similar_type == ECoreCorrType.E_CORE_TYPE_ROLLING.value:
        # pop参数window，默认使用g_rolling_corr_window
        window = kwargs.pop('window', g_rolling_corr_window)
        corr = rolling_corr(df, window=window)
    else:
        # 还是给个默认的corr计算np.corrcoef(df.T)
        corr = np.corrcoef(df.T)
    # 将计算结果的corr转换为pd.DataFrame对象，行和列索引都使用df.columns
    corr = pd.DataFrame(corr, index=df.columns, columns=df.columns)
    return corr


@ABuDTUtil.consume_time
def rolling_corr(df, ss=None, window=g_rolling_corr_window):
    """
    滑动窗口按时间权重计算相关系数

    :param df: pd.DataFrame对象或者pd.Series对象
    e.g.
                usA	    usAA	usAAC
    2015/7/27	0.76	-1.94	0.59
    2015/7/28	2.12	2.6	    1.3
    2015/7/29	-0.12	2.94	-1.34
    2015/7/30	1.41	-1.77	-4.04
    2015/7/31	-0.05	-1.1	1.39
    ......

    :param ss: pd.Series对象, ss的大小需要与df.shape[0]一样
    e.g.
                usA
    2015/7/27	0.76
    2015/7/28	2.12
    2015/7/29	-0.12
    2015/7/30	1.41
    2015/7/31	-0.05
    ......
    :param window: 窗口大小, 默认值g_rolling_corr_window，即60d
    :return: 当ss为None时返回df.shape[1]大小的相关系数二维方阵，否则，返回长度为df.shape[1]的相关系数一维数组
    """
    if window < 1 or window > df.shape[0]:
        raise TypeError('window out of index, must in [{},{}]'.format(1, df.shape[0]))
    rolling_window = list(zip(np.arange(df.shape[0] - window + 1), np.arange(window, df.shape[0] + 1)))
    """
        通过list(zip(np.arange(df.shape[0] - window + 1), np.arange(window, df.shape[0] + 1)))
        生成rolling_window，rolling_window形如下：
        [(0, 60), (1, 61), (2, 62), (3, 63), (4, 64), (5, 65), (6, 66), (7, 67), (8, 68),
        (9, 69), (10, 70), (11, 71), (12, 72), (13, 73), (14, 74), (15, 75), (16, 76),
        (17, 77), (18, 78), (19, 79), (20, 80), (21, 81), (22, 82), (23, 83), (24, 84),
         (25, 85), (26, 86), (27, 87), (28, 88),........]
    """
    weights = np.linspace(0, 1, len(rolling_window))
    # 随着时间的推移，过去的时间占得比重越来越少
    # noinspection PyUnresolvedReferences
    weights = weights / weights.sum()
    """
        weights即权重序列形如下所示：
        [0.      0.      0.      0.      0.      0.0001  0.0001  0.0001  0.0001
        0.0001  0.0001  0.0001  0.0001.........................................
        0.0044  0.0044  0.0044  0.0044  0.0044  0.0044  0.0044  0.0044  0.0045
        ........................................0.0045  0.0045  0.0045  0.0045]
    """
    corr = 0
    if ss is None:
        # 如果不使用pd_rolling_corr，需要copy，保证df == np.inf的会修改
        df = df.copy()
        df[df == np.inf] = 0
        # 迭代rolling_window下的时间窗口，使用np.corrcoef，比使用pd_rolling_corr效果高很多
        for (s, e) in rolling_window:
            # eg. rolling_window第一个即为np.corrcoef(df.iloc[0:60].T)
            window_corr = np.corrcoef(df.iloc[s:e].T)
            window_corr[np.isinf(window_corr) | np.isnan(window_corr)] = 0
            # 当前窗口下的相关系数乘以权重, window_corr * weights[s]为df.shape[1]大小的相关系数二维方阵
            corr += window_corr * weights[s]
    else:
        # 针对两个输入序列使用pd_rolling_corr
        window_corr = pd_rolling_corr(df, ss, window=window)
        window_corr.dropna(inplace=True, how='all')
        """
            pd_rolling_corr返回的window_corr形如, 即每一个子窗口corr:

            59     0.234309
            60     0.237821
            61     0.244905
            62     0.242731
            63     0.249227
            ...............
            499    0.154148
            500    0.138837
            501    0.133806
            502    0.135788
            503    0.138762
        """
        window_corr[window_corr == np.inf] = 0
        for ind in np.arange(window_corr.shape[0]):
            # 对应天的相关系数 ＊ 对应天的系统权重，长度为df.shape[1]的相关系数一维数组
            corr += window_corr.iloc[ind] * weights[ind]
    return corr


def spearmanr(a, b=None, axis=0, p_value=False):
    """
    如果需要计算p_value使用stats.spearmanr计算，否则使用rankdata配合使用np.apply_along_axis，
    进行spearmanr相关计算，因为计算p_value耗时
    :param a: 可迭代序列a
    :param b: 可迭代序列b
    :param axis: 系数计算作用轴方向
    :param p_value 是否需要计算p_value
    :return: p_value 是True是返回 scipy.cost_stats.SpearmanrResult
             p_value 是False 返回 np.array 的二维方阵
    """
    if p_value:
        # 需要计算p_value使用stats.spearmanr计算
        return stats.spearmanr(a=a, b=b, axis=axis)
    else:
        # 使用rankdata配合使用np.apply_along_axis
        a, outaxis = _chk_asarray(a, axis)
        ar = np.apply_along_axis(rankdata, outaxis, a)
        br = None
        if b is not None:
            b, axisout = _chk_asarray(b, axis)
            br = np.apply_along_axis(rankdata, axisout, b)
        # 返回 np.array 的二维方阵
        return np.corrcoef(ar, br, rowvar=outaxis)


def _chk_asarray(a, axis):
    """内部函数，为spearmanr下不需要计算p_value的情况下，为apply_along_axis转换数据"""
    if axis is None:
        a = np.ravel(a)
        outaxis = 0
    else:
        a = np.asarray(a)
        outaxis = axis

    if a.ndim == 0:
        a = np.atleast_1d(a)

    return a, outaxis
