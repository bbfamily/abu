# -*- encoding:utf-8 -*-
"""
    相关系数，相似度可视化模块
"""
from __future__ import print_function
from __future__ import absolute_import
from __future__ import division

import itertools
import math

import matplotlib.pyplot as plt
import numpy as np

from ..MarketBu import ABuSymbolPd
from ..UtilBu import ABuScalerUtil

# noinspection PyUnresolvedReferences
from ..CoreBu.ABuFixes import xrange
from ..CoreBu import ABuEnv
from ..UtilBu.ABuDTUtil import plt_show

"""预备颜色序列集，超出序列数量应使用itertools.cycle循环绘制"""
K_PLT_MAP_STYLE = [
    'b', 'c', 'g', 'k', 'm', 'r', 'y', 'b--', 'c--', 'g--', 'k--']


def draw_show_close(sorted_ret, target_count, show_cnt):
    """
    通过多少个交易日参数target_count，计算出make_kl_df的参数n_folds，
    使用ABuScalerUtil.scaler_std将show_cnt个最相似的股票价格序列进行
    标准化在一个数量值范围内可视化
    :param sorted_ret: 可迭代序列，元素形如('usTSLA', 1.0), ('usSINA', 0.45565379371028253).....
    :param target_count: 需要请求多少个交易日数据，int
    :param show_cnt: 可视化top show_cnt相关个价格走势
    """
    if show_cnt is None and not isinstance(show_cnt, int):
        return
    # 规避sorted_ret长度不够的问题
    show_cnt = min(show_cnt, len(sorted_ret))
    if show_cnt <= 0:
        return

    with plt_show():
        # 循环K_PLT_MAP_STYLE颜色集的颜色，绘制各个金融时间序列
        for x, cs_color in zip(xrange(0, show_cnt), itertools.cycle(K_PLT_MAP_STYLE)):
            # 通过多少个交易日参数target_count，计算出要请求几年的数据n_folds
            n_folds = int(math.ceil(target_count / ABuEnv.g_market_trade_year))
            # sorted_ret[x] : ('usTSLA', 1.0) -> sorted_ret[x][0]: usTSLA
            # FIXME 暂时忽略一个bug如果请求可视化时使用的是start，end，方式那么这里可视化的时间段就不符合了，需要传递完整的信息
            df = ABuSymbolPd.make_kl_df(sorted_ret[x][0], n_folds=n_folds)
            # 支可视化close_array
            close_array = df['close']
            if target_count < len(close_array):
                # 再次确认时间范围
                close_array = close_array[:int(target_count)]

            cs_np = np.array(close_array, dtype=np.float)
            # 使用ABuScalerUtil.scaler_std将序列进行标准化在一个数量值范围内可视化
            plt.plot(ABuScalerUtil.scaler_std(cs_np), cs_color, label=sorted_ret[x][0])
        plt.legend(loc='best', bbox_to_anchor=(1.05, 1), borderaxespad=0.)


def draw_show_close_array(sorted_ret, cs_array):
    """
    绘制参数中cs_array序列个金融时间序列
    :param sorted_ret: 可迭代序列，元素形如('usTSLA', 1.0), ('usSINA', 0.45565379371028253).....
    :param cs_array: 可迭代的价格序列
    """
    # 循环K_PLT_MAP_STYLE颜色集的颜色，绘制各个金融时间序列
    for x, (cs_np, cs_color) in enumerate(zip(cs_array, itertools.cycle(K_PLT_MAP_STYLE))):
        # sorted_ret[x] : ('usTSLA', 1.0) -> sorted_ret[x][0]: usTSLA
        plt.plot(cs_np, cs_color, label=sorted_ret[x][0], bbox_to_anchor=(1.05, 1), loc=3, borderaxespad=0.)

    plt.legend(loc='best')
    plt.show()
