# -*- encoding:utf-8 -*-
"""
    黄金分割及比例分割示例模块
"""

from __future__ import print_function
from __future__ import absolute_import
from __future__ import division

from collections import namedtuple

import matplotlib.pyplot as plt

from ..TLineBu import ABuTLExecute
from ..UtilBu.ABuDTUtil import plt_show

__author__ = '阿布'
__weixin__ = 'abu_quant'


def calc_golden(kl_pd, show=True):
    """
    只针对金融时间序列的收盘价格close序列，进行黄金分割及比例分割
    数值结果分析以及可视化
    :param kl_pd: 金融时间序列，pd.DataFrame对象
    :param show: 是否可视化黄金分割及比例分割结果
    :return: 黄金分割及比例分割结果组成的namedtuple数值对象
    """
    kl_close = kl_pd.close

    if not hasattr(kl_pd, 'name'):
        # 金融时间序列中如果有异常的没有name信息的补上一个unknown
        kl_pd.name = 'unknown'

    # 计算视觉黄金分割
    gd_382, gd_500, gd_618 = ABuTLExecute.find_golden_point(kl_pd.index, kl_close)
    # 计算统计黄金分割
    gex_382, gex_500, gex_618 = ABuTLExecute.find_golden_point_ex(kl_pd.index, kl_close)

    # below above 382, 618确定，即382，618上下底
    below618, above618 = ABuTLExecute.below_above_gen(gd_618, gex_618)
    below382, above382 = ABuTLExecute.below_above_gen(gd_382, gex_382)

    # 再次通过比例序列percents和find_percent_point寻找对应比例的位置字典pts_dict
    percents = [0.20, 0.25, 0.30, 0.70, 0.80, 0.90, 0.95]
    pts_dict = ABuTLExecute.find_percent_point(percents, kl_close)

    # 0.20, 0.25, 0.30只找最低的，即底部只要最低的
    below200, _ = ABuTLExecute.below_above_gen(*pts_dict[0.20])
    below250, _ = ABuTLExecute.below_above_gen(*pts_dict[0.25])
    below300, _ = ABuTLExecute.below_above_gen(*pts_dict[0.30])

    # 0.70, 0.80, 0.90, 0.95只找最高的，即顶部只要最高的
    _, above700 = ABuTLExecute.below_above_gen(*pts_dict[0.70])
    _, above800 = ABuTLExecute.below_above_gen(*pts_dict[0.80])
    _, above900 = ABuTLExecute.below_above_gen(*pts_dict[0.90])
    _, above950 = ABuTLExecute.below_above_gen(*pts_dict[0.95])

    if show:
        with plt_show():
            # 开始可视化黄金分割及比例分割结果
            plt.axes([0.025, 0.025, 0.95, 0.95])
            plt.plot(kl_close)

            # 0.70, 0.80, 0.90, 0.95，lw线条粗度递减
            plt.axhline(above950, lw=3.5, color='c')
            plt.axhline(above900, lw=3.0, color='y')
            plt.axhline(above800, lw=2.5, color='k')
            plt.axhline(above700, lw=2.5, color='m')

            # 中间层的618是带，有上下底
            plt.axhline(above618, lw=2, color='r')
            plt.axhline(below618, lw=1.5, color='r')
            plt.fill_between(kl_pd.index, above618, below618,
                             alpha=0.1, color="r")
            # 中间层的382是带，有上下底
            plt.axhline(above382, lw=1.5, color='g')
            plt.axhline(below382, lw=2, color='g')
            plt.fill_between(kl_pd.index, above382, below382,
                             alpha=0.1, color="g")

            # 0.20, 0.25, 0.30 lw线条粗度递曾
            plt.axhline(below300, lw=2.5, color='k')
            plt.axhline(below250, lw=3.0, color='y')
            plt.axhline(below200, lw=3.5, color='c')

            _ = plt.setp(plt.gca().get_xticklabels(), rotation=30)
            plt.legend([kl_pd.name, 'above950', 'above900', 'above800', 'above700', 'above618', 'below618',
                        'above382', 'below382', 'below300', 'below250', 'below200'], bbox_to_anchor=(1.05, 1), loc=2,
                       borderaxespad=0.)
            plt.title('between golden')

    return namedtuple('golden', ['g382', 'gex382', 'g500', 'gex500', 'g618',
                                 'gex618', 'above618', 'below618', 'above382', 'below382',
                                 'above950', 'above900', 'above800', 'above700', 'below300', 'below250', 'below200'])(
        gd_382, gex_382,
        gd_500, gex_500, gd_618, gex_618, above618, below618, above382, below382,
        above950, above900, above800, above700, below300, below250, below200)
