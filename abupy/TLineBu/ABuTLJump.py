# -*- encoding:utf-8 -*-
"""
    跳空缺口模块
"""

from __future__ import print_function
from __future__ import absolute_import
from __future__ import division

import itertools
from collections import namedtuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from ..MarketBu import ABuMarketDrawing
from ..CoreBu.ABuPdHelper import pd_resample
from ..UtilBu.ABuDateUtil import fmt_date
from ..UtilBu.ABuDTUtil import plt_show

__author__ = '阿布'
__weixin__ = 'abu_quant'

"""预备颜色序列集，超出序列数量应使用itertools.cycle循环绘制"""
K_PLT_MAP_STYLE = [
    'b', 'c', 'g', 'k', 'm', 'r', 'y', 'w']


def calc_jump(kl_pd, jump_diff_factor=1, show=True):
    """
    通过对比交易日当月的成交量，和当月的振幅来确定交易日当日的跳空阀值，
    分别组装跳空方向，跳空能量，跳空距离等数据进入pd.DataFrame对象返回
    :param kl_pd: 金融时间序列，pd.DataFrame对象
    :param jump_diff_factor: 参数通过设置jump_diff_factor来调节跳空阀值的大小，默认jump_diff_factor＝1
    :param show: 是否对结果跳空点进行可视化
    :return: pd.DataFrame对象
    """
    # 由于过程会修改金融时间序列，所以先copy一个再处理
    kl_pd = kl_pd.copy()
    # 将日change取abs变成日振幅保存在kl_pd新列abs_pct_change
    kl_pd['abs_pct_change'] = np.abs(kl_pd['p_change'])
    # 日振幅取平均做为第一层判断是否达成跳空的条件，即跳空最起码要振幅超过日振幅平均值
    change_ratio_min = kl_pd['abs_pct_change'].mean()

    # 提取月振幅volume_mean
    # TODO 做为参数可修改21d
    change_mean = pd_resample(kl_pd.abs_pct_change, '21D', how='mean')
    """
        eg: change_mean形如
        2014-07-23    0.7940
        2014-08-13    0.6536
        2014-09-03    0.8120
        2014-09-24    1.2673
        2014-10-15    1.1007
                       ...
        2016-04-13    1.2080
        2016-05-04    0.9093
        2016-05-25    0.6208
        2016-06-15    1.1831
        2016-07-06    0.6693
    """
    # 提取月成交量均值volume_mean
    volume_mean = pd_resample(kl_pd.volume, '21D', how='mean')
    """
        eg：volume_mean形如
        2014-07-23    1350679
        2014-08-13    1256093
        2014-09-03    1593358
        2014-09-24    1816544
        2014-10-15    2362897
                       ...
        2016-04-13    2341972
        2016-05-04    1633200
        2016-05-25    1372525
        2016-06-15    2071612
        2016-07-06    1136278
    """
    # 使用使用kl_pd没有resample之前的index和change_mean进行loc操作，为了把没有的index都变成nan
    change_mean = change_mean.loc[kl_pd.index]
    # 有nan之后开始填充nan
    change_mean.fillna(method='pad', inplace=True)
    # bfill再来一遍只是为了填充最前面的nan
    change_mean.fillna(method='bfill', inplace=True)
    """
        loc以及填充nan后change_mean形如：change_mean
        2014-07-23    0.7940
        2014-07-24    0.7940
        2014-07-25    0.7940
        2014-07-28    0.7940
        2014-07-29    0.7940
        2014-07-30    0.7940
        2014-07-31    0.7940
        2014-08-01    0.7940
        2014-08-04    0.7940
        2014-08-05    0.7940
                       ...
        2016-07-13    0.6693
        2016-07-14    0.6693
        2016-07-15    0.6693
        2016-07-18    0.6693
        2016-07-19    0.6693
        2016-07-20    0.6693
        2016-07-21    0.6693
        2016-07-22    0.6693
        2016-07-25    0.6693
        2016-07-26    0.6693
    """
    # 使用使用kl_pd没有resample之前的index和change_mean进行loc操作，为了把没有的index都变成nan
    volume_mean = volume_mean.loc[kl_pd.index]
    # 有nan之后开始填充nan
    volume_mean.fillna(method='pad', inplace=True)
    # bfill再来一遍只是为了填充最前面的nan
    volume_mean.fillna(method='bfill', inplace=True)
    """
        loc以及填充nan后volume_mean形如：change_mean
        2014-07-23    1350679.0
        2014-07-24    1350679.0
        2014-07-25    1350679.0
        2014-07-28    1350679.0
        2014-07-29    1350679.0
        2014-07-30    1350679.0
        2014-07-31    1350679.0
        2014-08-01    1350679.0
        2014-08-04    1350679.0
        2014-08-05    1350679.0
                        ...
        2016-07-13    1136278.0
        2016-07-14    1136278.0
        2016-07-15    1136278.0
        2016-07-18    1136278.0
        2016-07-19    1136278.0
        2016-07-20    1136278.0
        2016-07-21    1136278.0
        2016-07-22    1136278.0
        2016-07-25    1136278.0
        2016-07-26    1136278.0
    """
    jump_pd = pd.DataFrame()

    # 迭代金融时间序列，即针对每一个交易日分析跳空
    for kl_index in np.arange(0, kl_pd.shape[0]):
        today = kl_pd.iloc[kl_index]
        if today.abs_pct_change <= change_ratio_min:
            # 第一层判断：跳空最起码要振幅超过日振幅平均值
            continue

        date = fmt_date(today.date)
        if today.volume <= volume_mean.loc[date]:
            # 第二层判断：跳空当日的成交量起码要超过当月平均值
            continue

        # 获取今天对应的月振幅, 做为今天判断是否跳空的价格阀值百分比
        jump_threshold = np.abs(change_mean.loc[date])
        if today.pre_close == 0 or jump_threshold == 0:
            # 只是为避免异常数据
            continue

        # 计算跳空距离阀值，即以昨天收盘为基数乘以跳空阀值比例除100得到和高开低收相同单位的价格阀值jump_diff
        # 参数通过设置jump_diff_factor来调节跳空阀值的大小，默认jump_diff_factor＝1
        jump_diff = today.pre_close * jump_threshold / 100 * jump_diff_factor
        # 第三层判断：根据向上向下跳空选择跳空计算
        if today.p_change > 0 and (today.low - today.pre_close) > jump_diff:
            # 注意向上跳空判断使用today.low，向上跳空 1
            today['jump'] = 1
            # 月振幅跳空阀值
            today['jump_threshold'] = jump_threshold
            # 跳空距离阀值
            today['jump_diff'] = jump_diff
            # 计算出跳空缺口强度
            today['jump_power'] = (today.low - today.pre_close) / jump_diff

            jump_pd = jump_pd.append(today)
        elif today.p_change < 0 and (today.pre_close - today.high) > jump_diff:
            # 注意向下跳空判断使用today.high，向下跳空 －1
            today['jump'] = -1
            # 月振幅跳空阀值
            today['jump_threshold'] = jump_threshold
            # 跳空距离阀值
            today['jump_diff'] = jump_diff
            # 计算出跳空缺口强度
            today['jump_power'] = (today.pre_close - today.high) / jump_diff
            jump_pd = jump_pd.append(today)

    if show:
        # 通过plot_candle_form_klpd可视化跳空缺口，通过view_indexs参数
        ABuMarketDrawing.plot_candle_form_klpd(kl_pd, view_indexs=jump_pd.index)
    return jump_pd


# noinspection PyClassHasNoInit
class AbuJumpTuple(namedtuple('AbuJumpTuple',
                              ('date',
                               'direction',
                               'power',
                               'price'))):
    __slots__ = ()

    def __repr__(self):
        return "date:{}, direction:{}, power:{}, price:{}".format(
            self.date,
            self.direction,
            self.power, self.price)


def calc_jump_line(kl_pd, power_threshold=2.0, jump_diff_factor=1, show=True):
    """
    通过calc_jump计算kl_pd金融时间序列周期内跳空方向，跳空能量，跳空距离，
    通过跳空能量jump.jump_power大于阀值的跳空点组成AbuJumpTuple对象，
    AbuJumpTuple对象有跳空日期date，跳空方向direction，跳空能量power，跳空基数价格price组成，
    可视化AbuJumpTuple对象序列jump_lines中所有的跳空点

    :param kl_pd: 金融时间序列，pd.DataFrame对象
    :param power_threshold: 筛选跳空的点阀值，float，默认2.0
    :param jump_diff_factor: 透传给calc_jump，参数通过设置jump_diff_factor来调节跳空阀值的大小
    :param show: 是否可视化AbuJumpTuple对象序列中所有的跳空点
    :return: AbuJumpTuple对象序列
    """

    # 通过calc_jump计算kl_pd金融时间序列周期内跳空方向，跳空能量，跳空距离pd.DataFrame对象jump_pd
    jump_pd = calc_jump(kl_pd, jump_diff_factor=jump_diff_factor, show=False)
    jump_lines = []
    for jump_index in np.arange(0, jump_pd.shape[0]):
        jump = jump_pd.iloc[jump_index]
        # 通过跳空能量jump.jump_power大于阀值的跳空点组成AbuJumpTuple对象
        if jump.jump_power > power_threshold:
            # AbuJumpTuple对象有跳空日期date，跳空方向direction，跳空能量power，跳空基数价格price组成
            jump_tuple = AbuJumpTuple(jump_pd.index[jump_index],
                                      jump.jump, jump.jump_power,
                                      jump.pre_close)
            jump_lines.append(jump_tuple)
    if show:
        _show_jump_line(kl_pd, jump_lines)
    return jump_lines


def _show_jump_line(kl_pd, jump_lines):
    """
    可视化AbuJumpTuple对象序列jump_lines中所有的跳空点
    :param kl_pd: 金融时间序列，pd.DataFrame对象
    :param jump_lines: AbuJumpTuple对象序列
    """
    with plt_show():
        plt.plot(kl_pd.close)
        # 迭代跳空点，通过itertools.cycle(K_PLT_MAP_STYLE)形成不同的颜色
        for jump_tuple, cs_color in zip(jump_lines, itertools.cycle(K_PLT_MAP_STYLE)):
            # 跳空点位对应的价格上面绘制横线，label标注跳空能量
            plt.axhline(jump_tuple.price, color=cs_color, label='power:' + str(jump_tuple.power))
            # 跳空描述：日期：up／down， 根据jump_tuple.direction跳空方向
            jump_desc = '{} : {}'.format(jump_tuple.date, ' up ' if jump_tuple.direction > 0 else ' down ')
            # 再把这个跳空时间点上画一个圆圈进行标示
            plt.plot(jump_tuple.date, jump_tuple.price, 'ro', markersize=12, markeredgewidth=(1.0 * jump_tuple.power),
                     markerfacecolor='None', markeredgecolor=cs_color, label=jump_desc)

        plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
        plt.title('jump lines')


def calc_jump_line_weight(kl_pd, sw=(0.5, 0.5), power_threshold=2.0, jump_diff_factor=1, show=True):
    """
    通过calc_jump计算kl_pd金融时间序列周期内跳空方向，跳空能量，跳空距离，
    把每一个跳空点都转换为AbuJumpTuple对象，与calc_jump_line不同点是
    计算时间跳空能量分两部分组成 非时间加权的跳空能量 ＋ 时间加权的跳空能量，
    参数sw控制占比，sw[0]：控制非时间加权的跳空能量所最终占比，sw[1]：控制时间加权的跳空能量所最终占比
    最终加权能量大于阀值能量的进行能量替换，加入到结果序列中返回
    :param kl_pd:  金融时间序列，pd.DataFrame对象
    :param sw: tuple对象，sw[0]：控制非时间加权的跳空能量所最终占比，sw[1]：控制时间加权的跳空能量所最终占比
    :param power_threshold: 筛选跳空的点阀值，float，默认2.0
    :param jump_diff_factor: 透传给calc_jump，参数通过设置jump_diff_factor来调节跳空阀值的大小
    :param show: 是否可视化AbuJumpTuple对象序列中所有的跳空点
    :return: AbuJumpTuple对象序列
    """

    # 通过calc_jump计算kl_pd金融时间序列周期内跳空方向，跳空能量，跳空距离pd.DataFrame对象jump_pd
    jump_pd = calc_jump(kl_pd, jump_diff_factor=jump_diff_factor, show=False)
    ws_jump_lines = []
    for jump_index in np.arange(0, jump_pd.shape[0]):
        jump = jump_pd.iloc[jump_index]
        # 把每一个跳空点都转换为AbuJumpTuple
        jump_tuple = AbuJumpTuple(jump_pd.index[jump_index],
                                  jump.jump, jump.jump_power,
                                  jump.pre_close)

        # 拿出跳空日的index，之后拿对应的跳空权重值使用
        weight_index = kl_pd.index.tolist().index(jump_tuple.date)
        # 线性加权0-1kl_pd.shape[0]个, 之后对应时间序列
        weights = np.linspace(0, 1, kl_pd.shape[0])
        """
        eg: weights 形如
        array([ 0.    ,  0.002 ,  0.004 ,  0.006 ,  0.008 ,  0.0099,  0.0119,
                0.0139,  0.0159,  0.0179,  0.0199,  0.0219,  0.0239,  0.0258,
                0.0278,  0.0298,  0.0318,  0.0338,  0.0358,  0.0378,  0.0398,
                0.0417,  0.0437,  0.0457,  0.0477,  0.0497,  0.0517,  0.0537,
                0.0557,  0.0577,  0.0596,  0.0616,  0.0636,  0.0656,  0.0676,
                ......
                0.4453,  0.4473,  0.4493,  0.4513,  0.4533,  0.4553,  0.4573,
                0.4592,  0.4612,  0.4632,  0.4652,  0.4672,  0.4692,  0.4712,
                0.4732,  0.4751,  0.4771,  0.4791,  0.4811,  0.4831,  0.4851,
                0.4871,  0.4891,  0.4911,  0.493 ,  0.495 ,  0.497 ,  0.499 ,
                ......
                0.7097,  0.7117,  0.7137,  0.7157,  0.7177,  0.7197,  0.7217,
                0.7237,  0.7256,  0.7276,  0.7296,  0.7316,  0.7336,  0.7356,
                0.7376,  0.7396,  0.7416,  0.7435,  0.7455,  0.7475,  0.7495,
                0.7515,  0.7535,  0.7555,  0.7575,  0.7594,  0.7614,  0.7634,
                ......
                0.9463,  0.9483,  0.9503,  0.9523,  0.9543,  0.9563,  0.9583,
                0.9602,  0.9622,  0.9642,  0.9662,  0.9682,  0.9702,  0.9722,
                0.9742,  0.9761,  0.9781,  0.9801,  0.9821,  0.9841,  0.9861,
                0.9881,  0.9901,  0.992 ,  0.994 ,  0.996 ,  0.998 ,  1.    ])
        """

        # 计算时间跳空能量分两部分组成：
        #   sw[0]：控制非时间加权的跳空能量所最终占比
        #   sw[1]：控制时间加权的跳空能量所最终占比
        w_power = (jump_tuple.power * sw[0]) + (weights[weight_index] * jump_tuple.power * sw[1])
        if w_power > power_threshold:
            # 最终加权能量大于阀值能量的进行能量替换，加入到ws_jump_lines中
            # noinspection PyProtectedMember
            jump_tuple = jump_tuple._replace(power=w_power)
            ws_jump_lines.append(jump_tuple)
    if show:
        _show_jump_line(kl_pd, ws_jump_lines)
    return ws_jump_lines
