# -*- encoding:utf-8 -*-
"""

跳空缺口的筛选

由于现在在写一本关于股票量化方面的书，会涉及到相关文章里的一些内容，
出版社方面不希望我现在开源全部代码，但是一定会开源，最晚等书出版发行
以后，现在只能开源文章中涉及的部分代码，整个系统的开源会稍后，请谅解
我其实也觉着有人能看你的代码就已经很给面子了，但是。。。再次抱歉！！

"""
from __future__ import print_function

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

import itertools
from collections import namedtuple

import MarketDrawer


__author__ = 'BBFamily'

K_PLT_MAP_STYLE = [
    'b', 'c', 'g', 'k', 'm', 'r', 'y', 'w']


def calc_jump_line_weight(kl_pd, sw=(0.5, 0.5), show=True):
    """
        组成namedtuple jump array
        接口考虑跳空时间权重:使用线性时间权重分配sw[1]给予的权重
    """
    jump_pd = calc_jump(kl_pd, False)
    ws_jump_lines = []
    for jump_index in np.arange(0, jump_pd.shape[0]):
        jump = jump_pd.iloc[jump_index]

        jump_tuple = namedtuple('jump', ['date', 'direction', 'power', 'price'])(jump_pd.index[jump_index],
                                                                                 jump.jump, jump.jump_power,
                                                                                 jump.preClose)

        '''
            计算权重序列的index
        '''
        weight_index = kl_pd.index.tolist().index(jump_tuple.date)
        weights = np.linspace(0, 1, kl_pd.shape[0])

        w_power = (jump_tuple.power * sw[0]) + (weights[weight_index] * jump_tuple.power * sw[1])
        if w_power > 1.5:
            jump_tuple = jump_tuple._replace(power=w_power)
            ws_jump_lines.append(jump_tuple)
    if show:
        _show_jump_line(kl_pd, ws_jump_lines)
    return ws_jump_lines


def calc_jump_line(kl_pd, show=True):
    """
        跳口power > 2.0的组成namedtuple jump array
        接口不考虑跳空时间权重
    """
    jump_pd = calc_jump(kl_pd, False)
    jump_lines = []
    for jump_index in np.arange(0, jump_pd.shape[0]):
        jump = jump_pd.iloc[jump_index]
        if jump.jump_power > 2.0:
            jump_tuple = namedtuple('jump', ['date', 'direction', 'power', 'price'])(jump_pd.index[jump_index],
                                                                                     jump.jump, jump.jump_power,
                                                                                     jump.preClose)
            jump_lines.append(jump_tuple)

    if show:
        _show_jump_line(kl_pd, jump_lines)
    return jump_lines


def _show_jump_line(kl_pd, jump_lines):
    plt.plot(kl_pd.close)
    for jump_tuple, csColor in zip(jump_lines, itertools.cycle(K_PLT_MAP_STYLE)):
        plt.axhline(jump_tuple.price, color=csColor, label='power:' + str(jump_tuple.power))

        drt = ' up ' if jump_tuple.direction > 0 else ' down '
        jump_desc = str(jump_tuple.date) + drt
        plt.plot(jump_tuple.date, jump_tuple.price, 'ro', markersize=12, markeredgewidth=(1.0 * jump_tuple.power),
                 markerfacecolor='None', markeredgecolor=csColor, label=jump_desc)

    plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
    plt.title('jump lines')
    plt.show()


def calc_jump(kl_pd, show=True):
    """
        要使用np.abs(klPd['netChangeRatio'])去计算每个月的跳空阀值
    """
    kl_pd['abs_netChangeRatio'] = np.abs(kl_pd['netChangeRatio'])
    xd_mean = kl_pd.resample('21D').mean()
    net_change_ratio_mean = xd_mean['abs_netChangeRatio']
    volume_mean = xd_mean['volume']
    net_change_ratio_min = kl_pd['abs_netChangeRatio'].mean()
    """
        last_match_index只是为了提速
    """
    last_match_index = 0
    jump_pd = pd.DataFrame()
    for kl_index in np.arange(0, kl_pd.shape[0]):
        today = kl_pd.iloc[kl_index]

        if today.abs_netChangeRatio <= net_change_ratio_min:
            """
                只处理满足最小幅度的，也可提速
            """
            continue
        while net_change_ratio_mean.shape[0] > last_match_index \
                and kl_pd.index[kl_index] > net_change_ratio_mean.index[last_match_index]:
            last_match_index += 1

        if net_change_ratio_mean.shape[0] == last_match_index:
            """
                到最后了，倒回去一个，可优化代码
            """
            last_match_index -= 1

        if volume_mean[last_match_index] > today.volume:
            """
                首先量不达标的先排除了
            """
            continue

        jump_threshold = np.abs(net_change_ratio_mean[last_match_index])

        if today.preClose == 0 or jump_threshold == 0:
            continue

        # jump_threshold跳口的百分比, jump_diff需要调控的距离
        jump_diff = today.preClose * jump_threshold / 100
        if today.netChangeRatio > 0 and (today.low - today.preClose) > jump_diff:
            # if today.netChangeRatio > jump_threshold:
            """
                向上跳空 1
            """
            today['jump'] = 1
            today['jump_threshold'] = jump_threshold
            today['jump_diff'] = jump_diff
            """
                计算出跳空缺口强度
            """
            today['jump_power'] = (today.low - today.preClose) / jump_diff
            jump_pd = jump_pd.append(today)
        # elif (-today.netChangeRatio) > jump_threshold:
        elif today.netChangeRatio < 0 and (today.preClose - today.high) > jump_diff:
            """
                向下跳空 －1
            """
            today['jump'] = -1
            today['jump_threshold'] = jump_threshold
            today['jump_diff'] = jump_diff
            today['jump_power'] = (today.preClose - today.high) / jump_diff
            jump_pd = jump_pd.append(today)

    if show:
        MarketDrawer.plot_candle_form_klpd(kl_pd, view_indexs=jump_pd.index)
    return jump_pd
