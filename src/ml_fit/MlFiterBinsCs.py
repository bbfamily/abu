# -*- encoding:utf-8 -*-
"""
为了直观可视化制作qcut的bins点

由于现在在写一本关于股票量化方面的书，会涉及到相关文章里的一些内容，
出版社方面不希望我现在开源全部代码，但是一定会开源，最晚等书出版发行
以后，现在只能开源文章中涉及的部分代码，整个系统的开源会稍后，请谅解
我其实也觉着有人能看你的代码就已经很给面子了，但是。。。再次抱歉！！
"""
from __future__ import division
from __future__ import print_function

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import ZLog

__author__ = 'BBFamily'


def show_orders_hist(order_pd, s_list=None, q_default=10):

    if s_list is None:
        s_list = ['lowBkCnt', 'atr_std', 'jump_power', 'diff_days',
                  'wave_score1', 'wave_score2', 'wave_score3',
                  'deg_60WindowPd', 'deg_hisWindowPd', 'deg_windowPd']

    s_list = filter(lambda x: order_pd.columns.tolist().count(x) > 0, s_list)
    for sn in s_list:
        uq = len(np.unique(order_pd[sn]))
        if uq == 1:
            continue

        bins = 10
        bins = uq // 50 if uq // 50 > bins else bins
        order_pd[sn].hist(bins=bins)
        plt.show()

        try:
            cats = pd.qcut(order_pd[sn], q_default)
        except Exception:
            '''
                某一个数据超出q的数量导致无法分
            '''
            import pandas.core.algorithms as algos
            bins = algos.quantile(np.unique(order_pd[sn]), np.linspace(0, 1, q_default + 1))
            cats = pd.tools.tile._bins_to_cuts(order_pd[sn], bins, include_lowest=True)
            # ZLog.info(sn + ' qcut except use bins!')
        ZLog.info('{0} show hist and qcuts'.format(sn))
        ZLog.info(cats.value_counts())
