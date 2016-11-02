# -*- encoding:utf-8 -*-
"""

由于现在在写一本关于股票量化方面的书，会涉及到相关文章里的一些内容，
出版社方面不希望我现在开源全部代码，但是一定会开源，最晚等书出版发行
以后，现在只能开源文章中涉及的部分代码，整个系统的开源会稍后，请谅解
我其实也觉着有人能看你的代码就已经很给面子了，但是。。。再次抱歉！！

"""
import numpy as np
import matplotlib.pyplot as plt
import TLineAnalyse
from collections import namedtuple

__author__ = 'BBFamily'


def calc_mc_golden(kl_pd, mc_percent, loss_cnt):
    golden_tuple = namedtuple('golden', (
        'below200', 'below250', 'below300', 'below382', 'above618', 'above700', 'above800', 'above900', 'above950'))

    loss_percent = mc_percent[0: loss_cnt]
    win_percent = mc_percent[loss_cnt:]
    gg = sorted(calc_percent_golden(kl_pd, loss_percent, True)) + sorted(
        calc_percent_golden(kl_pd, win_percent, False))
    golden = golden_tuple(*gg)
    return golden


def calc_percent_golden(kl_pd, percents, want_min):
    dk = True if kl_pd.columns.tolist().count('close') > 0 else False
    uq_close = kl_pd.close if dk else kl_pd.price
    pts_dict = TLineAnalyse.find_percent_point(percents, uq_close)
    mum_func = np.minimum if want_min else np.maximum
    return [mum_func(*pts_dict[pt]) for pt in percents]


def calc_golden(kl_pd, show=True, only_be=False):
    dk = True if kl_pd.columns.tolist().count('close') > 0 else False
    uq_close = kl_pd.close if dk else kl_pd.price

    if not hasattr(kl_pd, 'name'):
        kl_pd.name = 'unknown'

    g_382, g_500, g_618 = TLineAnalyse.find_golden_point(kl_pd.index, uq_close)
    if show and not only_be:
        plt.axes([0.025, 0.025, 0.95, 0.95])
        plt.plot(uq_close) if dk else plt.plot(uq_close.values)
        plt.axhline(g_618, color='c')
        plt.axhline(g_500, color='r')
        plt.axhline(g_382, color='g')
        _ = plt.setp(plt.gca().get_xticklabels(), rotation=30)
        plt.legend([kl_pd.name, 'g618', 'g500', 'g382'])
        plt.title('mean golden')
        plt.show()

    gex_382, gex_500, gex_618 = TLineAnalyse.find_golden_point_ex(kl_pd.index, uq_close)
    if show and not only_be:
        plt.axes([0.025, 0.025, 0.95, 0.95])
        plt.plot(uq_close) if dk else plt.plot(uq_close.values)
        plt.axhline(gex_618, color='c')
        plt.axhline(gex_500, color='r')
        plt.axhline(gex_382, color='g')
        _ = plt.setp(plt.gca().get_xticklabels(), rotation=30)
        plt.legend([kl_pd.name, 'gex618', 'gex500', 'gex382'])
        plt.title('median golden')
        plt.show()

    above618 = np.maximum(g_618, gex_618)
    below618 = np.minimum(g_618, gex_618)
    above382 = np.maximum(g_382, gex_382)
    below382 = np.minimum(g_382, gex_382)

    percents = [0.20, 0.25, 0.30, 0.70, 0.80, 0.90, 0.95]
    # precents = np.linspace(0.0, 1.0, 0.05)
    pts_dict = TLineAnalyse.find_percent_point(percents, uq_close)

    # import pdb
    # pdb.set_trace()
    below200 = np.minimum(*pts_dict[0.20])
    below250 = np.minimum(*pts_dict[0.25])
    below300 = np.minimum(*pts_dict[0.30])

    above700 = np.maximum(*pts_dict[0.70])
    above800 = np.maximum(*pts_dict[0.80])
    above900 = np.maximum(*pts_dict[0.90])
    above950 = np.maximum(*pts_dict[0.95])

    if show:
        plt.axes([0.025, 0.025, 0.95, 0.95])
        plt.plot(uq_close) if dk else plt.plot(uq_close.values)

        plt.axhline(above950, lw=3.5, color='c')
        plt.axhline(above900, lw=3.0, color='y')
        plt.axhline(above800, lw=2.5, color='k')
        plt.axhline(above700, lw=2.5, color='m')

        plt.axhline(above618, lw=2, color='r')
        plt.axhline(below618, lw=1.5, color='r')
        plt.fill_between(kl_pd.index, above618, below618,
                         alpha=0.1, color="r")

        '''
            *************I AM HERE*************
        '''
        plt.axhline(above382, lw=1.5, color='g')
        plt.axhline(below382, lw=2, color='g')
        plt.fill_between(kl_pd.index, above382, below382,
                         alpha=0.1, color="g")

        plt.axhline(below300, lw=2.5, color='k')
        plt.axhline(below250, lw=3.0, color='y')
        plt.axhline(below200, lw=3.5, color='c')

        _ = plt.setp(plt.gca().get_xticklabels(), rotation=30)
        plt.legend([kl_pd.name, 'above950', 'above900', 'above800', 'above700', 'above618', 'below618',
                    'above382', 'below382', 'below300', 'below250', 'below200'], bbox_to_anchor=(1.05, 1), loc=2,
                   borderaxespad=0.)
        plt.title('between golden')
        plt.show()

    return namedtuple('golden', ['g382', 'gex382', 'g500', 'gex500', 'g618',
                                 'gex618', 'above618', 'below618', 'above382', 'below382',
                                 'above950', 'above900', 'above800', 'above700', 'below300', 'below250', 'below200'])(
        g_382, gex_382,
        g_500, gex_500, g_618, gex_618, above618, below618, above382, below382,
        above950, above900, above800, above700, below300, below250, below200)
