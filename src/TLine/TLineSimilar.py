# -*- encoding:utf-8 -*-
"""

由于现在在写一本关于股票量化方面的书，会涉及到相关文章里的一些内容，
出版社方面不希望我现在开源全部代码，但是一定会开源，最晚等书出版发行
以后，现在只能开源文章中涉及的部分代码，整个系统的开源会稍后，请谅解
我其实也觉着有人能看你的代码就已经很给面子了，但是。。。再次抱歉！！

"""
from __future__ import division
from __future__ import print_function

import copy

import NpUtil
import SimilarHelper
import SymbolPd
import ZLog
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from statsmodels.tsa.stattools import coint

__author__ = 'BBFamily'


def get_pdlist(sc):
    k__s_i_m__t_a_s_k__c_n_t = 3
    if sc.stop > k__s_i_m__t_a_s_k__c_n_t:
        ValueError('sc=slice > K_SIM_TASK_CNT!')
    pd_list = SimilarHelper.pd_list(k__s_i_m__t_a_s_k__c_n_t)
    pd_list = pd_list[sc]
    return pd_list


def get_sum_rank(pd_list, symbol):
    rank_pd = pd.DataFrame()
    for corr_ind, corr in enumerate(pd_list):
        corr_ss = corr[symbol].rank(ascending=False, method='first')
        col_name = SimilarHelper.maping_task_name(corr_ind).split('.')[0]
        rank_pd = rank_pd.join(pd.DataFrame(corr_ss.values, index=corr_ss.index, columns=[col_name]), how='outer')
    '''
        TODO: 全是等权重的计算，如有需要可分配权重计算
    '''
    return rank_pd.sum(axis=1)


def coint_similar(symbol, sc=slice(0, 2), show=True):
    """
        TODO: 稳定后 slice赋值给变量，sc直接＝相对应的变量
    """
    pd_list = get_pdlist(sc)
    sum_rank = get_sum_rank(pd_list, symbol)

    rank_head = sum_rank.sort_values(ascending=True)[1:100]

    kl_pd = SymbolPd.make_kfold_pd(symbol, n_folds=1)
    mul_pd = SymbolPd.make_kfold_mulpd(rank_head.index.tolist(), n_folds=1)
    coint_dict = {}
    for ind, cmp_symbol in enumerate(rank_head.index):
        klpd_cmp = mul_pd[cmp_symbol]
        if klpd_cmp is None:
            continue
        _, pvalue, _ = coint(kl_pd.close, klpd_cmp.close)
        if pvalue < 0.08:
            """
                记录index为了发现取多少个sort_values(ascending=True)[1:100]能
                有良好的数据
            """
            coint_dict[cmp_symbol] = (pvalue, ind)
    p_value_sorted = sorted(zip(coint_dict.values(), coint_dict.keys()))

    cmp_cnt = np.minimum(len(p_value_sorted), 10)
    symbols = [item[1] for item in p_value_sorted[:cmp_cnt]]

    mul_pd_it = mul_pd.swapaxes('items', 'minor')
    sd = mul_pd_it.items.tolist()
    sd.remove('close')
    """
        为了得到三维面板中干净的close列
    """
    close_panel = mul_pd_it.drop(sd)
    close_panel_pd = close_panel.loc['close'][symbols]

    if show:
        close_panel_pd_regular = NpUtil.regular_std(close_panel_pd)
        close_panel_pd_regular.plot()
        plt.title('close panel pd regular')
        plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
        plt.show()

    close_panel_pd_cp = copy.deepcopy(close_panel_pd)

    for col in close_panel_pd_cp.columns:
        """
            做一个一摸一样的pd就是为了得到投票权重表便于运算
        """
        close_panel_pd_cp[col] = kl_pd.close
    regular_diff = NpUtil.regular_std(close_panel_pd_cp - close_panel_pd)

    if show:
        regular_diff.plot()
        plt.title('regular diff')
        plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
        plt.show()

    """
        类似投票机制，获取投票coint的差值
            distance_votes
        2015-07-27   -14.724491
        2015-07-28   -12.712066
        2015-07-29   -11.945266
        2015-07-30   -13.801350
        2015-07-31   -13.520431
        2015-08-03   -11.381343
        2015-08-04    -9.486645
        2015-08-05   -11.319338
        2015-08-06    -6.517725
        2015-08-07    -9.103014
        2015-08-10    -5.025694
        ......................
    """
    distance_votes = regular_diff.sum(axis=1)
    votes_std = distance_votes.std()
    votes_mean = distance_votes.mean()
    above = votes_mean + votes_std
    below = votes_mean - votes_std
    if show:
        close_regular = NpUtil.regular_std(kl_pd.close)
        close_regular = (close_regular * distance_votes.max() / 2)
        close_regular.plot()
        distance_votes.plot()

        plt.axhline(votes_mean, color='r')
        plt.axhline(above, color='c')
        plt.axhline(below, color='g')

        plt.title('coint distance votes')
        plt.legend(['close regular', 'distance votes', 'votes mean', 'dvotes above', 'dvotes below'],
                   bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
        plt.show()


def calc_similar_top(symbol, sc=slice(0, 2), show=True, show_cnt=10):
    """
    按照等权重的方式计算相关性排名, 显示对比中会按照look_max方式对齐数据
    sc=slice 采用那几种相关性
    :param symbol:
    :param sc:
    :param show:
    :param show_cnt: 显示的数量
    :return:
    """
    pd_list = get_pdlist(sc)

    sum_rank = get_sum_rank(pd_list, symbol)

    rank_head = sum_rank.sort_values(ascending=True)[1: show_cnt + 1]
    if show:
        mul_pd = SymbolPd.make_kfold_mulpd(rank_head.index, n_folds=1)
        klpd = SymbolPd.make_kfold_pd(symbol, n_folds=1)
        for cmp_symbol in rank_head.index:
            klpd_symbol_nrm, klpd_cmp_symbol_nrm = NpUtil.two_mean_list(klpd.close,
                                                                        mul_pd[cmp_symbol].close, type_look='look_max')

            klpd_symbol_nrm.plot()
            klpd_cmp_symbol_nrm.plot()
            plt.legend([symbol, cmp_symbol])
            plt.title('similar draw')
            plt.show()

    return rank_head


def calc_similar(symbol, cmp_symbol, sc=slice(0, 2), show=True):
    """
        sc: 使用几个维度相似性验证的选择切片
        默认使用：
                    E_CORE_TASK_CG_PEARS  = 0
                    E_CORE_TASK_CG_SPERM  = 1
        如只想使用SPERM sc=slice(1, 2)

        对比的股票在rank中的位置分量
    """
    pd_list = get_pdlist(sc)

    sum_rank = get_sum_rank(pd_list, symbol)

    cmp_rank = sum_rank.sort_values(ascending=True).index.tolist().index(cmp_symbol)
    rank_score = 1 - cmp_rank / sum_rank.shape[0]
    if show:
        ZLog.info(symbol + ' similar rank score' + cmp_symbol + ' :' + str(rank_score))

        mul_pd = SymbolPd.make_kfold_mulpd([symbol, cmp_symbol])

        klpd_symbol = SymbolPd.get_n_year(mul_pd[symbol], from_year=2)
        klpd_cmp_symbol = SymbolPd.get_n_year(mul_pd[cmp_symbol], from_year=2)
        """
            缩放到同一个数量级
        """
        kl_pd_symbol_nrm, klpd_cmp_symbol_nrm = NpUtil.two_mean_list(klpd_symbol.close,
                                                                     klpd_cmp_symbol.close, type_look='look_max')

        kl_pd_symbol_nrm.plot()
        klpd_cmp_symbol_nrm.plot()
        plt.legend([symbol, cmp_symbol])
        plt.title('similar draw')
        plt.show()

        distance = (kl_pd_symbol_nrm - klpd_cmp_symbol_nrm)
        distance_mean = distance.mean()
        distance_std = distance.std()
        above = distance_mean + distance_std
        below = distance_mean - distance_std
        distance.plot()
        plt.axhline(distance_mean, color='r', linestyle='--')
        plt.axhline(above, color='c')
        plt.axhline(below, color='g')
        plt.title('similar distance')
        plt.legend(['distance', 'distance_mean', 'distance above', 'distance below'],
                   bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
        plt.show()

    return rank_score
