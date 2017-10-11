# -*- encoding:utf-8 -*-
"""
    相关系数上层技术线应用模块
"""

from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

import copy
import logging

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from statsmodels.tsa.stattools import coint

from ..CoreBu.ABuEnv import EMarketDataSplitMode
from ..CoreBu import ABuEnv
from ..MarketBu import ABuSymbolPd
from ..MarketBu.ABuSymbol import code_to_symbol
from ..SimilarBu import ABuSimilar
from ..SimilarBu import ECoreCorrType
from ..TradeBu.ABuBenchmark import AbuBenchmark
from ..UtilBu import ABuScalerUtil
from ..UtilBu.ABuProgress import do_clear_output
from ..SimilarBu.ABuSimilar import from_local
from ..TLineBu.ABuTLine import AbuTLine
from ..UtilBu.ABuDTUtil import plt_show

__author__ = '阿布'
__weixin__ = 'abu_quant'

g_top_corr_cnt = 300
g_coint_threshold = 0.38
g_coint_show_max = 10


def rank_corr_sum(corr_df_dict, symbol):
    """
    使用corr_df_dict各个相关性矩阵中symbol的rank值进行sum合并：
        eg：rank_pd
                           pears  sperm
            usBIDU           5.0    5.0
            usFB             8.0    8.0
            usGOOG           6.0    6.0
            usNOAH           1.0    1.0
            usSFUN           7.0    7.0
            usTSLA           9.0    9.0
            usVIPS           3.0    3.0
            usWUBA           4.0    4.0

        eg: rank_pd.sum(axis=1)
            usBIDU           10.0
            usFB             16.0
            usGOOG           12.0
            usNOAH            2.0
            usSFUN           14.0
            usTSLA           18.0
            usVIPS            6.0
            usWUBA            8.0
    :param corr_df_dict: 相关性矩阵组成的字典对象
    :param symbol: eg: 'usTSLA'
    :return: rank_pd.sum(axis=1)，pd.Series对象
    """
    rank_pd = pd.DataFrame()
    for corr_df_key in corr_df_dict:
        corr_df = corr_df_dict[corr_df_key]
        """
            eg：corr_df

                            usBIDU    usFB  usGOOG  usNOAH  usSFUN  usTSLA  usVIPS  usWUBA
            usBIDU         1.0000  0.3013  0.3690  0.4015  0.3680  0.3015  0.3706  0.4320
            usFB           0.3013  1.0000  0.6609  0.2746  0.1978  0.4080  0.2856  0.2438
            usGOOG         0.3690  0.6609  1.0000  0.3682  0.1821  0.3477  0.3040  0.2917
            usNOAH         0.4015  0.2746  0.3682  1.0000  0.3628  0.2178  0.4645  0.4488
            usSFUN         0.3680  0.1978  0.1821  0.3628  1.0000  0.2513  0.2843  0.4883
            usTSLA         0.3015  0.4080  0.3477  0.2178  0.2513  1.0000  0.2327  0.3340
            usVIPS         0.3706  0.2856  0.3040  0.4645  0.2843  0.2327  1.0000  0.4189
            usWUBA         0.4320  0.2438  0.2917  0.4488  0.4883  0.3340  0.4189  1.0000
        """

        if symbol not in corr_df:
            # TODO 在这里处理有点晚
            # print(corr_df.columns)
            # print(code_to_symbol(symbol).symbol_code)
            return None

        corr_rank = corr_df[symbol].rank(ascending=False, method='first')
        """
            eg： corr_rank
            usBIDU           5.0
            usFB             8.0
            usGOOG           6.0
            usNOAH           1.0
            usSFUN           7.0
            usTSLA           9.0
            usVIPS           3.0
            usWUBA           4.0
        """
        rank_pd = rank_pd.join(pd.DataFrame(corr_rank.values, index=corr_rank.index,
                                            columns=[corr_df_key]), how='outer')
        """
            eg：rank_pd
                           pears  sperm
            usBIDU           5.0    5.0
            usFB             8.0    8.0
            usGOOG           6.0    6.0
            usNOAH           1.0    1.0
            usSFUN           7.0    7.0
            usTSLA           9.0    9.0
            usVIPS           3.0    3.0
            usWUBA           4.0    4.0
        """

    # TODO: 全是等权重的计算，需要可分配权重计算的参数
    """
        eg: rank_pd.sum(axis=1)
        usBIDU           10.0
        usFB             16.0
        usGOOG           12.0
        usNOAH            2.0
        usSFUN           14.0
        usTSLA           18.0
        usVIPS            6.0
        usWUBA            8.0
    """
    # 清一下输出，太乱
    do_clear_output()
    return rank_pd.sum(axis=1)


@from_local
def calc_similar(symbol, cmp_symbol, sum_rank=None, corr_jobs=(ECoreCorrType.E_CORE_TYPE_PEARS,
                                                               ECoreCorrType.E_CORE_TYPE_SPERM), show=True):
    """
    使用ABuTLSimilar.calc_similar()函数计算返回的相关性数值是以目标股票所在市场为观察者，
    它不关心某一个股票具体相关性的数值的大小，calc_similar(a, b) 的工作流程如下：
    1.计算a与市场中所有股票的相关性
    2.将所有相关性进行rank排序
    3.查询股票b在rank序列中的位置，此位置值即为结果
    即ABuTLSimilar.calc_similar返回值由0至1，这样的好处是通过计算usTSLA与usAAPL在所有股票中的相似度水平，会更全局客观的体现相关性

    :param symbol: eg: 'usTSLA'
    :param cmp_symbol: 与symbol进行整体市场相关对比的symbol eg: 'usTSLA'
    :param sum_rank: 已经缓存了的sum_rank数据，
                        eg: sum_rank
                        usBIDU           10.0
                        usFB             16.0
                        usGOOG           12.0
                        usNOAH            2.0
                        usSFUN           14.0
                        usTSLA           18.0
                        usVIPS            6.0
                        usWUBA            8.0
    :param corr_jobs: 默认：corr_jobs=(ECoreCorrType.E_CORE_TYPE_PEARS, ECoreCorrType.E_CORE_TYPE_SPERM)
                      可以再添加更多jobs
                      eg：
                        corr_jobs=(ECoreCorrType.E_CORE_TYPE_PEARS, ECoreCorrType.E_CORE_TYPE_SPERM,
                                   ECoreCorrType.E_CORE_TYPE_SIGN, ECoreCorrType.E_CORE_TYPE_ROLLING)
                      注意每添加一种相关计算方法，耗时都会增加
    :param show: 是否进行可视化
    :return: rank_score (float: 0至1), sum_rank
    """

    cs_symbol = code_to_symbol(symbol)
    cs_cmp_symbol = code_to_symbol(cmp_symbol)

    if cs_symbol.market != cs_cmp_symbol.market:
        # 必须在同一个市场
        logging.info('{} and {} in different market!!!'.format(symbol, cmp_symbol))
        return

    symbol = cs_symbol.value
    cmp_symbol = cs_cmp_symbol.value
    if sum_rank is None:
        tmp_market = ABuEnv.g_market_target
        # 强制把市场设置为一样的
        ABuEnv.g_market_target = cs_symbol.market
        corr_df_dict = ABuSimilar.multi_corr_df(corr_jobs)
        # 恢复之前的市场
        ABuEnv.g_market_target = tmp_market
        """
            eg： corr_df_dict
            {'pears':
                            usBIDU    usFB  usGOOG  usNOAH  usSFUN  usTSLA  usVIPS  usWUBA
            usBIDU         1.0000  0.3013  0.3690  0.4015  0.3680  0.3015  0.3706  0.4320
            usFB           0.3013  1.0000  0.6609  0.2746  0.1978  0.4080  0.2856  0.2438
            usGOOG         0.3690  0.6609  1.0000  0.3682  0.1821  0.3477  0.3040  0.2917
            usNOAH         0.4015  0.2746  0.3682  1.0000  0.3628  0.2178  0.4645  0.4488
            usSFUN         0.3680  0.1978  0.1821  0.3628  1.0000  0.2513  0.2843  0.4883
            usTSLA         0.3015  0.4080  0.3477  0.2178  0.2513  1.0000  0.2327  0.3340
            usVIPS         0.3706  0.2856  0.3040  0.4645  0.2843  0.2327  1.0000  0.4189
            usWUBA         0.4320  0.2438  0.2917  0.4488  0.4883  0.3340  0.4189  1.0000

            'sperm':
                            usBIDU    usFB  usGOOG  usNOAH  usSFUN  usTSLA  usVIPS  usWUBA
            usBIDU         1.0000  0.3888  0.4549  0.4184  0.3747  0.3623  0.4333  0.4396
            usFB           0.3888  1.0000  0.7013  0.2927  0.2379  0.4200  0.3123  0.2216
            usGOOG         0.4549  0.7013  1.0000  0.3797  0.2413  0.3871  0.3922  0.3035
            usNOAH         0.4184  0.2927  0.3797  1.0000  0.3581  0.2066  0.4643  0.4382
            usSFUN         0.3747  0.2379  0.2413  0.3581  1.0000  0.2645  0.3890  0.4693
            usTSLA         0.3623  0.4200  0.3871  0.2066  0.2645  1.0000  0.2540  0.2801
            usVIPS         0.4333  0.3123  0.3922  0.4643  0.3890  0.2540  1.0000  0.4080
            usWUBA         0.4396  0.2216  0.3035  0.4382  0.4693  0.2801  0.4080  1.0000 }
        """
        sum_rank = rank_corr_sum(corr_df_dict, symbol)
        """
            eg: sum_rank
            usBIDU           10.0
            usFB             16.0
            usGOOG           12.0
            usNOAH            2.0
            usSFUN           14.0
            usTSLA           18.0
            usVIPS            6.0
            usWUBA            8.0
        """
        if sum_rank is None:
            logging.info('{} not in corr df!!!'.format(symbol))
            return None, None

    if cmp_symbol not in sum_rank.index:
        logging.info('{} not in sum_rank.index'.format(cmp_symbol))
        return None, None

    # sum_rank.sort_values(ascending=True)之后的结果index即是对比的排序结果值cmp_rank
    cmp_rank = sum_rank.sort_values(ascending=True).index.tolist().index(cmp_symbol)
    """
        eg: sum_rank.sort_values(ascending=True)
        usNOAH            2.0
        us_NYSE:.IXIC     4.0
        usVIPS            6.0
        usWUBA            8.0
        usBIDU           10.0
        usGOOG           12.0
        usSFUN           14.0
        usFB             16.0
        usTSLA           18.0
    """
    # 计算cmp_rank在整体sum_rank的比例位置rank_score
    rank_score = 1 - cmp_rank / sum_rank.shape[0]

    if show:
        log_func = logging.info if ABuEnv.g_is_ipython else print
        log_func('{} similar rank score {} : {}'.format(symbol, cmp_symbol, rank_score))

        # 通过make_kl_df序列的接口获取两个金融时间序列
        mul_pd = ABuSymbolPd.make_kl_df([symbol, cmp_symbol], n_folds=2)
        kl_pd = mul_pd[symbol]
        kl_pd_cmp = mul_pd[cmp_symbol]

        # 缩放到同一个数量级type_look='look_max'
        kl_pd, kl_pd_cmp = ABuScalerUtil.scaler_xy(kl_pd.close,
                                                   kl_pd_cmp.close,
                                                   type_look='look_max')

        with plt_show():
            # 首先可视化已经缩放到一个级别的两个金融序列
            kl_pd.plot()
            kl_pd_cmp.plot()
            plt.legend([symbol, cmp_symbol])
            plt.title('similar draw')

        distance = (kl_pd - kl_pd_cmp)
        # 通过distance构造技术线对象AbuTLine，可视化几个技术线
        line = AbuTLine(distance, line_name='{} distance {}'.format(symbol, cmp_symbol))
        line.show()
        # 可视化技术线拟合曲线及上下拟合通道曲线
        line.show_regress_trend_channel()
        # 可视化技术线'路程位移比'
        # line.show_shift_distance()
        # 对技术线阻力位和支撑位进行绘制
        # line.show_support_resistance_trend(show=False)

    return rank_score, sum_rank


@from_local
def calc_similar_top(symbol, sum_rank=None, corr_jobs=(ECoreCorrType.E_CORE_TYPE_PEARS,
                                                       ECoreCorrType.E_CORE_TYPE_SPERM), show=True, show_cnt=10):
    """
    使用corr_jobs种相关算法在env所在的市场中寻找与symbol最相关的show_cnt个，可视化
    show_cnt个
    :param symbol: eg: 'usTSLA'
    :param sum_rank: 已经缓存了的sum_rank数据，
                        eg: sum_rank
                        usBIDU           10.0
                        usFB             16.0
                        usGOOG           12.0
                        usNOAH            2.0
                        usSFUN           14.0
                        usTSLA           18.0
                        usVIPS            6.0
                        usWUBA            8.0
    :param corr_jobs: 默认：corr_jobs=(ECoreCorrType.E_CORE_TYPE_PEARS, ECoreCorrType.E_CORE_TYPE_SPERM)
                      可以再添加更多jobs
                      eg：
                        corr_jobs=(ECoreCorrType.E_CORE_TYPE_PEARS, ECoreCorrType.E_CORE_TYPE_SPERM,
                                   ECoreCorrType.E_CORE_TYPE_SIGN, ECoreCorrType.E_CORE_TYPE_ROLLING)
                      注意每添加一种相关计算方法，耗时都会增加
    :param show: 是否进行可视化
    :param show_cnt: 可视化最相关的top n的个数
    :return: 排序好的top 相关序列，pd.Series对象
                eg：
                    us_NYSE:.IXIC     4.0
                    usFB              6.0
                    usGOOG            8.0
                    usWUBA           11.0
                    usBIDU           11.0
                    usSFUN           14.0
                    usVIPS           16.0
                    usNOAH           18.0
    """
    cs = code_to_symbol(symbol)
    symbol = cs.value
    if sum_rank is None:
        # TODO 重复代码太多，提前头装饰器
        tmp_market = ABuEnv.g_market_target
        # 强制把市场设置为一样的
        ABuEnv.g_market_target = cs.market
        corr_df_dict = ABuSimilar.multi_corr_df(corr_jobs)
        # 恢复之前的市场
        ABuEnv.g_market_target = tmp_market
        """
            eg： corr_df_dict
            {'pears':
                            usBIDU    usFB  usGOOG  usNOAH  usSFUN  usTSLA  usVIPS  usWUBA
            usBIDU         1.0000  0.3013  0.3690  0.4015  0.3680  0.3015  0.3706  0.4320
            usFB           0.3013  1.0000  0.6609  0.2746  0.1978  0.4080  0.2856  0.2438
            usGOOG         0.3690  0.6609  1.0000  0.3682  0.1821  0.3477  0.3040  0.2917
            usNOAH         0.4015  0.2746  0.3682  1.0000  0.3628  0.2178  0.4645  0.4488
            usSFUN         0.3680  0.1978  0.1821  0.3628  1.0000  0.2513  0.2843  0.4883
            usTSLA         0.3015  0.4080  0.3477  0.2178  0.2513  1.0000  0.2327  0.3340
            usVIPS         0.3706  0.2856  0.3040  0.4645  0.2843  0.2327  1.0000  0.4189
            usWUBA         0.4320  0.2438  0.2917  0.4488  0.4883  0.3340  0.4189  1.0000

            'sperm':
                            usBIDU    usFB  usGOOG  usNOAH  usSFUN  usTSLA  usVIPS  usWUBA
            usBIDU         1.0000  0.3888  0.4549  0.4184  0.3747  0.3623  0.4333  0.4396
            usFB           0.3888  1.0000  0.7013  0.2927  0.2379  0.4200  0.3123  0.2216
            usGOOG         0.4549  0.7013  1.0000  0.3797  0.2413  0.3871  0.3922  0.3035
            usNOAH         0.4184  0.2927  0.3797  1.0000  0.3581  0.2066  0.4643  0.4382
            usSFUN         0.3747  0.2379  0.2413  0.3581  1.0000  0.2645  0.3890  0.4693
            usTSLA         0.3623  0.4200  0.3871  0.2066  0.2645  1.0000  0.2540  0.2801
            usVIPS         0.4333  0.3123  0.3922  0.4643  0.3890  0.2540  1.0000  0.4080
            usWUBA         0.4396  0.2216  0.3035  0.4382  0.4693  0.2801  0.4080  1.0000 }
        """
        sum_rank = rank_corr_sum(corr_df_dict, symbol)
        """
            eg: sum_rank
            usBIDU           10.0
            usFB             16.0
            usGOOG           12.0
            usNOAH            2.0
            usSFUN           14.0
            usTSLA           18.0
            usVIPS            6.0
            usWUBA            8.0
        """
        if sum_rank is None:
            logging.info('{} not in corr df!!!'.format(symbol))
            return

    show_cnt = sum_rank.shape[0] - 1 if show_cnt > sum_rank.shape[0] else show_cnt
    # sort_values后即为rank排序结果按照参数show_cnt个数进行返回，第一个是自身掠过
    rank_head = sum_rank.sort_values(ascending=True)[1: show_cnt + 1]
    if show:
        kl_pd = ABuSymbolPd.make_kl_df(symbol, n_folds=1)
        # 获取top show_cnt个symbol金融时间序列
        mul_pd = ABuSymbolPd.make_kl_df(rank_head.index, n_folds=1)

        for cmp_symbol in rank_head.index:
            # 迭代所有的收盘价格序列进行数据缩放
            kl_pd_close, kl_pd_cmp_close = ABuScalerUtil.scaler_xy(kl_pd.close,
                                                                   mul_pd[cmp_symbol].close,
                                                                   type_look='look_max')

            with plt_show():
                # 缩放后的数据进行可视化对比
                kl_pd_close.plot()
                kl_pd_cmp_close.plot()
                plt.legend([symbol, cmp_symbol])
                plt.title('similar draw')

    """
        eg：rank_head
        us_NYSE:.IXIC     4.0
        usFB              6.0
        usGOOG            8.0
        usWUBA           11.0
        usBIDU           11.0
        usSFUN           14.0
        usVIPS           16.0
        usNOAH           18.0
    """
    return rank_head


@from_local
def coint_similar(symbol, sum_rank=None, corr_jobs=(ECoreCorrType.E_CORE_TYPE_PEARS,
                                                    ECoreCorrType.E_CORE_TYPE_SPERM), show=True):
    """
    首先找到的是最相关的top个，从top n个最相关的再找协整，只考虑pvalue，因为已经是从top n个最相关的再找协整
    可视化整个过程

    :param symbol: eg: 'usTSLA'
    :param sum_rank: 已经缓存了的sum_rank数据，
                        eg: sum_rank
                        usBIDU           10.0
                        usFB             16.0
                        usGOOG           12.0
                        usNOAH            2.0
                        usSFUN           14.0
                        usTSLA           18.0
                        usVIPS            6.0
                        usWUBA            8.0
    :param corr_jobs: 默认：corr_jobs=(ECoreCorrType.E_CORE_TYPE_PEARS, ECoreCorrType.E_CORE_TYPE_SPERM)
                      可以再添加更多jobs
                      eg：
                        corr_jobs=(ECoreCorrType.E_CORE_TYPE_PEARS, ECoreCorrType.E_CORE_TYPE_SPERM,
                                   ECoreCorrType.E_CORE_TYPE_SIGN, ECoreCorrType.E_CORE_TYPE_ROLLING)
                      注意每添加一种相关计算方法，耗时都会增加
    :param show: 是否进行可视化
    """
    cs = code_to_symbol(symbol)
    symbol = cs.value
    if sum_rank is None:
        tmp_market = ABuEnv.g_market_target
        # 强制把市场设置为一样的
        ABuEnv.g_market_target = cs.market
        corr_df_dict = ABuSimilar.multi_corr_df(corr_jobs)
        # 恢复之前的市场
        ABuEnv.g_market_target = tmp_market
        """
            eg： corr_df_dict
            {'pears':
                            usBIDU    usFB  usGOOG  usNOAH  usSFUN  usTSLA  usVIPS  usWUBA
            usBIDU         1.0000  0.3013  0.3690  0.4015  0.3680  0.3015  0.3706  0.4320
            usFB           0.3013  1.0000  0.6609  0.2746  0.1978  0.4080  0.2856  0.2438
            usGOOG         0.3690  0.6609  1.0000  0.3682  0.1821  0.3477  0.3040  0.2917
            usNOAH         0.4015  0.2746  0.3682  1.0000  0.3628  0.2178  0.4645  0.4488
            usSFUN         0.3680  0.1978  0.1821  0.3628  1.0000  0.2513  0.2843  0.4883
            usTSLA         0.3015  0.4080  0.3477  0.2178  0.2513  1.0000  0.2327  0.3340
            usVIPS         0.3706  0.2856  0.3040  0.4645  0.2843  0.2327  1.0000  0.4189
            usWUBA         0.4320  0.2438  0.2917  0.4488  0.4883  0.3340  0.4189  1.0000

            'sperm':
                            usBIDU    usFB  usGOOG  usNOAH  usSFUN  usTSLA  usVIPS  usWUBA
            usBIDU         1.0000  0.3888  0.4549  0.4184  0.3747  0.3623  0.4333  0.4396
            usFB           0.3888  1.0000  0.7013  0.2927  0.2379  0.4200  0.3123  0.2216
            usGOOG         0.4549  0.7013  1.0000  0.3797  0.2413  0.3871  0.3922  0.3035
            usNOAH         0.4184  0.2927  0.3797  1.0000  0.3581  0.2066  0.4643  0.4382
            usSFUN         0.3747  0.2379  0.2413  0.3581  1.0000  0.2645  0.3890  0.4693
            usTSLA         0.3623  0.4200  0.3871  0.2066  0.2645  1.0000  0.2540  0.2801
            usVIPS         0.4333  0.3123  0.3922  0.4643  0.3890  0.2540  1.0000  0.4080
            usWUBA         0.4396  0.2216  0.3035  0.4382  0.4693  0.2801  0.4080  1.0000 }
        """
        sum_rank = rank_corr_sum(corr_df_dict, symbol)
        """
            eg: sum_rank
            usBIDU           10.0
            usFB             16.0
            usGOOG           12.0
            usNOAH            2.0
            usSFUN           14.0
            usTSLA           18.0
            usVIPS            6.0
            usWUBA            8.0
        """
        if sum_rank is None:
            logging.info('{} not in corr df!!!'.format(symbol))
            return None, None

    top_cnt = sum_rank.shape[0] if g_top_corr_cnt > sum_rank.shape[0] else g_top_corr_cnt
    # 首先找到的是最相关的top个
    rank_head = sum_rank.sort_values(ascending=True)[1:top_cnt]

    # 使用symbol做标尺
    benchmark = AbuBenchmark(symbol, n_folds=1)
    # benchmark做为数据标尺获取最相关的top个金融时间数据
    mul_pd = ABuSymbolPd.make_kl_df(rank_head.index, n_folds=1,
                                    data_mode=EMarketDataSplitMode.E_DATA_SPLIT_UNDO,
                                    benchmark=benchmark)

    coint_dict = {}
    for ind, cmp_symbol in enumerate(rank_head.index):
        if cmp_symbol not in mul_pd:
            continue

        klpd_cmp = mul_pd[cmp_symbol]
        if klpd_cmp is None:
            continue

        """
                coint返回值三个如下：
                coint_t : float
                    t-statistic of unit-root test on residuals
                pvalue : float
                    MacKinnon's approximate p-value based on MacKinnon (1994)
                crit_value : dict
                    Critical values for the test statistic at the 1 %, 5 %, and 10 %
                    levels.

                这里只考虑pvalue，因为已经是从top n个最相关的再找协整
        """
        _, pvalue, _ = coint(benchmark.kl_pd.close, klpd_cmp.close)
        if pvalue < g_coint_threshold:
            # pvalue小于阀值即加入coint_dict字典

            # 记录ind为了发现取多少个sort_values(ascending=True)[1:g_top_corr_cnt]能有良好的数据
            # 即为了事后调整g_top_corr_cnt使用，并非实际需要
            coint_dict[cmp_symbol] = (pvalue, ind)
    p_value_sorted = sorted(zip(coint_dict.values(), coint_dict.keys()))
    if len(p_value_sorted) == 0:
        logging.info(
            'len(p_value_sorted) == 0 please try change tl.similar.g_top_corr_cnt|tl.similar.g_coint_threshold!')
        return None, None

    if show:
        cmp_cnt = np.minimum(len(p_value_sorted), g_coint_show_max)
        # 只取item[1]，[0]是ind
        symbols = [item[1] for item in p_value_sorted[:cmp_cnt]]
        mul_pd_swap = mul_pd.swapaxes('items', 'minor')
        close_panel_pd = mul_pd_swap['close'][symbols]
        """
            转轴后只取收盘价格序列
            eg： close_panel_pd
                          usFB  usGOOG  usNOAH  usVIPS  usWUBA  us_NYSE:.IXIC
            2015-07-24   96.95  623.56   23.40  20.250   65.25       5088.629
            2015-07-27   94.17  627.26   22.16  19.990   62.89       5039.776
            2015-07-28   95.29  628.00   22.94  20.200   60.32       5089.207
            2015-07-29   96.99  631.93   23.35  20.260   59.89       5111.730
            2015-07-30   95.21  632.59   22.87  19.700   60.24       5128.785
            ...            ...     ...     ...     ...     ...            ...
            2016-07-20  121.92  741.19   25.11  13.630   48.17       5089.930
            2016-07-21  120.61  738.63   25.51  13.690   49.25       5073.900
            2016-07-22  121.00  742.74   25.50  13.510   49.21       5100.160
            2016-07-25  121.63  739.77   25.57  13.390   49.84       5097.628
            2016-07-26  121.64  740.92   24.75  13.655   50.36       5084.629
        """
        # 将数据scale到一个级别上，注意使用mean_how=True，避免极值的干扰
        close_panel_pd = ABuScalerUtil.scaler_matrix(close_panel_pd, mean_how=True)
        """
            ABuScalerUtil.scaler_matrix缩放后的数据矩阵如下所示
            eg： close_panel_pd
                             usFB     usGOOG     usNOAH     usVIPS     usWUBA
            2015-07-24  4451.7674  4311.1198  4477.3494  6601.2284  5980.4246
            2015-07-27  4324.1148  4336.7006  4240.0882  6516.4719  5764.1211
            2015-07-28  4375.5432  4341.8168  4389.3332  6584.9290  5528.5703
            2015-07-29  4453.6041  4368.9877  4467.7825  6604.4882  5489.1591
            ...               ...        ...        ...        ...        ...
            2016-07-20  5598.3443  5124.3808  4804.5404  4443.1972  4414.9740
            2016-07-21  5538.1915  5106.6817  4881.0762  4462.7564  4513.9603
            2016-07-22  5556.0995  5135.0971  4879.1628  4404.0788  4510.2942
            2016-07-25  5585.0280  5114.5633  4892.5566  4364.9604  4568.0362
            2016-07-26  5585.4872  5122.5141  4735.6581  4451.3468  4615.6963
        """
        # 可视化scaler_matrix操作后的close
        close_panel_pd.plot(figsize=ABuEnv.g_plt_figsize)
        plt.title('close panel pd scaler_matrix')
        plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
        plt.show()

        close_panel_pd_cp = copy.deepcopy(close_panel_pd)

        for col in close_panel_pd_cp.columns:
            """
                做一个一摸一样的pd就是为了得到投票权重表便于运算：

                     close_panel_pd_cp[col] = benchmark.kl_pd.close

                将所有数据列都使用标尺的数据进行替换，结果是每一列的数据都相同，
                比如这样，列数据都和标尺一样
                            usFB  usGOOG  usNOAH  usVIPS  usWUBA  us_NYSE:.IXIC
                2015-07-24  265.41  265.41  265.41  265.41  265.41         265.41
                2015-07-27  253.01  253.01  253.01  253.01  253.01         253.01
                2015-07-28  264.82  264.82  264.82  264.82  264.82         264.82
                2015-07-29  263.82  263.82  263.82  263.82  263.82         263.82
                2015-07-30  266.79  266.79  266.79  266.79  266.79         266.79
                ...            ...     ...     ...     ...     ...            ...
                2016-07-20  228.36  228.36  228.36  228.36  228.36         228.36
                2016-07-21  220.50  220.50  220.50  220.50  220.50         220.50
                2016-07-22  222.27  222.27  222.27  222.27  222.27         222.27
                2016-07-25  230.01  230.01  230.01  230.01  230.01         230.01
                2016-07-26  225.93  225.93  225.93  225.93  225.93         225.93
            """
            close_panel_pd_cp[col] = benchmark.kl_pd.close
        """
            将复刻后的close_panel_pd_cp与原始close_panel_pd求差后，再进行scaler_std
            ABuScalerUtil.scaler_std(close_panel_pd_cp - close_panel_pd):

                  usFB  usGOOG  usNOAH  usVIPS  usWUBA  us_NYSE:.IXIC
            2015-07-24  0.9705  1.7793  0.7405 -1.6987 -1.9294        -1.0803
            2015-07-27  1.2277  1.6619  1.1473 -1.6270 -1.5697        -0.8853
            2015-07-28  1.1393  1.6826  0.8987 -1.6831 -1.1334        -1.0866
            2015-07-29  0.9629  1.5955  0.7550 -1.7035 -1.0656        -1.2124
            2015-07-30  1.1519  1.5906  0.9265 -1.5197 -1.1169        -1.2878
            ...            ...     ...     ...     ...     ...            ...
            2016-07-21 -1.5539 -0.8188 -0.0710  0.3755  0.5784        -1.2418
            2016-07-22 -1.5899 -0.9012 -0.0644  0.4354  0.5879        -1.3728
            2016-07-25 -1.6371 -0.8138 -0.0746  0.4819  0.4997        -1.3179
            2016-07-26 -1.6473 -0.8509  0.2018  0.3922  0.4085        -1.2702
        """
        regular_diff = ABuScalerUtil.scaler_std(close_panel_pd_cp - close_panel_pd)
        regular_diff.plot(figsize=ABuEnv.g_plt_figsize)
        plt.title('regular diff')
        plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
        plt.show()
        """
            distance_votes = regular_diff.sum(axis=1):

            投票机制，获取投票coint的差值
                distance_votes
            distance_votes
            2015-07-24   -1.2181
            2015-07-27   -0.0451
            2015-07-28   -0.1825
            2015-07-29   -0.6682
            2015-07-30   -0.2555
                           ...
            2016-07-20   -2.5541
            2016-07-21   -2.7316
            2016-07-22   -2.9049
            2016-07-25   -2.8618
            2016-07-26   -2.7658
            ......................
        """
        distance_votes = regular_diff.sum(axis=1)

        votes_std = distance_votes.std()
        votes_mean = distance_votes.mean()
        above = votes_mean + votes_std
        below = votes_mean - votes_std
        close_regular = ABuScalerUtil.scaler_std(benchmark.kl_pd.close)
        close_regular = (close_regular * distance_votes.max() / 2)

        with plt_show():
            # noinspection PyUnresolvedReferences
            close_regular.plot()
            distance_votes.plot()

            plt.axhline(votes_mean, color='r')
            plt.axhline(above, color='c')
            plt.axhline(below, color='g')

            plt.title('coint distance votes')
            plt.legend(['close regular', 'distance votes', 'votes mean', 'dvotes above', 'dvotes below'],
                       bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
    return p_value_sorted, sum_rank
