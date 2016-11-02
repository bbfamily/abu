# -*- encoding:utf-8 -*-

"""

由于现在在写一本关于股票量化方面的书，会涉及到相关文章里的一些内容，
出版社方面不希望我现在开源全部代码，但是一定会开源，最晚等书出版发行
以后，现在只能开源文章中涉及的部分代码，整个系统的开源会稍后，请谅解
我其实也觉着有人能看你的代码就已经很给面子了，但是。。。再次抱歉！！

"""
from __future__ import division

import ast
import itertools
import random
from collections import namedtuple

import matplotlib.pyplot as plt
# 不能删除projection='3d'需要
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import pandas as pd
import scipy.optimize as sco

import NpUtil
import SymbolPd
import TLineGolden
import ZLog

from sklearn.externals.joblib import Parallel
from sklearn.externals.joblib import delayed
from ProcessMonitor import add_process_wrapper

__author__ = 'BBFamily'

g_mc_loss_cnt = 4
g_mc_win_cnt = 5

"""
    最后由几个最优化参数可视化找出的参数
    由函数中mc_golden选择是否开启参数还
    是继续使用默认参数
"""
g_mc_percent = [0.21828571428571428, 0.27285714285714285, 0.3274285714285714, 0.382, 0.7271428571428571,
                0.7817142857142857, 0.8362857142857143, 0.8908571428571428, 0.9454285714285714]


def show_golden_process(w_rate, symbol=None, mc_golden=False, bp=24.3, pf_cnt=200, loop_cnt=20000, outter_loop=1):
    if symbol is None:
        kl_pd = SymbolPd.make_kfold_pd('usNOAH')[-82:-40]
    else:
        kl_pd = SymbolPd.make_kfold_pd(symbol)

    golden = TLineGolden.calc_mc_golden(kl_pd, g_mc_percent, g_mc_loss_cnt) if mc_golden else TLineGolden.calc_golden(
        kl_pd)

    while outter_loop > 0:
        outter_loop -= 1
        profits = []
        for _ in np.arange(loop_cnt):
            wl = init_golden_w_full(golden, bp) if mc_golden else init_golden_wl(golden, bp)
            if wl is None:
                ZLog.info('init_golden_wl out of bp range!')
                return

            sp = 0
            while wl is not None:
                supports = wl['supports']
                resistances = wl['resistances']
                w = np.random.binomial(1, w_rate)

                '''
                    sp = resistances[-1]
                    sp = supports[0]
                    -1, 0都行，反正这里走到有意义的sp时都是一个了
                '''
                if w:
                    sp = resistances[-1]
                else:
                    sp = supports[0]

                wl = golden_map_wl_grid(w, wl)
            else:
                """
                    －10 默认交易成本（手续费）
                """
                pf = pf_cnt * (sp - bp) - 10
                profits.append(pf)
        profits = pd.Series(profits)
        show = (outter_loop == 0)
        NpUtil.calc_regress_ang(profits.cumsum(), show)
        if show:
            profits.hist()


def show_golden_nb_process(w_rate, symbol=None, mc_golden=False, bp=24.3, pf_cnt=200, loop_cnt=20000, outter_loop=1):
    import numba as nb
    f_nb = nb.jit(show_golden_process)
    return f_nb(w_rate, symbol, mc_golden, bp, pf_cnt, loop_cnt, outter_loop)


def show_golden_sco_process(w_rate, how='wl_money_rate', symbol=None, max_iter=5000, bp=24.3, pf_cnt=200,
                            loop_cnt=20000):
    if symbol is None:
        kl_pd = SymbolPd.make_kfold_pd('usNOAH')[-82:-40]
    else:
        kl_pd = SymbolPd.make_kfold_pd(symbol)

    """
        初始化猜测,不用很多大概就行, 范围内均匀分布就行
    """
    loss_percents = np.linspace(0.0, 0.382, 8)
    win_percents = np.linspace(0.618, 1.0, 8)
    guess_win_percent = random.sample(win_percents, g_mc_win_cnt)
    guess_loss_percent = random.sample(loss_percents, g_mc_loss_cnt)

    def min_func(percent):
        golden = TLineGolden.calc_mc_golden(kl_pd, percent, g_mc_loss_cnt)
        profits = []

        for _ in np.arange(loop_cnt):
            wl = init_golden_w_full(golden, bp)
            sp = 0
            while wl is not None:
                supports = wl['supports']
                resistances = wl['resistances']
                w = np.random.binomial(1, w_rate)
                if w:
                    sp = resistances[-1]
                else:
                    sp = supports[0]

                wl = golden_map_wl_grid(w, wl)
            else:
                pf = pf_cnt * (sp - bp) - 10
                profits.append(pf)

        profits = pd.DataFrame(profits, columns=['profits'])
        profits['win'] = np.where(profits > 0, 1, 0)
        wl_money_rate = np.abs(
            profits[profits['win'] > 0]['profits'].mean() / profits[profits['win'] <= 0]['profits'].mean())
        wl_rate = profits.win.value_counts()[1] / profits.win.value_counts().sum()
        sum_money = profits['profits'].cumsum().iloc[-1]
        return np.array([wl_money_rate, wl_rate, sum_money])

    def min_func_st(p_how):

        def min_func_wl_money_rate(percent):
            """
            最优每次输赢比例
            :return:
            """
            return -min_func(percent)[0]

        def min_func_wl_rate(percent):
            """
            最优总输赢比例
            :return:
            """
            return -min_func(percent)[1]

        def min_func_sum_money(percent):
            """
            最优最后盈利
            :return:
            """
            return -min_func(percent)[2]

        if p_how == 'wl_money_rate':
            return min_func_wl_money_rate
        elif p_how == 'wl_rate':
            return min_func_wl_rate
        else:
            return min_func_sum_money

    bnds = list((0, 0.382) for _ in range(g_mc_loss_cnt)) + list((0.618, 1) for _ in range(g_mc_win_cnt))
    opts = sco.minimize(min_func_st(how), guess_loss_percent + guess_win_percent, method='SLSQP',
                        bounds=bnds, options={'maxiter': max_iter})
    return opts, min_func


def _golden_mc_process_profits(profits_dict):
    wl_money_rates = []
    wl_rates = []
    sum_moneys = []
    keys = []
    for k, profits in profits_dict.items():
        """
            wl_money_rates: 每次输输赢钱的比例
            wl_rates      : 总输赢比例
            sum_money     : loop之后最后的money
        """
        profits = pd.DataFrame(profits, columns=['profits'])
        profits['win'] = np.where(profits > 0, 1, 0)
        wl_money_rate = np.abs(
            profits[profits['win'] > 0]['profits'].mean() / profits[profits['win'] <= 0]['profits'].mean())
        wl_rate = profits.win.value_counts()[1] / profits.win.value_counts().sum()
        sum_money = profits['profits'].cumsum().iloc[-1]
        wl_money_rates.append(wl_money_rate)
        wl_rates.append(wl_rate)
        sum_moneys.append(sum_money)
        keys.append(k)

    # sum_moneys_nm = NpUtil.regular_std(np.array(sum_moneys))
    # sum_moneys_nm = (sum_moneys_nm + 3) / 6

    cmap = plt.get_cmap('jet', 20)
    cmap.set_under('gray')
    fig, ax = plt.subplots()
    cax = ax.scatter(wl_money_rates, wl_rates, c=sum_moneys, cmap=cmap, vmin=np.min(sum_moneys),
                     vmax=np.max(sum_moneys))
    fig.colorbar(cax, label='sum_moneys', extend='min')
    # plt.scatter(wl_money_rates, wl_rates, c=sum_moneys, marker='o')
    # plt.colorbar(label='sum_moneys')
    plt.grid(True)
    plt.xlabel('wl_money_rates')
    plt.ylabel('wl_rates')
    plt.show()

    fig = plt.figure(figsize=(9, 6))
    ax = fig.add_subplot(111, projection='3d')
    ax.view_init(30, 60)
    ax.scatter(wl_money_rates, wl_rates, sum_moneys, zdir='z', c='b', marker='^')
    ax.set_xlabel('wl_money_rates')
    ax.set_ylabel('wl_rates')
    ax.set_zlabel('sum_moneys')
    plt.show()

    return pd.DataFrame([wl_money_rates, wl_rates, sum_moneys, keys],
                        index=['wl_money_rates', 'wl_rates', 'sum_moneys', 'keys']).T


def _golden_mc_process_cmp(kl_pd, loss_percent, win_percent, golden_tuple, profits_dict, w_rate,
                           bp, pf_cnt, p_outter_loop, loop_cnt, show=False):
    """
        置换出对应的股价
    """
    gg = sorted(TLineGolden.calc_percent_golden(kl_pd, loss_percent, True)) + sorted(
        TLineGolden.calc_percent_golden(kl_pd, win_percent, False))
    """
        numba不支持 *args, **kwargs, 只能这样gg[0], gg[1], gg[2], gg[3], gg[4], gg[5], gg[6], gg[7], gg[8]了
    """
    # golden = golden_tuple(*gg)
    golden = golden_tuple(gg[0], gg[1], gg[2], gg[3], gg[4], gg[5], gg[6], gg[7], gg[8])
    p__key = sorted(loss_percent.tolist() + win_percent.tolist())
    p_key = str(p__key)
    if p_key in profits_dict:
        """
            过滤重复的猜测
        """
        return

    outter_loop = p_outter_loop
    while outter_loop > 0:
        outter_loop -= 1
        profits = []
        for _ in np.arange(loop_cnt):
            wl = init_golden_w_full(golden, bp)
            if wl is None:
                ZLog.info('init_golden_wl out of bp range!')

            sp = 0
            while wl is not None:
                supports = wl['supports']
                resistances = wl['resistances']
                w = np.random.binomial(1, w_rate)
                if w:
                    sp = resistances[-1]
                else:
                    sp = supports[0]

                wl = golden_map_wl_grid(w, wl)
            else:
                pf = pf_cnt * (sp - bp) - 10
                profits.append(pf)
        profits = pd.Series(profits)
        profits_dict[p_key] = profits
        if show:
            plt.plot(kl_pd.close)

            plt.axhline(golden.above950, lw=3.5, color='c')
            plt.axhline(golden.above900, lw=3.0, color='y')
            plt.axhline(golden.above800, lw=2.5, color='k')
            plt.axhline(golden.above700, lw=2.5, color='m')
            plt.axhline(golden.above618, lw=2, color='r')
            plt.fill_between(kl_pd.index, golden.above618, golden.above950,
                             alpha=0.1, color="r")

            plt.axhline(golden.below382, lw=2, color='g')
            plt.axhline(golden.below300, lw=2.5, color='k')
            plt.axhline(golden.below250, lw=3.0, color='y')
            plt.axhline(golden.below200, lw=3.5, color='c')
            plt.fill_between(kl_pd.index, golden.below200, golden.below382,
                             alpha=0.1, color="g")
            _ = plt.setp(plt.gca().get_xticklabels(), rotation=30)
            plt.legend(['close', 'above950', 'above900', 'above800', 'above700', 'above618',
                        'below382', 'below300', 'below250', 'below200'], bbox_to_anchor=(1.05, 1),
                       loc=2,
                       borderaxespad=0.)
            plt.title('between golden')
            plt.show()
            NpUtil.calc_regress_ang(profits.cumsum(), show)
            plt.show()
            profits.hist()
            plt.show()


def check_golden_mc_result(filter_ret, w_rate, symbol=None, bp=24.3, pf_cnt=200, loop_cnt=20000, p_outter_loop=1):
    """
    最后可视化验证少量的筛选结果

    :param filter_ret:
    :param w_rate:
    :param symbol:
    :param bp:
    :param pf_cnt:
    :param loop_cnt:
    :param p_outter_loop:
    :return:
    """
    if symbol is None:
        kl_pd = SymbolPd.make_kfold_pd('usNOAH')[-82:-40]
    else:
        kl_pd = SymbolPd.make_kfold_pd(symbol)

    for loc in np.arange(filter_ret.shape[0]):
        percents = ast.literal_eval(filter_ret['keys'].iloc[loc])
        ZLog.info(loc)
        ZLog.info(percents)
        loss_percent = np.array(percents[0:4])
        win_percent = np.array(percents[4:9])
        profits_dict = {}
        golden_tuple = namedtuple('golden', (
            'below200', 'below250', 'below300', 'below382', 'above618', 'above700', 'above800', 'above900', 'above950'))

        _golden_mc_process_cmp(kl_pd, loss_percent, win_percent, golden_tuple, profits_dict, w_rate,
                               bp, pf_cnt, p_outter_loop, loop_cnt, show=True)


def show_golden_mc_process(w_rate, symbol=None, max_iter=5000, bp=24.3, pf_cnt=200, loop_cnt=20000, p_outter_loop=1):
    """
    使用模特卡洛方式分析寻找最优参数
    :param w_rate:
    :param symbol:
    :param max_iter:
    :param bp:
    :param pf_cnt:
    :param loop_cnt:
    :param p_outter_loop:
    :return:
    """
    profits_dict = {}
    loss_percents = np.linspace(0.0, 0.382, 100)
    win_percents = np.linspace(0.618, 1.0, 100)
    golden_tuple = namedtuple('golden', (
        'below200', 'below250', 'below300', 'below382', 'above618', 'above700', 'above800', 'above900', 'above950'))

    if symbol is None:
        kl_pd = SymbolPd.make_kfold_pd('usNOAH')[-82:-40]
    else:
        kl_pd = SymbolPd.make_kfold_pd(symbol)

    for _ in np.arange(max_iter):
        loss_percent = np.random.choice(loss_percents, g_mc_loss_cnt, replace=False)
        win_percent = np.random.choice(win_percents, g_mc_win_cnt, replace=False)

        _golden_mc_process_cmp(kl_pd, loss_percent, win_percent, golden_tuple, profits_dict, w_rate,
                               bp, pf_cnt, p_outter_loop, loop_cnt, show=False)
    return _golden_mc_process_profits(profits_dict)


@add_process_wrapper
def _do_mc_product_process(sub_lws, w_rate, symbol=None, bp=24.3, pf_cnt=200, loop_cnt=20000,
                           p_outter_loop=1):
    """
    独立进程对子任务的完成
    :param sub_lws:
    :param w_rate:
    :param symbol:
    :param bp:
    :param pf_cnt:
    :param loop_cnt:
    :param p_outter_loop:
    :return:
    """
    profits_dict = {}
    golden_tuple = namedtuple('golden', (
        'below200', 'below250', 'below300', 'below382', 'above618', 'above700', 'above800', 'above900', 'above950'))

    if symbol is None:
        kl_pd = SymbolPd.make_kfold_pd('usNOAH')[-82:-40]
    else:
        kl_pd = SymbolPd.make_kfold_pd(symbol)

    for loss_percent, win_percent in sub_lws:
        """
            golden_mc_process_cmp里面统一array
        """
        loss_percent = np.array(loss_percent)
        win_percent = np.array(win_percent)
        _golden_mc_process_cmp(kl_pd, loss_percent, win_percent, golden_tuple, profits_dict, w_rate,
                               bp, pf_cnt, p_outter_loop, loop_cnt, show=False)
    return profits_dict


def show_golden_mc_product_process(w_rate, n_jobs=1, symbol=None, max_iter=8, bp=24.3, pf_cnt=200, loop_cnt=20000,
                                   p_outter_loop=1):
    """
    使用模特卡洛方式分析寻找最优参数, 排列最合所有元素
    :param n_jobs:
    :param w_rate:
    :param symbol:
    :param max_iter:
    :param bp:
    :param pf_cnt:
    :param loop_cnt:
    :param p_outter_loop:
    :return:
    """

    """
        ⚠️控制max_iter, 15以上就会有上亿种排列组合了
    """
    loss_percents = np.linspace(0.0, 0.382, max_iter)
    win_percents = np.linspace(0.618, 1.0, max_iter)
    ls = list(itertools.combinations(loss_percents, g_mc_loss_cnt))
    ws = list(itertools.combinations(win_percents, g_mc_win_cnt))
    lws = list(itertools.product(ls, ws))

    # 暂时默认8核cpu,且把0及其它都归结为-1的范畴
    n_jobs = 8 if n_jobs <= 0 else n_jobs
    process_lws = []
    if n_jobs > 1:
        group_adjacent = lambda a, k: zip(*([iter(a)] * k))
        process_lws = group_adjacent(lws, n_jobs)
        # 将剩下的再放进去
        sf = -(len(lws) % n_jobs)
        if sf < 0:
            process_lws.append(lws[sf:])
    else:
        process_lws.append(lws)

    parallel = Parallel(
        n_jobs=n_jobs, verbose=0, pre_dispatch='2*n_jobs')

    out = parallel(
        delayed(_do_mc_product_process)(sub_lws, w_rate, symbol, bp, pf_cnt, loop_cnt,
                                        p_outter_loop)
        for sub_lws in process_lws)

    profits_dict = {}
    for sub_profits_dict in out:
        profits_dict.update(sub_profits_dict)

    return _golden_mc_process_profits(profits_dict)


def show_golden_mc_nb_product_process(w_rate, n_jobs=1, symbol=None, max_iter=8, bp=24.3, pf_cnt=200, loop_cnt=20000,
                                      p_outter_loop=1):
    import numba as nb
    f_nb = nb.jit(show_golden_mc_product_process)
    return f_nb(w_rate, n_jobs, symbol, max_iter, bp, pf_cnt, loop_cnt, p_outter_loop)


def show_golden_mc_nb_process(w_rate, symbol=None, max_iter=5000, bp=24.3, pf_cnt=200, loop_cnt=20000, p_outter_loop=1):
    import numba as nb
    f_nb = nb.jit(show_golden_mc_process)
    return f_nb(w_rate, symbol, max_iter, bp, pf_cnt, loop_cnt, p_outter_loop)


def init_golden_w_full(golden, bp):
    """
    只取最完成的

    :param bp:
    :param golden:
    :return:
    """
    if bp < golden.below382 or bp > golden.above618:
        return None

    mapds = {golden.below200: 'below200', golden.below250: 'below250', golden.below300: 'below300',
             golden.below382: 'below382',
             golden.above618: 'above618', golden.above700: 'above700', golden.above800: 'above800',
             golden.above900: 'above900', golden.above950: 'above950'}

    supports = [golden.below200, golden.below250, golden.below300, golden.below382]
    resistances = [golden.above618, golden.above700, golden.above800, golden.above900, golden.above950]

    kwargs = {'supports': supports, 'resistances': resistances, 'mapds': mapds}

    return kwargs


def init_golden_wl(golden, bp):
    """
    完整影射，最后选择factor sell type需要
    :param golden:
    :param bp:
    :return:
    """
    mapds = {golden.below200: 'below200', golden.below250: 'below250', golden.below300: 'below300',
             golden.below382: 'below382',
             golden.above618: 'above618', golden.above700: 'above700', golden.above800: 'above800',
             golden.above900: 'above900',
             golden.above950: 'above950'}

    if golden.below382 <= bp < golden.above382:
        '''
            买入跌回去，降低最高止盈
        '''
        supports = [golden.below200, golden.below250, golden.below300, golden.below382]
        resistances = [golden.above618, golden.above700, golden.above800, golden.above900]
    elif golden.above382 <= bp < golden.below618:
        '''
            最高止盈，最低止损
        '''
        supports = [golden.below200, golden.below250, golden.below300, golden.below382]
        resistances = [golden.above618, golden.above700, golden.above800, golden.above900, golden.above950]

    elif golden.below618 <= bp < golden.above618:
        '''
            单边升高止损
        '''
        supports = [golden.below250, golden.below300, golden.below382]
        resistances = [golden.above618, golden.above700, golden.above800, golden.above900, golden.above950]
    else:
        return None
    kwargs = {'supports': supports, 'resistances': resistances, 'mapds': mapds}

    return kwargs


def golden_map_wl_grid(win, map_dict):
    if map_dict is None or not map_dict.has_key('supports') or not map_dict.has_key('resistances') \
            or not map_dict.has_key('mapds'):
        raise ValueError('golden_map_wl_grid kwargs is None or not kwargs.has_key!')

    supports = map_dict['supports']
    resistances = map_dict['resistances']
    mapds = map_dict['mapds']
    if win:
        jsr = len(resistances)
        if jsr > 1:
            supports.append(resistances.pop(0))
            jsr -= 1
            supports = supports[-jsr:]
        else:
            return None
    else:
        jss = len(supports)
        if jss > 1:
            resistances.insert(0, supports.pop())
            resistances = resistances[:jss]
        else:
            return None

    map_dict = {'supports': supports, 'resistances': resistances, 'mapds': mapds}
    return map_dict
