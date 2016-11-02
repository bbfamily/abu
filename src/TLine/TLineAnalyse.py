# -*- encoding:utf-8 -*-
"""

对矢量曲线分析提取骨架
提取趋势，阻力，支撑位
注意有些方法需要不断拟合
等运算，速度会有问题，不
适合grid情况下全量的测试

由于现在在写一本关于股票量化方面的书，会涉及到相关文章里的一些内容，
出版社方面不希望我现在开源全部代码，但是一定会开源，最晚等书出版发行
以后，现在只能开源文章中涉及的部分代码，整个系统的开源会稍后，请谅解
我其实也觉着有人能看你的代码就已经很给面子了，但是。。。再次抱歉！！

"""
from __future__ import print_function

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from scipy import optimize
from scipy.interpolate import interp1d
from scipy import stats

from sklearn.cluster import KMeans
from sklearn import metrics

import TLineDrawer
import TargetStockHelper
import SymbolPd

import warnings

__author__ = 'BBFamily'


def find_trend_point(x_org, y_org, show=False):
    min_list = find_min_local(x_org, y_org)
    max_list = []
    y_org.index = x_org

    for index, xd in enumerate(min_list):
        xd = int(xd)
        if index == 0:
            ss = y_org.iloc[:xd].argmax()
            max_list.append(ss)
            continue
        sc = y_org.iloc[min_list[index - 1]:xd].argmax()
        max_list.append(sc)
        if index == len(min_list) - 1:
            sc = y_org.iloc[xd:].argmax()
            max_list.append(sc)
    max_list.append(x_org[len(x_org) - 1])
    max_list = sorted(set(max_list))
    if show:
        TLineDrawer.plot_xy_with_other(x_org, y_org, 'o', min_list, max_list)
    return min_list, max_list


def find_percent_point(percents, y_org):
    cs_max = y_org.max()
    cs_min = y_org.min()

    """
        pt * 100 0.80 * 100 80.0
    """
    return {pt: (stats.scoreatpercentile(y_org, np.round(pt * 100, 1)),
                 (cs_max - cs_min) * pt + cs_min) for pt in percents}


def find_golden_point_ex(x_org, y_org, show=False):
    sp382 = stats.scoreatpercentile(y_org, 38.2)
    sp618 = stats.scoreatpercentile(y_org, 61.8)
    sp50 = stats.scoreatpercentile(y_org, 50.0)

    if show:
        x_org = np.array(x_org)
        sp382_list = [[x_org.min(), x_org.max()], [sp382, sp382]]
        sp618_list = [[x_org.min(), x_org.max()], [sp618, sp618]]
        sp50_list = [[x_org.min(), x_org.max()], [sp50, sp50]]

        TLineDrawer.plot_xy_with_other_x_y(x_org, y_org, '-', sp382_list, sp50_list, sp618_list)
    return sp382, sp50, sp618


def find_golden_point(x_org, y_org, show=False):
    cs_max = y_org.max()
    cs_min = y_org.min()

    sp382 = (cs_max - cs_min) * 0.382 + cs_min
    sp618 = (cs_max - cs_min) * 0.618 + cs_min
    sp50 = (cs_max - cs_min) * 0.5 + cs_min
    if show:
        x_org = np.array(x_org)
        sp382_list = [[x_org.min(), x_org.max()], [sp382, sp382]]
        sp618_list = [[x_org.min(), x_org.max()], [sp618, sp618]]
        sp50_list = [[x_org.min(), x_org.max()], [sp50, sp50]]

        TLineDrawer.plot_xy_with_other_x_y(x_org, y_org, '-', sp382_list, sp50_list, sp618_list)
    return sp382, sp50, sp618


def poly_bfgs(min_list, x, y_org, n):
    n_min_list = min_list
    min_list = []
    y = y_org
    start = 0
    end = n - 1
    p_fit = int(len(x) / 2)
    if len(n_min_list) > 0:
        start = n_min_list[0]
        end = n_min_list[len(n_min_list) - 1]

        x = x[start:end + 1]
        y = [y_org[min_x] for min_x in x]

        p_fit = int(len(n_min_list) / 2)

    p = np.polynomial.Chebyshev.fit(x, y, p_fit)
    for index in xrange(start, end + 1, 1):
        x_min_local = int(optimize.fmin_bfgs(p, index, disp=0)[0])
        if 0 < x_min_local < n:
            min_list.append(x_min_local)
    min_list = sorted(set(min_list))
    return min_list


def kmean_clusters(min_list, d, show, thresh):
    np.random.seed(0)

    k_rng = range(1, len(min_list))
    est = [KMeans(n_clusters=k).fit(d) for k in k_rng]

    silhouette_score = [
        metrics.silhouette_score(d, e.labels_, metric='euclidean') for e in est[1:]]
    within_sum_squares = [e.inertia_ for e in est]

    diff_sq = [sq / within_sum_squares[0] for sq in within_sum_squares]

    diff_sq_pd = pd.Series(diff_sq)

    k_list = list(k_rng)
    select_k = k_list[len(k_list) - 1]
    thresh_pd = diff_sq_pd[diff_sq_pd < thresh]
    if thresh_pd.shape[0] > 0:
        select_k = k_list[thresh_pd.index[0]]

    if show:
        TLineDrawer.plot_elow_k_choice(k_rng, silhouette_score, within_sum_squares, select_k)

    select_est = est[select_k - 1]
    y_kmean = select_est.predict(d)
    return y_kmean, select_k


def find_min_local(x_org, y_org, show=False, loop_cnt=2, thresh=0.06):
    x = x_org
    # y = y_org
    n = len(x)
    min_list = []
    while loop_cnt > 0:
        loop_cnt -= 1
        min_list = poly_bfgs(min_list, x, y_org, n)
        if show:
            TLineDrawer.plot_xy_with_other(x_org, y_org, 'o', min_list)

    if len(min_list) <= 1:
        warnings.warn("len(minList) <= 1")
        return min_list

    d = np.array([min_list, [y_org[min_x] for min_x in min_list]]).T
    y_kmean, select_k = kmean_clusters(min_list, d, show, thresh)

    if show:
        TLineDrawer.plot_xy_with_scatter_color(x_org, y_org, d, y_kmean)

    d_pd = pd.DataFrame(d, columns=['X', 'Y'])
    d_pd['Cluster'] = y_kmean
    d_pd_min = d_pd.groupby(['Cluster', 'Y', 'X'])['X', 'Y'].min()

    min_list = [d_pd_min.loc[x, :].values[0][0] for x in xrange(0, select_k)]
    if show:
        TLineDrawer.plot_xy_with_other(x_org, y_org, 'o', min_list)
    return sorted(set(min_list))


def random_unit_test(extra_fun=None, extra_parm=None):
    np.random.seed(26)

    market_symbols = TargetStockHelper.stock_code_to_market_list(SymbolPd.K_TARGET_SYMBOL)

    choice_symbols = np.random.choice(market_symbols, 10)
    print(choice_symbols)
    for symbol in choice_symbols:
        pd_symbol = SymbolPd.make_pd(symbol)
        x = list(xrange(0, pd_symbol.shape[0]))
        y = pd_symbol['close']

        if len(x) <= 1:
            continue

        if extra_fun is None:
            find_trend_point(x, y, show=True)
        else:
            if extra_parm is None:
                extra_fun(x, y)
            else:
                extra_fun(x, y, extra_parm)


def special_unit_test(special):
    pd_symbol = SymbolPd.make_pd(special)
    print(pd_symbol.head())
    x = list(xrange(0, pd_symbol.shape[0]))
    y = pd_symbol['close']
    find_trend_point(x, y, show=True)


def show_process(capital_pd):
    n = capital_pd.shape[0]
    x = xrange(0, n)
    y = capital_pd['capital_blance']
    p = np.polynomial.Chebyshev.fit(x, y, n / 2)
    plt.plot(x, y, '', x, p(x), '-')
    plt.show()

    min_list = []
    for index in xrange(0, n, 1):
        # xmin_local = optimize.fminbound(p, 100, 200)
        x_min_local = int(optimize.fmin_bfgs(p, index, disp=0)[0])
        if 0 < x_min_local < n:
            min_list.append(x_min_local)
    print(min_list)

    plt.plot(min_list, capital_pd['capital_blance'][min_list], 'o')
    plt.show()

    measures = capital_pd['capital_blance'][min_list].values
    linear_interp = interp1d(min_list, measures)
    computed_time = np.arange(min_list[0], min_list[len(min_list) - 1] + 1, 1)
    linear_results = linear_interp(computed_time)

    plt.plot(min_list, measures, 'o', label='measures')
    plt.plot(computed_time, linear_results, label='linear interp')

    p2 = np.polynomial.Chebyshev.fit(min_list, measures, len(min_list) / 2)
    plt.plot(x, y, '')
    plt.plot(min_list, [p2(min_x) for min_x in min_list], 'o', label='measures')
    plt.show()

    n_min_list = []
    for index in xrange(min_list[0], min_list[len(min_list) - 1] + 1, 1):
        # xmin_local = optimize.fminbound(p, 100, 200)
        x_min_local = int(optimize.fmin_bfgs(p2, index, disp=0)[0])
        if 0 < x_min_local < n:
            n_min_list.append(x_min_local)
    n_min_list = sorted(set(n_min_list))
    print(n_min_list)
    plt.plot(n_min_list, capital_pd['capital_blance'][n_min_list], 'o')

    p3 = np.polynomial.Chebyshev.fit(
        n_min_list, capital_pd['capital_blance'][n_min_list], 1)
    plt.plot(x, y, '', x, p3(x), '-')
    plt.plot(n_min_list, capital_pd['capital_blance'][n_min_list], 'o')
    plt.show()
