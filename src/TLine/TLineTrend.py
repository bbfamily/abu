# -*- encoding:utf-8 -*-
"""

上升趋势线

"""
from __future__ import print_function
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

import TLineDrawer
import TLineAnalyse

import NpUtil

from sklearn.linear_model import LinearRegression
from sklearn.cluster import KMeans
from collections import namedtuple

__author__ = 'BBFamily'

K_TREND_KMEAN = 0
K_TREND_REG = 1


def calc_reg_fit(kl_pd, show=True):
    dk = True if kl_pd.columns.tolist().count('close') > 0 else False
    uq_close = kl_pd.close if dk else kl_pd.price

    asset = pd.DataFrame(uq_close.values, index=np.arange(0, len(uq_close)))
    reg_params = NpUtil.regress_y(asset)
    x = np.arange(0, len(uq_close))

    a = reg_params[0]
    b = reg_params[1]
    reg_y_fit = x * b + a

    min_ind = (asset.values.T - reg_y_fit).argmin()
    below = x[min_ind] * b + a - asset.values[min_ind]
    reg_y_bwlow = x * b + a - below

    max_ind = (asset.values.T - reg_y_fit).argmax()
    above = x[max_ind] * b + a - asset.values[max_ind]
    reg_y_above = x * b + a - above

    asset_std = asset.std().values[0]
    reg_y_below_std = x * b + a - asset_std
    reg_y_above_std = x * b + a + asset_std

    if show:
        plt.axes([0.025, 0.025, 0.95, 0.95])
        plt.plot(asset)
        plt.plot(x, reg_y_fit, 'c')
        plt.plot(x, reg_y_bwlow, 'r')
        plt.plot(x, reg_y_above, 'g')

        plt.plot(x, reg_y_below_std, 'k')
        plt.plot(x, reg_y_above_std, 'm')

        plt.title('reg fit')
        plt.legend([kl_pd.name, 'reg_y_fit', 'reg_y_bwlow', 'reg_y_above', 'reg_y_bwlow_std', 'reg_y_above_std'],
                   bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
        plt.show()

    distance_fit = (asset.values.T - reg_y_fit)[0]
    std_fit = distance_fit.std()
    distance_mean = distance_fit.mean()
    distance_std_above = distance_mean + std_fit
    distance_std_below = distance_mean - std_fit

    distance_max = distance_fit.max()
    distance_min = distance_fit.min()

    if show:
        plt.axes([0.025, 0.025, 0.95, 0.95])
        plt.ylim([distance_min - std_fit / 3, distance_max + std_fit / 3])
        plt.plot(distance_fit)
        plt.axhline(distance_std_above, color='m')
        plt.axhline(distance_mean, color='c')
        plt.axhline(distance_std_below, color='k')
        plt.axhline(distance_max, color='g')
        plt.axhline(distance_min, color='r')

        plt.title('reg fit distance')
        plt.legend([kl_pd.name, 'distance_std_above', 'distance_mean',
                    'distance_std_below', 'distance_max', 'distance_min'], bbox_to_anchor=(1.05, 1), loc=2,
                   borderaxespad=0.)
        plt.show()

    return namedtuple('reg_fit', ['now', 'mean', 'above', 'below', 'distance_max', 'distance_min'])(distance_fit[-1],
                                                                                                    distance_mean,
                                                                                                    distance_std_above,
                                                                                                    distance_std_below,
                                                                                                    distance_max,
                                                                                                    distance_min)


def _split_xy_trend(x_fit, y_fit):
    y_fit_pd = pd.DataFrame(y_fit)
    y_fit_pd_sft = y_fit_pd.shift(1)
    diff_fit = y_fit_pd - y_fit_pd_sft
    diff_fit = diff_fit[diff_fit < 0].dropna()

    split_index = diff_fit.index
    other = []
    if len(split_index) > 0:
        for index, split in enumerate(split_index):
            if index == len(split_index) - 1:
                x_fit_sub = x_fit[split:].reshape(-1, 1)
                y_fit_sub = y_fit[split:].reshape(-1, 1)
                if x_fit_sub.shape[0] > 1:
                    other.append([x_fit_sub, y_fit_sub])
                if index == 0:
                    x_fit_sub = x_fit[0:split].reshape(-1, 1)
                    y_fit_sub = y_fit[0:split].reshape(-1, 1)
                else:
                    x_fit_sub = x_fit[split_index[index - 1]:split].reshape(-1, 1)
                    y_fit_sub = y_fit[split_index[index - 1]:split].reshape(-1, 1)
            elif index == 0:
                x_fit_sub = x_fit[0:split].reshape(-1, 1)
                y_fit_sub = y_fit[0:split].reshape(-1, 1)
            else:
                x_fit_sub = x_fit[split_index[index - 1]:split].reshape(-1, 1)
                y_fit_sub = y_fit[split_index[index - 1]:split].reshape(-1, 1)
            if x_fit_sub.shape[0] > 1:
                other.append([x_fit_sub, y_fit_sub])
    else:
        # xFit.shape[1] > 1:
        other.append([x_fit.reshape(-1, 1), y_fit.reshape(-1, 1)])
    return other


def _do_trend_plot(x_org, y_org, other, mode):
    # print other
    do_other = []
    if mode == K_TREND_REG:
        for xy in other:
            if len(xy) == 2:
                linreg = LinearRegression()
                linreg.fit(xy[0], xy[1])
                x_pred = np.array(x_org).reshape(-1, 1)
                y_pred = linreg.predict(x_pred)
                x_pred = np.ravel(x_pred)
                y_pred = np.ravel(y_pred)
                do_other.append([x_pred, y_pred, '-'])
            do_other.append([xy[0], xy[1], 'ro'])
    elif mode == K_TREND_KMEAN:
        for xy in other:
            if len(xy) == 2:
                x_pred = xy[0]
                y_pred = xy[1]
                if x_pred.shape[0] == 2:
                    pass
                elif x_pred.shape[0] == 3:
                    d = np.array([np.ravel(x_pred), np.ravel(y_pred)]).T
                    d_pd = pd.DataFrame(d, columns=['X', 'Y'])
                    d_pd.sort(['X'])
                    x_pred = d_pd.iloc[0:2, 0:1].values
                    y_pred = d_pd.iloc[0:2, 1:2].values
                else:
                    print('KMeans effect')
                    select_k = 2
                    kmeans = KMeans(n_clusters=select_k, init='random')
                    d = np.array([np.ravel(x_pred), np.ravel(y_pred)]).T

                    kmeans.fit(d)
                    y_kmeans = kmeans.predict(d)

                    d_pd = pd.DataFrame(d, columns=['X', 'Y'])
                    d_pd['Cluster'] = y_kmeans
                    d_pd_min = d_pd.groupby(['Cluster', 'Y', 'X'])['X', 'Y'].min()
                    x_pred = [d_pd_min.loc[x, :].values[0][0] for x in xrange(0, select_k)]
                    y_pred = [y_org[x] for x in x_pred]
                    x_pred = np.array(x_pred).reshape(-1, 1)
                    y_pred = np.array(y_pred).reshape(-1, 1)

                linreg = LinearRegression()
                linreg.fit(x_pred, y_pred)
                x_pred = np.array(x_org).reshape(-1, 1)
                y_pred = linreg.predict(x_pred)
                x_pred = np.ravel(x_pred)
                y_pred = np.ravel(y_pred)

                # _calcXyAngle(xPred, yPred)
                do_other.append([x_pred, y_pred, '-'])
            do_other.append([xy[0], xy[1], 'ro'])

    TLineDrawer.plot_xy_with_mark(x_org, y_org, *do_other)


def plot_up_trend(x_org, y_org, mode=K_TREND_REG):
    min_list = TLineAnalyse.find_min_local(x_org, y_org)

    x_fit = np.array(min_list)
    y_fit = [y_org[pt] for pt in min_list]
    y_fit = np.array(y_fit)

    other = _split_xy_trend(x_fit, y_fit)
    _do_trend_plot(x_org, y_org, other, mode)
