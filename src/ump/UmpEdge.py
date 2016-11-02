# -*- encoding:utf-8 -*-
"""

由于现在在写一本关于股票量化方面的书，会涉及到相关文章里的一些内容，
出版社方面不希望我现在开源全部代码，但是一定会开源，最晚等书出版发行
以后，现在只能开源文章中涉及的部分代码，整个系统的开源会稍后，请谅解
我其实也觉着有人能看你的代码就已经很给面子了，但是。。。再次抱歉！！

"""
from __future__ import division

import ZCommonUtil
import ZEnv
import ZLog
from MlFiter import MlFiterClass
from UmpBase import CachedUmpManager
from sklearn.mixture import GMM
from sklearn.metrics.pairwise import pairwise_distances
import sklearn.preprocessing as preprocessing
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

__author__ = 'BBFamily'


class UmpEdgeClass(object):
    dump_clf_manager = CachedUmpManager()

    def __init__(self, orders_pd, predict=False):
        if not predict:
            orders_pd_tmp = orders_pd[orders_pd['result'] <> 0]
            """
                有结果也有可能是白玩 ['profit'] <> 0
            """
            orders_pd_tmp = orders_pd_tmp[orders_pd_tmp['profit'] <> 0]
            order_has_ret = orders_pd_tmp.filter(
                ['profit', 'profit_cg', 'atr_std', 'deg_hisWindowPd', 'deg_windowPd', 'deg_60WindowPd',
                 'wave_score1', 'wave_score2', 'wave_score3'])

            matrix = order_has_ret.as_matrix()

            y = matrix[:, :2]
            x = matrix[:, 2:]
            self.fiter = MlFiterClass(x, y, order_has_ret)
        self.scaler = preprocessing.StandardScaler()
        self.top_loss_ss = []
        self.top_win_ss = []

    def dump_file_fn(self):
        return ZEnv.g_project_root + '/data/cache/ump_edge'

    def dump_clf(self):
        # x_scale_param = self.scaler.fit(self.fiter.x)
        # filter_scale_x = self.scaler.fit_transform(self.fiter.x, x_scale_param)
        dump_clf = {'top_loss_ss': self.top_loss_ss, 'top_win_ss': self.top_win_ss,
                    'fiter_df': self.fiter.df, 'fiter_x': self.fiter.x}

        ZCommonUtil.dump_pickle(dump_clf, self.dump_file_fn())

    def predict(self, **kwargs):

        dump_clf = UmpEdgeClass.dump_clf_manager.get_ump(self)

        x = np.array([kwargs[col] for col in dump_clf['fiter_df'].columns[2:-3]])

        x = x.reshape(1, -1)
        con_x = np.concatenate((x, dump_clf['fiter_x']), axis=0)

        x_scale_param = self.scaler.fit(con_x)
        con_x = self.scaler.fit_transform(con_x, x_scale_param)

        distance_min_ind = pairwise_distances(con_x[0].reshape(1, -1), con_x[1:],
                                              metric='euclidean').argmin()
        '''
            置换出可以作为分类输入的x
        '''
        ss = dump_clf['fiter_df'].iloc[distance_min_ind]['ss']
        if ss in dump_clf['top_loss_ss']:
            return -1
        elif ss in dump_clf['top_win_ss']:
            return 1
        return 0

    def gmm_component_filter(self, nc=20, threshold=0.72, show=True):
        clf = GMM(nc, n_iter=500, random_state=3).fit(self.fiter.y)
        ss = clf.predict(self.fiter.y)

        self.fiter.df['p_rk_cg'] = self.fiter.df['profit_cg'].rank()
        self.fiter.df['ss'] = ss

        win_top = len(self.fiter.df['profit_cg']) - len(self.fiter.df['profit_cg']) * 0.25
        loss_top = len(self.fiter.df['profit_cg']) * 0.25
        self.fiter.df['rk'] = 0
        self.fiter.df['rk'] = np.where(self.fiter.df['p_rk_cg'] > win_top, 1, self.fiter.df['rk'])
        self.fiter.df['rk'] = np.where(self.fiter.df['p_rk_cg'] < loss_top, -1, self.fiter.df['rk'])

        xt = pd.crosstab(self.fiter.df['ss'], self.fiter.df['rk'])
        xt_pct = xt.div(xt.sum(1).astype(float), axis=0)

        if show:
            xt_pct.plot(
                figsize=(16, 8),
                kind='bar',
                stacked=True,
                title=str('ss') + ' -> ' + str('result'))
            plt.xlabel(str('ss'))
            plt.ylabel(str('result'))

            ZLog.info(xt_pct[xt_pct[-1] > threshold])
            ZLog.info(xt_pct[xt_pct[1] > threshold])

        self.top_loss_ss = xt_pct[xt_pct[-1] > threshold].index
        self.top_win_ss = xt_pct[xt_pct[1] > threshold].index
        return xt, xt_pct
