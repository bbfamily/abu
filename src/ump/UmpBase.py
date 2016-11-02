# -*- encoding:utf-8 -*-
"""

由于现在在写一本关于股票量化方面的书，会涉及到相关文章里的一些内容，
出版社方面不希望我现在开源全部代码，但是一定会开源，最晚等书出版发行
以后，现在只能开源文章中涉及的部分代码，整个系统的开源会稍后，请谅解
我其实也觉着有人能看你的代码就已经很给面子了，但是。。。再次抱歉！！

"""
from __future__ import division

import copy
import MlFiterSnn
import MlFiterTensorFlow
import ZCommonUtil
import ZLog

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.optimize as sco

from mpl_toolkits.mplot3d import Axes3D
from sklearn.mixture import GMM
from Decorator import warnings_filter

from abc import ABCMeta, abstractmethod
import six

__author__ = 'BBFamily'


class CachedUmpManager:
    def __init__(self):
        # import weakref
        # self._cache = weakref.WeakValueDictionary()
        self._cache = dict()

    def get_ump(self, ump):
        name = ump.dump_file_fn()
        if name not in self._cache:
            dump_clf_with_ind = ZCommonUtil.load_pickle(name)
            self._cache[name] = dump_clf_with_ind
        else:
            dump_clf_with_ind = self._cache[name]
        return dump_clf_with_ind

    def clear(self):
        self._cache.clear()


class UmpBaseClass(six.with_metaclass(ABCMeta, object)):
    dump_clf_manager = CachedUmpManager()

    @abstractmethod
    def dump_file_fn(self):
        pass

    def __init__(self, orders_pd, fiter_cls, predict=False, **kwarg):
        self.orders_pd = orders_pd
        # 不再特定
        # self.fiter = MlFiterGoldenPd.MlFiterGoldenPdClass(orderPd=self.orders_pd, dummies=False)
        self.fiter_cls = fiter_cls

        if predict:
            return

        self.fiter = fiter_cls(orderPd=self.orders_pd, **kwarg)
        """
            默认svm
        """
        self.fiter().estimator.svc()
        # self.golden_fiter().estimator.random_forest_classifier(n_estimators=100)
        # self.golden_fiter().estimator.adaboost_classifier(n_estimators=200)
        self.df = None

    def dump_clf(self, llps):
        """
        :param llps: cprs[(so.cprs['lps'] < 0) & (so.cprs['lms'] < -0.0)]
        你所需要的符合筛选条件的cprs
        :return:
        """
        dump_clf_with_ind = {}
        for rt_ind in llps.index:
            component, sub_ind = rt_ind.split('_')
            component = int(component)
            sub_ind = int(sub_ind)
            dump_clf_with_ind[rt_ind] = (self.rts[component][0], sub_ind)
        ZCommonUtil.dump_pickle(dump_clf_with_ind, self.dump_file_fn())

    def hit_cnt(self, x):
        dump_clf_with_ind = UmpBaseClass.dump_clf_manager.get_ump(self)
        hit_cnt = 0
        for clf, ind in dump_clf_with_ind.values():
            ss = clf.predict(x)
            if ss == ind:
                hit_cnt += 1
        return hit_cnt

    def predict(self, x, need_ind_cnt=1):
        dump_clf_with_ind = UmpBaseClass.dump_clf_manager.get_ump(self)
        count_ind = 0
        for clf, ind in dump_clf_with_ind.values():
            ss = clf.predict(x)
            if ss == ind:
                count_ind += 1
                if need_ind_cnt == count_ind:
                    return 0
        return 1

    def predict_kwargs(self, w_col, need_ind_cnt=1, **kwargs):
        for col in w_col:
            if col not in kwargs:
                ZLog.info('judge kwargs error!')
                return

        x = np.array([kwargs[col] for col in w_col])
        x = x.reshape(1, -1)

        return self.predict(x, need_ind_cnt) == 1

    def predict_hit_kwargs(self, w_col, **kwargs):
        for col in w_col:
            if col not in kwargs:
                ZLog.info('judge kwargs error!')
                return

        x = np.array([kwargs[col] for col in w_col])
        x = x.reshape(1, -1)

        return self.hit_cnt(x)

    @warnings_filter
    def show_general(self, use_fiter=False):
        order_has_ret_fit = self.fiter.order_has_ret if use_fiter else self.orders_pd[self.orders_pd['result'] <> 0]

        ZLog.info('all fit order = ' + str(order_has_ret_fit.shape))

        xt = order_has_ret_fit.result.value_counts()
        ZLog.info('win rate = ' + str(xt[1] / xt.sum()))
        ZLog.info('profit_cg.sum() = ' + str(order_has_ret_fit.profit_cg.sum()))

        order_has_ret_fit.sort_values('buy Date')['profit_cg'].cumsum().plot(grid=True, title='profit_cg cumsum')

        profit_cg_win_mean = order_has_ret_fit[order_has_ret_fit['profit_cg'] > 0].profit_cg.mean()
        profit_cg_loss_mean = order_has_ret_fit[order_has_ret_fit['profit_cg'] < 0].profit_cg.mean()
        ZLog.info('win mean = {0} loss_mean = {1} '.format(profit_cg_win_mean, profit_cg_loss_mean))
        plt.show()

    def show_snn_tt(self):
        MlFiterSnn.SnnClass.do_snn_tt(self.fiter().x, self.fiter().y, n_folds=10, print_loss=True)

    def show_knn_tt(self):
        MlFiterTensorFlow.KnnTF.do_tf_tt(self.fiter().x, self.fiter().y, n_folds=10)

    def show_mnn_tt(self):
        MlFiterTensorFlow.MnnTF.do_tf_tt(self.fiter().x, self.fiter().y, n_folds=10)

    def parse_rt(self, rt, order_has_ret, rt_ind):
        clf = rt[0]
        ss = clf.predict(self.fiter().x)
        self.df['ss'] = ss
        """
            如果不是对应序列原始单子的就的匹配查找
        """

        # noinspection PyUnusedLocal
        def match_profit(x, p_order_has_ret):
            match = p_order_has_ret[(p_order_has_ret.index == x.name) &
                                    (p_order_has_ret.atr_std == x.atr_std) &
                                    (p_order_has_ret.wave_score1 == x.wave_score1) &
                                    (p_order_has_ret.deg_windowPd == x.deg_windowPd) &
                                    (p_order_has_ret.deg_60WindowPd == x.deg_60WindowPd) &
                                    (p_order_has_ret.deg_hisWindowPd == x.deg_hisWindowPd)]
            pf_cg = match.profit_cg
            return pf_cg.values[0]

        ind = rt_ind
        loss_rate = self.df[self.df['ss'] == ind]['result'].value_counts()[0] / self.df[self.df['ss'] == ind][
            'result'].value_counts().sum()

        loss_cnt = self.df[self.df['ss'] == ind].shape[0]
        df_cpt = self.df[self.df['ss'] == ind]
        df_cpt['profit'] = df_cpt.apply(lambda x: order_has_ret.ix[int(x.ind)].profit_cg, axis=1)
        # 如果不是对应序列原始单子的就的匹配查找
        # df_ind['prfit'] = df_ind.apply(match_profit, axis=1, args=(order_has_ret,))
        p_mean = df_cpt['profit'].mean()
        p_sum = df_cpt['profit'].sum()

        return loss_cnt, loss_rate, p_sum, p_mean, df_cpt

    def show_parse_rt(self, rt):
        """
        直观显示某一个rt的crosstab div图
        :param rt:
        :return:
        """
        clf = rt[0]
        ss = clf.predict(self.fiter().x)
        self.df['ss'] = ss
        xt = pd.crosstab(self.df['ss'], self.df['result'])
        xt_pct = xt.div(xt.sum(1).astype(float), axis=0)
        xt_pct.plot(
            figsize=(16, 8),
            kind='bar',
            stacked=True,
            title=str('ss') + ' -> ' + str('result'))
        plt.xlabel(str('ss'))
        plt.ylabel(str('result'))

    def _gmm_parse_rt_plot(self, show=True):
        rts = self.rts
        lcs = []
        lrs = []
        lps = []
        lms = []
        cps = []
        nts = {}
        for cp, rt in rts.items():
            for rt_ind in rt[1]:
                loss_cnt, loss_rate, loss_ps, loss_pm, df_cpt = self.parse_rt(rt, self.fiter.order_has_ret,
                                                                              rt_ind)
                lcs.append(loss_cnt)
                lrs.append(loss_rate)
                lps.append(loss_ps)
                lms.append(loss_pm)
                """
                    component + '_' + gmm fit index
                """
                cps_key = '{0}_{1}'.format(cp, rt_ind)
                cps.append(cps_key)
                nts[cps_key] = df_cpt

        if show:
            cmap = plt.get_cmap('jet', 20)
            cmap.set_under('gray')
            fig, ax = plt.subplots()
            cax = ax.scatter(lrs, lcs, c=lps, cmap=cmap, vmin=np.min(lps),
                             vmax=np.max(lps))
            fig.colorbar(cax, label='lps', extend='min')
            plt.grid(True)
            plt.xlabel('lrs')
            plt.ylabel('lcs')
            plt.show()

            fig = plt.figure(figsize=(9, 6))
            ax = fig.gca(projection='3d')
            ax.view_init(30, 60)
            ax.scatter3D(lrs, lcs, lps, c=lms, s=50, cmap='spring')
            ax.set_xlabel('lrs')
            ax.set_ylabel('lcs')
            ax.set_zlabel('lms')
            plt.show()

        cprs = pd.DataFrame([lcs, lrs, lps, lms], index=['lcs', 'lrs', 'lps', 'lms'], columns=cps).T
        return cprs, nts

    def choose_cprs_component(self, llps):
        """
        :param llps: cprs[(so.cprs['lps'] < 0) & (so.cprs['lms'] < -0.0)]
        你所需要的符合筛选条件的cprs
        :return:
        """
        if not hasattr(self, 'cprs'):
            raise ValueError('gmm_component_filter not exe!!!! ')

        nts_pd = pd.DataFrame()
        for nk in llps.index:
            nts_pd = nts_pd.append(self.nts[nk])
        nts_pd = nts_pd.drop_duplicates(subset='ind', keep='last')
        ZLog.info('nts_pd.shape = {0}'.format(nts_pd.shape))
        loss_rate = nts_pd.result.value_counts()[0] / nts_pd.result.value_counts().sum()
        win_rate = nts_pd.result.value_counts()[1] / nts_pd.result.value_counts().sum()
        ZLog.info('nts_pd loss rate = {0}'.format(loss_rate))

        improved = (nts_pd.shape[0] / self.fiter.order_has_ret.shape[0]) * (loss_rate - win_rate)
        ZLog.info('improved rate = {0}'.format(improved))

        xt = self.fiter.order_has_ret.result.value_counts()
        ZLog.info('predict win rate = ' + str(xt[1] / xt.sum() + improved))

        nts_pd.sort_index()['profit'].cumsum().plot()
        plt.show()

    def gmm_component_filter(self, p_ncs=None, threshold=0.65, show=True):
        """
        :param p_ncs: 分类范围,
        :param threshold: 选择阀值
        :param show:
        :return:
        """
        ncs = p_ncs
        if ncs is None:
            ncs = np.arange(40, 85)
        df = copy.deepcopy(self.fiter().df)
        """
            添加一个索引序列，方便之后快速查找原始单据
        """
        df['ind'] = np.arange(0, df.shape[0])
        rts = {}
        for component in ncs:
            clf = GMM(component, n_iter=500, random_state=3).fit(self.fiter().x)
            ss = clf.predict(self.fiter().x)
            df['ss'] = ss
            xt = pd.crosstab(df[
                                 'ss'], df['result'])
            xt_pct = xt.div(xt.sum(1).astype(float), axis=0)
            if len(xt_pct[xt_pct[0] > threshold].index) > 0:
                rts[component] = (clf, xt_pct[xt_pct[0] > threshold].index)

        self.rts = rts
        self.df = df
        self.cprs, self.nts = self._gmm_parse_rt_plot(show)

        if show:
            self.cprs['lps'].plot(kind='bar')
            plt.show()
            self.cprs['lcs'].plot(kind='bar')
            plt.show()
        return self.cprs

    def brust_min(self):
        """
        全局最优
        :return:
        """
        cprs = self.cprs
        optv = sco.brute(self.min_func_improved, ((round(cprs['lps'].min(), 2), 0, 0.5), (round(cprs['lms'].min(), 2),
                                                                                          round(cprs['lms'].max(), 3),
                                                                                          0.01),
                                                  (round(cprs['lrs'].min(), 2), round(cprs['lrs'].max(), 2), 0.1)),
                         finish=None)
        return optv

    def sco_min(self, guess):
        """
        局部最优借
        :param guess:
        :return:
        """
        cprs = self.cprs
        bnds = ((round(cprs['lps'].min(), 3), round(cprs['lps'].max(), 3)),
                (round(cprs['lms'].min(), 3), round(cprs['lms'].max(), 3)),
                (round(cprs['lrs'].min(), 3), round(cprs['lrs'].max(), 3)))

        optv = sco.minimize(self.min_func_improved, guess, method='BFGS',
                            bounds=bnds)
        return optv

    def min_func(self, lpmr):
        cprs = self.cprs
        nts = self.nts

        llps = cprs[(cprs['lps'] <= lpmr[0]) & (cprs['lms'] <= lpmr[1]) & (cprs['lrs'] >= lpmr[2])]

        nts_pd = pd.DataFrame()
        for nk in llps.index:
            nts_pd = nts_pd.append(nts[nk])
        if nts_pd.empty:
            return np.array([0.0001, 0])
        nts_pd = nts_pd.drop_duplicates(subset='ind', keep='last')

        num = nts_pd.shape[0]
        loss_rate = nts_pd.result.value_counts()[0] / nts_pd.result.value_counts().sum()
        win_rate = nts_pd.result.value_counts()[1] / nts_pd.result.value_counts().sum()
        improved = (nts_pd.shape[0] / self.fiter.order_has_ret.shape[0]) * (loss_rate - win_rate)
        # print improved
        return np.array([improved, num])

    def min_func_improved(self, lpmr):
        """
            求最大提高，min负数
        """
        return -self.min_func(lpmr)[0]
