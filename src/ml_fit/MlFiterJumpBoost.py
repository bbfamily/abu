# -*- encoding:utf-8 -*-
"""

由于现在在写一本关于股票量化方面的书，会涉及到相关文章里的一些内容，
出版社方面不希望我现在开源全部代码，但是一定会开源，最晚等书出版发行
以后，现在只能开源文章中涉及的部分代码，整个系统的开源会稍后，请谅解
我其实也觉着有人能看你的代码就已经很给面子了，但是。。。再次抱歉！！

"""

from __future__ import division
from __future__ import print_function

import itertools

import numpy as np
import pandas as pd
from sklearn import metrics

import ZLog
from MlFiterBoost import MlFiterBoostClass

__author__ = 'BBFamily'


class MlFiterJumpBoostClass(MlFiterBoostClass):
    s_dd_threshold = 20

    def _boost_dummies_weights(self, order, above, below, weights, w_v=0.001, offset=True):
        for bd in self.bd_list:
            query = order[bd.name]
            q_list = list()
            q_list.append(query)
            q_ind = sorted(itertools.chain(bd.cat_list, q_list)).index(query)
            predict = bd.clf.predict(q_ind)

            mv = w_v
            if predict >= above:
                result = 1
                if offset:
                    mv = np.maximum(predict - above, w_v)
            elif predict <= below:
                result = 0

                if offset:
                    mv = np.maximum(below - predict, w_v)
            else:
                continue

            if result == order.result:
                weights[bd.name] += mv
            else:
                weights[bd.name] -= mv

    def _boost_dummies_predict(self, order, weights, below):
        reslut_ss = pd.Series()
        for bd in self.bd_list:
            query = order[bd.name]
            q_list = list()
            q_list.append(query)
            q_ind = sorted(itertools.chain(bd.cat_list, q_list)).index(query)
            predict = bd.clf.predict(q_ind)
            reslut_ss[bd.name] = predict
        predict = np.dot(reslut_ss.values.reshape(-1, 1).T, weights.values.reshape(-1, 1))
        predict = predict[0]

        result = 1
        if predict <= below:
            result = 0

        return result

    def recall_score(self):
        order_diff_days = self.orderPd[
            (self.orderPd['diff_days'] > 0) & (self.orderPd['diff_days'] <= self.dd_threshold)]

        below_Line = self.below * 0.9
        predict_both = order_diff_days.apply(self._boost_dummies_predict, axis=1, args=(self.weights_both, below_Line,))
        predict_above = order_diff_days.apply(self._boost_dummies_predict, axis=1,
                                              args=(self.weights_above, below_Line,))
        predict_below = order_diff_days.apply(self._boost_dummies_predict, axis=1,
                                              args=(self.weights_below, below_Line,))

        order_diff_days['predict_above'] = predict_above
        order_diff_days['predict_below'] = predict_below
        order_diff_days['predict_both'] = predict_both

        # predict_plus = np.array([predict_both, predict_above, predict_below]).T.sum(axis=1)
        # order_diff_days['predict'] = np.where(predict_plus <= 0, 0, 1)
        # predict_loss_pd_plus = order_diff_days[predict_plus <= 0]
        # ZLog.info(predict_loss_pd_plus.shape)
        # ZLog.info(metrics.accuracy_score(predict_loss_pd_plus['result'], predict_loss_pd_plus['predict']))

        predict_loss_pd = order_diff_days[order_diff_days['predict_both'] == 0]
        ZLog.info('predict_both')
        ZLog.info(predict_loss_pd.shape)
        ZLog.info(metrics.accuracy_score(predict_loss_pd['result'], predict_loss_pd['predict_both']))
        ZLog.newline()

        predict_loss_pd = order_diff_days[order_diff_days['predict_below'] == 0]
        ZLog.info('predict_below')
        ZLog.info(predict_loss_pd.shape)
        ZLog.info(metrics.accuracy_score(predict_loss_pd['result'], predict_loss_pd['predict_below']))
        ZLog.newline()

        predict_loss_pd = order_diff_days[order_diff_days['predict_above'] == 0]
        ZLog.info('predict_above')
        ZLog.info(predict_loss_pd.shape)
        ZLog.info(metrics.accuracy_score(predict_loss_pd['result'], predict_loss_pd['predict_above']))
        ZLog.newline()

        return order_diff_days

    def fit(self):
        order_diff_days = self.orderPd[
            (self.orderPd['diff_days'] > 0) & (self.orderPd['diff_days'] <= self.dd_threshold)]

        # weights_eq = pd.Series(np.ones(len(jump_boost.bd_list))/len(jump_boost.bd_list), index=[bd.name for bd in jump_boost.bd_list])

        weights_above = pd.Series(np.ones(len(self.bd_list)) / len(self.bd_list),
                                  index=[bd.name for bd in self.bd_list])
        order_diff_days.apply(self._boost_dummies_weights, axis=1, args=(self.above, self.above, weights_above))
        self.weights_above = weights_above / weights_above.sum()
        ZLog.info('weights_above')
        ZLog.info(self.weights_above)
        ZLog.newline()

        weights_below = pd.Series(np.ones(len(self.bd_list)) / len(self.bd_list),
                                  index=[bd.name for bd in self.bd_list])
        order_diff_days.apply(self._boost_dummies_weights, axis=1, args=(self.below, self.below, weights_below))
        weights_below = weights_below + weights_below.mean()
        self.weights_below = weights_below / weights_below.sum()
        ZLog.info('weights_below')
        ZLog.info(self.weights_below)
        ZLog.newline()

        weights_both = pd.Series(np.ones(len(self.bd_list)) / len(self.bd_list), index=[bd.name for bd in self.bd_list])
        order_diff_days.apply(self._boost_dummies_weights, axis=1, args=(self.above, self.below, weights_both))
        self.weights_both = weights_both / weights_both.sum()
        ZLog.info('weights_both')
        ZLog.info(self.weights_both)

    def calc_above(self):
        orderPd = self.orderPd
        order_outer_diff_days = orderPd[(orderPd['diff_days'] == 0) | (orderPd['diff_days'] > self.dd_threshold)]
        self.above = order_outer_diff_days.result.value_counts()[
                         1].sum() / order_outer_diff_days.result.value_counts().sum()
        ZLog.info('above win rate: ' + str(self.above))
        return self.above

    def calc_below(self):
        order_diff_days = self.orderPd[
            (self.orderPd['diff_days'] > 0) & (self.orderPd['diff_days'] <= self.dd_threshold)]
        self.below = order_diff_days.result.value_counts()[1].sum() / order_diff_days.result.value_counts().sum()
        ZLog.info('below win rate: ' + str(self.below))
        return self.below

    def make_boost(self, **kwarg):
        self.dd_threshold = MlFiterJumpBoostClass.s_dd_threshold
        if kwarg.has_key('dd_threshold'):
            self.dd_threshold = kwarg['dd_threshold']

        order_diff_days = self.orderPd[
            (self.orderPd['diff_days'] > 0) & (self.orderPd['diff_days'] <= self.dd_threshold)]
        self.orderPd['result'] = np.where(self.orderPd['result'] == -1, 0, 1)

        self.make_boost_dummies(order_diff_days, order_diff_days.jump_power,
                                prefix='power_dummies', regex='result|power_dummies*')

        self.make_boost_dummies(order_diff_days, order_diff_days.wave_score1,
                                prefix='ws1_dummies', regex='result|ws1_dummies*')
        self.make_boost_dummies(order_diff_days, order_diff_days.wave_score2,
                                prefix='ws2_dummies', regex='result|ws2_dummies*')
        self.make_boost_dummies(order_diff_days, order_diff_days.wave_score3,
                                prefix='ws3_dummies', regex='result|ws3_dummies*')

        # self.make_boost_dummies(order_diff_days, order_diff_days.deg_hisWindowPd, 
        #     prefix='dh_dummies', regex='result|dh_dummies*')
        self.make_boost_dummies(order_diff_days, order_diff_days.deg_windowPd,
                                prefix='dw_dummies', regex='result|dw_dummies*')
        self.make_boost_dummies(order_diff_days, order_diff_days.deg_60WindowPd,
                                prefix='d60_dummies', regex='result|d60_dummies*')
