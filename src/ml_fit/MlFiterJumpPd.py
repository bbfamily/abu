# -*- encoding:utf-8 -*-
"""

xy解析

由于现在在写一本关于股票量化方面的书，会涉及到相关文章里的一些内容，
出版社方面不希望我现在开源全部代码，但是一定会开源，最晚等书出版发行
以后，现在只能开源文章中涉及的部分代码，整个系统的开源会稍后，请谅解
我其实也觉着有人能看你的代码就已经很给面子了，但是。。。再次抱歉！！

"""
from __future__ import division
from __future__ import print_function

import numpy as np
import pandas as pd

import MlFiterDhpVerify
from MlFiterDhpPd import MlFiterDhpPdClass

__author__ = 'BBFamily'

# g_w_col = ['atr_std', 'deg_hisWindowPd', 'deg_windowPd', 'deg_60WindowPd', 'jump_power', 'diff_days',
#            'wave_score1', 'wave_score2', 'wave_score3']
#
# g_regex_d = 'dd_dummies*|power_dummies*|ws1_dummies*|ws2_dummies*|' \
#             'ws3_dummies*|atr_dummies*|dh_dummies*|dw_dummies*|d60_dummies*'
#
# g_regex_v = 'diff_days|jump_power|wave_score1|wave_score2|wave_score3|atr_std|' \
#             'deg_hisWindow|deg_windowPd|deg_60WindowPd'


g_w_col = ['jump_power', 'diff_days']

g_regex_d = 'dd_dummies*|power_dummies*'

g_regex_v = 'diff_days|jump_power'


class MlFiterJumpPdClass(MlFiterDhpPdClass):
    """
        默认jump diff dd_threshold 天鼠
        可通过kwarg传入
    """
    s_dd_threshold = 20

    @classmethod
    def dump_dict_extend(cls):
        return dict(dd_threshold=MlFiterJumpPdClass.s_dd_threshold)

    @classmethod
    def verify_process(cls, order_pd, only_jd=False, first_local=False, tn_threshold=800):
        """
        :param order_pd:
        :param only_jd: 使用以序列化的只进行judge
        :param first_local: 优先使用本地分类器
        :param tn_threshold:
        :return:
        """
        from MlFiterJumpJudge import MlFiterJumpJudgeClass
        judge_cls = MlFiterJumpJudgeClass
        make_x_func = lambda order: dict(deg_hisWindowPd=order.deg_hisWindowPd, deg_windowPd=order.deg_windowPd,
                                         deg_60WindowPd=order.deg_60WindowPd, diff_days=order.diff_days,
                                         jump_power=order.jump_power, lowBkCnt=order.lowBkCnt, atr_std=order.atr_std,
                                         wave_score1=order.wave_score1,
                                         wave_score2=order.wave_score2, wave_score3=order.wave_score3)

        def make_order_func(p_order_pd):
            in_order_has_ret = p_order_pd[(p_order_pd['diff_days'] > 0) & (p_order_pd['diff_days'] <= 20)]
            in_order_has_ret = in_order_has_ret[in_order_has_ret['result'] <> 0]
            in_order_has_ret['result'] = np.where(in_order_has_ret['result'] == -1, 0, 1)
            return in_order_has_ret

        return MlFiterDhpVerify.verify_process(cls, judge_cls, make_x_func, make_order_func, order_pd=order_pd,
                                               only_jd=only_jd,
                                               first_local=first_local, tn_threshold=tn_threshold)

    @classmethod
    def dummies_xy(cls, order_has_ret):
        bins = [-np.inf, 1.0, 1.1, 1.2, 1.3, 1.5, 1.7, 2.0, 2.5, 5.0, np.inf]
        cats = pd.cut(order_has_ret.jump_power, bins)
        jump_power_dummies = pd.get_dummies(cats, prefix='power_dummies')
        order_has_ret = pd.concat([order_has_ret, jump_power_dummies], axis=1)

        # cats = pd.qcut(order_has_ret.diff_days, 3)
        bins = [-np.inf, 3, 6, 9, 12, 14, 16, 18, np.inf]
        cats = pd.cut(order_has_ret.diff_days, bins)
        diff_days_dummies = pd.get_dummies(cats, prefix='dd_dummies')
        order_has_ret = pd.concat([order_has_ret, diff_days_dummies], axis=1)


        # bins = [-np.inf, 1, 2, 3, 4, 7, 11, 15, np.inf]
        # cats = pd.cut(order_has_ret.lowBkCnt, bins)
        # low_bk_cnt_dummies = pd.get_dummies(cats, prefix='bkcnt_dummies')
        # order_has_ret = pd.concat([order_has_ret, low_bk_cnt_dummies], axis=1)
        # cats = pd.qcut(order_has_ret.lowBkCnt, 2)

        # bins = [-np.inf, 0.0, 0.1, 0.2, 0.4, 0.50, 0.85, 1.0, 1.2, np.inf]
        # cats = pd.cut(order_has_ret.wave_score1, bins)
        # wave_score1_dummies = pd.get_dummies(cats, prefix='ws1_dummies')
        # order_has_ret = pd.concat([order_has_ret, wave_score1_dummies], axis=1)
        #
        # cats = pd.cut(order_has_ret.wave_score2, bins)
        # wave_score2_dummies = pd.get_dummies(cats, prefix='ws2_dummies')
        # order_has_ret = pd.concat([order_has_ret, wave_score2_dummies], axis=1)
        #
        # cats = pd.cut(order_has_ret.wave_score3, bins)
        # wave_score3_dummies = pd.get_dummies(cats, prefix='ws3_dummies')
        # order_has_ret = pd.concat([order_has_ret, wave_score3_dummies], axis=1)
        #
        # bins = [-np.inf, 0.0, 0.2, 0.4, 0.6, 0.8, 1.0, np.inf]
        # cats = pd.cut(order_has_ret.atr_std, bins)
        # atr_dummies = pd.get_dummies(cats, prefix='atr_dummies')
        # order_has_ret = pd.concat([order_has_ret, atr_dummies], axis=1)
        #
        # bins = [-np.inf, -20, -12, -7, -3, 0, 3, 7, 12, 20, np.inf]
        # cats = pd.cut(order_has_ret.deg_hisWindowPd, bins)
        # deg_his_window_dummies = pd.get_dummies(cats, prefix='dh_dummies')
        # order_has_ret = pd.concat([order_has_ret, deg_his_window_dummies], axis=1)
        #
        # cats = pd.cut(order_has_ret.deg_windowPd, bins)
        # deg_window_dummies = pd.get_dummies(cats, prefix='dw_dummies')
        # order_has_ret = pd.concat([order_has_ret, deg_window_dummies], axis=1)
        #
        # cats = pd.cut(order_has_ret.deg_60WindowPd, bins)
        # deg_60window_dummies = pd.get_dummies(cats, prefix='d60_dummies')
        # order_has_ret = pd.concat([order_has_ret, deg_60window_dummies], axis=1)

        return order_has_ret

    def make_xy(self, **kwarg):
        self.make_dhp_xy(**kwarg)

        if kwarg is None or not kwarg.has_key('orderPd'):
            raise ValueError('kwarg is None or not kwarg.has_key ordersPd')
        order_pd = kwarg['orderPd']

        self.dd_threshold = MlFiterJumpPdClass.s_dd_threshold
        if kwarg.has_key('dd_threshold'):
            self.dd_threshold = kwarg['dd_threshold']

        order_pd = order_pd[(order_pd['diff_days'] > 0) & (order_pd['diff_days'] <= self.dd_threshold)]
        order_has_ret = order_pd[order_pd['result'] <> 0]
        order_has_ret['result'] = np.where(order_has_ret['result'] == -1, 0, 1)

        """
            将没有处理过的原始有效数据保存一份
        """
        self.order_has_ret = order_has_ret

        if self.dummies:
            order_has_ret = MlFiterJumpPdClass.dummies_xy(order_has_ret)

        regex = g_regex_d if self.dummies else g_regex_v
        regex = 'result|' + regex

        self.do_make_xy(order_has_ret, regex)
