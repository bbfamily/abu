# -*- encoding:utf-8 -*-
"""

因子deg具体，xy解析

由于现在在写一本关于股票量化方面的书，会涉及到相关文章里的一些内容，
出版社方面不希望我现在开源全部代码，但是一定会开源，最晚等书出版发行
以后，现在只能开源文章中涉及的部分代码，整个系统的开源会稍后，请谅解
我其实也觉着有人能看你的代码就已经很给面子了，但是。。。再次抱歉！！

"""
from __future__ import division
from __future__ import print_function

import numpy as np
import pandas as pd
from MlFiterDhpPd import MlFiterDhpPdClass

__author__ = 'BBFamily'


g_w_col = ['deg_hisWindowPd', 'deg_windowPd', 'deg_60WindowPd']
g_regex_d = 'dh_dummies*|dw_dummies*|d60_dummies*'
g_regex_v = 'deg_hisWindow|deg_windowPd|deg_60WindowPd'


class MlFiterDegPdClass(MlFiterDhpPdClass):
    @classmethod
    def dump_dict_extend(cls):
        return None


    @classmethod
    def dummies_xy(cls, order_has_ret):
        """
        bins的选择不是胡来,根据binscs可视化数据结果进行
        :param order_has_ret:
        :return:
        """
        bins = [-np.inf, -20, -12, -7, -3, 0, 3, 7, 12, 20, np.inf]
        cats = pd.cut(order_has_ret.deg_hisWindowPd, bins)
        deg_his_window_dummies = pd.get_dummies(cats, prefix='dh_dummies')
        order_has_ret = pd.concat([order_has_ret, deg_his_window_dummies], axis=1)

        cats = pd.cut(order_has_ret.deg_windowPd, bins)
        deg_window_dummies = pd.get_dummies(cats, prefix='dw_dummies')
        order_has_ret = pd.concat([order_has_ret, deg_window_dummies], axis=1)

        cats = pd.cut(order_has_ret.deg_60WindowPd, bins)
        deg_60window_dummies = pd.get_dummies(cats, prefix='d60_dummies')
        order_has_ret = pd.concat([order_has_ret, deg_60window_dummies], axis=1)

        return order_has_ret

    def make_xy(self, **kwarg):
        self.make_dhp_xy(**kwarg)

        if kwarg is None or 'orderPd' not in kwarg:
            raise ValueError('kwarg is None or not kwarg.has_key ordersPd')

        order_pd = kwarg['orderPd']
        order_has_ret = order_pd[order_pd['result'] <> 0]
        order_has_ret['result'] = np.where(order_has_ret['result'] == -1, 0, 1)

        """
            将没有处理过的原始有效数据保存一份
        """
        self.order_has_ret = order_has_ret

        if self.dummies:
            order_has_ret = MlFiterDegPdClass.dummies_xy(order_has_ret)

        regex = g_regex_d if self.dummies else g_regex_v
        regex = 'result|' + regex

        self.do_make_xy(order_has_ret, regex)


