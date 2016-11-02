#-*- encoding:utf-8 -*-
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

from MlFiterPd import MlFiterPdClass

__author__ = 'BBFamily'


class MlFiterDlwPdClass(MlFiterPdClass):

    def make_xy(self, **kwarg):
        if kwarg is None or not kwarg.has_key('orderPd'):
            raise ValueError('kwarg is None or not kwarg.has_key ordersPd')

        order_pd = kwarg['orderPd']
        order_has_ret = order_pd[order_pd['result'] <> 0]
        order_has_ret['result'] = np.where(order_has_ret['result'] == -1, 0, 1)
        deg_df = order_has_ret.filter(
            regex='result|deg_hisWindowPd|deg_windowPd|deg_60WindowPd|lowBkCnt|wave_score1|wave_score2|wave_score3')
        deg_np = deg_df.as_matrix()

        self.y = deg_np[:, 0]
        self.x = deg_np[:, 1:]
        self.df = deg_df
        self.np = deg_np
