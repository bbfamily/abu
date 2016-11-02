# -*- encoding:utf-8 -*-
"""

由于现在在写一本关于股票量化方面的书，会涉及到相关文章里的一些内容，
出版社方面不希望我现在开源全部代码，但是一定会开源，最晚等书出版发行
以后，现在只能开源文章中涉及的部分代码，整个系统的开源会稍后，请谅解
我其实也觉着有人能看你的代码就已经很给面子了，但是。。。再次抱歉！！

"""
from __future__ import division
from __future__ import print_function

import ast
from abc import ABCMeta, abstractmethod
from collections import namedtuple

import numpy as np
import pandas as pd
from sklearn import tree

import ZLog
import six

__author__ = 'BBFamily'


class BoostDummies(namedtuple('BoostDummies',
                              (
                                      'name',
                                      'X',
                                      'y',
                                      'cat_list', 'clf'))):
    __slots__ = ()

    def __repr__(self):
        return "max: {0:.5f}, std: {1:.5f}, min: {2:.5f}, name: {3}".format(
            self.y.max(),
            self.y.std(),
            self.y.min(),
            self.name)


class MlFiterBoostClass(six.with_metaclass(ABCMeta, object)):
    s_qcut_bins = 100

    def __init__(self, **kwarg):
        """
            通过kwarg可以修改qcut_bins的值
        """
        if kwarg is None or 'orderPd' not in kwarg:
            raise ValueError('kwarg is None or not kwarg.has_key ordersPd')

        orderPd = kwarg['orderPd']
        self.orderPd = orderPd[orderPd['result'] <> 0]

        self.qcut_bins = MlFiterBoostClass.s_qcut_bins
        if 'qcut_bins' in kwarg:
            self.qcut_bins = kwarg['qcut_bins']
        self.bd_list = []
        self.above = 1.0
        self.below = 0.0

        self.make_boost(**kwarg)
        self.calc_above()
        self.calc_below()

    def make_boost_dummies(self, orderPd, cats_ss, prefix, regex):
        try:
            cats = pd.qcut(cats_ss, self.qcut_bins)
        except Exception, e:
            '''
                某一个数据超出q的数量导致无法分
            '''
            import pandas.core.algorithms as algos
            bins = algos.quantile(np.unique(cats_ss), np.linspace(0, 1, self.qcut_bins + 1))
            cats = pd.tools.tile._bins_to_cuts(cats_ss, bins, include_lowest=True)
            ZLog.info(prefix + ' qcut except use bins!')

        dummies = pd.get_dummies(cats, prefix=prefix)
        dummies_pd = pd.concat([orderPd, dummies], axis=1)
        df = dummies_pd.filter(regex=regex)

        features = df.columns.values[1:]
        ss = pd.Series()
        for feature in features:
            fr = df.groupby(feature)[df.columns.values[0]].value_counts()[1.0]
            ss[feature] = fr[1] / fr.sum()
        '''
            ss:
            ws3_dummies_[-0.847, -0.498]       0.462963
            ws3_dummies_(-0.498, -0.418]       0.528302
            ws3_dummies_(-0.418, -0.353]       0.566038
            ws3_dummies_(-0.353, -0.303]       0.444444
            ws3_dummies_(-0.303, -0.268]       0.358491
        '''
        X = pd.factorize(ss.index)[0].reshape(-1, 1)
        y = ss.values

        # mark_x = pd.factorize(ss.index)[1]

        def cat_to_tuple(cat):
            cat = cat.replace('[', '(')
            cat = cat.replace(']', ')')
            return ast.literal_eval(cat)[0]

        '''
            cat_list 最后组成一位数组，方便之后predict
            [-0.847,
             -0.498,
             -0.418,
             -0.353,
             -0.303
        '''
        cat_list = sorted(map(cat_to_tuple, cats.value_counts().index))

        '''
            暂时没有必要用复杂分类器
        '''
        clf = tree.DecisionTreeRegressor(random_state=1)
        clf.fit(X, y)
        self.bd_list.append(BoostDummies(cats_ss.name, X, y, cat_list, clf))

    @abstractmethod
    def fit(self):
        pass

    @abstractmethod
    def calc_above(self):
        pass

    @abstractmethod
    def calc_below(self):
        pass

    @abstractmethod
    def make_boost(self, **kwarg):
        pass
