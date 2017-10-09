# -*- encoding:utf-8 -*-
"""回测结果评分模块"""
from __future__ import print_function
from __future__ import absolute_import
from __future__ import division
from abc import ABCMeta, abstractmethod
from collections import namedtuple

import numpy as np
import pandas as pd

from ..CoreBu.ABuFixes import six
from .ABuMetricsBase import AbuMetricsBase

__author__ = '阿布'
__weixin__ = 'abu_quant'


# noinspection PyClassHasNoInit
class AbuScoreTuple(namedtuple('AbuScoreTuple',
                               ('orders_pd',
                                'action_pd',
                                'capital',
                                'benchmark',
                                'buy_factors',
                                'sell_factors',
                                'stock_picks'))):
    """namedtuple扩展类，在GridSearch中包装回测参数及结果"""

    __slots__ = ()

    def __repr__(self):
        return "orders_pd:{}\naction_pd:{}\ncapital:{}\nbenchmark:{}\n" \
               "buy_factors:{}\nsell_factors:{}\nstock_picks:{}".format(
                self.orders_pd.info() if self.orders_pd is not None else 'zero order',
                self.action_pd.info() if self.action_pd is not None else 'zero action', self.capital, self.benchmark,
                self.buy_factors, self.sell_factors, self.stock_picks)


class AbuBaseScorer(six.with_metaclass(ABCMeta, object)):
    """针对GridSearch的score_tuple_array进行评分抽象基类"""

    # noinspection PyUnresolvedReferences
    def __init__(self, score_tuple_array, *arg, **kwargs):
        """
        :param score_tuple_array: 承接GridSearch返回的AbuScoreTuple对象序列
        :param kwargs: 可选weights代表评分项权重， 可选metrics_class代表交易目标度量类
        """
        self.score_tuple_array = score_tuple_array
        self.score_dict = {}
        self.weights_cnt = -1
        # 设置度量项抽取函数select_score_func，度量名称columns_name, weights_cnt
        self._init_self_begin(arg, *arg, **kwargs)

        # 检测_init_self_begin中必须要子类设置的有没有设置ok
        if not hasattr(self, 'select_score_func'):
            raise RuntimeError('_init_self_begin must set select_score_func')
        if not hasattr(self, 'columns_name'):
            raise RuntimeError('_init_self_begin must set columns_name')

        # 如果有设置权重就分配权重否则等权重
        if 'weights' in kwargs and kwargs['weights'] is not None and len(kwargs['weights']) == self.weights_cnt:
            self.weights = kwargs['weights']
        else:
            self.weights = self.weights_cnt * [1. / self.weights_cnt, ]

        # metrics_class = kwargs.pop('metrics_class', AbuMetricsBase)
        if 'metrics_class' in kwargs and kwargs['metrics_class'] is not None \
                and issubclass(kwargs['metrics_class'], AbuMetricsBase):
            self.metrics_class = kwargs['metrics_class']
        else:
            self.metrics_class = AbuMetricsBase

        valid_score_tuple_array = []
        for ind, score_tuple in enumerate(self.score_tuple_array):
            # 一个一个的进行度量
            metrics = self.metrics_class(score_tuple.orders_pd, score_tuple.action_pd, score_tuple.capital,
                                         score_tuple.benchmark)
            if metrics.valid:
                metrics.fit_metrics()
                # 使用子类_init_self_begin中设置的select_score_func方法选取
                self.score_dict[ind] = self.select_score_func(metrics)
                valid_score_tuple_array.append(score_tuple)
        # 把筛选出来有交易结果的重新放到score_tuple_array中
        self.score_tuple_array = valid_score_tuple_array

        # 将score_dict转换DataFrame并且转制
        self.score_pd = pd.DataFrame(self.score_dict).T
        # 设置度量指标名称
        self.score_pd.columns = self.columns_name
        """
            一般的任务是将score_pd中需要反转的度量结果进行反转
            所以要在rank前做self._init_self_end
        """
        self._init_self_end(arg, *arg, **kwargs)

        # 分数每一项都由0-1
        score_ls = np.linspace(0, 1, self.score_pd.shape[0])
        for cn in self.columns_name:
            # 每一项的结果rank后填入对应项
            score = score_ls[(self.score_pd[cn].rank().values - 1).astype(int)]
            self.score_pd['score_' + cn] = score

        scores = self.score_pd.filter(regex='score_*')
        # 根据权重计算最后的得分
        self.score_pd['score'] = scores.apply(lambda s: (s * self.weights).sum(), axis=1)

    @abstractmethod
    def _init_self_begin(self, *arg, **kwargs):
        """子类需要实现，设置度量项抽取函数select_score_func，度量名称columns_name，weights_cnt"""
        pass

    @abstractmethod
    def _init_self_end(self, *arg, **kwargs):
        """子类需要实现，一般的任务是将score_pd中需要反转的度量结果进行反转"""
        pass

    def fit_score(self):
        """对度量结果按照score排序，返回排序后的score列"""
        self.score_pd.sort_values(by='score', inplace=True)
        return self.score_pd['score']

    def __call__(self):
        """call self.fit_score"""
        return self.fit_score()


class WrsmScorer(AbuBaseScorer):
    def _init_self_begin(self, *arg, **kwargs):
        """胜率，策略收益，策略sharpe值，策略最大回撤组成select_score_func"""

        self.select_score_func = lambda metrics: [metrics.win_rate, metrics.algorithm_period_returns,
                                                  metrics.algorithm_sharpe,
                                                  metrics.max_drawdown]
        self.columns_name = ['win_rate', 'returns', 'sharpe', 'max_drawdown']
        self.weights_cnt = len(self.columns_name)

    def _init_self_end(self, *arg, **kwargs):
        """
        _init_self_end这里一般的任务是将score_pd中需要反转的反转，默认是数据越大越好，有些是越小越好，
        类似make_scorer(xxx, greater_is_better=True)中的参数greater_is_better的作用：

                            sign = 1 if greater_is_better else -1

        WrsmScorer中max_drawdown虽然是越小越好，但由于本身值即是负数形式，所以不用反转数据
                win_rate	returns	sharpe	max_drawdown	score_win_rate	score_returns	score_sharpe	score_max_drawdown	score
        0	0.307087	0.256354	0.922678	-0.110116	0.189935	0.814123	0.601461	0.511769	0.529322
        1	0.307087	0.256354	0.922678	-0.110116	0.189935	0.814123	0.601461	0.511769	0.529322
        2	0.307087	0.256354	0.922678	-0.110116	0.189935	0.814123	0.601461	0.511769	0.529322
        3	0.307087	0.256354	0.922678	-0.110116	0.189935	0.814123	0.601461	0.511769	0.529322
        4	0.307087	0.256354	0.922678	-0.110116	0.189935	0.814123	0.601461	0.511769	0.529322
        """
        pass


class DemoScorer(AbuBaseScorer):
    def _init_self_begin(self, *arg, **kwargs):
        """胜率，策略收益，手续费组成select_score_func"""

        self.select_score_func = lambda metrics: [metrics.win_rate, metrics.algorithm_period_returns,
                                                  metrics.commission_sum]
        self.columns_name = ['win_rate', 'returns', 'commission']
        self.weights_cnt = len(self.columns_name)

    def _init_self_end(self, *arg, **kwargs):
        """
        _init_self_end这里一般的任务是将score_pd中需要反转的反转，默认是数据越大越好，有些是越小越好，
        类似make_scorer(xxx, greater_is_better=True)中的参数greater_is_better的作用：

                            sign = 1 if greater_is_better else -1
        """
        pass


def make_scorer(score_tuple_array, sc_class, **kwargs):
    """
    score对外接口模块函数
    :param score_tuple_array: 承接GridSearch返回的AbuScoreTuple对象序列
    :param sc_class: 指定进行评分的具体评分类，AbuBaseScorer子类，非实例对象
    :param kwargs: AbuBaseScorer中init的参数，可选weights代表评分项权重， 可选metrics_class代表交易目标度量类
    :return: 通过AbuBaseScorer __call__ 调用sc_class.fit_score()，
             返回fit_score返回值，即self.score_pd.sort_values(by='score')['score']
    """
    return sc_class(score_tuple_array, **kwargs)()
