# -*- encoding:utf-8 -*-
"""

最上业务逻辑层的🐔类

由于现在在写一本关于股票量化方面的书，会涉及到相关文章里的一些内容，
出版社方面不希望我现在开源全部代码，但是一定会开源，最晚等书出版发行
以后，现在只能开源文章中涉及的部分代码，整个系统的开源会稍后，请谅解
我其实也觉着有人能看你的代码就已经很给面子了，但是。。。再次抱歉！！

"""

from __future__ import division
from __future__ import print_function

from abc import ABCMeta, abstractmethod

import ZLog
import six
from MlFiter import MlFiterClass

__author__ = 'BBFamily'


class MlFiterPdClass(six.with_metaclass(ABCMeta, object)):
    '''
        认为所有的业务需求的self.df满足 0 ＝ y， 1: ＝ X
    '''

    def __init__(self, **kwarg):
        self.make_xy(**kwarg)
        if not hasattr(self, 'x') or not hasattr(self, 'y') \
                or not hasattr(self, 'df'):
            raise ValueError('make_xy failed! x, y not exist!')
        self.fiter = MlFiterClass(self.x, self.y, self.df)

    def fit(self):
        return self.fiter.fit()

    @abstractmethod
    def make_xy(self, **kwarg):
        pass

    def show_process(self, pdf=True, cas=True, css=True, ras=True,
                     ttxy=True, plc=True, pra=True, pcm=True, pvt=True, pmps=True, pmrs=True):
        ZLog.info(self.fiter.importances_coef_pd())
        if pdf:
            self.fiter.plot_decision_function()
        if cas:
            ZLog.newline()
            ZLog.info('cross_val_accuracy_score')
            ZLog.info(self.fiter.cross_val_accuracy_score())
        if css:
            ZLog.newline()
            ZLog.info('cross_val_mean_squared_score')
            ZLog.info(self.fiter.cross_val_mean_squared_score())
        if ras:
            ZLog.newline()
            ZLog.info('cross_val_roc_auc_score')
            ZLog.info(self.fiter.cross_val_roc_auc_score())
        if ttxy:
            ZLog.newline()
            ZLog.info('train_test_split_xy')
            self.fiter.train_test_split_xy()
        if plc:
            ZLog.newline()
            ZLog.info('plot_learning_curve')
            self.fiter.plot_learning_curve()
        if pra:
            ZLog.newline()
            ZLog.info('plot_roc_estimator')
            self.fiter.plot_roc_estimator()
        if pcm:
            ZLog.newline()
            ZLog.info('plot_confusion_matrices')
            self.fiter.plot_confusion_matrices()
        if pvt:
            ZLog.newline()
            ZLog.info('plot_visualize_tree')
            self.fiter.plot_visualize_tree()
        if pmps:
            ZLog.newline()
            ZLog.info('prob_maximum_precision_score')
            self.fiter.prob_maximum_precision_score()
        if pmrs:
            ZLog.newline()
            ZLog.info('prob_maximum_recall_score')
            self.fiter.prob_maximum_recall_score()

    def __call__(self):
        '''
            方便外面直接call，不用每次去get
        '''
        return self.fiter

        # def estimator_info_wrapper(self, func):
        #     @functools.wraps(func)
        #     def wrapper(*args, **kwargs):
        #         fiter = self.fiter.get_fiter()
        #         ZLog.info(format(fiter.__class__.__name__, '*^58s'))
        #         return func(*args, **kwargs)
        #     return wrapper
