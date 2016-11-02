# -*- encoding:utf-8 -*-
"""

Judge jump

由于现在在写一本关于股票量化方面的书，会涉及到相关文章里的一些内容，
出版社方面不希望我现在开源全部代码，但是一定会开源，最晚等书出版发行
以后，现在只能开源文章中涉及的部分代码，整个系统的开源会稍后，请谅解
我其实也觉着有人能看你的代码就已经很给面子了，但是。。。再次抱歉！！

"""
from __future__ import division
import numpy as np
import pandas as pd

import ZLog
from MlFiterJudge import MlFiterJudgeClass

__author__ = 'BBFamily'


class MlFiterDhpJudgeClass(MlFiterJudgeClass):
    def __repr__(self):
        if not hasattr(self, 'estimator') or not hasattr(self, 'prob_threshold') \
                or not hasattr(self, 'dummies') \
                or not hasattr(self, 'invoke_hmm') \
                or not hasattr(self, 'invoke_pca'):
            return 'not init parms!'

        return "estimator: {0}, prob_threshold: {1:.5f}, dummies: {2}, " \
               "invoke_hmm: {3}, invoke_pca: {4}".format(
            self.estimator,
            self.prob_threshold,
            self.dummies,
            self.invoke_hmm,
            self.invoke_pca)

    __str__ = __repr__

    def judge_cls(self):
        """
        子类返回自己需要作为判别器的类
        :return:
        """
        return MlFiterJudgeClass

    def _serialize_file_name(self):
        """
        继续需要子类去实现
        :return:
        """
        raise RuntimeError('_serialize_file_name must imp')

    def judge(self, **kwargs):
        """
        继续需要子类去实现
        :return:
        """
        raise RuntimeError('judge(self, **kwargs) must imp')

    def do_judge(self, w_col, regex_dummies, pd_class, **kwargs):
        if not hasattr(self, 'estimator') or not hasattr(self, 'prob_threshold') \
                or not hasattr(self, 'dummies') \
                or not hasattr(self, 'invoke_hmm') or not hasattr(self, 'invoke_pca'):
            '''
                暂时只info，如果有必要需要raise exception
            '''
            ZLog.info('not estimator or prob or dhp')
            return True

        w = np.array([kwargs[col] for col in w_col])
        w = w.reshape(1, -1)

        prob_threshold = self.prob_threshold
        estimator = self.estimator
        dummies = self.dummies
        invoke_hmm = self.invoke_hmm
        invoke_pca = self.invoke_pca

        df = None
        if dummies or invoke_hmm:
            df = pd.DataFrame(w)
            df.columns = w_col

        if dummies and df is not None:
            df_dummies = pd_class.dummies_xy(df)
            regex = regex_dummies
            df = df_dummies.filter(regex=regex)
            w = df.as_matrix()

        if invoke_hmm:
            '''
                只是置换出hmm形式的x值，这里的df没有修改，暂时也没有必要
            '''
            w = pd_class.hmm_predict(self, w).reshape(1, -1)
        elif invoke_pca:
            '''
                elif 互斥
            '''
            w = pd_class.pca_predict(self, w).reshape(1, -1)

        prob = estimator.predict_proba(w)[:, 1][0]
        if prob > prob_threshold:
            return True
        return False
