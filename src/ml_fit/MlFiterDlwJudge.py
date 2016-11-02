# -*- encoding:utf-8 -*-
"""

Judge dlw

由于现在在写一本关于股票量化方面的书，会涉及到相关文章里的一些内容，
出版社方面不希望我现在开源全部代码，但是一定会开源，最晚等书出版发行
以后，现在只能开源文章中涉及的部分代码，整个系统的开源会稍后，请谅解
我其实也觉着有人能看你的代码就已经很给面子了，但是。。。再次抱歉！！

"""
import ZEnv
import numpy as np

import ZLog
from MlFiterJudge import MlFiterJudgeClass

__author__ = 'BBFamily'


class MlFiterDlwJudgeClass(MlFiterJudgeClass):
    K_GOLDEN_DEG_PROB = 'prob_threshold'

    def _serialize_file_name(self):
        return ZEnv.g_project_root + '/data/cache/dlw'

    def judge(self, **kwargs):

        if not kwargs.has_key('deg_hisWindowPd') \
                or not kwargs.has_key('deg_windowPd') \
                or not kwargs.has_key('deg_60WindowPd') \
                or not kwargs.has_key('lowBkCnt') \
                or not kwargs.has_key('wave_score1') \
                or not kwargs.has_key('wave_score2') \
                or not kwargs.has_key('wave_score3'):
            ZLog.info('judge dlw kwargs error!')
            return

        if not hasattr(self, 'estimator') or not hasattr(self, MlFiterDlwJudgeClass.K_GOLDEN_DEG_PROB):
            '''
                暂时只info，如果有必要需要raise exception
            '''
            ZLog.info('not estimator or prob')
            return

        w = np.array([kwargs['deg_hisWindowPd'], kwargs['deg_windowPd'], kwargs['deg_60WindowPd'], kwargs['lowBkCnt'],
                      kwargs['wave_score1'], kwargs['wave_score2'], kwargs['wave_score3']])

        prob_threshold = self.prob_threshold
        estimator = self.estimator
        prob = estimator.predict_proba(w.reshape(1, -1))[:, 1][0]
        if prob > prob_threshold:
            return True
        return False
