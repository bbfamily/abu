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

from abc import abstractmethod
from collections import OrderedDict

import numpy as np
import pandas as pd

import NpUtil
import ZLog
from MlFiterCreater import MlFiterCreaterClass
from MlFiterPd import MlFiterPdClass
import MlFiterExcute

from sklearn.externals.joblib import Parallel
from sklearn.externals.joblib import delayed
from ProcessMonitor import add_process_wrapper

__author__ = 'BBFamily'


@add_process_wrapper
def do_hmm_xy(n_components, p_x):
    cn = str(n_components) + '_hmm'
    hmm_fiter = MlFiterCreaterClass()
    hmm_fiter.gaussian_hmm(n_components=n_components, covariance_type='diag', n_iter=5000)
    hmm_fiter.hmm.fit(p_x)
    sequence = hmm_fiter.hmm.predict(p_x)
    return cn, sequence


class MlFiterDhpPdClass(MlFiterPdClass):
    """
        组合上有 dmmies, hmm, pca可拆选的

        子类需要实现的方法
        1 dummies_xy:
            确定需要离散的值及范围
        2 make_xy:
            make_xy中要开头调make_dhp_xy
    """
    s_dummies = False
    s_invoke_hmm = False
    s_invoke_pca = False

    s_judges = dict()

    '''
        作为hmm初始n_components设置
    '''
    s_hmm_slice = slice(7, 64)

    @classmethod
    @abstractmethod
    def dummies_xy(cls, order_has_ret):
        pass

    @classmethod
    @abstractmethod
    def dump_dict_extend(cls):
        """
        补充需要序列化的dict
        如果没有任何要补充的返回None
        :return: dict
        """
        pass

    def make_xy(self, **kwarg):
        raise RuntimeError('make_xy is not init!')

    @classmethod
    def _do_dump_process(cls, judge_cls, order_pd, dummies, invoke_hmm, invoke_pca, tn_threshold, show):
        """
        :param judge_cls:
        :param order_pd:
        :param dummies:
        :param invoke_hmm:
        :param invoke_pca:
        :param tn_threshold:
        :param show:
        :return:
        """

        """
            非常重要, 就有这样才能保证使用同的n个kf,不会出现
            不同切分对应不同kf造成的过拟合假象
        """
        MlFiterExcute.g_enable_kf_cache = True

        fiter_pd = cls(orderPd=order_pd, dummies=dummies, invoke_hmm=invoke_hmm,
                       invoke_pca=invoke_pca)
        fiter_pd().estimator.random_forest_classifier(n_estimators=200)

        thresholds = np.linspace(0.05, 0.4, 8)
        for threshold in thresholds:
            max_v = fiter_pd().prob_maximum_recall_score(threshold=threshold, show=show)
            '''
                exp max_v ((0.68611346108571414, 1095), 0.36122448979591837)
            '''
            if max_v[0][1] > tn_threshold:
                break

        judge = judge_cls()

        how = cls.make_how(dummies, invoke_hmm, invoke_pca)
        judge.set_how(how)
        '''
            如果有hmm的分类器就咬存起来
        '''

        dump_dict = dict(prob_threshold=max_v[1],
                         dummies=dummies,
                         invoke_hmm=invoke_hmm, invoke_pca=invoke_pca)

        ext = cls.dump_dict_extend()
        if ext is not None:
            dump_dict.update(ext)

        '''
            不再存储hmm的具体分类器了，只保存两个影射矩阵
        '''
        # hmm_dump_dict = cls.hmm_dump(jump_pd)
        # dump_dict.update(hmm_dump_dict)
        # sequence_dict = cls.sequence_dump(jump_pd)
        # dump_dict.update(sequence_dict)
        if hasattr(fiter_pd, 'pca_x'):
            dump_dict.update({'pca_x': getattr(fiter_pd, 'pca_x')})

        if hasattr(fiter_pd, 'hmm_x'):
            dump_dict.update({'hmm_x': getattr(fiter_pd, 'hmm_x')})

        if hasattr(fiter_pd, 'ml_x'):
            dump_dict.update({'ml_x': getattr(fiter_pd, 'ml_x')})

        judge.dump_estimator_fiter(fiter_pd, **dump_dict)

        MlFiterExcute.g_enable_kf_cache = False

        return judge, fiter_pd

    @classmethod
    def dump_process(cls, judge_cls, order_pd, tn_threshold=1000, show=False, first_local=False):

        d_judge = None
        if first_local and judge_cls.exist_local_estimator(cls, dummies=True, invoke_hmm=False, invoke_pca=False):
            """
                返回读取的本地分类器与判别器
            """
            ZLog.info('d_judge load with local estimator')
        else:
            d_judge = cls._do_dump_process(judge_cls, order_pd, dummies=True, invoke_hmm=False,
                                           invoke_pca=False, tn_threshold=tn_threshold, show=show)

        v_judge = None
        if first_local and judge_cls.exist_local_estimator(cls, dummies=False, invoke_hmm=False, invoke_pca=False):
            """
                返回读取的本地分类器与判别器
            """
            ZLog.info('v_judge load with local estimator')
        else:
            v_judge = cls._do_dump_process(judge_cls, order_pd, dummies=False, invoke_hmm=False,
                                           invoke_pca=False, tn_threshold=tn_threshold, show=show)

        '''
            s_hmm_slice    = slice(7, 64)
            需要很长时间，而且由于mac版本bug，没办法多进程并行，需要做时在做
        '''
        dm_judge = None
        if first_local and judge_cls.exist_local_estimator(cls, dummies=True, invoke_hmm=True, invoke_pca=False):
            """
                返回读取的本地分类器与判别器
            """
            ZLog.info('v_judge load with local estimator')
        else:
            dm_judge = cls._do_dump_process(judge_cls, order_pd, dummies=True, invoke_hmm=True,
                                            invoke_pca=False, tn_threshold=tn_threshold, show=show)

        vm_judge = None
        if first_local and judge_cls.exist_local_estimator(cls, dummies=False, invoke_hmm=True, invoke_pca=False):
            """
                返回读取的本地分类器与判别器
            """
            ZLog.info('vm_judge load with local estimator')
        else:
            vm_judge = cls._do_dump_process(judge_cls, order_pd, dummies=False, invoke_hmm=True,
                                            invoke_pca=False, tn_threshold=tn_threshold, show=show)

        vp_judge = None
        if first_local and judge_cls.exist_local_estimator(cls, dummies=False, invoke_hmm=False, invoke_pca=True):
            """
                返回读取的本地分类器与判别器
            """
            ZLog.info('vp_judge load with local estimator')
        else:
            vp_judge = cls._do_dump_process(judge_cls, order_pd, dummies=False, invoke_hmm=False,
                                            invoke_pca=True, tn_threshold=tn_threshold, show=show)

        dp_judge = None
        if first_local and judge_cls.exist_local_estimator(cls, dummies=True, invoke_hmm=False, invoke_pca=True):
            """
                返回读取的本地分类器与判别器
            """
            ZLog.info('dp_judge load with local estimator')
        else:
            dp_judge = cls._do_dump_process(judge_cls, order_pd, dummies=True, invoke_hmm=False,
                                            invoke_pca=True, tn_threshold=tn_threshold, show=show)

        return d_judge, v_judge, dm_judge, vm_judge, vp_judge, dp_judge

    @classmethod
    def do_predict_process(cls, judge_cls, dummies, invoke_hmm, invoke_pca, **kwargs):
        how = cls.make_how(dummies, invoke_hmm, invoke_pca)

        if cls.s_judges.has_key(how):
            judge = cls.s_judges[how]
        else:
            judge = judge_cls()
            judge.set_how(how)
            judge.load_estimator()
            '''
                保存类全局缓存，对于频繁判断提诉
            '''
            cls.s_judges[how] = judge
        return judge.judge(**kwargs)

    @classmethod
    def predict_process(cls, judge_cls, **kwargs):
        # d_ret = cls.do_predict_process(judge_cls, True, False, False, **kwargs)
        # v_ret = cls.do_predict_process(judge_cls, False, False, False, **kwargs)
        dm_ret = cls.do_predict_process(judge_cls, True, True, False, **kwargs)
        # vm_ret = cls.do_predict_process(judge_cls, False, True, False, **kwargs)
        # dp_ret = cls.do_predict_process(judge_cls, True, False, True, **kwargs)
        # vp_ret = cls.do_predict_process(judge_cls, False, False, True, **kwargs)

        return dm_ret > 0.1
        # return (d_ret + v_ret + dm_ret + vm_ret + dp_ret + vp_ret) > 0

    @classmethod
    def make_how(cls, dummies, invoke_hmm, invoke_pca):
        how = ''
        if dummies:
            how += '_dummies'
        if invoke_hmm:
            how += '_invoke_hmm'
        elif invoke_pca:
            '''
            pca与hmm互斥，且优先级小, so elif
            '''
            how += '_invoke_pca'
        return how

    @classmethod
    def pca_predict(cls, hoo, x):
        from sklearn.metrics.pairwise import pairwise_distances
        if not hasattr(hoo, 'pca_x') or not hasattr(hoo, 'ml_x'):
            raise RuntimeError('pca_x or ml_x miss!!')

        distance_min_ind = pairwise_distances(x.reshape(1, -1), hoo.pca_x[0:],
                                              metric='euclidean').argmin()
        '''
            置换出可以作为分类输入的x
        '''
        w = hoo.ml_x[distance_min_ind]

        return w

    @classmethod
    def hmm_predict(cls, hoo, x):
        """
            precict并不使用之前给hmm的estimation
            在读取的两个矩阵做映射关系，作出外面需要
            的x数据
        """
        from sklearn.metrics.pairwise import pairwise_distances
        if not hasattr(hoo, 'hmm_x') or not hasattr(hoo, 'ml_x'):
            raise RuntimeError('hmm_x or ml_x miss!!')

        distance_min_ind = pairwise_distances(x.reshape(1, -1), hoo.hmm_x[0:],
                                              metric='euclidean').argmin()

        '''
            置换出可以作为分类输入的x
        '''
        w = hoo.ml_x[distance_min_ind]

        return w

    def do_make_xy(self, order_has_ret, regex):
        df = order_has_ret.filter(regex=regex)

        matrix = df.as_matrix()
        self.y = matrix[:, 0]
        self.x = matrix[:, 1:]

        if self.invoke_hmm:
            '''
                hmm在dummies之后的数据二次处理，2*2
            '''
            df = self.hmm_xy(self.x, df)
            '''
                hmm 后重新定义xy，df
            '''
            df = df.filter(regex='result|hmm')
            matrix = df.as_matrix()
            self.y = matrix[:, 0]
            self.x = matrix[:, 1:]

            '''
                将结果hmm的x存起来，对应输入的x计算distance最小的hmm_x再此map ml_x
                setattr只为和普通属性分要序列化的属性
            '''
            setattr(self, 'ml_x', self.x)

        elif self.invoke_pca:
            self.x = self.pca_xy(self.x)

            matrix = np.concatenate([self.y.reshape(-1, 1), self.x], axis=1)
            df = pd.DataFrame(matrix)
            '''
                setattr只为和普通属性分要序列化的属性
            '''
            setattr(self, 'ml_x', self.x)

        self.df = df
        self.np = matrix

    def make_dhp_xy(self, **kwarg):
        """
            子类需要继续扩展需求,实现make_xy
            :param kwarg:
            :return:
        """
        self.dummies = MlFiterDhpPdClass.s_dummies
        if kwarg.has_key('dummies'):
            self.dummies = kwarg['dummies']

        self.invoke_hmm = MlFiterDhpPdClass.s_invoke_hmm
        if kwarg.has_key('invoke_hmm'):
            self.invoke_hmm = kwarg['invoke_hmm']

        self.invoke_pca = MlFiterDhpPdClass.s_invoke_pca
        if kwarg.has_key('invoke_pca'):
            self.invoke_pca = kwarg['invoke_pca']

    # def hmm_xy(self, X, orderPd, sce=s_hmm_slice):
    #     for n_components in np.arange(sce.start, sce.stop):
    #         cn = str(n_components) + '_hmm'
    #
    #         hmm_fiter = MlFiterCreaterClass()
    #         hmm_fiter.gaussian_hmm(n_components=n_components, covariance_type='diag', n_iter=5000)
    #
    #         hmm_fiter.hmm.fit(X)
    #         sequence = hmm_fiter.hmm.predict(X)
    #
    #         orderPd[cn] = sequence
    #     '''
    #         原始v或者dummies之后的存起来，map predict clac ditance
    #     '''
    #     setattr(self, 'hmm_x', X)
    #     return orderPd
    def hmm_xy(self, x, order_pd, sce=s_hmm_slice, n_jobs=-1):
        # n_jobs = 1
        parallel = Parallel(
            n_jobs=n_jobs, verbose=0, pre_dispatch='2*n_jobs')

        out = parallel(delayed(do_hmm_xy)(n_components, x)
                       for n_components in np.arange(sce.start, sce.stop))

        for ret in out:
            order_pd[ret[0]] = ret[1]
        '''
            原始v或者dummies之后的存起来，map predict clac ditance
        '''
        setattr(self, 'hmm_x', x)
        return order_pd

    def pca_xy(self, x):
        setattr(self, 'pca_x', x)

        pca = MlFiterCreaterClass().pca_func()
        nmx = x if self.dummies else np.apply_along_axis(NpUtil.regular_std, 0, x)
        pca.fit(nmx)
        x_trans = pca.fit_transform(nmx)

        return x_trans
