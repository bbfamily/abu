# -*- encoding:utf-8 -*-
"""

Judge的🐔类

由于现在在写一本关于股票量化方面的书，会涉及到相关文章里的一些内容，
出版社方面不希望我现在开源全部代码，但是一定会开源，最晚等书出版发行
以后，现在只能开源文章中涉及的部分代码，整个系统的开源会稍后，请谅解
我其实也觉着有人能看你的代码就已经很给面子了，但是。。。再次抱歉！！

"""
from abc import ABCMeta, abstractmethod

import ZCommonUtil
import six

try:
    import cPickle as pickle
except ImportError:
    import pickle

from MlFiterPd import MlFiterPdClass

__author__ = 'BBFamily'


class MlFiterJudgeClass(six.with_metaclass(ABCMeta, object)):
    @abstractmethod
    def _serialize_file_name(self):
        pass

    @abstractmethod
    def judge(self, **kwargs):
        pass

    def set_how(self, how):
        self.how = how

    @classmethod
    def exist_local_estimator(cls, est_cls, dummies, invoke_hmm, invoke_pca):
        """
        查询是否有缓存分类器
        :return:
        """
        judge = cls()
        how = est_cls.make_how(dummies, invoke_hmm, invoke_pca)
        judge.set_how(how)
        fn = judge._serialize_file_name()
        return ZCommonUtil.file_exist(fn)

    def dump_estimator_fiter(self, fiter, **kwargs):
        if not isinstance(fiter, MlFiterPdClass):
            raise TypeError('dump_estimator_fiter isinstance(fiter, MlFiterPdClass)!')

        estimator = fiter.fit()
        self.__dump_estimator(estimator, **kwargs)

    def __dump_estimator(self, estimator, **kwargs):
        """
            一定要确保estimator进来之前fit了，不然judge时crash
            :param estimator:
            :param kwargs:
            :return:
        """
        dump_dict = {'estimator': estimator}
        if kwargs is not None and len(kwargs) > 0:
            dump_dict.update(kwargs)

        for key, value in dump_dict.items():
            if not hasattr(self, key):
                setattr(self, key, value)

        fn = self._serialize_file_name()
        ZCommonUtil.ensure_dir(fn)
        fw = open(fn, 'w')
        pickle.dump(dump_dict, fw)
        fw.close()

    def load_estimator(self):
        fn = self._serialize_file_name()

        if not ZCommonUtil.file_exist(fn):
            return None

        fr = open(fn)
        ret = pickle.load(fr)
        fr.close()

        for key, value in ret.items():
            """
                setattr给self
            """
            setattr(self, key, value)

        return ret
