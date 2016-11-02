# -*- encoding:utf-8 -*-
"""

Judge jump

由于现在在写一本关于股票量化方面的书，会涉及到相关文章里的一些内容，
出版社方面不希望我现在开源全部代码，但是一定会开源，最晚等书出版发行
以后，现在只能开源文章中涉及的部分代码，整个系统的开源会稍后，请谅解
我其实也觉着有人能看你的代码就已经很给面子了，但是。。。再次抱歉！！

"""
import ZEnv
import ZLog
from MlFiterDhpJudge import MlFiterDhpJudgeClass
from MlFiterGoldenPd import MlFiterGoldenPdClass
import MlFiterGoldenPd

__author__ = 'BBFamily'


class MlFiterGoldenJudgeClass(MlFiterDhpJudgeClass):
    def _serialize_file_name(self):
        fn = ZEnv.g_project_root + '/data/cache/golden'
        if hasattr(self, 'how'):
            fn = fn + self.how
        return fn

    def judge(self, **kwargs):
        for w in MlFiterGoldenPd.g_w_col:
            if w not in kwargs:
                ZLog.info('judge golden kwargs error!')
                return

        regex = MlFiterGoldenPd.g_regex_d

        """
            要保持和mertics做的pd一样的顺序
        """
        w_col = MlFiterGoldenPd.g_w_col

        pd_class = MlFiterGoldenPdClass
        return self.do_judge(w_col, regex, pd_class, **kwargs)
