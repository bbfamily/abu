# -*- encoding:utf-8 -*-
"""

由于现在在写一本关于股票量化方面的书，会涉及到相关文章里的一些内容，
出版社方面不希望我现在开源全部代码，但是一定会开源，最晚等书出版发行
以后，现在只能开源文章中涉及的部分代码，整个系统的开源会稍后，请谅解
我其实也觉着有人能看你的代码就已经很给面子了，但是。。。再次抱歉！！

"""
from __future__ import division

import ZEnv
from UmpBase import UmpBaseClass

__author__ = 'BBFamily'


class UmpMainClass(UmpBaseClass):
    def dump_file_fn(self):
        return ZEnv.g_project_root + '/data/cache/ump_main_' + str(self.fiter_cls.__name__).lower()


