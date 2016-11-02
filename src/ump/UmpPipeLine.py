# -*- encoding:utf-8 -*-
"""

由于现在在写一本关于股票量化方面的书，会涉及到相关文章里的一些内容，
出版社方面不希望我现在开源全部代码，但是一定会开源，最晚等书出版发行
以后，现在只能开源文章中涉及的部分代码，整个系统的开源会稍后，请谅解
我其实也觉着有人能看你的代码就已经很给面子了，但是。。。再次抱歉！！

"""
from __future__ import division

import ZLog
from UmpEdge import UmpEdgeClass
from UmpMain import UmpMainClass
from UmpJump import UmpJumpClass

__author__ = 'BBFamily'

K_MAIN_UMP_HIT_THRESHOLD = 20
K_JUMP_UMP_HIT_THRESHOLD = 5


class UmpPipeLineClass(object):
    def __init__(self, ump_dict):
        self.ump_dict = ump_dict

        self.main_umps = []
        self.jump_ump = None
        self.edge_ump = None
        for ump, ump_args in self.ump_dict.items():
            if isinstance(ump, UmpMainClass):
                self.main_umps.append((ump, ump_args))
            elif isinstance(ump, UmpJumpClass):
                self.jump_ump = (ump, ump_args)
            elif isinstance(ump, UmpEdgeClass):
                self.edge_ump = (ump, ump_args)
            else:
                raise TypeError('what a ump type set!')

    def do_pipe_line_predict(self, **kwargs):
        return self.learn_pipe_line_predict(**kwargs)[0]

    def learn_pipe_line_predict(self, **kwargs):
        """
        :param kwargs:
        :return:
        """
        main_hit_cnt = 0
        for ump, ump_args in self.main_umps:
            if 'cons' in ump_args:
                cons_func = ump_args['cons']
                if not cons_func(kwargs):
                    """
                        执行限制条件func, exp
                        UmpJumpClass(None, MlFiterJumpPdClass, predict=True): {'w_col': MlFiterJumpPd.g_w_col,
                        'need_ind_cnt': 1, 'cons': lambda order: order['diff_days'] < 21}
                    """
                    continue
            if 'w_col' not in ump_args:
                raise ValueError('MISS w_col not in ump_args!!!')
            main_hit_cnt += ump.predict_hit_kwargs(ump_args['w_col'], **kwargs)

        jump_hit_cnt = 0
        if self.jump_ump is not None:
            ump, ump_args = self.jump_ump
            if 'cons' in ump_args:
                cons_func = ump_args['cons']
                if cons_func(kwargs) and 'w_col' in ump_args:
                    jump_hit_cnt = ump.predict_hit_kwargs(ump_args['w_col'], **kwargs)

        if self.edge_ump is not None:
            ump, ump_args = self.edge_ump
            edge_pred = ump.predict(**kwargs)

            """
                所有参数通过learn_ump_hit_cnt，learn_ump_predict
                学习可视化而来，微调不要对数据过拟合, if逻辑有重复可优化
                但为外面ml提供细的类别暂时不要优化
            """
            if main_hit_cnt >= 20 and edge_pred <> 1:
                return 0, edge_pred, 0

            if jump_hit_cnt >= 5 and edge_pred <> 1:
                return 0, edge_pred, 1

            if jump_hit_cnt > 1 and main_hit_cnt > 10:
                return 0, edge_pred, 2

            if edge_pred == 1 and (jump_hit_cnt >= 3 and main_hit_cnt >= 6):
                return 0, edge_pred, 3

            if edge_pred == 1 and main_hit_cnt >= 25:
                return 0, edge_pred, 4

            if edge_pred == 0 and (jump_hit_cnt >= 2 and main_hit_cnt >= 4):
                return 0, edge_pred, 5

            if edge_pred == 0 and main_hit_cnt >= 15:
                return 0, edge_pred, 6

            if edge_pred == -1 and (main_hit_cnt >= 1 or jump_hit_cnt >= 1):
                return 0, edge_pred, 7

            return 1, edge_pred, 8
        return 1, -1, -1

    def learn_ump_hit_cnt(self, **kwargs):
        """
        只作为辅助生成学习数据函数
        :param kwargs:
        :return:
        """
        hit_cnt_dict = {}
        for ump, ump_args in self.ump_dict.items():
            if 'w_col' in ump_args:
                if 'cons' in ump_args:
                    cons_func = ump_args['cons']
                    if not cons_func(kwargs):
                        hit_cnt_dict[ump.dump_file_fn() + '_hit'] = 0
                        continue
                if not hasattr(ump, 'predict_hit_kwargs'):
                    raise TypeError('predict_hit_kwargs gone!')
                hit_cnt = ump.predict_hit_kwargs(ump_args['w_col'], **kwargs)
                hit_cnt_dict[ump.dump_file_fn() + '_hit'] = hit_cnt
        return hit_cnt_dict

    def learn_ump_predict(self, **kwargs):
        """
        只作为辅助生成学习数据函数
        :param kwargs:
        :return:
        """
        predict_dict = {}
        for ump, ump_args in self.ump_dict.items():
            if 'w_col' in ump_args:
                if 'cons' in ump_args:
                    cons_func = ump_args['cons']
                    if not cons_func(kwargs):
                        predict_dict[ump.dump_file_fn() + '_predict'] = True
                        continue
                need_ind_cnt = 1
                if 'need_ind_cnt' in ump_args:
                    need_ind_cnt = ump_args['need_ind_cnt']
                if not hasattr(ump, 'predict_kwargs'):
                    raise TypeError('predict_kwargs gone!')
                predict = ump.predict_kwargs(ump_args['w_col'], need_ind_cnt, **kwargs)
                predict_dict[ump.dump_file_fn() + '_predict'] = predict
        return predict_dict
