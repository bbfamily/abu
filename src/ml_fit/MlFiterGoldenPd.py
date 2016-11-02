# -*- encoding:utf-8 -*-
"""

xy解析

由于现在在写一本关于股票量化方面的书，会涉及到相关文章里的一些内容，
出版社方面不希望我现在开源全部代码，但是一定会开源，最晚等书出版发行
以后，现在只能开源文章中涉及的部分代码，整个系统的开源会稍后，请谅解
我其实也觉着有人能看你的代码就已经很给面子了，但是。。。再次抱歉！！

"""
from __future__ import division
from __future__ import print_function

import ZEnv
import numpy as np
import pandas as pd

import MlFiterCreater
import MlFiterDhpVerify
import MlFiterExcute
import ZCommonUtil
from MlFiterDhpPd import MlFiterDhpPdClass

__author__ = 'BBFamily'

K_GOLDEN_VERIFY_DEC_TREE = ZEnv.g_project_root + '/data/cache/golden_verify_dec_tree'

g_enable_verify_tree = True

g_w_col = ['atr_std', 'deg_hisWindowPd', 'deg_windowPd', 'deg_60WindowPd', 'wave_score1']
g_regex_d = 'ws1_dummies*|atr_dummies*|dh_dummies*|dw_dummies*|d60_dummies*'
g_regex_v = 'wave_score1|atr_std|deg_hisWindow|deg_windowPd|deg_60WindowPd'


class MlFiterGoldenPdClass(MlFiterDhpPdClass):
    @classmethod
    def dump_dict_extend(cls):
        return None

    @classmethod
    def verify_process(cls, order_pd, only_jd=False, first_local=False, tn_threshold=800):
        """
        :param order_pd:
        :param only_jd: 使用以序列化的只进行judge
        :param first_local: 优先使用本地分类器
        :param tn_threshold:
        :return:
        """
        from MlFiterGoldenJudge import MlFiterGoldenJudgeClass
        judge_cls = MlFiterGoldenJudgeClass
        make_x_func = lambda order: dict(deg_hisWindowPd=order.deg_hisWindowPd, deg_windowPd=order.deg_windowPd,
                                         deg_60WindowPd=order.deg_60WindowPd, atr_std=order.atr_std,
                                         wave_score1=order.wave_score1)

        def make_order_func(p_order_pd):
            in_order_has_ret = p_order_pd[p_order_pd['result'] <> 0]
            in_order_has_ret['result'] = np.where(in_order_has_ret['result'] == -1, 0, 1)
            return in_order_has_ret

        jd_ret, order_has_ret = MlFiterDhpVerify.verify_process(cls, judge_cls, make_x_func, make_order_func,
                                                                order_pd=order_pd,
                                                                only_jd=only_jd,
                                                                first_local=first_local, tn_threshold=tn_threshold)

        if g_enable_verify_tree:
            cls.dump_verify_result(order_has_ret)

        return jd_ret, order_has_ret

    @classmethod
    def dump_verify_result(cls, order_has_ret):
        regex = 'result|d_ret|v_ret|dm_ret|vm_ret|dp_ret|vp_ret'
        ttr = order_has_ret.filter(regex=regex)
        matrix = ttr.as_matrix()
        y = matrix[:, 0]
        x = matrix[:, 1:]

        feature_names = ttr.columns[1:]
        dec_tree = MlFiterCreater.MlFiterCreaterClass().decision_tree_classifier(max_depth=len(feature_names))
        dec_tree.fit(x, y)

        MlFiterExcute.graphviz_tree(dec_tree, feature_names, x, y)
        ZCommonUtil.dump_pickle(dec_tree, K_GOLDEN_VERIFY_DEC_TREE)

    @classmethod
    def predict_process(cls, judge_cls, **kwargs):
        if not g_enable_verify_tree:
            return super(MlFiterGoldenPdClass, cls).predict_process(judge_cls, **kwargs)

        d_ret = cls.do_predict_process(judge_cls, True, False, False, **kwargs)
        v_ret = cls.do_predict_process(judge_cls, False, False, False, **kwargs)
        dm_ret = cls.do_predict_process(judge_cls, True, True, False, **kwargs)
        vm_ret = cls.do_predict_process(judge_cls, False, True, False, **kwargs)
        dp_ret = cls.do_predict_process(judge_cls, True, False, True, **kwargs)
        vp_ret = cls.do_predict_process(judge_cls, False, False, True, **kwargs)

        # if K_GOLDEN_VERIFY_DEC_TREE in cls.s_judges:
        #     dec_tree = cls.s_judges[K_GOLDEN_VERIFY_DEC_TREE]
        # else:
        #     dec_tree = ZCommonUtil.load_pickle(K_GOLDEN_VERIFY_DEC_TREE)
        #     cls.s_judges[K_GOLDEN_VERIFY_DEC_TREE] = dec_tree
        # return dec_tree.predict(np.array([d_ret, v_ret, dm_ret, vm_ret, dp_ret, vp_ret]).reshape(1, -1))
        return [d_ret, v_ret, dm_ret, vm_ret, dp_ret, vp_ret]

    @classmethod
    def dummies_xy(cls, order_has_ret):
        """
        bins的选择不是胡来,根据binscs可视化数据结果进行
        :param order_has_ret:
        :return:
        """
        bins = [-np.inf, 0.0, 0.1, 0.2, 0.4, 0.50, 0.85, 1.0, 1.2, np.inf]
        cats = pd.cut(order_has_ret.wave_score1, bins)
        wave_score1_dummies = pd.get_dummies(cats, prefix='ws1_dummies')
        order_has_ret = pd.concat([order_has_ret, wave_score1_dummies], axis=1)

        bins = [-np.inf, 0.0, 0.2, 0.4, 0.6, 0.8, 1.0, np.inf]
        cats = pd.cut(order_has_ret.atr_std, bins)
        atr_dummies = pd.get_dummies(cats, prefix='atr_dummies')
        order_has_ret = pd.concat([order_has_ret, atr_dummies], axis=1)

        bins = [-np.inf, -20, -12, -7, -3, 0, 3, 7, 12, 20, np.inf]

        cats = pd.cut(order_has_ret.deg_hisWindowPd, bins)
        deg_his_window_dummies = pd.get_dummies(cats, prefix='dh_dummies')
        order_has_ret = pd.concat([order_has_ret, deg_his_window_dummies], axis=1)

        cats = pd.cut(order_has_ret.deg_windowPd, bins)
        deg_window_dummies = pd.get_dummies(cats, prefix='dw_dummies')
        order_has_ret = pd.concat([order_has_ret, deg_window_dummies], axis=1)

        cats = pd.cut(order_has_ret.deg_60WindowPd, bins)
        deg_60window_dummies = pd.get_dummies(cats, prefix='d60_dummies')
        order_has_ret = pd.concat([order_has_ret, deg_60window_dummies], axis=1)

        return order_has_ret

    def make_xy(self, **kwarg):
        self.make_dhp_xy(**kwarg)

        if kwarg is None or 'orderPd' not in kwarg:
            raise ValueError('kwarg is None or not kwarg.has_key ordersPd')

        order_pd = kwarg['orderPd']
        order_has_ret = order_pd[order_pd['result'] <> 0]
        order_has_ret['result'] = np.where(order_has_ret['result'] == -1, 0, 1)

        """
            将没有处理过的原始有效数据保存一份
        """
        self.order_has_ret = order_has_ret

        if self.dummies:
            order_has_ret = MlFiterGoldenPdClass.dummies_xy(order_has_ret)
        regex = g_regex_d if self.dummies else g_regex_v
        regex = 'result|' + regex

        self.do_make_xy(order_has_ret, regex)
