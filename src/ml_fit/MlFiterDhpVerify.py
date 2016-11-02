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

import matplotlib.pyplot as plt
import numpy as np
from sklearn import metrics

import ZLog

__author__ = 'BBFamily'


def verify_process(est_cls, judge_cls, make_x_func, make_order_func, order_pd, only_jd=False, first_local=False,
                   tn_threshold=800):
    """
    :param est_cls:
    :param judge_cls:
    :param make_x_func:
    :param make_order_func:
    :param order_pd:
    :param only_jd: 使用以序列化的只进行judge
    :param first_local: 优先使用本地分类器
    :param tn_threshold:
    :return:
    """
    if not only_jd:
        _, _, _, _, _, _ = est_cls.dump_process(judge_cls, order_pd, tn_threshold, True, first_local=first_local)

    def apply_judge(order, p_make_x_func):
        x = p_make_x_func(order)
        """
            离散不使用隐因子
        """
        d_ret = est_cls.do_predict_process(judge_cls, True, False, False, **x)
        """
            连续不使用隐因子
        """
        v_ret = est_cls.do_predict_process(judge_cls, False, False, False, **x)

        """
            离散使用隐因子
        """
        dm_ret = est_cls.do_predict_process(judge_cls, True, True, False, **x)
        """
            连续使用隐因子
        """
        vm_ret = est_cls.do_predict_process(judge_cls, False, True, False, **x)

        """
            离散使用pca
        """
        dp_ret = est_cls.do_predict_process(judge_cls, True, False, True, **x)
        """
            连续使用pca
        """
        vp_ret = est_cls.do_predict_process(judge_cls, False, False, True, **x)
        return d_ret, v_ret, dm_ret, vm_ret, dp_ret, vp_ret

    order_has_ret = make_order_func(order_pd)
    jd_ret = order_pd.apply(apply_judge, axis=1, args=(make_x_func,))

    order_has_ret['d_ret'] = [1 if ret[0] else 0 for ret in jd_ret]
    order_has_ret['v_ret'] = [1 if ret[1] else 0 for ret in jd_ret]
    order_has_ret['dm_ret'] = [1 if ret[2] else 0 for ret in jd_ret]
    order_has_ret['vm_ret'] = [1 if ret[3] else 0 for ret in jd_ret]
    order_has_ret['dp_ret'] = [1 if ret[4] else 0 for ret in jd_ret]
    order_has_ret['vp_ret'] = [1 if ret[5] else 0 for ret in jd_ret]

    v_ret_result = metrics.accuracy_score(order_has_ret[order_has_ret['v_ret'] == 0]['result'],
                                          order_has_ret[order_has_ret['v_ret'] == 0]['v_ret'])
    ZLog.info('v_ret_result: ' + str(v_ret_result))

    d_ret_result = metrics.accuracy_score(order_has_ret[order_has_ret['d_ret'] == 0]['result'],
                                          order_has_ret[order_has_ret['d_ret'] == 0]['d_ret'])
    ZLog.info('d_ret_result: ' + str(d_ret_result))

    dp_ret_result = metrics.accuracy_score(order_has_ret[order_has_ret['dp_ret'] == 0]['result'],
                                           order_has_ret[order_has_ret['dp_ret'] == 0]['dp_ret'])
    ZLog.info('dp_ret_result: ' + str(dp_ret_result))

    vp_ret_result = metrics.accuracy_score(order_has_ret[order_has_ret['vp_ret'] == 0]['result'],
                                           order_has_ret[order_has_ret['vp_ret'] == 0]['vp_ret'])
    ZLog.info('vp_ret_result: ' + str(vp_ret_result))

    dm_ret_result = metrics.accuracy_score(order_has_ret[order_has_ret['dm_ret'] == 0]['result'],
                                           order_has_ret[order_has_ret['dm_ret'] == 0]['dm_ret'])
    ZLog.info('dm_ret_result: ' + str(dm_ret_result))

    vm_ret_result = metrics.accuracy_score(order_has_ret[order_has_ret['vm_ret'] == 0]['result'],
                                           order_has_ret[order_has_ret['vm_ret'] == 0]['vm_ret'])
    ZLog.info('vm_ret_result: ' + str(vm_ret_result))

    ZLog.newline(fill_cnt=58)

    v_ret_result_all = metrics.accuracy_score(order_has_ret['result'], order_has_ret['v_ret'])
    ZLog.info('v_ret_result_all: ' + str(v_ret_result_all))
    d_ret_result_all = metrics.accuracy_score(order_has_ret['result'], order_has_ret['d_ret'])
    ZLog.info('d_ret_result_all: ' + str(d_ret_result_all))
    dp_ret_result_all = metrics.accuracy_score(order_has_ret['result'], order_has_ret['dp_ret'])
    ZLog.info('dp_ret_result_all: ' + str(dp_ret_result_all))
    vp_ret_result_all = metrics.accuracy_score(order_has_ret['result'], order_has_ret['vp_ret'])
    ZLog.info('vp_ret_result_all: ' + str(vp_ret_result_all))
    dm_ret_result_all = metrics.accuracy_score(order_has_ret['result'], order_has_ret['dm_ret'])
    ZLog.info('dm_ret_result_all: ' + str(dm_ret_result_all))
    vm_ret_result_all = metrics.accuracy_score(order_has_ret['result'], order_has_ret['vm_ret'])
    ZLog.info('vm_ret_result_all: ' + str(vm_ret_result_all))

    ZLog.newline(fill_cnt=58)
    order_has_ret['vdmret'] = order_has_ret['d_ret'] + order_has_ret['v_ret'] + order_has_ret['dp_ret'] + order_has_ret[
        'vp_ret']
    order_has_ret['vdmret'].value_counts().plot(kind='barh')
    plt.title('vdmret barh')
    plt.show()

    ((order_has_ret['vdmret'] == 1) & (order_has_ret['v_ret'] == 1)).value_counts().plot(kind='bar')
    plt.title('v_ret == 1')
    plt.show()

    ((order_has_ret['vdmret'] == 1) & (order_has_ret['d_ret'] == 1)).value_counts().plot(kind='bar')
    plt.title('d_ret == 1')
    plt.show()

    ((order_has_ret['vdmret'] == 1) & (order_has_ret['vm_ret'] == 1)).value_counts().plot(kind='bar')
    plt.title('vm_ret == 1')
    plt.show()

    ((order_has_ret['vdmret'] == 1) & (order_has_ret['dm_ret'] == 1)).value_counts().plot(kind='bar')
    plt.title('dm_ret == 1')
    plt.show()

    ((order_has_ret['vdmret'] == 1) & (order_has_ret['dp_ret'] == 1)).value_counts().plot(kind='bar')
    plt.title('dp_ret == 1')
    plt.show()

    ((order_has_ret['vdmret'] == 1) & (order_has_ret['vp_ret'] == 1)).value_counts().plot(kind='bar')
    plt.title('vp_ret == 1')
    plt.show()

    final_result = metrics.accuracy_score(order_has_ret[order_has_ret['vdmret'] == 0]['result'],
                                          order_has_ret[order_has_ret['vdmret'] == 0]['vdmret'])
    ZLog.info('final_result: ' + str(final_result))

    order_has_ret['vdmret_one'] = np.where(order_has_ret['vdmret'] == 1, 0, 1)
    final_one_result = metrics.accuracy_score(order_has_ret[order_has_ret['vdmret_one'] == 0]['result'],
                                              order_has_ret[order_has_ret['vdmret_one'] == 0]['vdmret_one'])
    ZLog.info('final_one_result: ' + str(final_one_result))

    order_has_ret['vdmret_two'] = np.where(order_has_ret['vdmret'] == 2, 0, 1)
    final_two_result = metrics.accuracy_score(order_has_ret[order_has_ret['vdmret_two'] == 0]['result'],
                                              order_has_ret[order_has_ret['vdmret_two'] == 0]['vdmret_two'])
    ZLog.info('final_two_result: ' + str(final_two_result))

    order_has_ret['vdmret_three'] = np.where(order_has_ret['vdmret'] == 3, 0, 1)
    final_three_result = metrics.accuracy_score(order_has_ret[order_has_ret['vdmret_three'] == 0]['result'],
                                                order_has_ret[order_has_ret['vdmret_three'] == 0]['vdmret_three'])
    ZLog.info('final_three_result: ' + str(final_three_result))

    order_has_ret['vdmret_four'] = np.where(order_has_ret['vdmret'] == 4, 0, 1)
    final_four_result = metrics.accuracy_score(order_has_ret[order_has_ret['vdmret_four'] == 0]['result'],
                                               order_has_ret[order_has_ret['vdmret_four'] == 0]['vdmret_four'])
    ZLog.info('final_four_result: ' + str(final_four_result))

    return jd_ret, order_has_ret
