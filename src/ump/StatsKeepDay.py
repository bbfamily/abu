# -*- encoding:utf-8 -*-
"""

由于现在在写一本关于股票量化方面的书，会涉及到相关文章里的一些内容，
出版社方面不希望我现在开源全部代码，但是一定会开源，最晚等书出版发行
以后，现在只能开源文章中涉及的部分代码，整个系统的开源会稍后，请谅解
我其实也觉着有人能看你的代码就已经很给面子了，但是。。。再次抱歉！！

"""
from __future__ import division

import numpy as np
from scipy import stats
from sklearn.externals.joblib import Parallel
from sklearn.externals.joblib import delayed

import Capital
import ShowMsg
import SymbolPd
import ZLog
from ProcessMonitor import add_process_wrapper

__author__ = 'BBFamily'


@add_process_wrapper
def _do_keep_day_maxths(order_last_bond, cap, keep95):
    def stats_max_winth(order):
        try:
            bd = order['buy Date']
            kp = order['keep_days']
            '''
                n_folds=3多来一个，避免由于cap对齐造成的缺失
            '''
            kl_pd = SymbolPd.make_kfold_pd(order.Symbol, cap=cap)
            b_key = kl_pd[kl_pd['date'] == bd].key.values[0]
            '''
                bKey + 1
                第一天的return不计算，factor赞按照mean买入，可以认为
                符合正态分布，均值是0
            '''
            maxth = kl_pd[b_key + 1: b_key + kp + 1].netChangeRatio.cumsum().max()
            maxth95 = maxth if kp >= keep95 or (b_key + keep95 + 1) > kl_pd.shape[0] \
                else kl_pd[b_key + 1:b_key + keep95 + 1].netChangeRatio.cumsum().max()
        except Exception, e:
            # import pdb
            # pdb.set_trace()
            ShowMsg.show_msg('Exception', e.__str__)
            ZLog.debug(e.__str__)
            ZLog.debug(bd)
            ZLog.debug(kl_pd)
            return 0, 0

        return maxth, maxth95

    maxths = order_last_bond.apply(stats_max_winth, axis=1)

    return maxths


def keep_day_maxths(order_pds, n_jobs=-1):
    """
        只有＝＝2最后的如期算出的keep days 才是正确的
    """
    order__last_bond = order_pds[(order_pds.symbol_index == 2) & (order_pds['result'] <> 0)]

    """
        类似主成分取95%的keep days
    """
    keep95 = int(stats.scoreatpercentile(order__last_bond.keep_days, 95))

    """
        每个进程一次处理的200
    """
    handel_cnt = 200

    """
        要来个cap不然会有对其问题
    """
    cap = Capital.CapitalClass(100000)

    parallel = Parallel(
        n_jobs=n_jobs, verbose=0, pre_dispatch='2*n_jobs')

    out = parallel(delayed(_do_keep_day_maxths)(order__last_bond[ed_ind - handel_cnt:ed_ind], cap, keep95)
                   for ed_ind in np.arange(handel_cnt, order__last_bond.shape[0], handel_cnt))

    return out, order__last_bond
