# -*- encoding:utf-8 -*-
"""
    多支交易对象进行择时操作封装模块，内部通过AbuPickTimeWorker进行
    择时，包装完善前后工作，包括多进程下的进度显示，错误处理捕获，结果
    处理等事务
"""

from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import logging

import numpy as np
import pandas as pd
from enum import Enum

from .ABuPickTimeWorker import AbuPickTimeWorker
from ..CoreBu.ABuEnvProcess import add_process_env_sig
from ..TradeBu import ABuTradeExecute
from ..TradeBu import ABuTradeProxy
from ..TradeBu.ABuKLManager import AbuKLManager
from ..UtilBu.ABuProgress import AbuMulPidProgress

__author__ = '阿布'
__weixin__ = 'abu_quant'


class EFitError(Enum):
    """
        择时操作的错误码
    """

    # 择时操作正常完成，且至少生成一个order
    FIT_OK = 0
    # 择时对象数据获取错误
    NET_ERROR = 1
    # 择时对象数据错误
    DATE_ERROR = 2
    # 择时操作正常完成，但没有生成一个order
    NO_ORDER_GEN = 3
    # 其它错误
    OTHER_ERROR = 4


def _do_pick_time_work(capital, buy_factors, sell_factors, kl_pd, benchmark, draw=False,
                       show_info=False, show_pg=False):
    """
    内部方法：包装AbuPickTimeWorker进行fit，分配错误码，通过trade_summary生成orders_pd，action_pd
    :param capital: AbuCapital实例对象
    :param buy_factors: 买入因子序列
    :param sell_factors: 卖出因子序列
    :param kl_pd: 金融时间序列
    :param benchmark: 交易基准对象，AbuBenchmark实例对象
    :param draw: 是否绘制在对应的金融时间序列上的交易行为
    :param show_info: 是否显示在整个金融时间序列上的交易结果
    :param show_pg: 是否择时内部启动进度条，适合单进程或者每个进程里只有一个symbol进行择时
    :return:
    """
    if kl_pd is None or kl_pd.shape[0] == 0:
        return None, EFitError.NET_ERROR

    pick_timer_worker = AbuPickTimeWorker(capital, kl_pd, benchmark, buy_factors, sell_factors)
    if show_pg:
        pick_timer_worker.enable_task_pg()
    pick_timer_worker.fit()

    if len(pick_timer_worker.orders) == 0:
        # 择时金融时间序列拟合操作后，没有任何order生成
        return None, EFitError.NO_ORDER_GEN

    # 生成关键的orders_pd与action_pd
    orders_pd, action_pd, _ = ABuTradeProxy.trade_summary(pick_timer_worker.orders, kl_pd, draw=draw,
                                                          show_info=show_info)

    # 最后生成list是因为tuple无法修改导致之后不能灵活处理
    return [orders_pd, action_pd], EFitError.FIT_OK


@add_process_env_sig
def do_symbols_with_same_factors(target_symbols, benchmark, buy_factors, sell_factors, capital,
                                 apply_capital=True, kl_pd_manager=None,
                                 show=False, back_target_symbols=None, func_factors=None, show_progress=True):
    """
    输入为多个择时交易对象，以及相同的择时买入，卖出因子序列，对多个交易对象上实施相同的因子
    :param target_symbols: 多个择时交易对象序列
    :param benchmark: 交易基准对象，AbuBenchmark实例对象
    :param buy_factors: 买入因子序列
    :param sell_factors: 卖出因子序列
    :param capital: AbuCapital实例对象
    :param apply_capital: 是否进行资金对象的融合，多进程环境下将是False
    :param kl_pd_manager: 金融时间序列管理对象，AbuKLManager实例
    :param show: 是否显示每个交易对象的交易细节
    :param back_target_symbols:  补位targetSymbols为了忽略网络问题及数据不足导致的问题
    :param func_factors: funcFactors在内层解开factors dicts为了do_symbols_with_diff_factors
    :param show_progress: 进度条显示，默认True
    """
    if kl_pd_manager is None:
        kl_pd_manager = AbuKLManager(benchmark, capital)

    def _batch_symbols_with_same_factors(p_buy_factors, p_sell_factors):
        r_orders_pd = None
        r_action_pd = None
        r_all_fit_symbols_cnt = 0
        # 启动多进程进度显示AbuMulPidProgress
        with AbuMulPidProgress(len(target_symbols), 'pick times complete', show_progress=show_progress) as progress:
            for epoch, target_symbol in enumerate(target_symbols):
                # 如果symbol只有一个就不show了，留给下面_do_pick_time_work中show_pg内部显示进度
                if len(target_symbols) > 1:
                    # 如果要绘制交易细节就不要clear了
                    progress.show(epoch + 1, clear=not show)

                if func_factors is not None and callable(func_factors):
                    # 针对do_symbols_with_diff_factors mul factors等情况嵌入可变因子
                    p_buy_factors, p_sell_factors = func_factors(target_symbol)
                try:
                    kl_pd = kl_pd_manager.get_pick_time_kl_pd(target_symbol)
                    ret, fit_error = _do_pick_time_work(capital, p_buy_factors, p_sell_factors, kl_pd, benchmark,
                                                        draw=show, show_info=show,
                                                        show_pg=(len(target_symbols) == 1 and show_progress))
                except Exception as e:
                    logging.exception(e)
                    continue

                if ret is None and back_target_symbols is not None:
                    # 择时结果错误或者没有order生成的情况下，如果有补位序列，择从序列中pop出一个，进行补位
                    if fit_error is not None and fit_error == EFitError.NO_ORDER_GEN:
                        # 没有order生成的要统计进去
                        r_all_fit_symbols_cnt += 1
                    while True:
                        if len(back_target_symbols) <= 0:
                            break
                        # pop出来代替原先的target
                        target_symbol = back_target_symbols.pop()
                        kl_pd = kl_pd_manager.get_pick_time_kl_pd(target_symbol)
                        ret, fit_error = _do_pick_time_work(capital, p_buy_factors, p_sell_factors, kl_pd, benchmark,
                                                            draw=show, show_info=show)
                        if fit_error == EFitError.NO_ORDER_GEN:
                            r_all_fit_symbols_cnt += 1
                        if ret is not None:
                            break
                if ret is None:
                    continue
                r_all_fit_symbols_cnt += 1
                # 连接每一个交易对象生成的orders_pd和action_pd
                r_orders_pd = ret[0] if r_orders_pd is None else pd.concat([r_orders_pd, ret[0]])
                r_action_pd = ret[1] if r_action_pd is None else pd.concat([r_action_pd, ret[1]])
        return r_orders_pd, r_action_pd, r_all_fit_symbols_cnt

    orders_pd, action_pd, all_fit_symbols_cnt = _batch_symbols_with_same_factors(buy_factors, sell_factors)
    if orders_pd is not None and action_pd is not None:

        # 要sort'Date', 'action'两项，不然之后的行apply_action_to_capital后有问题
        # noinspection PyUnresolvedReferences
        action_pd = action_pd.sort_values(['Date', 'action'])
        action_pd.index = np.arange(0, action_pd.shape[0])
        # noinspection PyUnresolvedReferences
        orders_pd = orders_pd.sort_values(['buy_date'])
        if apply_capital:
            # 如果非多进程环境下开始融合资金对象
            ABuTradeExecute.apply_action_to_capital(capital, action_pd, kl_pd_manager, show_progress=show_progress)

    return orders_pd, action_pd, all_fit_symbols_cnt


def do_symbols_with_diff_factors(target_symbols, benchmark, factor_dict, capital, apply_capital=True,
                                 kl_pd_manager=None,
                                 show=False,
                                 back_target_symbols=None):
    """
        输入为多个择时交易对象，每个交易对象有属于自己的买入，卖出因子，
        在factor_dict中通过对象唯一标识进行提取
    """

    def _func_factors(target_symbol):
        """
            定义do_symbols_with_same_factors中使用的对交易因子dict进行解包的方法
        """
        sub_dict = factor_dict[target_symbol]
        buy_factors = sub_dict['buy_factors']
        sell_factors = sub_dict['sell_factors']
        return buy_factors, sell_factors

    # 通过funcFactors在内层解开factors dict
    return do_symbols_with_same_factors(target_symbols, benchmark, None, None, capital, apply_capital=apply_capital,
                                        kl_pd_manager=kl_pd_manager,
                                        show=show,
                                        back_target_symbols=back_target_symbols,
                                        func_factors=_func_factors)
