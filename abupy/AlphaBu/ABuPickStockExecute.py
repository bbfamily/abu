# -*- encoding:utf-8 -*-
"""
    包装选股worker进行，完善前后工作
"""

from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

from .ABuPickStockWorker import AbuPickStockWorker
from ..CoreBu.ABuEnvProcess import add_process_env_sig
from ..MarketBu.ABuMarket import split_k_market
from ..TradeBu.ABuKLManager import AbuKLManager
from ..CoreBu.ABuFixes import ThreadPoolExecutor

__author__ = '阿布'
__weixin__ = 'abu_quant'


@add_process_env_sig
def do_pick_stock_work(choice_symbols, benchmark, capital, stock_pickers):
    """
    包装AbuPickStockWorker进行选股
    :param choice_symbols: 初始备选交易对象序列
    :param benchmark: 交易基准对象，AbuBenchmark实例对象
    :param capital: 资金类AbuCapital实例化对象
    :param stock_pickers: 选股因子序列
    :return:
    """
    kl_pd_manager = AbuKLManager(benchmark, capital)
    stock_pick = AbuPickStockWorker(capital, benchmark, kl_pd_manager, choice_symbols=choice_symbols,
                                    stock_pickers=stock_pickers)
    stock_pick.fit()
    return stock_pick.choice_symbols


@add_process_env_sig
def do_pick_stock_thread_work(choice_symbols, benchmark, capital, stock_pickers, n_thread):
    """包装AbuPickStockWorker启动线程进行选股"""
    result = []

    def when_thread_done(r):
        result.extend(r.result())

    with ThreadPoolExecutor(max_workers=n_thread) as pool:
        thread_symbols = split_k_market(n_thread, market_symbols=choice_symbols)
        for symbols in thread_symbols:
            future_result = pool.submit(do_pick_stock_work, symbols, benchmark, capital, stock_pickers)
            future_result.add_done_callback(when_thread_done)

    return result
