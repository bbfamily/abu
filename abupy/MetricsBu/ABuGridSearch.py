# -*- encoding:utf-8 -*-
"""
    借鉴sklearn GridSearch，针对买入因子，卖出因子，选股因子最合进行最优寻找分析
"""

from __future__ import print_function
from __future__ import absolute_import
from __future__ import division
import operator
from collections import Mapping
from functools import reduce
from ..CoreBu.ABuFixes import partial
from itertools import product

import logging
import numpy as np

from ..TradeBu.ABuBenchmark import AbuBenchmark
from ..TradeBu.ABuCapital import AbuCapital
from ..TradeBu.ABuKLManager import AbuKLManager
from .ABuMetricsScore import AbuScoreTuple, WrsmScorer, make_scorer
from ..AlphaBu.ABuPickStockMaster import AbuPickStockMaster
from ..AlphaBu.ABuPickTimeMaster import AbuPickTimeMaster
from ..CoreBu.ABuEnvProcess import add_process_env_sig, AbuEnvProcess
from ..CoreBu.ABuParallel import delayed, Parallel
from ..CoreBu import ABuEnv
from ..CoreBu.ABuEnv import EMarketDataFetchMode
from ..UtilBu import ABuProgress

__author__ = '阿布'
__weixin__ = 'abu_quant'


class ParameterGrid(object):
    """参数进行product辅助生成类"""

    def __init__(self, param_grid):
        if isinstance(param_grid, Mapping):
            param_grid = [param_grid]
        self.param_grid = param_grid

    def __iter__(self):
        """迭代参数组合实现"""
        for p in self.param_grid:
            items = sorted(p.items())
            if not items:
                yield {}
            else:
                keys, values = zip(*items)
                for v in product(*values):
                    params = dict(zip(keys, v))
                    yield params

    def __len__(self):
        """参数组合长度实现"""
        product_mul = partial(reduce, operator.mul)
        return sum(product_mul(len(v) for v in p.values()) if p else 1
                   for p in self.param_grid)

    def __getitem__(self, ind):
        """通过index方式获取某个参数组合实现"""
        for sub_grid in self.param_grid:
            if not sub_grid:
                if ind == 0:
                    return {}
                else:
                    ind -= 1
                    continue
            keys, values_lists = zip(*sorted(sub_grid.items())[::-1])
            sizes = [len(v_list) for v_list in values_lists]
            total = np.product(sizes)

            if ind >= total:
                ind -= total
            else:
                out = {}
                for key, v_list, n in zip(keys, values_lists, sizes):
                    ind, offset = divmod(ind, n)
                    out[key] = v_list[offset]
                return out

        raise IndexError('ParameterGrid index out of range')


def _check_param_grid(param_grid):
    """检测迭代序列是否可进行grid"""

    if hasattr(param_grid, 'items'):
        param_grid = [param_grid]

    for p in param_grid:
        for v in p.values():
            if isinstance(v, np.ndarray) and v.ndim > 1:
                raise ValueError("Parameter array should be one-dimensional.")

            check = [isinstance(v, k) for k in (list, tuple, np.ndarray)]
            if True not in check:
                raise ValueError("Parameter values should be a list.")

            if len(v) == 0:
                raise ValueError("Parameter values should be a non-empty "
                                 "list.")


@add_process_env_sig
def grid_mul_func(read_cash, benchmark, buy_factors, sell_factors, stock_pickers, choice_symbols, kl_pd_manager=None):
    """
    针对输入的买入因子，卖出因子，选股因子及其它参数，进行两年历史交易回测，返回结果包装AbuScoreTuple对象
    :param read_cash: 初始化资金数(int)
    :param benchmark: 交易基准对象，AbuBenchmark实例对象
    :param buy_factors: 买入因子序列
    :param sell_factors: 卖出因子序列
    :param stock_pickers: 选股因子序列
    :param choice_symbols: 初始备选交易对象序列
    :param kl_pd_manager: 金融时间序列管理对象，AbuKLManager实例
    :return: AbuScoreTuple对象
    """

    # 通过初始化资金数，交易基准对象构造资金管理对象capital
    capital = AbuCapital(read_cash, benchmark)

    # 由于grid_mul_func以处于多任务运行环境，所以不内部不再启动多任务，使用1个进程选股
    n_process_pick_stock = 1
    # 由于grid_mul_func以处于多任务运行环境，所以不内部不再启动多任务，使用1个进程择时
    n_process_pick_time = 1
    # 由于grid_mul_func以处于多任务运行环境，所以不内部不再启动多任务，使用1个进程数据收集
    n_process_kl = 1

    if stock_pickers is not None:
        # 有选股因子序列首选进行选股
        choice_symbols = AbuPickStockMaster.do_pick_stock_with_process(capital, benchmark,
                                                                       stock_pickers,
                                                                       choice_symbols=choice_symbols,
                                                                       n_process_pick_stock=n_process_pick_stock)

    if choice_symbols is None or len(choice_symbols) == 0:
        logging.info('pick stock result is zero!')
        return None

    # 通过买入因子，卖出因子等进行择时操作
    orders_pd, action_pd, all_fit_symbols_cnt = AbuPickTimeMaster.do_symbols_with_same_factors_process(
        choice_symbols, benchmark,
        buy_factors, sell_factors, capital, kl_pd_manager=kl_pd_manager, n_process_kl=n_process_kl,
        n_process_pick_time=n_process_pick_time)

    # 将最终结果包装为AbuScoreTuple对象
    return AbuScoreTuple(orders_pd, action_pd, capital, benchmark, buy_factors, sell_factors,
                         stock_pickers)


# noinspection PyAttributeOutsideInit
class GridSearch(object):
    """最优grid search对外接口类"""

    def __init__(self, read_cash, choice_symbols, stock_pickers_product=None,
                 buy_factors_product=None, sell_factors_product=None, score_weights=None, metrics_class=None):
        """
        :param read_cash: 初始化资金数(int)
        :param choice_symbols: 初始备选交易对象序列
        :param stock_pickers_product: 选股因子product之后的序列
        :param buy_factors_product: 买入因子product之后的序列
        :param sell_factors_product: 卖出因子product之后的序列
        :param score_weights: make_scorer中设置的评分权重
        :param metrics_class: make_scorer中设置的度量类
        """
        self.read_cash = read_cash
        self.benchmark = AbuBenchmark()
        self.kl_pd_manager = AbuKLManager(self.benchmark, AbuCapital(self.read_cash, self.benchmark))
        self.choice_symbols = choice_symbols
        self.stock_pickers_product = [None] if stock_pickers_product is None else stock_pickers_product
        self.buy_factors_product = [None] if buy_factors_product is None else buy_factors_product
        self.sell_factors_product = [None] if sell_factors_product is None else sell_factors_product
        self.score_weights = score_weights
        self.metrics_class = metrics_class

    def fit(self, score_class=WrsmScorer, n_jobs=-1):
        """
        开始寻找最优因子参数组合，费时操作，迭代所有因子组合进行交易回测，回测结果进行评分
        :param score_class: 对回测结果进行评分的评分类，AbuBaseScorer类型，非对象，只传递类信息
        :param n_jobs: 默认回测并行的任务数，默认-1, 即启动与cpu数量相同的进程数
        :return: (scores: 评分结果dict， score_tuple_array: 因子组合序列)
        """

        if len(self.stock_pickers_product) == 1 and self.stock_pickers_product[0] is None:
            # 如果没有设置选股因子，外层统一进行交易数据收集，之所以是1，以为在__init__中[None]的设置
            need_batch_gen = self.kl_pd_manager.filter_pick_time_choice_symbols(self.choice_symbols)
            self.kl_pd_manager.batch_get_pick_time_kl_pd(need_batch_gen, n_process=ABuEnv.g_cpu_cnt)

        # 只有E_DATA_FETCH_FORCE_LOCAL才进行多任务模式，否则回滚到单进程模式n_jobs = 1
        if n_jobs != 1 and ABuEnv.g_data_fetch_mode != EMarketDataFetchMode.E_DATA_FETCH_FORCE_LOCAL:
            # 1. hdf5多进程还容易写坏数据
            # 2. MAC OS 10.9 之后并行联网＋numpy 系统bug crash，卡死等问题
            logging.info('batch get only support E_DATA_FETCH_FORCE_LOCAL for Parallel!')
            n_jobs = 1

        parallel = Parallel(
            n_jobs=n_jobs, verbose=0, pre_dispatch='2*n_jobs')

        pass_kl_pd_manager = None
        if len(self.stock_pickers_product) == 1 and self.stock_pickers_product[0] is None:
            pass_kl_pd_manager = self.kl_pd_manager

        # 暂时关闭多进程进度条，太多, 注意这种全局设置一定要在AbuEnvProcess初始化之前完成
        ABuProgress.g_show_ui_progress = False
        # 多任务环境下的内存环境拷贝对象AbuEnvProcess
        p_nev = AbuEnvProcess()
        # 多层迭代各种类型因子，没一种因子组合作为参数启动一个新进程，运行grid_mul_func
        out_abu_score_tuple = parallel(
            delayed(grid_mul_func)(self.read_cash, self.benchmark, buy_factors, sell_factors, stock_pickers,
                                   self.choice_symbols, pass_kl_pd_manager, env=p_nev)
            for stock_pickers in self.stock_pickers_product for buy_factors in self.buy_factors_product for
            sell_factors in self.sell_factors_product)
        ABuProgress.g_show_ui_progress = True

        # 返回的AbuScoreTuple序列转换score_tuple_array
        score_tuple_array = list(out_abu_score_tuple)
        # 使用ABuMetricsScore中make_scorer对多个参数组合的交易结果进行评分，详情阅读ABuMetricsScore模块
        scores = make_scorer(score_tuple_array, score_class, weights=self.score_weights,
                             metrics_class=self.metrics_class)
        # 评分结果最好的赋予best_score_tuple_grid
        self.best_score_tuple_grid = score_tuple_array[scores.index[-1]]
        return scores, score_tuple_array
