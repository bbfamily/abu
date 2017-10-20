# -*- encoding:utf-8 -*-
"""借鉴sklearn GridSearch，针对买入因子，卖出因子，选股因子最合进行最优寻找分析"""

from __future__ import print_function
from __future__ import absolute_import
from __future__ import division
import operator
from collections import Mapping
from functools import reduce
from ..CoreBu.ABuFixes import partial
from itertools import product, chain

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
from ..UtilBu.ABuProgress import AbuMulPidProgress
from ..MarketBu.ABuMarket import split_k_market
from ..MarketBu.ABuDataCheck import check_symbol_data
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
def grid_search_mul_process(read_cash, benchmark, factors, choice_symbols, kl_pd_manager=None):
    """
    针对输入的买入因子，卖出因子，选股因子及其它参数，进行两年历史交易回测，返回结果包装AbuScoreTuple对象
    :param read_cash: 初始化资金数(int)
    :param benchmark: 交易基准对象，AbuBenchmark实例对象
    :param factors: 买入因子，卖出因子，选股因子product 最外层tuple->dict对象->字典values->list->list对象->dict对象
            eg:
                (
                    {'buy_factors': [{'class': x1}, {'class': x2}]
                    'sell_factors': [{'class': Y1}, {'class': Y2}],
                    'stock_pickers': [{'class': Z1}, {'class': Z2}]},

                    {'buy_factors': [{'class': xx1}, {'class': xx2}]
                    'sell_factors': [{'class': Yy1}, {'class': Yy2}],
                    'stock_pickers': [{'class': Zz1}, {'class': Zz2}]
                    .................................................
                )
    :param choice_symbols: 初始备选交易对象序列
    :param kl_pd_manager: 金融时间序列管理对象，AbuKLManager实例
    :return: AbuScoreTuple对象
    """
    # 由于grid_search_mul_process以处于多任务运行环境，所以不内部不再启动多任务，使用1个进程选股
    n_process_pick_stock = 1
    # 由于grid_search_mul_process以处于多任务运行环境，所以不内部不再启动多任务，使用1个进程择时
    n_process_pick_time = 1
    # 由于grid_search_mul_process以处于多任务运行环境，所以不内部不再启动多任务，使用1个进程数据收集
    n_process_kl = 1
    # 每一个任务子进程中返回的由AbuScoreTuple组成的独立结果对象，进程承接层使用chain.from_iterable摊开展平
    result_tuple_array = []
    # 如果因子组合的个数大于4组显示外层的进度条，否则显示内层子进程择时进程条
    show_outer_progress = True if len(factors) >= 4 else False
    with AbuMulPidProgress(len(factors), 'grid search total progress', show_progress=show_outer_progress) as progress:
        progress.display_step = 1
        for epoch, factor in enumerate(factors):
            progress.show(epoch + 1)
            buy_factors = factor['buy_factors']
            sell_factors = factor['sell_factors']
            stock_pickers = factor['stock_pickers']

            # 通过初始化资金数，交易基准对象构造资金管理对象capital
            capital = AbuCapital(read_cash, benchmark)
            if stock_pickers is not None:
                # 有选股因子序列首选进行选股
                choice_symbols = \
                    AbuPickStockMaster.do_pick_stock_with_process(capital, benchmark,
                                                                  stock_pickers,
                                                                  choice_symbols=choice_symbols,
                                                                  n_process_pick_stock=n_process_pick_stock)

            if choice_symbols is None or len(choice_symbols) == 0:
                logging.info('pick stock result is zero!')
                result_tuple_array.append(AbuScoreTuple(None, None, capital, benchmark, buy_factors, sell_factors,
                                                        stock_pickers))
                continue

            # 通过买入因子，卖出因子等进行择时操作
            orders_pd, action_pd, all_fit_symbols_cnt = AbuPickTimeMaster.do_symbols_with_same_factors_process(
                choice_symbols, benchmark,
                buy_factors, sell_factors, capital, kl_pd_manager=kl_pd_manager, n_process_kl=n_process_kl,
                n_process_pick_time=n_process_pick_time, show_progress=not show_outer_progress)

            # 将最终结果包装为AbuScoreTuple对象
            result_tuple = AbuScoreTuple(orders_pd, action_pd, capital, benchmark, buy_factors, sell_factors,
                                         stock_pickers)
            result_tuple_array.append(result_tuple)

    return result_tuple_array


# noinspection PyAttributeOutsideInit
class GridSearch(object):
    """最优grid search对外接口类"""

    @classmethod
    def combine_same_factor_class(cls, factors):
        """
        合并不同的class factor到符合grid search格式的因子参数组合：

        eg：
            org_factor = [ {
                  'class': AClass,
                  'xd' : 20,
                  'past_factor': 2,
                  'up_deg_threshold': 3
             }, {
                  'class': AClass,
                  'xd' : 30,
                  'past_factor': 3,
                  'up_deg_threshold': 4
             }, {
                  'class': BClass,
                  'xd' : 20,
                  'past_factor': 2,
                  'down_deg_threshold': -2
            }, {
                  'class': BClass,
                  'xd' : 30,
                  'past_factor': 3,
                  'down_deg_threshold': -4
            }]

            转换合并后结果：

            [{'class': [AClass],
              'down_deg_threshold': [-4, -2],
              'past_factor': [2, 3],
              'xd': [20, 30]},
             {'class': [BClass],
              'past_factor': [2, 3],
              'up_deg_threshold': [3, 4],
              'xd': [20, 30]}]
        :param factors: 转换前多个买入或者卖出策略因子组成的list容器对象
        :return: 转换后符合grid search格式的策略因子组成的list容器对象
        """
        #  转换后符合grid search格式的策略因子组成的list容器对象, 首先筛选出独立参数策略
        combine_factor_list = list(filter(lambda factor: isinstance(factor['class'], list), factors))

        # 把需要再次组合参数的进行筛选
        factors = list(filter(lambda factor: not isinstance(factor['class'], list), factors))

        # 先找出唯一的class集合序列
        # noinspection PyTypeChecker
        unique_class_set = set([factor['class'] for factor in factors])

        for class_value in unique_class_set:
            # 一个一个唯一的class筛选出来
            unique_class_factors = list(filter(lambda factor: factor['class'] == class_value, factors))
            # 每一个唯一的class筛选出来所有的字典key
            all_keys = set([factor_key for factor in unique_class_factors for factor_key in factor.keys()])
            combine_factor = dict()
            for factor_key in all_keys:
                # 将有相同key的组成一个序列
                factor_grid_list = set(
                    [factor[factor_key] for factor in unique_class_factors if factor_key in factor.keys()])
                # 新的dict符合grid search标准
                combine_factor[factor_key] = list(factor_grid_list)
            combine_factor_list.append(combine_factor)
        return combine_factor_list

    @classmethod
    def show_top_constraints_metrics(cls, constraints, scores, score_tuple_array, top_cnt=10):
        direction = int(top_cnt / abs(top_cnt))
        # 需要使用-direction因为分数是从小－>大排序的
        scores = scores[::-direction]
        result_top = constraints(scores, score_tuple_array, top_cnt)

        from ..MetricsBu.ABuMetricsBase import AbuMetricsBase
        for top_tuple in result_top:
            logging.info(u'买入策略:{}'.format(top_tuple.buy_factors))
            logging.info(u'卖出策略:{}'.format(top_tuple.sell_factors))
            AbuMetricsBase.show_general(top_tuple.orders_pd, top_tuple.action_pd, top_tuple.capital,
                                        top_tuple.benchmark,
                                        returns_cmp=True, only_info=True)
            logging.info('\n')

    @classmethod
    def show_top_score_metrics(cls, scores, score_tuple_array, top_cnt=10):
        """
        类方法，根据grid_search寻找最优参数的grid结果返回的scores序列与AbuScoreTuple对象序列
        显示top n个度量结果
        :param scores: grid_search结果返回的scores序列
        :param score_tuple_array: grid_search结果返回的AbuScoreTuple对象序列
        :param top_cnt: 显示的top度量结果个数，默认10个， 如果传递的top_cnt为负数，eg：-10 则取出最不好的10个结
        """
        # 如果传递的top_cnt为负数，eg：-10 则取出最不好的10个结果
        direction = int(top_cnt / abs(top_cnt))
        # 需要使用-direction因为分数是从小－>大排序的
        top_n = scores[::-direction][:abs(top_cnt)]
        from ..MetricsBu.ABuMetricsBase import AbuMetricsBase
        for top in top_n.index:
            top_tuple = score_tuple_array[top]
            logging.info(u'买入策略:{}'.format(top_tuple.buy_factors))
            logging.info(u'卖出策略:{}'.format(top_tuple.sell_factors))
            AbuMetricsBase.show_general(top_tuple.orders_pd, top_tuple.action_pd, top_tuple.capital,
                                        top_tuple.benchmark,
                                        returns_cmp=True, only_info=True)
            logging.info('\n')

    @classmethod
    def grid_search(cls, choice_symbols, buy_factors, sell_factors, read_cash=10000000,
                    score_weights=None, metrics_class=None):
        """
        类方法: 不gird选股因子，只使用买入因子和卖出因子序列的gird product行为
        :param choice_symbols: 初始备选交易对象序列
        :param buy_factors: 买入因子grid序列或者直接为独立买入因子grid
        :param sell_factors: 卖出因子grid序列或者直接为独立卖出因子grid
        :param read_cash: 初始化资金数(int), 默认10000000
        :param score_weights: make_scorer中设置的评分权重
        :param metrics_class: make_scorer中设置的度量类
        """

        scores = None
        score_tuple_array = None
        restore_data_fetch = None

        if ABuEnv.g_data_fetch_mode != EMarketDataFetchMode.E_DATA_FETCH_FORCE_LOCAL:
            restore_data_fetch = ABuEnv.g_data_fetch_mode
            ABuEnv.g_data_fetch_mode = EMarketDataFetchMode.E_DATA_FETCH_FORCE_LOCAL

        if check_symbol_data(choice_symbols):
            def factor_grid(factors):
                if isinstance(factors, dict):
                    # 独立因子grid, 确保参数为序列，且形成独立grid list
                    factors_grid = [{bf_key: factors[bf_key] if isinstance(factors[bf_key],
                                                                           list) else [factors[bf_key]]
                                     for bf_key in factors.keys()}]
                elif isinstance(factors, list):
                    # 如果传递进来的本身就是序列，需要对序列内容做参数监测，
                    factors_grid = []
                    for factor in factors:
                        factor_dict = {bf_key: factor[bf_key] if isinstance(factor[bf_key],
                                                                            list) else [factor[bf_key]]
                                       for bf_key in factor.keys()}
                        factors_grid.append(factor_dict)
                else:
                    raise TypeError('factors must be dict or list not {}'.format(type(factors)))
                return factors_grid

            # print('buy_factors', buy_factors)
            buy_factors_grid = factor_grid(buy_factors)
            # print('buy_factors_grid', buy_factors_grid)
            sell_factors_grid = factor_grid(sell_factors)

            from ..MetricsBu.ABuGridHelper import gen_factor_grid, K_GEN_FACTOR_PARAMS_BUY, K_GEN_FACTOR_PARAMS_SELL

            buy_factors_product = gen_factor_grid(K_GEN_FACTOR_PARAMS_BUY, buy_factors_grid)
            # print('buy_factors_product', buy_factors_product)

            sell_factors_product = gen_factor_grid(K_GEN_FACTOR_PARAMS_SELL, sell_factors_grid)

            logging.info(u'卖出因子参数共有{}种组合方式'.format(len(sell_factors_product)))
            logging.info(u'卖出因子组合0: 形式为{}'.format(sell_factors_product[0]))

            logging.info(u'买入因子参数共有{}种组合方式'.format(len(buy_factors_product)))
            logging.info(u'买入因子组合0: 形式为{}'.format(buy_factors_product[0]))

            logging.info(u'买入卖出组合形式共: {} X {} = {}种'.format(
                len(buy_factors_product), len(sell_factors_product),
                len(buy_factors_product) * len(sell_factors_product)))
            # return buy_factors_product, sell_factors_product
            gs = cls(read_cash, choice_symbols, buy_factors_product=buy_factors_product,
                     sell_factors_product=sell_factors_product, score_weights=score_weights,
                     metrics_class=metrics_class)
            scores, score_tuple_array = gs.fit(n_jobs=-1)
            best_score_tuple_grid = gs.best_score_tuple_grid
            from ..MetricsBu.ABuMetricsBase import AbuMetricsBase
            logging.info(u'最佳买入因子参数组合：{}'.format(best_score_tuple_grid.buy_factors))
            logging.info(u'最佳卖出因子参数组合：{}'.format(best_score_tuple_grid.sell_factors))
            logging.info('*' * 100)
            AbuMetricsBase.show_general(best_score_tuple_grid.orders_pd, best_score_tuple_grid.action_pd,
                                        best_score_tuple_grid.capital, best_score_tuple_grid.benchmark,
                                        returns_cmp=True, only_info=True)
        if restore_data_fetch is not None:
            ABuEnv.g_data_fetch_mode = restore_data_fetch

        return scores, score_tuple_array

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

        pass_kl_pd_manager = None
        if len(self.stock_pickers_product) == 1 and self.stock_pickers_product[0] is None:
            # 如果没有设置选股因子，外层统一进行交易数据收集，之所以是1，以为在__init__中[None]的设置
            need_batch_gen = self.kl_pd_manager.filter_pick_time_choice_symbols(self.choice_symbols)
            # grid多进程symbol数量大于40才使用多进程，否则单进程执行
            self.kl_pd_manager.batch_get_pick_time_kl_pd(need_batch_gen,
                                                         n_process=ABuEnv.g_cpu_cnt if len(need_batch_gen) > 40 else 1)
            pass_kl_pd_manager = self.kl_pd_manager

        if n_jobs <= 0:
            # 因为下面要根据n_jobs来split_k_market
            n_jobs = ABuEnv.g_cpu_cnt

        # 只有E_DATA_FETCH_FORCE_LOCAL才进行多任务模式，否则回滚到单进程模式n_jobs = 1
        if n_jobs != 1 and ABuEnv.g_data_fetch_mode != EMarketDataFetchMode.E_DATA_FETCH_FORCE_LOCAL:
            # 1. hdf5多进程还容易写坏数据
            # 2. MAC OS 10.9 之后并行联网＋numpy 系统bug crash，卡死等问题
            logging.info('batch get only support E_DATA_FETCH_FORCE_LOCAL for Parallel!')
            n_jobs = 1

        factors_product = [{'buy_factors': item[0], 'sell_factors': item[1], 'stock_pickers': item[2]} for item in
                           product(self.buy_factors_product, self.sell_factors_product, self.stock_pickers_product)]

        # 将factors切割为n_jobs个子序列，这样可以每个进程处理一个子序列
        process_factors = split_k_market(n_jobs, market_symbols=factors_product)
        # 因为切割会有余数，所以将原始设置的进程数切换为分割好的个数, 即32 -> 33 16 -> 17
        n_jobs = len(process_factors)
        parallel = Parallel(
            n_jobs=n_jobs, verbose=0, pre_dispatch='2*n_jobs')
        # 多任务环境下的内存环境拷贝对象AbuEnvProcess
        p_nev = AbuEnvProcess()
        # 多层迭代各种类型因子，没一种因子组合作为参数启动一个新进程，运行grid_search_mul_process
        out_abu_score_tuple = parallel(
            delayed(grid_search_mul_process)(self.read_cash, self.benchmark, factors,
                                             self.choice_symbols, pass_kl_pd_manager, env=p_nev)
            for factors in process_factors)

        # 都完事时检测一下还有没有ui进度条
        ABuProgress.do_check_process_is_dead()
        # 返回的AbuScoreTuple序列转换score_tuple_array, 即摊开多个子结果序列eg: ([], [], [], [])->[]
        score_tuple_array = list(chain.from_iterable(out_abu_score_tuple))
        # 使用ABuMetricsScore中make_scorer对多个参数组合的交易结果进行评分，详情阅读ABuMetricsScore模块
        scores = make_scorer(score_tuple_array, score_class, weights=self.score_weights,
                             metrics_class=self.metrics_class)
        # 评分结果最好的赋予best_score_tuple_grid
        self.best_score_tuple_grid = score_tuple_array[scores.index[-1]]
        return scores, score_tuple_array
