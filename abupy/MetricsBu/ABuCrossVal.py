# -*- encoding:utf-8 -*-
"""策略验证模块"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import logging
from itertools import chain

import numpy as np
import pandas as pd

from ..SimilarBu.ABuSimilar import find_similar_with_folds
from ..CoreBu import ABuEnv
from ..CoreBu.ABuEnv import EMarketTargetType, EMarketDataFetchMode
from ..MarketBu.ABuSymbol import IndexSymbol
from ..MarketBu.ABuDataCheck import all_market_env_check
from ..CoreBu.ABuEnvProcess import add_process_env_sig, AbuEnvProcess
from ..UtilBu.ABuProgress import AbuMulPidProgress, AbuProgress
from ..CoreBu.ABuParallel import delayed, Parallel
from ..UtilBu import ABuProgress
from ..TradeBu.ABuBenchmark import AbuBenchmark
from ..TradeBu.ABuCapital import AbuCapital
from ..AlphaBu.ABuPickTimeMaster import AbuPickTimeMaster
from ..MetricsBu.ABuMetricsBase import AbuMetricsBase
from ..SimilarBu.ABuSimilar import ECoreCorrType
from ..MarketBu import ABuMarketDrawing

__author__ = '阿布'
__weixin__ = 'abu_quant'


@add_process_env_sig
def cross_val_mul_process(vc, cv, corr_series, benchmark, buy_factors, sell_factors, cash=10000000):
    """
    :param vc: 多进程调度层分配的相关度范围 eg：(0.564, 1.0]
    :param cv: 交叉验证的数量级，eg：10，内部根据cv进行多次从相关度范围内随机抽取symbol进行回测
    :param corr_series: 多进程调度层传递的相关pd.Series对象，index为symbol，value为corr值
    :param benchmark: 交易基准对象，AbuBenchmark实例对象
    :param buy_factors: 买入因子组合
    :param sell_factors: 卖出因子组合
    :param cash: 初始化资金数(int)，默认10000000
    """
    # 由于cross_val_mul_process以处于多任务运行环境，所以不内部不再启动多任务，使用1个进程择时
    n_process_pick_time = 1
    # 由于cross_val_mul_process以处于多任务运行环境，所以不内部不再启动多任务，使用1个进程数据收集
    n_process_kl = 1
    # 进程承接层使用chain.from_iterable摊开展平
    metrics_array = []
    """
        根据多进程调度层分配的相关度范围对symbol进行筛选：
        eg：
            vc = (0.564, 1.0]
            vc.left = 0.564
            vc.right = 1.0
    """
    symbol_vc = corr_series[(corr_series > vc.left) & (corr_series <= vc.right)].index
    with AbuMulPidProgress(cv, 'cross val progress') as progress:
        progress.display_step = 1
        for epoch in np.arange(0, cv):
            progress.show(epoch + 1)
            # 通过初始化资金数，交易基准对象构造资金管理对象capital
            capital = AbuCapital(cash, benchmark)
            # 在满足相关度范围的symbol_vc里面随机抽取cv个symbol
            choice_symbols = np.random.choice(symbol_vc, cv, replace=False)
            # 通过买入因子，卖出因子等进行择时操作
            orders_pd, action_pd, _ = AbuPickTimeMaster.do_symbols_with_same_factors_process(
                choice_symbols, benchmark,
                buy_factors, sell_factors, capital, n_process_kl=n_process_kl,
                n_process_pick_time=n_process_pick_time, show_progress=False)
            # 使用AbuMetricsBase对回测结果进行度量
            metrics = AbuMetricsBase(orders_pd, action_pd, capital, benchmark)
            metrics.fit_metrics()
            """
                度量结果添加到返回数组中添加的对象为tuple：
                tuple = (回测结果metrics对象，多进程调度层分配的相关度范围，本轮随机抽取的symbol)
            """
            metrics_array.append((metrics, vc, choice_symbols))

    return metrics_array


# noinspection PyProtectedMember
class AbuCrossVal(object):
    """对策略根据相关性进行交叉多次验证实现类"""

    def __init__(self, market=None, corr_type=ECoreCorrType.E_CORE_TYPE_PEARS):
        """
        :param market: 进行验证的市场，默认None将使用env中设置的市场
        :param corr_type: 相关系数计算方法参数
        """
        # None则服从ABuEnv.g_market_target市场设置
        self.market = ABuEnv.g_market_target if market is None else market
        # ipython notebook下使用logging.info
        self.log_func = logging.info if ABuEnv.g_is_ipython else print
        self.corr_type = corr_type

    def _find_or_cache_similar(self, n_folds, benchmark, enable_cache):
        """
            根据是否有相关度数据缓存获取相关数据，通过hasattr查询类中是否有对应市场缓存，如果没有
            使用find_similar_with_folds获取相关数据通过setattr设置为类变量
        """
        cache_similar_key = '{}_{}_similar_cache'.format(self.market.value, n_folds)
        if enable_cache and hasattr(self, cache_similar_key):
            # 查询是否存在类缓存
            self.log_func('load similar from cache！')
            return getattr(self, cache_similar_key)

        self.log_func('begin similar work...')
        # 使用find_similar_with_folds获取相关数据
        sorted_corr = find_similar_with_folds(benchmark, n_folds=n_folds, corr_type=self.corr_type)
        # 通过setattr设置为类变量
        setattr(self, cache_similar_key, sorted_corr)
        self.log_func('end similar work...')
        return sorted_corr

    def _do_cross_corr(self, buy_factors, sell_factors, benchmark, corr_series, cv, n_folds):
        """
        多进程调度层：通过pd.qcut将相关数据corr_series切分成cv个
        相关范围段，启动cv个进程分别在每个进程中对相关范围段symbol
        进行回测，汇总多进程的执行结果使用chain.from_iterable将结果
        摊平
        :param buy_factors: 买入因子组合
        :param sell_factors: 卖出因子组合
        :param benchmark: 交易基准对象，AbuBenchmark实例对
        :param corr_series: 多进程调度层传递的相关pd.Series对象，index为symbol，value为corr值
        :param cv: 交叉验证的数量级，eg：10，内部根据cv进行多次从相关度范围内随机抽取symbol进行回测
        :param n_folds: 交叉验证的回测历史年数，需要确保本地有缓存对应的年数数据存在
        """
        cats = pd.qcut(corr_series, cv)
        corr_vc = cats.value_counts()
        """
            eg: corr_vc

                (0.564, 1.0]        551
                (0.486, 0.564]      551
                (0.427, 0.486]      551
                (0.377, 0.427]      551
                (0.321, 0.377]      551
                (0.268, 0.321]      551
                (0.201, 0.268]      551
                (0.128, 0.201]      551
                (0.0588, 0.128]     551
                (-0.984, 0.0588]    551
        """
        # 回测历史时间周期设置只依赖标尺AbuBenchmark的构造时间长度
        benchmark = AbuBenchmark(benchmark, n_folds=n_folds)
        parallel = Parallel(
            n_jobs=cv, verbose=0, pre_dispatch='2*n_jobs')
        # 多任务环境下的内存环境拷贝对象AbuEnvProcess
        p_nev = AbuEnvProcess()

        out_cross_val = parallel(
            delayed(cross_val_mul_process)(vc, cv, corr_series, benchmark, buy_factors,
                                           sell_factors, env=p_nev)
            for vc in corr_vc.index)
        # 摊开多个子结果序列eg: ([], [], [], [])->[]
        self.metrics_array = list(chain.from_iterable(out_cross_val))
        self.out_cross_val = out_cross_val
        # 都完事时检测一下还有没有ui进度条
        ABuProgress.do_check_process_is_dead()
        # 输出验证整体结果
        self.show_cross_val_all()

    def fit(self, buy_factors, sell_factors, cv=10, market=None, enable_cache=True, n_folds=5):
        """
        主执行函数：
        1. 首先计算全市场所有symbol与大盘指标的相关度
        2. 通过pd.qcut将相关数据corr_series切分成cv个相关范围段
        3. 启动cv个进程分别在每个进程中对相关范围段symbol进行回测
        4. 汇总多进程的执行结果使用chain.from_iterable将结果摊平
        :param buy_factors: 买入因子组合
        :param sell_factors: 卖出因子组合
        :param cv: cv默认为为10
        :param market: 进行验证的市场，默认None将使用env中设置的市场
        :param enable_cache: 是否重复使用相关度数据缓存
        :param n_folds: 交叉验证的回测历史年数，需要确保本地有缓存对应的年数数据存在
        """

        if ABuEnv._g_enable_example_env_ipython:
            # 只支持非沙盒本地数据模式
            self.log_func('cross val only support local data, sandbox data now!')
            self.log_func('please use abupy.env.disable_example_env_ipython() close sandbox mode!')
            return

        # 相关性全市场数据监测
        if not all_market_env_check():
            return

        if market is not None:
            self.market = market

        if self.market == EMarketTargetType.E_MARKET_TARGET_US:
            # 美股
            benchmark = IndexSymbol.IXIC
        elif ABuEnv.g_market_target == EMarketTargetType.E_MARKET_TARGET_HK:
            # 港股
            benchmark = IndexSymbol.HSI
        elif ABuEnv.g_market_target == EMarketTargetType.E_MARKET_TARGET_CN:
            # a股
            benchmark = IndexSymbol.SH
        else:
            raise RuntimeError('cross val only support us, cn, hk market!')

        restore_market = ABuEnv.g_market_target
        # 临时切换市场，都完事后需要再切换回来
        ABuEnv.g_market_target = self.market

        # 需要强制走本地数据，体高效率以及对比度公正
        restore_date_mode = ABuEnv.g_data_fetch_mode
        # 临时强制走本地数据，都完事后需要再切换回来
        ABuEnv.g_data_fetch_mode = EMarketDataFetchMode.E_DATA_FETCH_FORCE_LOCAL

        sorted_corr = self._find_or_cache_similar(n_folds, benchmark, enable_cache)
        """
            eg:
                sorted_corr:

                [(us_NYSE:.IXIC, 0.99999999999999989),
                 ('usTQQQ', 0.98232461061643761),
                 ('usQQQ', 0.98189324977534143),
                 ('usVONG', 0.97282023230260084),
                 ('usDGRW', 0.92769329708593817),
                 ('usTDIV', 0.90129474970808454),
                 ('usVTHR', 0.89865579645001936),
                 ('usCFO', 0.89842456369113732),
                 ('usPNQI', 0.88093679204010267),
                 ('usTY', 0.86932098539827651)]

                corr_series:

                us.IXIC    1.0000
                usTQQQ     0.9823
                usQQQ      0.9819
                usONEQ     0.9795
                usVONG     0.9728
                usQQEW     0.9681
                usQQXT     0.9428
                usVONE     0.9389
                usDGRW     0.9277
                usSKYY     0.9211
        """
        corr_series = pd.Series([corr[1] for corr in sorted_corr], index=[corr[0] for corr in sorted_corr])
        # 把第一个大盘symbol跳过
        corr_series = corr_series[1:]
        self._do_cross_corr(buy_factors, sell_factors, benchmark, corr_series, cv, n_folds)

        # 恢复之前的市场设置以及数据获取模式
        ABuEnv.g_market_target = restore_market
        ABuEnv.g_data_fetch_mode = restore_date_mode

    def plot_all_cross_val_orders(self):
        """通过fit函数进行相关交叉验证后，将验证的所有交易单保存在本地"""
        with AbuProgress(len(self.metrics_array), 0, 'save and plot orders to png file') as progess:
            for index, metrics in enumerate(self.metrics_array):
                progess.show(index + 1)
                if metrics is not None and metrics[0].valid and metrics[0].orders_pd.shape[0] > 0:
                    ABuMarketDrawing.plot_candle_from_order(metrics[0].orders_pd, save=True)
        self.log_func('all orders plot and save to png complete! path={}'.format(ABuMarketDrawing.save_dir_name()))

    def show_cross_val_se(self, start=0, end=1):
        """
           显示通过fit函数进行验证后得到的metrics_array中的：
           1. 相关度范围段
           2. 相关度范围段随机抽取的symbol
           3. symbol对应的度量结果

           默认只显示一个，设置参数start， start的值调整显示数量和范围
        """
        for metrics in self.metrics_array[start:end]:
            if metrics is not None and metrics[0].valid:
                self.log_func(u'回测symbol:{}'.format(metrics[2]))
                self.log_func(u'回测symbol与大盘相关度范围:{}'.format(metrics[1]))
                metrics[0].plot_order_returns_cmp()
                self.log_func('\n')

    def show_cross_val_all(self):
        """
            显示通过fit函数进行验证后得到的metrics_array中所有交易的度量结果：
            统计所有交易数量，加权计算所有交易的
            1. 买入后卖出的交易总数量
            2. 胜率
            3. 平均获利期望
            4. 平均亏损期望
            5. 盈亏比
            6. 所有交易收益比例和
            7. 所有交易总盈亏和
        """

        def _show_metrics(metrics_array, p_title):
            all_deal_cnt = 0
            all_win_rate = 0
            all_gains_mean = 0
            all_losses_mean = 0
            all_win_loss_profit_rate = 0
            all_profit_cg = 0
            all_profit = 0
            for metrics in metrics_array:
                metrics = metrics[0]
                if metrics is not None and metrics.valid:
                    deal_cnt = metrics.order_has_ret.shape[0]
                    all_deal_cnt += deal_cnt
                    all_win_rate += metrics.win_rate * deal_cnt
                    all_gains_mean += metrics.gains_mean * deal_cnt
                    all_losses_mean += metrics.losses_mean * deal_cnt
                    all_win_loss_profit_rate += metrics.win_loss_profit_rate * deal_cnt
                    all_profit_cg += metrics.order_has_ret.profit_cg.sum()
                    all_profit += metrics.all_profit

            self.log_func(p_title)
            self.log_func(u'买入后卖出的交易总数量:{}'.format(all_deal_cnt))
            self.log_func(u'胜率:{:.4f}%'.format(all_win_rate / all_deal_cnt * 100))
            self.log_func(u'平均获利期望:{:.4f}%'.format(all_gains_mean / all_deal_cnt * 100))
            self.log_func(u'平均亏损期望:{:.4f}%'.format(all_losses_mean / all_deal_cnt * 100))
            self.log_func(u'盈亏比:{:.4f}'.format(all_win_loss_profit_rate / all_deal_cnt))
            self.log_func(u'所有交易收益比例和:{:.4f} '.format(all_profit_cg))
            self.log_func(u'所有交易总盈亏和:{:.4f} '.format(all_profit))
            self.log_func('\n')

        _show_metrics(self.metrics_array, u'所有交叉验证交易度量结果如下：')

        for metrics_cv in self.out_cross_val:
            if len(metrics_cv) > 0:
                title = u'与大盘相关度范围:{}验证结果如下：'.format(metrics_cv[0][1])
                _show_metrics(metrics_cv, title)
