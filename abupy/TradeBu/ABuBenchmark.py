# -*- encoding:utf-8 -*-
"""
    基准模块，基准的作用在于交易时间范围确定，交易时间序列对齐，
    抛弃异常时间序列，交易市场范围限制，以及对比与策略的度量结果等作用
"""

from __future__ import print_function
from __future__ import absolute_import
from __future__ import division

from ..CoreBu.ABuEnv import EMarketDataSplitMode, EMarketTargetType
from ..MarketBu import ABuSymbolPd
from ..MarketBu.ABuSymbol import IndexSymbol, Symbol
from ..CoreBu import ABuEnv
from ..CoreBu.ABuBase import PickleStateMixin
from ..CoreBu.ABuFixes import six

__author__ = '阿布'
__weixin__ = 'abu_quant'


class AbuBenchmark(PickleStateMixin):
    """基准类，混入PickleStateMixin，因为在abu.store_abu_result_tuple会进行对象本地序列化"""

    def __init__(self, benchmark=None, start=None, end=None, n_folds=2, rs=True, benchmark_kl_pd=None):
        if benchmark_kl_pd is not None and hasattr(benchmark_kl_pd, 'name'):
            """从金融时间序列直接构建"""
            self.benchmark = benchmark_kl_pd.name
            self.start = benchmark_kl_pd.iloc[0].date
            self.end = benchmark_kl_pd.iloc[-1].date
            self.n_folds = n_folds
            self.kl_pd = benchmark_kl_pd
            return

        if benchmark is None:
            if ABuEnv.g_market_target == EMarketTargetType.E_MARKET_TARGET_US:
                # 美股
                benchmark = IndexSymbol.IXIC
            elif ABuEnv.g_market_target == EMarketTargetType.E_MARKET_TARGET_HK:
                # 港股
                benchmark = IndexSymbol.HSI
            elif ABuEnv.g_market_target == EMarketTargetType.E_MARKET_TARGET_CN:
                # a股
                benchmark = IndexSymbol.SH
            elif ABuEnv.g_market_target == EMarketTargetType.E_MARKET_TARGET_FUTURES_CN:
                # 国内期货
                benchmark = IndexSymbol.BM_FUTURES_CN
            elif ABuEnv.g_market_target == EMarketTargetType.E_MARKET_TARGET_TC:
                # 币类市场
                benchmark = IndexSymbol.TC_INX
            elif ABuEnv.g_market_target == EMarketTargetType.E_MARKET_TARGET_OPTIONS_US:
                # 美股期权暂时也以IXIC做为标尺，最好是外部参数中的benchmark设置
                benchmark = IndexSymbol.IXIC
            elif ABuEnv.g_market_target == EMarketTargetType.E_MARKET_TARGET_FUTURES_GLOBAL:
                # 国际期货暂时也以BM_FUTURES_GB做为标尺
                benchmark = IndexSymbol.BM_FUTURES_GB
            else:
                raise TypeError('benchmark is None AND g_market_target ERROR!')

        self.benchmark = benchmark
        self.start = start
        self.end = end
        self.n_folds = n_folds
        # 基准获取数据使用data_mode=EMarketDataSplitMode.E_DATA_SPLIT_SE，即不需要对齐其它，只需要按照时间切割
        self.kl_pd = ABuSymbolPd.make_kl_df(benchmark, data_mode=EMarketDataSplitMode.E_DATA_SPLIT_SE,
                                            n_folds=n_folds,
                                            start=start, end=end)

        if rs and self.kl_pd is None:
            # 如果基准时间序列都是none，就不要再向下运行了
            raise ValueError('CapitalClass init benchmark kl_pd is None')

    def unpick_extend_work(self, state):
        """完成 PickleStateMixin中__setstate__结束之前的工作，为kl_pd.name赋予准确的benchmark"""
        if isinstance(self.benchmark, Symbol):
            self.kl_pd.name = self.benchmark.value
        elif isinstance(self.benchmark, six.string_types):
            self.kl_pd.name = self.benchmark

    def __str__(self):
        """打印对象显示：benchmark n_folds"""
        return 'benchmark is {}, n_folds = {}'.format(self.kl_pd.name, self.n_folds)

    __repr__ = __str__

"""
    # 如果需要本地序列化很多，需要考虑存贮空间可使用LazyFunc.
    from ..UtilBu.ABuLazyUtil import LazyFunc

    class AbuBenchmark(object):
        def __init__(self, benchmark=None, start=None, end=None, n_folds=2):
            if benchmark is None:
                if ABuEnv.g_market_target == EMarketTargetType.E_MARKET_TARGET_US:
                    benchmark = IndexSymbol.IXIC
                elif ABuEnv.g_market_target == EMarketTargetType.E_MARKET_TARGET_HK:
                    benchmark = IndexSymbol.HSI
                else:
                    benchmark = IndexSymbol.SH
            self.benchmark = benchmark

            self.n_folds = n_folds
            self.start = start
            self.end = end

    @LazyFunc
    def kl_pd(self):
        kl_pd = ABuSymbolPd.make_kl_df(self.benchmark, data_mode=EMarketDataSplitMode.E_DATA_SPLIT_SE,
                                       n_folds=self.n_folds,
                                       start=self.start, end=self.end)

        if kl_pd is None:
            raise ValueError('CapitalClass init benchmark kl_pd is None')
        return kl_pd
"""
