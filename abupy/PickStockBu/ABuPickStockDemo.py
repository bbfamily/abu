# -*- encoding:utf-8 -*-
"""
    选股示例因子：价格选股因子
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np

from .ABuPickStockBase import AbuPickStockBase, reversed_result
from ..TLineBu.ABuTL import AbuTLine
from ..CoreBu.ABuEnv import EMarketDataSplitMode
from ..MarketBu import ABuSymbolPd
from ..TradeBu import AbuBenchmark

__author__ = '阿布'
__weixin__ = 'abu_quant'


class AbuPickStockShiftDistance(AbuPickStockBase):
    """位移路程比选股因子示例类"""

    def _init_self(self, **kwargs):
        """通过kwargs设置位移路程比选股条件，配置因子参数"""
        self.threshold_sd = kwargs.pop('threshold_sd', 2.0)
        self.threshold_max_cnt = kwargs.pop('threshold_max_cnt', 4)
        self.threshold_min_cnt = kwargs.pop('threshold_min_cnt', 1)

    @reversed_result
    def fit_pick(self, kl_pd, target_symbol):
        """开始根据位移路程比边际参数进行选股"""

        pick_line = AbuTLine(kl_pd.close, 'shift distance')
        shift_distance = pick_line.show_shift_distance(step_x=1.2, show_log=False, show=False)
        shift_distance = np.array(shift_distance)
        # show_shift_distance返回的参数为四组数据，最后一组是每个时间段的位移路程比值
        sd_arr = shift_distance[:, -1]
        # 大于阀值的进行累加和计算
        # noinspection PyUnresolvedReferences
        threshold_cnt = (sd_arr >= self.threshold_sd).sum()
        # 边际条件参数开始生效
        if self.threshold_max_cnt > threshold_cnt >= self.threshold_min_cnt:
            return True
        return False

    def fit_first_choice(self, pick_worker, choice_symbols, *args, **kwargs):
        raise NotImplementedError('AbuPickStockShiftDistance fit_first_choice unsupported now!')


class AbuPickStockNTop(AbuPickStockBase):
    """根据一段时间内的涨幅选取top N个"""

    def _init_self(self, **kwargs):
        """通过kwargs设置选股条件，配置因子参数"""
        # 选股参数symbol_pool：进行涨幅比较的top n个symbol
        self.symbol_pool = kwargs.pop('symbol_pool', [])
        # 选股参数n_top：选取前n_top个symbol, 默认3
        self.n_top = kwargs.pop('n_top', 3)
        # 选股参数direction_top：选取前n_top个的方向，即选择涨的多的，还是选择跌的多的
        self.direction_top = kwargs.pop('direction_top', 1)

    @reversed_result
    def fit_pick(self, kl_pd, target_symbol):
        """开始根据参数进行选股"""
        if len(self.symbol_pool) == 0:
            # 如果没有传递任何参照序列symbol，择默认为选中
            return True
        # 定义lambda函数计算周期内change
        kl_change = lambda p_kl: \
            p_kl.iloc[-1].close / p_kl.iloc[0].close if p_kl.iloc[0].close != 0 else 0

        cmp_top_array = []
        kl_pd.name = target_symbol
        # AbuBenchmark直接传递一个kl
        benchmark = AbuBenchmark(benchmark_kl_pd=kl_pd)
        for symbol in self.symbol_pool:
            if symbol != target_symbol:
                # 使用benchmark模式进行获取
                kl = ABuSymbolPd.make_kl_df(symbol, data_mode=EMarketDataSplitMode.E_DATA_SPLIT_UNDO,
                                            benchmark=benchmark)
                # kl = ABuSymbolPd.make_kl_df(symbol, start=start, end=end)
                if kl is not None and kl.shape[0] > kl_pd.shape[0] * 0.75:
                    # 需要获取实际交易日数量，避免停盘等错误信号
                    cmp_top_array.append(kl_change(kl))

        if self.n_top > len(cmp_top_array):
            # 如果结果序列不足n_top个，直接认为选中
            return True

        # 与选股方向相乘，即结果只去top
        cmp_top_array = np.array(cmp_top_array) * self.direction_top
        # 计算本源的周期内涨跌幅度
        target_change = kl_change(kl_pd) * self.direction_top
        # sort排序小－》大, 非inplace
        cmp_top_array.sort()
        # [::-1]大－》小
        # noinspection PyTypeChecker
        if target_change > cmp_top_array[::-1][self.n_top - 1]:
            # 如果比排序后的第self.n_top位置上的大就认为选中
            return True
        return False

    def fit_first_choice(self, pick_worker, choice_symbols, *args, **kwargs):
        raise NotImplementedError('AbuPickStockNTop fit_first_choice unsupported now!')
