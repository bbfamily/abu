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
