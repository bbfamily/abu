# -*- encoding:utf-8 -*-
"""
    选股示例因子：价格选股因子
"""
from __future__ import print_function
from __future__ import absolute_import
from __future__ import division

from .ABuPickStockBase import AbuPickStockBase, reversed_result
import numpy as np

__author__ = '阿布'
__weixin__ = 'abu_quant'


class AbuPickStockPriceMinMax(AbuPickStockBase):
    """价格选股因子示例类"""
    def _init_self(self, **kwargs):
        """通过kwargs设置选股价格边际条件，配置因子参数"""

        # 暂时与base保持一致不使用kwargs.pop('a', default)方式
        # fit_pick中选择 > 最小(threshold_price_min), 默认负无穷，即默认所有都符合
        self.threshold_price_min = -np.inf
        if 'threshold_price_min' in kwargs:
            # 最小价格阀值
            self.threshold_price_min = kwargs['threshold_price_min']

        # fit_pick中选择 < 最大(threshold_price_max), 默认正无穷，即默认所有都符合
        self.threshold_price_max = np.inf
        if 'threshold_price_max' in kwargs:
            # 最大价格阀值
            self.threshold_price_max = kwargs['threshold_price_max']

    @reversed_result
    def fit_pick(self, kl_pd, target_symbol):
        """开始根据自定义价格边际参数进行选股"""
        if kl_pd.close.max() < self.threshold_price_max and kl_pd.close.min() > self.threshold_price_min:
            # kl_pd.close的最大价格 < 最大价格阀值 且 kl_pd.close的最小价格 > 最小价格阀值
            return True
        return False

    def fit_first_choice(self, pick_worker, choice_symbols, *args, **kwargs):
        raise NotImplementedError('AbuPickStockPriceMinMax fit_first_choice unsupported now!')
