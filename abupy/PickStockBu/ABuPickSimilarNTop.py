# -*- encoding:utf-8 -*-
"""
    选股示例因子：相似度选股因子，主要示例fit_first_choice
"""

from __future__ import print_function
from __future__ import absolute_import
from __future__ import division

import numpy as np

from ..UtilBu import ABuDateUtil
from ..TLineBu import ABuTLSimilar
from .ABuPickStockBase import AbuPickStockBase, reversed_result
from ..SimilarBu.ABuSimilar import find_similar_with_se, ECoreCorrType

"""外部可通过如：abupy.ps.similar_top = 300修改默认值"""
g_pick_similar_n_top = 100


# noinspection PyAttributeOutsideInit
class AbuPickSimilarNTop(AbuPickStockBase):
    """相似度选股因子示例类"""

    def _init_self(self, **kwargs):
        """通过kwargs设置相似度选股边际条件，相似度计算方法，返回目标数量等，配置因子参数"""

        # 暂时与base保持一致不使用kwargs.pop('a', default)方式
        # 设置目标相似度交易对象
        self.similar_stock = kwargs['similar_stock']

        # 设置目标相似度选取top个数，只在fit_first_choice中使用，默认top100, 即前100个最相似的股票
        self.n_top = g_pick_similar_n_top
        if 'n_top' in kwargs:
            self.n_top = kwargs['n_top']

        # 设置fit_pick中使用的最小相似边际条件
        self.threshold_similar_min = -np.inf
        if 'threshold_similar_min' in kwargs:
            self.threshold_similar_min = kwargs['threshold_similar_min']

        # 设置fit_pick中使用的最大相似边际条件
        self.threshold_similar_max = np.inf
        if 'threshold_similar_max' in kwargs:
            self.threshold_similar_max = kwargs['threshold_similar_max']

        # 相似度是否使用时间加权计算，如果使用速度会慢
        self.rolling = False
        if 'rolling' in kwargs:
            self.rolling = kwargs['rolling']

        # 相似度计算使用算法设置
        self.corr_type = ECoreCorrType.E_CORE_TYPE_PEARS
        if 'corr_type' in kwargs:
            self.corr_type = kwargs['corr_type']
        # 相似度rank缓存
        self.s_sum_rank = None

    @reversed_result
    def fit_pick(self, kl_pd, target_symbol):
        """开始根据自定义相似度边际条件进行选股"""

        # 由于外层worker需要不断迭代symbol使用同一个因子对象进行选股，所以这里缓存了相似度rank结果，只计算一次
        similar_rank, self.s_sum_rank = ABuTLSimilar.calc_similar(self.similar_stock, target_symbol,
                                                                  self.s_sum_rank,
                                                                  show=False)
        # 边际筛选
        if self.threshold_similar_min < similar_rank < self.threshold_similar_max:
            return True
        return False

    def fit_first_choice(self, pick_worker, choice_symbols, *args, **kwargs):
        """
        因子相似度批量选股接口
        :param pick_worker: 选股worker，AbuPickStockWorker实例对象
        :param choice_symbols: 初始备选交易对象序列
        :return:
        """

        # 获取交易目标选股阶段的金融时间序列
        similar_kl_pd = pick_worker.kl_pd_manager.get_pick_stock_kl_pd(self.similar_stock, self.xd, self.min_xd)
        if similar_kl_pd is None or len(similar_kl_pd) == 0:
            return []

        # 选股阶段的金融时间序列最后一个日期作为similar_end
        similar_end = ABuDateUtil.timestamp_to_str(similar_kl_pd.index[-1])
        # 选股阶段的金融时间序列第一个个日期作为similar_start
        similar_start = ABuDateUtil.timestamp_to_str(similar_kl_pd.index[0])
        # 通过ABuSimilar模块中的find_similar_with_se计算与交易目标的相似度rank dict
        net_cg_ret = find_similar_with_se(self.similar_stock, similar_start, similar_end,
                                          rolling=self.rolling, show=False,
                                          corr_type=self.corr_type)
        # 取相似度结果的n_top个，作为选股结果
        similar_top_choice = [ss[0] for ss in net_cg_ret[1:self.n_top + 1]]
        # 通过集合选取在初始备选交易对象序列和相似度选股序列中的子序列
        return list(set(similar_top_choice) & set(choice_symbols))
