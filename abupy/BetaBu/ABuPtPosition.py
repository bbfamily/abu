# -*- encoding:utf-8 -*-
"""示例仓位管理：示例价格位置仓位模块"""

from __future__ import division
from __future__ import absolute_import
from __future__ import print_function

from scipy import stats

from .ABuPositionBase import AbuPositionBase


class AbuPtPosition(AbuPositionBase):
    """
        示例价格位置仓位管理类：

        根据买入价格在之前一段时间的价格位置来决策仓位大小

        假设过去一段时间的价格为[10, 9, 8, 7, 6, 5, 4, 3, 2, 1]
        如果当前买入价格为2元：则买入仓位配比很高(认为均值回复有很大向上空间)
        如果当前买入价格为9元：则买入仓位配比很低(认为均值回复向上空间比较小)
    """

    def fit_position(self, factor_object):
        """
        针对均值回复类型策略的仓位管理：
        根据当前买入价格在过去一段金融序列中的价格rank位置来决定仓位
        fit_position计算的结果是买入多少个单位（股，手，顿，合约）
        :param factor_object: ABuFactorBuyBases子类实例对象
        :return:买入多少个单位（股，手，顿，合约）
        """

        # self.kl_pd_buy为买入当天的数据，获取之前的past_day_cnt天数据
        last_kl = factor_object.past_today_kl(self.kl_pd_buy, self.past_day_cnt)
        if last_kl is None or last_kl.empty:
            precent_pos = self.pos_base
        else:
            # 使用percentileofscore计算买入价格在过去的past_day_cnt天的价格位置
            precent_pos = stats.percentileofscore(last_kl.close, self.bp)
            precent_pos = (1 + (self.mid_precent - precent_pos) / 100) * self.pos_base
        # 最大仓位限制，依然受上层最大仓位控制限制，eg：如果算出全仓，依然会减少到75%，如修改需要修改最大仓位值
        precent_pos = self.pos_max if precent_pos > self.pos_max else precent_pos
        # 结果是买入多少个单位（股，手，顿，合约）
        return self.read_cash * precent_pos / self.bp * self.deposit_rate

    def _init_self(self, **kwargs):
        """价格位置仓位控制管理类初始化设置"""
        # 默认平均仓位比例0.10，即10%
        self.pos_base = kwargs.pop('pos_base', 0.10)
        # 默认获取之前金融时间序列的长短数量
        self.past_day_cnt = kwargs.pop('past_day_cnt', 20)
        # 默认的比例中值，一般不需要设置
        self.mid_precent = kwargs.pop('mid_precent', 50.0)
