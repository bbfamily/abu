# -*- encoding:utf-8 -*-
"""
    买入择时示例因子：黄金分割线买入择时因子
"""

from __future__ import absolute_import
from __future__ import print_function
from __future__ import division


from ..TLineBu import ABuTLGolden
from ..CoreBu.ABuPdHelper import pd_rolling_mean
from .ABuFactorBuyBase import AbuFactorBuyBase, BuyCallMixin

__author__ = '阿布'
__weixin__ = 'abu_quant'


# noinspection PyAttributeOutsideInit
class AbuFactorBuyGolden(AbuFactorBuyBase, BuyCallMixin):
    """示例黄金分割线买入择时因子，混入BuyCallMixin，即向上突破触发买入event"""

    def _init_self(self, **kwargs):
        """
            kwargs中必须: 突破参数xd 比如20，30，40天...作为分割周期
            kwargs中可选参数: 突破参数ma_day(int)，作为在择时突破中的买入event条件
            kwargs中可选参数: 短暂停留阀值stay_day(int)，作为在择时突破中的买入event条件
        """
        # 黄金分割参数xd， 比如20，30，40天...作为分割周期
        self.xd = kwargs['xd']
        # 突破参数ma_day(int), 默认5：代表使用5日移动均线
        self.ma_day = kwargs.pop('ma_day', 5)
        # 短暂停留阀值stay_day(int)，默认进入1天即算
        self.stay_day = kwargs.pop('stay_day', 1)
        # 交易目标短暂停留的标志
        self.below_stay_days = 0
        # 在输出生成的orders_pd中显示的名字
        self.factor_name = '{}:{}'.format(self.__class__.__name__, self.xd)

    def fit_day(self, today):
        """
        策略属于均值回复类型策略：当价格在黄金分割线低档位附近徘徊且有向上回复的趋势时买入目标
        :param today: 当前驱动的交易日金融时间序列数据
        :return:
        """
        # 忽略不符合买入的天（统计周期内前xd天）
        if self.today_ind < self.xd - 1:
            return None

        # 切片从前xd开始直到当前交易日为止的金融序列数据
        window_pd = self.kl_pd[self.today_ind - self.xd + 1: self.today_ind + 1]
        # 在切片的金融序列数据上计算黄金分割档位值，具体阅读ABuTLGolden.calc_golden
        golden = ABuTLGolden.calc_golden(window_pd, show=False)

        """
            目标买入区域在below382－>above382之间，一旦 < below382彻底放弃一段时间的买入观察，
            一旦进入below382－>above382区间也不是立即执行买单，需要在此区域短暂停留后有向上的趋势
            再买入
        """
        if today['close'] < golden.below382:
            # 进入< below382 彻底放弃一段时间的买入观察，放弃周期＝self.xd
            self.below_stay_days = 0
            self.skip_days = self.xd
        elif golden.above382 > today['close'] >= golden.below382:
            # 作为交易目标短暂停留的标志
            self.below_stay_days += 1

        def ma_break_func():
            """
               分别计算xd天的移动平均价格和初始化中的ma_day天的移动平均价格，
               买入条件是ma_xd > ma_cross, 即快线突破慢线
            """
            # 快线尾巴
            ma_cross = pd_rolling_mean(window_pd.close, window=self.ma_day, min_periods=self.ma_day)[-1]
            # 慢线尾巴
            ma_xd = pd_rolling_mean(window_pd.close, window=self.xd, min_periods=self.xd)[-1]
            if ma_cross > ma_xd:
                # 快线突破慢线
                return True
            return False

        """
            self.below_stay_days >= self.stay_day: 代表进入below382－>above382区域短暂停留，
                                                   可以通过_init_self参数微调短暂停留阀值stay_day
            today['close'] >= golden.above382 or ma_break_func(): 代表有向上的趋势
        """
        if self.below_stay_days >= self.stay_day and (today['close'] >= golden.above382 or ma_break_func()):
            # 重置交易目标短暂停留的标志
            self.below_stay_days = 0
            # 放弃一段时间的买入观察, 放弃周期＝self.xd/2
            self.skip_days = self.xd / 2
            # 生成买单, 由于使用了今天的收盘价格做为策略信号判断，所以信号发出后，只能明天买
            return self.buy_tomorrow()
        return None
