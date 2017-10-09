# -*- encoding:utf-8 -*-
"""
    买入择时示例因子：动态自适应双均线策略
"""

from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import math

import numpy as np

from .ABuFactorBuyBase import AbuFactorBuyXD, BuyCallMixin
from ..IndicatorBu.ABuNDMa import calc_ma_from_prices
from ..CoreBu.ABuPdHelper import pd_resample
from ..TLineBu.ABuTL import AbuTLine

__author__ = '阿布'
__weixin__ = 'abu_quant'


# noinspection PyAttributeOutsideInit
class AbuDoubleMaBuy(AbuFactorBuyXD, BuyCallMixin):
    """示例买入动态自适应双均线策略"""

    def _init_self(self, **kwargs):
        """
            kwargs中可选参数：fast: 均线快线周期，默认不设置，使用自适应动态快线
            kwargs中可选参数：slow: 均线慢线周期，默认不设置，使用自适应动态慢线
            kwargs中可选参数：resample_max: 动态慢线可设置参数重采样周期最大值，默认100，即动态慢线最大100
            kwargs中可选参数：resample_min: 动态慢线可设置参数重采样周期最小值，默认10，即动态慢线最小10
            kwargs中可选参数：change_threshold：动态慢线可设置参数代表慢线的选取阀值，默认0.12
        """

        # 均线快线周期，默认使用5天均线
        self.ma_fast = kwargs.pop('fast', -1)
        self.dynamic_fast = False
        if self.ma_fast == -1:
            self.ma_fast = 5
            self.dynamic_fast = True

        # 均线慢线周期，默认使用60天均线
        self.ma_slow = kwargs.pop('slow', -1)
        self.dynamic_slow = False
        if self.ma_slow == -1:
            self.ma_slow = 60
            self.dynamic_slow = True
        # 动态慢线可设置参数重采样周期最大值，默认90
        self.resample_max = kwargs.pop('resample_max', 100)
        # 动态慢线可设置参数重采样周期最小值，默认10
        self.resample_min = kwargs.pop('resample_min', 10)
        # 动态慢线可设置参数代表慢线的选取阀值，默认0.12
        self.change_threshold = kwargs.pop('change_threshold', 0.12)

        if self.ma_fast >= self.ma_slow:
            # 慢线周期必须大于快线
            raise ValueError('ma_fast >= self.ma_slow !')

        # xd周期数据需要比ma_slow大一天，这样计算ma就可以拿到今天和昨天两天的ma，用来判断金叉，死叉
        kwargs['xd'] = self.ma_slow + 1
        # 设置好xd后可以直接使用基类针对xd的初始化
        super(AbuDoubleMaBuy, self)._init_self(**kwargs)

        # 在输出生成的orders_pd中显示的名字
        self.factor_name = '{}:fast={},slow={}'.format(self.__class__.__name__, self.ma_fast, self.ma_slow)

    def _dynamic_calc_fast(self, today):
        """
            根据大盘最近一个月走势震荡程度，动态决策快线的值，规则如下：
            如果大盘最近一个月走势使用：
                    一次拟合可以表达：fast＝slow * 0.05 eg: slow=60->fast=60*0.05=3
                    二次拟合可以表达：fast＝slow * 0.15 eg: slow=60->fast=60*0.15=9
                    三次拟合可以表达：fast＝slow * 0.3 eg: slow=60->fast=60*0.3=18
                    四次及以上拟合可以表达：fast＝slow * 0.5 eg: slow=60->fast=60*0.5=30
        """
        # 策略中拥有self.benchmark，即交易基准对象，AbuBenchmark实例对象，benchmark.kl_pd即对应的市场大盘走势
        benchmark_df = self.benchmark.kl_pd
        # 拿出大盘的今天
        benchmark_today = benchmark_df[benchmark_df.date == today.date]
        if benchmark_today.empty:
            # 默认值为慢线的0.15
            return math.ceil(self.ma_slow * 0.15)

        # 要拿大盘最近一个月的走势，准备切片的start，end
        end_key = int(benchmark_today.iloc[0].key)
        start_key = end_key - 20
        if start_key < 0:
            # 默认值为慢线的0.15
            return math.ceil(self.ma_slow * 0.15)
        # 使用切片切出从今天开始向前20天的数据
        benchmark_month = benchmark_df[start_key:end_key + 1]
        # 通过大盘最近一个月的收盘价格做为参数构造AbuTLine对象
        benchmark_month_line = AbuTLine(benchmark_month.close, 'benchmark month line')
        # 计算这个月最少需要几次拟合才能代表走势曲线
        least = benchmark_month_line.show_least_valid_poly(show=False)
        if least == 1:
            # 一次拟合可以表达：fast＝slow * 0.05 eg: slow=60->fast=60*0.05=3
            return math.ceil(self.ma_slow * 0.05)
        elif least == 2:
            # 二次拟合可以表达：fast＝slow * 0.15 eg: slow=60->fast=60*0.15=9
            return math.ceil(self.ma_slow * 0.15)
        elif least == 3:
            # 三次拟合可以表达：fast＝slow * 0.3 eg: slow=60->fast=60*0.3=18
            return math.ceil(self.ma_slow * 0.3)
        else:
            # 四次及以上拟合可以表达：fast＝slow * 0.5 eg: slow=60->fast=60*0.5=30
            return math.ceil(self.ma_slow * 0.5)

    def _dynamic_calc_slow(self, today):
        """
            动态决策慢线的值，规则如下：

            切片最近一段时间的金融时间序列，对金融时间序列进行变换周期重新采样，
            对重新采样的结果进行pct_change处理，对pct_change序列取abs绝对值，
            对pct_change绝对值序列取平均，即算出重新采样的周期内的平均变化幅度，

            上述的变换周期由10， 15，20，30....进行迭代, 直到计算出第一个重新
            采样的周期内的平均变化幅度 > 0.12的周期做为slow的取值
        """
        last_kl = self.past_today_kl(today, self.resample_max)
        if last_kl.empty:
            # 返回慢线默认值60
            return 60

        for slow in np.arange(self.resample_min, self.resample_max, 5):
            rule = '{}D'.format(slow)
            change = abs(pd_resample(last_kl.close, rule, how='mean').pct_change()).mean()
            """
                eg: pd_resample(last_kl.close, rule, how='mean')

                    2014-07-23    249.0728
                    2014-09-03    258.3640
                    2014-10-15    240.8663
                    2014-11-26    220.1552
                    2015-01-07    206.0070
                    2015-02-18    198.0932
                    2015-04-01    217.9791
                    2015-05-13    251.3640
                    2015-06-24    266.4511
                    2015-08-05    244.3334
                    2015-09-16    236.2250
                    2015-10-28    222.0441
                    2015-12-09    222.0574
                    2016-01-20    177.2303
                    2016-03-02    226.8766
                    2016-04-13    230.6000
                    2016-05-25    216.7596
                    2016-07-06    222.6420

                    abs(pd_resample(last_kl.close, rule, how='mean').pct_change())

                    2014-09-03    0.037
                    2014-10-15    0.068
                    2014-11-26    0.086
                    2015-01-07    0.064
                    2015-02-18    0.038
                    2015-04-01    0.100
                    2015-05-13    0.153
                    2015-06-24    0.060
                    2015-08-05    0.083
                    2015-09-16    0.033
                    2015-10-28    0.060
                    2015-12-09    0.000
                    2016-01-20    0.202
                    2016-03-02    0.280
                    2016-04-13    0.016
                    2016-05-25    0.060
                    2016-07-06    0.027

                    abs(pd_resample(last_kl.close, rule, how='mean').pct_change()).mean():

                    0.080
            """
            if change > self.change_threshold:
                """
                    返回第一个大于change_threshold的slow,
                    change_threshold默认为0.12，以周期突破的策略一般需要在0.08以上，0.12是为快线留出套利空间
                """
                return slow
        # 迭代np.arange(min, max, 5)都不符合就返回max
        return self.resample_max

    def fit_month(self, today):
        # fit_month即在回测策略中每一个月执行一次的方法
        if self.dynamic_slow:
            # 一定要先动态算ma_slow，因为动态计算fast依赖slow
            self.ma_slow = self._dynamic_calc_slow(today)
        if self.dynamic_fast:
            # 动态计算快线
            self.ma_fast = self._dynamic_calc_fast(today)
        # 动态重新计算后，改变在输出生成的orders_pd中显示的名字
        self.factor_name = '{}:fast={},slow={}'.format(self.__class__.__name__, self.ma_fast, self.ma_slow)
        # import logging
        # logging.debug('{}:{}-fast={}|slow={}'.format(self.kl_pd.name, today.date, self.ma_fast, self.ma_slow))

    def fit_day(self, today):
        """双均线买入择时因子，信号快线上穿慢行形成金叉做为买入信号"""
        # 计算快线
        fast_line = calc_ma_from_prices(self.xd_kl.close, int(self.ma_fast), min_periods=1)
        # 计算慢线
        slow_line = calc_ma_from_prices(self.xd_kl.close, int(self.ma_slow), min_periods=1)

        if len(fast_line) >= 2 and len(slow_line) >= 2:
            # 今天的快线值
            fast_today = fast_line[-1]
            # 昨天的快线值
            fast_yesterday = fast_line[-2]
            # 今天的慢线值
            slow_today = slow_line[-1]
            # 昨天的慢线值
            slow_yesterday = slow_line[-2]

            if slow_yesterday >= fast_yesterday and fast_today > slow_today:
                # 快线上穿慢线, 形成买入金叉，使用了今天收盘价格，明天买入
                return self.buy_tomorrow()

    """可以选择是否覆盖AbuFactorBuyXD中的buy_tomorrow来增大交易频率，默认基类中self.skip_days = self.xd降低了频率"""
    # def buy_tomorrow(self):
    #     return self.make_buy_order(self.today_ind)
