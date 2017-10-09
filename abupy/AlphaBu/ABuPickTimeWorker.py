# -*- encoding:utf-8 -*-
"""
    择时具体工作者，整合金融时间序列，买入因子，卖出因子，资金类进行
    择时操作，以时间驱动择时事件的发生
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import copy

import numpy as np

from ..MarketBu import ABuSymbolPd
from ..FactorBuyBu.ABuFactorBuyBase import AbuFactorBuyBase
from ..FactorSellBu.ABuFactorSellBase import AbuFactorSellBase
from .ABuPickBase import AbuPickTimeWorkBase
# noinspection PyUnresolvedReferences
from ..CoreBu.ABuFixes import filter
from ..UtilBu.ABuProgress import AbuMulPidProgress

__author__ = '阿布'
__weixin__ = 'abu_quant'

"""
    是否使用自然周，自然月，默认开启，如需关闭使用下面代码：
    abupy.alpha.pick_time_worker.g_natural_long_task = False
"""
g_natural_long_task = True


# noinspection PyAttributeOutsideInit
class AbuPickTimeWorker(AbuPickTimeWorkBase):
    """择时类"""

    def __init__(self, cap, kl_pd, benchmark, buy_factors, sell_factors):
        """
        :param cap: 资金类AbuCapital实例化对象
        :param kl_pd: 择时时间段交易数据
        :param benchmark: 交易基准对象，AbuBenchmark实例对象
        :param buy_factors: 买入因子序列，序列中的对象为dict，每一个dict针对一个具体因子
        :param sell_factors: 卖出因子序列，序列中的对象为dict，每一个dict针对一个具体因子
        """
        self.capital = cap
        # 回测阶段kl
        self.kl_pd = kl_pd
        # 合并加上回测之前1年的数据，为了生成特征数据
        self.combine_kl_pd = ABuSymbolPd.combine_pre_kl_pd(self.kl_pd, n_folds=1)
        # 如特别在乎效率性能，打开下面注释的方式，只在g_enable_ml_feature模式下开启, 注释上一行
        # self.combine_kl_pd = ABuSymbolPd.combine_pre_kl_pd(self.kl_pd,
        #                                                    n_folds=1) if ABuEnv.g_enable_ml_feature else None
        # 传递给因子系列，因子内部可有选择性使用
        self.benchmark = benchmark
        # 初始化买入因子列表
        self.init_buy_factors(buy_factors)
        # 初始化卖出因子列表
        self.init_sell_factors(sell_factors)
        # 根据因子是否支持周，月任务属性，筛选周月任务因子对象列表，在初始化时做，提高时间驱动效率
        self.filter_long_task_factors()
        # 择时最终买入卖出行为列表，列表中每一个对象都为AbuOrder对象
        self.orders = list()
        # 择时进度条，默认空, 即不打开，不显示择时进度
        self.task_pg = None

    def __str__(self):
        """打印对象显示：买入因子列表＋卖出因子列表"""
        return 'buy_factors:{}\nsell_factors:{}'.format(self.buy_factors, self.sell_factors)

    __repr__ = __str__

    def enable_task_pg(self):
        """启动择时内部任务进度条"""
        if self.kl_pd is not None and hasattr(self.kl_pd, 'name') and len(self.kl_pd) > 120:
            self.task_pg = AbuMulPidProgress(len(self.kl_pd), 'pick {} times'.format(self.kl_pd.name))
            self.task_pg.init_ui_progress()
            self.task_pg.display_step = 42

    def _week_task(self, today):
        """
        周任务：使用self.week_buy_factors，self.week_sell_factors进行迭代
        不需再使用hasattr进行是否支持判断fit_week
        """
        # 优先执行买入择时因子专属卖出择时因子，而且即使买入因子被锁但附属于买入因子的卖出不能锁
        self._task_attached_sell(today, how='week')

        # 周任务中不建议生成买单，执行卖单，全部在日任务完成，如需判断通过today.exec_week，today.exec_month
        for sell_factor in self.week_sell_factors:
            sell_factor.fit_week(today, self.orders)

        # 执行买入择时因子专属选股因子，决策是否封锁择时买入因子，注意需要从self.buy_factors遍历不是self.week_buy_factors
        self._task_attached_ps(today, is_week=True)

        for buy_factor in self.week_buy_factors:
            if not buy_factor.lock_factor:
                # 如果买入因子没有被封锁执行任务
                buy_factor.fit_week(today)

    def _month_task(self, today):
        """
        月任务：使用self.month_buy_factors，self.month_sell_factors进行迭代
        不需再使用hasattr进行是否支持判断fit_month
        """
        # 优先执行买入择时因子专属卖出择时因子，而且即使买入因子被锁但附属于买入因子的卖出不能锁
        self._task_attached_sell(today, how='month')

        # 月任务中不建议生成买单，执行卖单，全部在日任务完成，如需判断通过today.exec_week，today.exec_month
        for sell_factor in self.month_sell_factors:
            sell_factor.fit_month(today, self.orders)

        # 执行择时因子专属选股因子，决策是否封锁择时买入因子，注意需要从self.buy_factors遍历不是self.month_buy_factors
        self._task_attached_ps(today, is_week=False)

        # 执行带有fit_month的择时买入因子
        for buy_factor in self.month_buy_factors:
            if not buy_factor.lock_factor:
                # 如果买入因子没有被封锁执行任务
                buy_factor.fit_month(today)

    def _day_task(self, today):
        """
        日任务：迭代买入卖出因子序列进行择时
        :param today: 今日的交易数据
        :return:
        """
        # 优先执行买入择时因子专属卖出择时因子，不受买入因子是否被锁的影响
        self._task_attached_sell(today, how='day')

        # 注意回测模式下始终非高频，非当日买卖，不区分美股，A股市场，卖出因子要先于买入因子的执行
        for sell_factor in self.sell_factors:
            # 迭代卖出因子，每个卖出因子针对今日交易数据，已经所以交易单进行择时
            sell_factor.read_fit_day(today, self.orders)

        # 买入因子行为要在卖出因子下面，否则为高频日交易模式
        for buy_factor in self.buy_factors:
            # 如果择时买入因子没有被封锁执行任务
            if not buy_factor.lock_factor:
                # 迭代买入因子，每个因子都对今天进行择时，如果生成order加入self.orders
                order = buy_factor.read_fit_day(today)
                if order and order.order_deal:
                    self.orders.append(order)

    def _task_attached_sell(self, today, how):
        """专属择时买入因子的择时卖出因子任务：日任务择时卖出因子， 周任务择时卖出因子，月任务择时卖出因子"""

        for buy_factor in self.buy_factors:
            # 筛选出当前买入因子所对应的所有单子, 注意这里使用buy_factor_class不是buy_factor，buy_factor带参数做为唯一标示
            factor_orders = list(filter(lambda order: order.buy_factor_class == buy_factor.__class__.__name__,
                                        self.orders))
            if len(factor_orders) == 0:
                # 当前因子没有对应单子
                continue

            # TODO 不要使用字符串进行eq比对
            for sell_factor in buy_factor.sell_factors:
                if how == 'day':
                    # 所有日任务都要用read_fit_day，且一定存在
                    sell_factor.read_fit_day(today, factor_orders)
                elif how == 'week' and hasattr(sell_factor, 'fit_week'):
                    # 周任务，可选择
                    sell_factor.fit_week(today, factor_orders)
                elif how == 'month'and hasattr(sell_factor, 'fit_month'):
                    # 月任务，可选择
                    sell_factor.fit_month(today, factor_orders)

    def _task_attached_ps(self, today, is_week):
        """专属择时买入因子的选股因子任务：周任务选股因子，月任务选股因子"""
        for buy_factor in self.buy_factors:
            # 不能使用today.exec_week或者today.exec_month来判定，因为有可能都时true
            buy_factor.fit_ps_week(today) if is_week else buy_factor.fit_ps_month(today)

    def _task_loop(self, today):
        """
        开始时间驱动，进行日任务，周任务，月任务，
        如果使用自然周，就会在每个周五进行择时操作
        自然月在每个月末最后一天进行择时，否则就以
        天数作为触发条件，这个时候定性任务本身的性质
        只是以时间跨度作为阀值，触发条件
        :param today: 对self.kl_pd apply操作，且axis＝1结果为一天的交易数据
        :return:
        """
        if self.task_pg is not None:
            self.task_pg.show()

        day_cnt = today.key
        # 判断是否执行周任务, 返回结果赋予today对象
        today.exec_week = today.week_task == 1 if g_natural_long_task else day_cnt % 5 == 0
        # 判断是否执行月任务, 返回结果赋予today对象
        today.exec_month = today.month_task == 1 if g_natural_long_task else day_cnt % 20 == 0

        if day_cnt == 0 and not today.exec_week:
            # 如果是择时第一天，且没有执行周任务，需要初始化买入因子专属周任务选股池子
            self._task_attached_ps(today, is_week=True)
        if day_cnt == 0 and not today.exec_month:
            # 如果是择时第一天，且没有执行月任务，需要初始化买入因子专属月任务选股池子
            self._task_attached_ps(today, is_week=False)

        if today.exec_month:
            # 执行因子月任务
            self._month_task(today)
        if today.exec_week:
            # 执行因子周任务
            self._week_task(today)
        # 执行择时因子日任务
        self._day_task(today)

    # noinspection PyTypeChecker
    def fit(self, *args, **kwargs):
        """
            根据交易数据，因子等输入数据，拟合择时
        """
        if g_natural_long_task:
            """如果要进行自然周，自然月择时任务，需要在kl_pd中添加自然周，自然月标记"""
            # 自然周: 每个周五进行标记
            self.kl_pd['week_task'] = np.where(self.kl_pd.date_week == 4, 1, 0)
            """
                自然月: 即前后两个日期，相互减，得到的数 > 60 必然为月末，20140801 - 20140731
                没有使用时间api，因为这样做运行效率快
                self.kl_pd.shift(-1)['date'] - self.kl_pd['date']
                ->
                >>>>
                2014-07-28     1.0
                2014-07-29     1.0
                2014-07-30     1.0
                2014-07-31    70.0
                2014-08-01     3.0
                2014-08-04     1.0
                2014-08-05     1.0
                >>>
                2014-08-22     3.0
                2014-08-25     1.0
                2014-08-26     1.0
                2014-08-27     1.0
                2014-08-28     1.0
                2014-08-29    73.0
                2014-09-02     1.0
                2014-09-03     1.0
                >>>>
            """
            self.kl_pd['month_task'] = np.where(self.kl_pd.shift(-1)['date'] - self.kl_pd['date'] > 60, 1, 0)
        # 通过pandas apply进行交易日递进择时
        self.kl_pd.apply(self._task_loop, axis=1)

        if self.task_pg is not None:
            self.task_pg.close_ui_progress()

    def init_sell_factors(self, sell_factors):
        """
        通过sell_factors实例化各个卖出因子
        :param sell_factors: list中元素为dict，每个dict为因子的构造元素，如class，构造参数等
        :return:
        """
        self.sell_factors = list()

        if sell_factors is None:
            return

        for factor_class in sell_factors:
            if factor_class is None:
                continue
            if 'class' not in factor_class:
                # 必须要有需要实例化的类信息
                raise ValueError('factor class key must name class !!!')

            factor_class_cp = copy.deepcopy(factor_class)
            # pop出类信息后剩下的都为类需要的参数
            class_fac = factor_class_cp.pop('class')
            # 整合capital，kl_pd等实例化因子对象
            factor = class_fac(self.capital, self.kl_pd, self.combine_kl_pd, self.benchmark, **factor_class_cp)

            if not isinstance(factor, AbuFactorSellBase):
                # 因子对象类型检测
                raise TypeError('factor must base AbuFactorSellBase')
            # 添加到卖出因子序列
            self.sell_factors.append(factor)

    def init_buy_factors(self, buy_factors):
        """
        通过buy_factors实例化各个买入因子
        :param buy_factors: list中元素为dict，每个dict为因子的构造元素，如class，构造参数等
        :return:
        """
        self.buy_factors = list()

        if buy_factors is None:
            return

        for factor_class in buy_factors:
            if factor_class is None:
                continue

            if 'class' not in factor_class:
                # 必须要有需要实例化的类信息
                raise ValueError('factor class key must name class !!!')

            factor_class_cp = copy.deepcopy(factor_class)
            # pop出类信息后剩下的都为类需要的参数
            class_fac = factor_class_cp.pop('class')
            # 整合capital，kl_pd等实例化因子对象
            factor = class_fac(self.capital, self.kl_pd, self.combine_kl_pd, self.benchmark, **factor_class_cp)

            if not isinstance(factor, AbuFactorBuyBase):
                # 因子对象类型检测
                raise TypeError('factor must base AbuFactorBuyBase')
            # 添加到买入因子序列
            self.buy_factors.append(factor)

    def filter_long_task_factors(self):
        """
        根据每一个因子是否有fit_week筛选周任务因子
        根据每一个因子是否有fit_month筛选月任务因子
        在初始化时完成筛选工作，避免在时间序列中迭代
        不断的进行hasattr判断是否支持
        """
        self.week_buy_factors = list(filter(lambda buy_factor: hasattr(buy_factor, 'fit_week'),
                                            self.buy_factors))
        self.month_buy_factors = list(filter(lambda buy_factor: hasattr(buy_factor, 'fit_month'),
                                             self.buy_factors))

        self.week_sell_factors = list(filter(lambda sell_factor: hasattr(sell_factor, 'fit_week'),
                                             self.sell_factors))
        self.month_sell_factors = list(filter(lambda sell_factor: hasattr(sell_factor, 'fit_month'),
                                              self.sell_factors))
