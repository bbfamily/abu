# -*- encoding:utf-8 -*-
"""
    示例买入择时因子
"""

from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import numpy as np
import pandas as pd

from .ABuFactorBuyBase import AbuFactorBuyBase, AbuFactorBuyXD, AbuFactorBuyTD, BuyCallMixin
from .ABuFactorBuyBreak import AbuFactorBuyBreak
from ..TLineBu.ABuTL import AbuTLine
from ..FactorBuyBu.ABuBuyFactorWrap import AbuLeastPolyWrap

__author__ = '阿布'
__weixin__ = 'abu_quant'


# noinspection PyAttributeOutsideInit
class AbuSDBreak(AbuFactorBuyXD, BuyCallMixin):
    """示例买入因子： 在AbuFactorBuyBreak基础上进行降低交易频率，提高系统的稳定性处理"""

    def _init_self(self, **kwargs):
        """
        :param kwargs: kwargs可选参数poly值，poly在fit_month中和每一个月大盘计算的poly比较，
        若是大盘的poly大于poly认为走势震荡，poly默认为2
        """
        super(AbuSDBreak, self)._init_self(**kwargs)
        # poly阀值，self.poly在fit_month中和每一个月大盘计算的poly比较，若是大盘的poly大于poly认为走势震荡
        self.poly = kwargs.pop('poly', 2)
        # 是否封锁买入策略进行择时交易
        self.lock = False

    def fit_month(self, today):
        # fit_month即在回测策略中每一个月执行一次的方法
        # 策略中拥有self.benchmark，即交易基准对象，AbuBenchmark实例对象，benchmark.kl_pd即对应的市场大盘走势
        benchmark_df = self.benchmark.kl_pd
        # 拿出大盘的今天
        benchmark_today = benchmark_df[benchmark_df.date == today.date]
        if benchmark_today.empty:
            return 0
        # 要拿大盘最近一个月的走势，准备切片的start，end
        end_key = int(benchmark_today.iloc[0].key)
        start_key = end_key - 20
        if start_key < 0:
            return False

        # 使用切片切出从今天开始向前20天的数据
        benchmark_month = benchmark_df[start_key:end_key + 1]
        # 通过大盘最近一个月的收盘价格做为参数构造AbuTLine对象
        benchmark_month_line = AbuTLine(benchmark_month.close, 'benchmark month line')
        # 计算这个月最少需要几次拟合才能代表走势曲线
        least = benchmark_month_line.show_least_valid_poly(show=False)

        if least >= self.poly:
            # 如果最少的拟合次数大于阀值self.poly，说明走势成立，大盘非震荡走势，解锁交易
            self.lock = False
        else:
            # 如果最少的拟合次数小于阀值self.poly，说明大盘处于震荡走势，封锁策略进行交易
            self.lock = True

    def fit_day(self, today):
        if self.lock:
            # 如果封锁策略进行交易的情况下，策略不进行择时
            return None

        # 今天的收盘价格达到xd天内最高价格则符合买入条件
        if today.close == self.xd_kl.close.max():
            return self.buy_tomorrow()


@AbuLeastPolyWrap()
class AbuTwoDayBuy(AbuFactorBuyTD, BuyCallMixin):
    """示例AbuLeastPolyWrap，混入BuyCallMixin，即向上突破触发买入event"""

    def _init_self(self, **kwargs):
        """简单示例什么都不编写了"""
        pass

    def fit_day(self, today):
        """
        针对每一个交易日拟合买入交易策略，今天涨，昨天涨就买
        :param today: 当前驱动的交易日金融时间序列数据
        :return:
        """
        # 今天的涨幅
        td_change = today.p_change
        # 昨天的涨幅
        yd_change = self.yesterday.p_change

        if td_change > 0 and 0 < yd_change < td_change:
            # 连续涨两天, 且今天的涨幅比昨天还高 －>买入, 用到了今天的涨幅，只能明天买
            return self.buy_tomorrow()
        return None


class AbuWeekMonthBuy(AbuFactorBuyBase, BuyCallMixin):
    """策略示例每周买入一次或者每一个月买入一次"""

    def _init_self(self, **kwargs):
        """kwargs可选参数：is_buy_month，bool默认True一个月买入一次, False一周买入一次"""
        self.is_buy_month = kwargs.pop('is_buy_month', True)

    def fit_day(self, today):
        """
        :param today: 当前驱动的交易日金融时间序列数据
        """
        if self.is_buy_month and today.exec_month or not self.is_buy_month and today.exec_week:
            # 没有用到今天的任何数据，直接今天买入
            return self.buy_today()


class AbuFactorBuyBreakUmpDemo(AbuFactorBuyBreak):
    """示例组织裁判进行更复杂的综合裁决 扩展AbuFactorBuyBreak组织裁判进行更复杂的综合裁决"""

    def make_ump_block_decision(self, ml_feature_dict):
        """
        进行裁判之间的拦截配合, 简单示例，只要加起来大于2个就算配合成功，拦截
        :param ml_feature_dict: 需要决策的当前买入时刻交易特征dict
        :return: bool, 对ml_feature_dict所描述的交易特征是否进行拦截
        """

        ump = self.ump_manger
        # 统计角度主裁对应这次交易命中的分类簇个数
        deg_hit_cnt = ump.ump_main_deg.predict_hit_kwargs(**ml_feature_dict)
        # 统计跳空主裁对应这次交易命中的分类簇个数
        jump_hit_cnt = ump.ump_main_jump.predict_hit_kwargs(**ml_feature_dict)
        # 统计波动主裁对应这次交易命中的分类簇个数
        wave_hit_cnt = ump.ump_main_wave.predict_hit_kwargs(**ml_feature_dict)
        # 统计价格主裁对应这次交易命中的分类簇个数
        price_hit_cnt = ump.ump_main_price.predict_hit_kwargs(**ml_feature_dict)
        # 进行裁判之间的拦截配合, 简单示例，只要加起来大于2个就算配合成功，拦截
        if deg_hit_cnt + jump_hit_cnt + wave_hit_cnt + price_hit_cnt > 2:
            return True
        return False


class AbuFactorBuyBreakReocrdHitDemo(AbuFactorBuyBreak):
    """示例让裁判自己学习怎么配合，自己做出最正确的判断"""

    def make_ump_block_decision(self, ml_feature_dict):
        """
        即是可以再次根据裁判之间的配合数据进行训练学习，让裁判自己学习怎么配合，自己做出最正确的判断，
        而不是像上面的示例使用固定值3来做为裁决阀值，AbuFactorBuyBreakReocrdHitDemo类似
        AbuFactorBuyBreakUmpDemo但是不对交易进行决策，只是把每一个裁判的对应交易命中的分类簇个数进行记录，更新在特征数据里
        :param ml_feature_dict: 需要决策的当前买入时刻交易特征dict
        :return: bool, 对ml_feature_dict所描述的交易特征是否进行拦截
        """
        ump = self.ump_manger
        # 统计角度主裁对应这次交易命中的分类簇个数
        deg_hit_cnt = ump.ump_main_deg.predict_hit_kwargs(**ml_feature_dict)
        # 统计跳空主裁对应这次交易命中的分类簇个数
        jump_hit_cnt = ump.ump_main_jump.predict_hit_kwargs(**ml_feature_dict)
        # 统计波动主裁对应这次交易命中的分类簇个数
        wave_hit_cnt = ump.ump_main_wave.predict_hit_kwargs(**ml_feature_dict)
        # 统计价格主裁对应这次交易命中的分类簇个数
        price_hit_cnt = ump.ump_main_price.predict_hit_kwargs(**ml_feature_dict)

        ml_feature_dict.update({'deg_hit_cnt': deg_hit_cnt, 'jump_hit_cnt': jump_hit_cnt,
                                'wave_hit_cnt': wave_hit_cnt, 'price_hit_cnt': price_hit_cnt})

        return False


class AbuFactorBuyBreakHitPredictDemo(AbuFactorBuyBreak):
    """
        继续继承AbuFactorBuyBreak复写make_ump_block_decision，
        区别是使用AbuFactorBuyBreakReocrdHitDemo的学习成果hit_ml
        对几个裁判这次交易命中的分类簇个数组成矢量特征进行predict，
        拦截预测结果为-1的交易
    """

    def _init_self(self, **kwargs):
        """
            与AbuFactorBuyBreak基本相同，唯一区别是关键子参数中添加了通过AbuFactorBuyBreakUmpDemo记录训练好的决策器
            self.hit_ml = kwargs['hit_ml']
        """
        super(AbuFactorBuyBreakHitPredictDemo, self)._init_self(**kwargs)
        # 添加了通过AbuFactorBuyBreakUmpDemo记录训练好的决策器
        self.hit_ml = kwargs['hit_ml']

    def make_ump_block_decision(self, ml_feature_dict):
        """
        用回测的数据进行训练后再次反过来指导回测，结果是没有意义的，
        这里的示例只是为了容易理解什么叫做：让裁判自己学习怎么配合，
        自己做出最正确的判断，更详细完整的示例会在之后的章节中示例讲解，
        请关注公众号的更新提醒
        :param ml_feature_dict: 需要决策的当前买入时刻交易特征dict
        :return: bool, 对ml_feature_dict所描述的交易特征是否进行拦截
        """
        ump = self.ump_manger
        # 统计角度主裁对应这次交易命中的分类簇个数
        deg_hit_cnt = ump.ump_main_deg.predict_hit_kwargs(**ml_feature_dict)
        # 统计跳空主裁对应这次交易命中的分类簇个数
        jump_hit_cnt = ump.ump_main_jump.predict_hit_kwargs(**ml_feature_dict)
        # 统计波动主裁对应这次交易命中的分类簇个数
        wave_hit_cnt = ump.ump_main_wave.predict_hit_kwargs(**ml_feature_dict)
        # 统计价格主裁对应这次交易命中的分类簇个数
        price_hit_cnt = ump.ump_main_price.predict_hit_kwargs(**ml_feature_dict)

        result = self.hit_ml.predict([deg_hit_cnt, jump_hit_cnt, wave_hit_cnt, price_hit_cnt])[0]
        if result == -1:
            return True
        return False


class AbuBTCDayBuy(AbuFactorBuyBase, BuyCallMixin):
    """
        比特币日交易策略：

        1. 买入条件1: 当日这100个股票60%以上都是上涨的
        2. 买入条件2: 使用在第12节：机器学习与比特币示例中编写的：信号发出今天比特币会有大行情
    """

    def _init_self(self, **kwargs):
        from ..MarketBu import ABuSymbolPd

        # 市场中与btc最相关的top个股票
        self.btc_similar_top = kwargs.pop('btc_similar_top')
        # 超过多少个相关股票今天趋势相同就买入
        self.btc_vote_val = kwargs.pop('btc_vote_val', 0.60)

        def _collect_kl(sim_line):
            """在初始化中将所有相关股票的对应时间的k线数据进行收集"""
            start = self.kl_pd.iloc[0].date
            end = self.kl_pd.iloc[-1].date
            kl = ABuSymbolPd.make_kl_df(sim_line.symbol, start=start, end=end)
            self.kl_dict[sim_line.symbol] = kl

        self.kl_dict = {}
        # k线数据进行收集到类字典对象self.kl_dict中
        self.btc_similar_top.apply(_collect_kl, axis=1)

    # noinspection PyUnresolvedReferences
    def fit_day(self, today):
        """
        :param today: 当前驱动的交易日金融时间序列数据
        """
        # 忽略不符合买入的天（统计周期内前两天, 因为btc的机器学习特证需要三天交易数据）
        if self.today_ind < 2:
            return None

        # 今天，昨天，前天三天的交易数据进行特证转换
        btc = self.kl_pd[self.today_ind - 2:self.today_ind + 1]
        # 三天的交易数据进行转换后得到btc_today_x
        btc_today_x = self.make_btc_today(btc)

        # btc_ml并没有在这里传入，实际如果要使用，需要对外部的btc_ml进行本地序列化后，构造读取本地
        # 买入条件2: 使用在第12节：机器学习与比特币示例中编写的：信号发出今天比特币会有大行情
        if btc_ml.predict(btc_today_x):
            # 买入条件1: 当日这100个股票60%以上都是上涨的
            vote_val = self.similar_predict(today.date)
            if vote_val > self.btc_vote_val:
                # 没有使用当天交易日的close等数据，且btc_ml判断的大波动是当日，所以当日买入
                return self.buy_today()

    # noinspection PyUnresolvedReferences
    def make_btc_today(self, sib_btc):
        """构造比特币三天数据特证"""
        from ..UtilBu import ABuScalerUtil

        sib_btc['big_wave'] = (sib_btc.high - sib_btc.low) / sib_btc.pre_close > 0.55
        sib_btc['big_wave'] = sib_btc['big_wave'].astype(int)
        sib_btc_scale = ABuScalerUtil.scaler_std(
            sib_btc.filter(['open', 'close', 'high', 'low', 'volume', 'pre_close',
                            'ma5', 'ma10', 'ma21', 'ma60', 'atr21', 'atr14']))
        # 把标准化后的和big_wave，date_week连接起来
        sib_btc_scale = pd.concat([sib_btc['big_wave'], sib_btc_scale, sib_btc['date_week']], axis=1)

        # 抽取第一天，第二天的大多数特征分别改名字以one，two为特征前缀，如：one_open，one_close，two_ma5，two_high.....
        a0 = sib_btc_scale.iloc[0].filter(['open', 'close', 'high', 'low', 'volume', 'pre_close',
                                           'ma5', 'ma10', 'ma21', 'ma60', 'atr21', 'atr14', 'date_week'])
        a0.rename(index={'open': 'one_open', 'close': 'one_close', 'high': 'one_high', 'low': 'one_low',
                         'volume': 'one_volume', 'pre_close': 'one_pre_close',
                         'ma5': 'one_ma5', 'ma10': 'one_ma10', 'ma21': 'one_ma21',
                         'ma60': 'one_ma60', 'atr21': 'one_atr21', 'atr14': 'one_atr14',
                         'date_week': 'one_date_week'}, inplace=True)

        a1 = sib_btc_scale.iloc[1].filter(['open', 'close', 'high', 'low', 'volume', 'pre_close',
                                           'ma5', 'ma10', 'ma21', 'ma60', 'atr21', 'atr14', 'date_week'])
        a1.rename(index={'open': 'two_open', 'close': 'two_close', 'high': 'two_high', 'low': 'two_low',
                         'volume': 'two_volume', 'pre_close': 'two_pre_close',
                         'ma5': 'two_ma5', 'ma10': 'two_ma10', 'ma21': 'two_ma21',
                         'ma60': 'two_ma60', 'atr21': 'two_atr21', 'atr14': 'two_atr14',
                         'date_week': 'two_date_week'}, inplace=True)
        # 第三天的特征只使用'open', 'low', 'pre_close', 'date_week'，该名前缀today，如today_open，today_date_week
        a2 = sib_btc_scale.iloc[2].filter(['big_wave', 'open', 'low', 'pre_close', 'date_week'])
        a2.rename(index={'open': 'today_open', 'low': 'today_low',
                         'pre_close': 'today_pre_close',
                         'date_week': 'today_date_week'}, inplace=True)
        # 将抽取改名字后的特征连接起来组合成为一条新数据，即3天的交易数据特征－>1条新的数据
        btc_today = pd.DataFrame(pd.concat([a0, a1, a2], axis=0)).T

        # 开始将周几进行离散处理
        dummies_week_col = btc_ml.df.filter(regex='(^one_date_week_|^two_date_week_|^today_date_week_)').columns
        dummies_week_df = pd.DataFrame(np.zeros((1, len(dummies_week_col))), columns=dummies_week_col)

        # 手动修改每一天的one hot
        one_day_key = 'one_date_week_{}'.format(btc_today.one_date_week.values[0])
        dummies_week_df[one_day_key] = 1
        two_day_key = 'two_date_week_{}'.format(btc_today.two_date_week.values[0])
        dummies_week_df[two_day_key] = 1
        today_day_key = 'today_date_week_{}'.format(btc_today.today_date_week.values[0])
        dummies_week_df[today_day_key] = 1
        btc_today.drop(['one_date_week', 'two_date_week', 'today_date_week'], inplace=True, axis=1)
        btc_today = pd.concat([btc_today, dummies_week_df], axis=1)
        return btc_today.as_matrix()[:, 1:]

    def similar_predict(self, today_date):
        """与比特币在市场中最相关的top100个股票已各自今天的涨跌结果进行投票"""

        def _predict_vote(sim_line, _today_date):
            kl = self.kl_dict[sim_line.symbol]
            if kl is None:
                return -1 * sim_line.vote_direction > 0
            kl_today = kl[kl.date == _today_date]
            if kl_today is None or kl_today.empty:
                return -1 * sim_line.vote_direction > 0
            # 需要 * sim_line.vote_direction，因为负相关的存在
            return kl_today.p_change.values[0] * sim_line.vote_direction > 0

        vote_result = self.btc_similar_top.apply(_predict_vote, axis=1, args={today_date, })
        # 将所有投票结果进行统计，得到与比特币最相关的这top100个股票的今天投票结果
        vote_val = 1 - vote_result.value_counts()[False] / vote_result.value_counts().sum()
        return vote_val
