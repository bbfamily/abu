# -*- encoding:utf-8 -*-
"""
    买入择时示例因子：突破买入择时因子
"""

from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

from .ABuFactorBuyBase import AbuFactorBuyBase, BuyCallMixin
from .ABuFactorBuyBreak import AbuFactorBuyBreak
from ..TLineBu.ABuTL import AbuTLine
from ..FactorBuyBu.ABuBuyFactorWrap import AbuLeastPolyWrap

__author__ = '阿布'
__weixin__ = 'abu_quant'


# noinspection PyAttributeOutsideInit
class AbuSDBreak(AbuFactorBuyBase, BuyCallMixin):
    """示例买入因子： 在AbuFactorBuyBreak基础上进行降低交易频率，提高系统的稳定性处理"""

    def _init_self(self, **kwargs):
        # 外部可以设置poly阀值，self.poly在fit_month中和每一个月大盘计算的poly比较，若是大盘的poly大于poly认为走势震荡
        self.poly = kwargs.pop('poly', 2)
        # 是否封锁买入策略进行择时交易
        self.lock = False

        # 下面的代码和AbuFactorBuyBase的实现一摸一样
        self.xd = kwargs['xd']
        self.skip_days = 0
        self.factor_name = '{}:{}'.format(self.__class__.__name__, self.xd)

    def fit_month(self, today):
        # fit_month即在回测策略中每一个月执行一次的方法
        # 策略中拥有self.benchmark，即交易基准对象，AbuBenchmark实例对象，benchmark.kl_pd即对应的市场大盘走势
        benchmark_df = self.benchmark.kl_pd
        # 拿出大盘的今天
        benchmark_today = benchmark_df[benchmark_df.date == today.date]
        if benchmark_today.empty:
            return 0
        # 要拿大盘最近一个月的走势，准备切片的start，end
        end_key = int(benchmark_today.ix[0].key)
        start_key = end_key - 20
        if start_key < 0:
            return 0

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

        # 下面的代码和AbuFactorBuyBase的实现一摸一样
        day_ind = int(today.key)
        if day_ind < self.xd - 1 or day_ind >= self.kl_pd.shape[0] - 1:
            return None
        if self.skip_days > 0:
            self.skip_days -= 1
            return None
        if today.close == self.kl_pd.close[day_ind - self.xd + 1:day_ind + 1].max():
            self.skip_days = self.xd
            return self.make_buy_order(day_ind)
        return None


@AbuLeastPolyWrap()
class AbuTwoDayBuy(AbuFactorBuyBase, BuyCallMixin):
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
        # key是金融时间序列索引
        day_ind = int(today.key)
        # 忽略不符合买入的天（统计周期内前第1天及最后一天）
        if day_ind == 0 or day_ind >= self.kl_pd.shape[0] - 1:
            return None

        # 今天的涨幅
        td_change = today.p_change
        # 昨天的涨幅
        yd_change = self.kl_pd.ix[day_ind - 1].p_change

        if td_change > 0 and 0 < yd_change < td_change:
            # 连续涨两天, 且今天的涨幅比昨天还高 －>买入
            return self.make_buy_order(day_ind)
        return None


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
        # 突破参数 xd， 比如20，30，40天...突破, 不要使用kwargs.pop('xd', 20), 明确需要参数xq
        self.xd = kwargs['xd']
        # 忽略连续创新高，比如买入后第二天又突破新高，忽略
        self.skip_days = 0
        # 在输出生成的orders_pd中显示的名字
        self.factor_name = '{}:{}'.format(self.__class__.__name__, self.xd)

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
