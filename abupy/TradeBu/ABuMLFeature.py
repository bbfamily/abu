# -*- encoding:utf-8 -*-
"""
    内置特征定义，以及用户特征扩展，定义模块
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import ast
import datetime
import os

import numpy as np

from ..CoreBu import ABuEnv
# noinspection PyUnresolvedReferences
from ..CoreBu.ABuFixes import xrange, range, six
from ..MarketBu import ABuMarketDrawing
from ..TLineBu import ABuTLAtr
from ..TLineBu import ABuTLJump
from ..TLineBu import ABuTLWave
from ..UtilBu import ABuRegUtil
from ..UtilBu import ABuStrUtil

__author__ = '阿布'
__weixin__ = 'abu_quant'

# 内置特征，趋势角度
g_deg_keys = [21, 42, 60, ABuEnv.g_market_trade_year]
# 内置特征，价格rank
g_price_rank_keys = [60, 90, 120, ABuEnv.g_market_trade_year]
# 内置特征，波动周期定义
g_wave_xd = 42
# 内置特征，波动取样个数
g_wave_key_cnt = 3
# 内置特征，atr周期定义
g_atr_xd = 42
# 快照周期
g_take_snap_shot_xd = 60


class BuyFeatureMixin(object):
    """
        买入特征标识混入，与BuyUmpMixin不同，具体feature类可能属于多个类别
        即可能同时混入BuyFeatureMixin和SellFeatureMixin
    """
    _feature_buy = True
    _feature_buy_prefix = 'buy_'


class SellFeatureMixin(object):
    """
        卖出特征标识混入，与SellUmpMixin不同，具体feature类可能属于多个类别
        即可能同时混入BuyFeatureMixin和SellFeatureMixin
    """
    _feature_sell = True
    _feature_sell_prefix = 'sell_'


class AbuFeatureBase(object):
    """特征构造基类"""

    def support_buy_feature(self):
        """是否支持买入特征构建"""
        return getattr(self, "_feature_buy", False) is True

    def support_sell_feature(self):
        """是否支持卖出特征构建"""
        return getattr(self, "_feature_sell", False) is True

    def check_support(self, buy_feature):
        """
        根据参数buy_feature检测是否支持特征构建
        :param buy_feature: 是否是买入特征构造（bool）
        """
        if buy_feature and not self.support_buy_feature:
            raise TypeError('feature support buy must subclass BuyFeatureMixin!!!')
        if not buy_feature and not self.support_sell_feature:
            raise TypeError('feature support buy must subclass SellFeatureMixin!!!')

    def feature_prefix(self, buy_feature, check=True):
        """
        根据buy_feature决定返回_feature_buy_prefix或者_feature_sell_prefix，目的是在calc_feature中构成唯一key
        :param buy_feature: 是否是买入特征构造（bool）
        :param check: 是否需要检测是否支持特征构建
        :return:
        """
        if check:
            self.check_support(buy_feature)
        return getattr(self, '_feature_buy_prefix') if buy_feature else getattr(self, '_feature_sell_prefix')

    def __str__(self):
        """打印对象显示：class name, support_buy_feature support_sell_feature, get_feature_keys"""
        return '{}:is_buy_feature:{} is_sell_feature:{} feature: {}'.format(self.__class__.__name__,
                                                                            self.support_buy_feature(),
                                                                            self.support_sell_feature(),
                                                                            self.get_feature_keys(
                                                                                self.support_buy_feature()))

    __repr__ = __str__

    def get_feature_ump_keys(self, ump_cls):
        """
        根据ump_cls，返回对应的get_feature_keys
        :param ump_cls: AbuUmpEdgeBase子类，参数为类，非实例对象
        :return: 键值对字典中的key序列
        """
        is_buy_ump = getattr(ump_cls, "_ump_type_prefix") == 'buy_'
        return self.get_feature_keys(buy_feature=is_buy_ump)

    def get_feature_keys(self, buy_feature):
        """
        子类主要需要实现的函数，定义feature的列名称
        :param buy_feature: 是否是买入特征构造（bool）
        :return: 键值对字典中的key序列
        """
        raise NotImplementedError('NotImplementedError get_feature_keys!!!')

    def calc_feature(self, kl_pd, combine_kl_pd, day_ind, buy_feature):
        """
        子类主要需要实现的函数，根据买入或者卖出时的金融时间序列，以及交易日信息构造特征
        :param kl_pd: 择时阶段金融时间序列
        :param combine_kl_pd: 合并择时阶段之前1年的金融时间序列
        :param day_ind: 交易发生的时间索引，即对应self.kl_pd.key
        :param buy_feature: 是否是买入特征构造（bool）
        :return: 构造特征的键值对字典
        """
        raise NotImplementedError('NotImplementedError calc_feature!!!')


class AbuFeatureDeg(AbuFeatureBase, BuyFeatureMixin, SellFeatureMixin):
    """角度特征，支持买入，卖出"""

    def __init__(self):
        """
            默认21, 42, 60, 250日走势角度特征，如外部修改，直接使用类似如下：
                       abupy.feature.g_deg_keys = [10, 20, 30, 40, 50]
        """

        # frozenset包一下，一旦定下来就不能修改，否则特征对不上
        self.deg_keys = frozenset(g_deg_keys)

    def get_feature_keys(self, buy_feature):
        """
        迭代生成所有走势角度特征feature的列名称定, 使用feature_prefix区分买入，卖出前缀key
        :param buy_feature: 是否是买入特征构造（bool）
        :return: 角度特征的键值对字典中的key序列
        """
        return ['{}deg_ang{}'.format(self.feature_prefix(buy_feature=buy_feature), dk) for dk in self.deg_keys]

    def calc_feature(self, kl_pd, combine_kl_pd, day_ind, buy_feature):
        """
        根据买入或者卖出时的金融时间序列，以及交易日信息构造拟合角度特征
        :param kl_pd: 择时阶段金融时间序列
        :param combine_kl_pd: 合并择时阶段之前1年的金融时间序列
        :param day_ind: 交易发生的时间索引，即对应self.kl_pd.key
        :param buy_feature: 是否是买入特征构造（bool）
        :return: 构造角度特征的键值对字典
        """

        # 返回的角度特征键值对字典
        deg_dict = {}
        for dk in self.deg_keys:
            # 迭代预设角度周期，计算构建特征
            if day_ind - dk >= 0:
                # 如果择时时间序列够提取特征，使用kl_pd截取特征交易周期收盘价格
                deg_close = kl_pd[day_ind - dk + 1:day_ind + 1].close
            else:
                # 如果择时时间序列不够提取特征，使用combine_kl_pd截取特征交易周期，首先截取直到day_ind的时间序列
                combine_kl_pd = combine_kl_pd.loc[:kl_pd.index[day_ind]]
                # 如combine_kl_pd长度大于特征周期长度－> 截取combine_kl_pd[-dk:].close，否则取combine_kl_pd所有交易收盘价格
                deg_close = combine_kl_pd[-dk:].close if combine_kl_pd.shape[0] > dk else combine_kl_pd.close

            # 使用截取特征交易周期收盘价格deg_close做为参数，通过calc_regress_deg计算趋势拟合角度
            ang = ABuRegUtil.calc_regress_deg(deg_close, show=False)
            # 标准化拟合角度值
            ang = 0 if np.isnan(ang) else round(ang, 3)
            # 角度特征键值对字典添加拟合角度周期key和对应的拟合角度值
            deg_dict['{}deg_ang{}'.format(self.feature_prefix(buy_feature=buy_feature), dk)] = ang
        return deg_dict


class AbuFeaturePrice(AbuFeatureBase, BuyFeatureMixin, SellFeatureMixin):
    """价格rank特征，支持买入，卖出"""

    def __init__(self):
        """
            默认60, 90, 120, 250日价格rank特征，如外部修改，直接使用类似如下：
                    abupy.feature.g_price_rank_keys = [10, 20, 30, 40，50]
        """

        # frozenset包一下，一旦定下来就不能修改，否则特征对不上
        self.price_rank_keys = frozenset(g_price_rank_keys)

    def get_feature_keys(self, buy_feature):
        """
        迭代生成所有价格rank特征feature的列名称, 使用feature_prefix区分买入，卖出前缀key
        :param buy_feature: 是否是买入特征构造（bool）
        :return: 价格rank特征的键值字典中的key序列
        """
        return ['{}price_rank{}'.format(self.feature_prefix(buy_feature=buy_feature), dk) for dk in
                self.price_rank_keys]

    def calc_feature(self, kl_pd, combine_kl_pd, day_ind, buy_feature):
        """
        根据买入或者卖出时的金融时间序列，以及交易日信息构造价格rank特征
        :param kl_pd: 择时阶段金融时间序列
        :param combine_kl_pd: 合并择时阶段之前1年的金融时间序列
        :param day_ind: 交易发生的时间索引，即对应self.kl_pd.key
        :param buy_feature: 是否是买入特征构造（bool）
        :return: 构造价格rank特征的键值对字典
        """
        # 返回的价格rank特征键值对字典
        price_rank_dict = {}
        for dk in self.price_rank_keys:
            # 迭代预设价格rank周期，计算构建特征
            if day_ind - dk >= 0:
                # 如果择时时间序列够提取特征，使用kl_pd截取特征交易周期收盘价格
                price_close = kl_pd[day_ind - dk + 1:day_ind + 1].close
            else:
                # 如果择时时间序列不够提取特征，使用combine_kl_pd截取特征交易周期，首先截取直到day_ind的时间序列
                combine_kl_pd = combine_kl_pd.loc[:kl_pd.index[day_ind]]
                # 如combine_kl_pd长度大于特征周期长度－> 截取combine_kl_pd[-dk:].close，否则取combine_kl_pd所有交易收盘价格
                price_close = combine_kl_pd[-dk:].close if combine_kl_pd.shape[0] > dk else combine_kl_pd.close
            """
                price_close.rank()的结果是所有价格的排名

                ...................
                2016-07-20    256.0
                2016-07-21    200.0
                2016-07-22    214.0
                2016-07-25    266.0
                2016-07-26    239.0

                -> price_close.rank()[-1]的结果是买入时刻或者卖出时刻的排名

                239.0

                －> price_close.rank()[-1] / price_close.rank().shape[0] 的结果即为：
                买入时刻或者卖出时刻的排名在周期中的排名位置，值由0-1

                eg: price_close.rank()[-1] / price_close.rank().shape[0]
                -> 239.0 / 504 = 0.47420634920634919, 即代表买入或者卖出时价格在特征周期中的位置
            """
            price_rank = price_close.rank()[-1] / price_close.rank().shape[0]
            # 标准化价格rank值
            price_rank = 0 if np.isnan(price_rank) else round(price_rank, 3)
            # 价格rank特征键值对字典添加价格rank周期key和对应的价格rank值
            price_rank_dict['{}price_rank{}'.format(self.feature_prefix(buy_feature=buy_feature), dk)] = price_rank
        return price_rank_dict


class AbuFeatureWave(AbuFeatureBase, BuyFeatureMixin, SellFeatureMixin):
    """波动特征，支持买入，卖出"""

    def __init__(self):
        """
            默认42日做为波动计算周期，如外部修改，直接使用类似如下：
                abupy.feature.g_wave_xd = 21
        """
        self.wave_xd = g_wave_xd
        self.wave_key_cnt = g_wave_key_cnt

    def get_feature_keys(self, buy_feature):
        """
        迭代生成所有波动特征feature的列名称, 使用feature_prefix区分买入，卖出前缀key
        :param buy_feature: 是否是买入特征构造（bool）
        :return: 波动特征的键值对字典中的key序列
        """
        return ['{}wave_score{}'.format(self.feature_prefix(buy_feature=buy_feature),
                                        xd_ind) for xd_ind in list(range(1, self.wave_key_cnt + 1))]

    def calc_feature(self, kl_pd, combine_kl_pd, day_ind, buy_feature):
        """
        根据买入或者卖出时的金融时间序列，以及交易日信息构造波动特征
        :param kl_pd: 择时阶段金融时间序列
        :param combine_kl_pd: 合并择时阶段之前1年的金融时间序列
        :param day_ind: 交易发生的时间索引，即对应self.kl_pd.key
        :param buy_feature: 是否是买入特征构造（bool）
        :return: 构造波动特征的键值对字典
        """

        # 构建波动特征周期固定，即一年的交易日数量
        wave_wide = ABuEnv.g_market_trade_year
        if day_ind - wave_wide >= 0:
            # 如果择时时间序列够提取特征，使用kl_pd截取特征交易周期收盘价格
            wave_df = kl_pd[day_ind - wave_wide + 1:day_ind + 1]
        else:
            # 如果择时时间序列不够提取特征，使用combine_kl_pd截取特征交易周期，首先截取直到day_ind的时间序列
            combine_kl_pd = combine_kl_pd.loc[:kl_pd.index[day_ind]]
            # 如combine_kl_pd长度大于特征周期长度－> 截取combine_kl_pd[-wave_wide:].close，否则取combine_kl_pd所有交易收盘价格
            wave_df = combine_kl_pd[-wave_wide:] if combine_kl_pd.shape[0] > wave_wide else combine_kl_pd

        # 返回的波动特征键值对字典
        wave_dict = {}
        for xd_ind in xrange(1, self.wave_key_cnt + 1):
            # wave_df固定为一年交易时间序列，xd是calc_wave_st内部计算rolling std的window值，详ABuTLWave.calc_wave_std
            wave = ABuTLWave.calc_wave_std(wave_df, xd=xd_ind * self.wave_xd, show=False)
            wave_score = wave.score
            # 标准化波动特征值
            wave_score = 0 if np.isnan(wave_score) else round(wave_score, 3)
            # 波动特征键值对字典添加波动特征key和对应的波动特征值
            wave_dict['{}wave_score{}'.format(self.feature_prefix(buy_feature=buy_feature), xd_ind)] = wave_score
        return wave_dict


class AbuFeatureAtr(AbuFeatureBase, BuyFeatureMixin):
    """atr特征，支持买入"""

    def __init__(self):
        """
            默认42日做为atr特征计算周期，如外部修改，直接使用类似如下：
                abupy.feature.g_atr_xd = 21
        """
        self.atr_xd = g_atr_xd
        self.atr_key = 'atr_std'

    def get_feature_keys(self, buy_feature):
        """
        返回对应的atr特征key值，虽然只有一个固定的atr key值，也返回序列，保持接口统一
        :param buy_feature: 是否是买入特征构造（bool）
        :return: atr特征的键值对字典中的key序列
        """
        return ['{}{}'.format(self.feature_prefix(buy_feature=buy_feature), self.atr_key)]

    def calc_feature(self, kl_pd, combine_kl_pd, day_ind, buy_feature):
        """
        根据买入或者卖出时的金融时间序列，以及交易日信息构造atr特征
        :param kl_pd: 择时阶段金融时间序列
        :param combine_kl_pd: 合并择时阶段之前1年的金融时间序列
        :param day_ind: 交易发生的时间索引，即对应self.kl_pd.key
        :param buy_feature: 是否是买入特征构造（bool）
        :return: 构造atr特征的键值对字典
        """

        # 构建atr特征周期固定，即一年的交易日数量
        atr_wide = ABuEnv.g_market_trade_year
        if day_ind - atr_wide >= 0:
            # 如果择时时间序列够提取特征，使用kl_pd截取特征交易周期收盘价格
            atr_df = kl_pd[day_ind - atr_wide + 1:day_ind + 1]
        else:
            # 如果择时时间序列不够提取特征，使用combine_kl_pd截取特征交易周期，首先截取直到day_ind的时间序列
            combine_kl_pd = combine_kl_pd.loc[:kl_pd.index[day_ind]]
            # 如combine_kl_pd长度大于特征周期长度－> 截取combine_kl_pd[-atr_wide:].close，否则取combine_kl_pd所有交易收盘价格
            atr_df = combine_kl_pd[-atr_wide:] if combine_kl_pd.shape[0] > atr_wide else combine_kl_pd

        # 返回的atr特征键值对字典
        atr_dict = {}
        # 计算atr特征，详见ABuTLAtr.calc_atr_std
        atr_std = ABuTLAtr.calc_atr_std(atr_df, xd=self.atr_xd, show=False)
        atr_score = atr_std.score
        # 标准化atr特征
        atr_score = 0 if np.isnan(atr_score) else round(atr_score, 3)
        # atr特征键值对字典添加atr特征key和对应的atr特征值
        atr_dict['{}{}'.format(self.feature_prefix(buy_feature=buy_feature), self.atr_key)] = atr_score
        return atr_dict


class AbuFeatureJump(AbuFeatureBase, BuyFeatureMixin, SellFeatureMixin):
    """跳空特征，支持买入，卖出"""

    def __init__(self):
        """
            跳空的特征key为:
            jump_down_power: 向下跳空的能量
            diff_down_days : 向下跳空距离买入或卖出的时间间隔
            jump_up_power  : 向上跳空的能量
            diff_up_days   : 向上跳空距离买入或卖出的时间间隔
        """
        self.jump_keys = frozenset(['jump_down_power', 'diff_down_days', 'jump_up_power', 'diff_up_days'])

    def get_feature_keys(self, buy_feature):
        """
        返回对应的跳空特征key值
        :param buy_feature: 是否是买入特征构造（bool）
        :return: 跳空特征的键值对字典中的key序列
        """
        return ['{}{}'.format(self.feature_prefix(buy_feature=buy_feature), jk) for jk in self.jump_keys]

    def calc_feature(self, kl_pd, combine_kl_pd, day_ind, buy_feature):
        """
        根据买入或者卖出时的金融时间序列，以及交易日信息构造跳空特征
        :param kl_pd: 择时阶段金融时间序列
        :param combine_kl_pd: 合并择时阶段之前1年的金融时间序列
        :param day_ind: 交易发生的时间索引，即对应self.kl_pd.key
        :param buy_feature: 是否是买入特征构造（bool）
        :return: 构造跳空特征的键值对字典
        """

        # 构建跳空特征周期固定，即一年的交易日数量
        jump_wide = ABuEnv.g_market_trade_year
        if day_ind - jump_wide >= 0:
            # 如果择时时间序列够提取特征，使用kl_pd截取特征交易周期收盘价格
            jump_df = kl_pd[day_ind - jump_wide + 1:day_ind + 1]
        else:
            # 如果择时时间序列不够提取特征，使用combine_kl_pd截取特征交易周期，首先截取直到day_ind的时间序列
            combine_kl_pd = combine_kl_pd.loc[:kl_pd.index[day_ind]]
            # 如combine_kl_pd长度大于特征周期长度－> 截取combine_kl_pd[-jump_wide:].close，否则取combine_kl_pd所有交易收盘价格
            jump_df = combine_kl_pd[-jump_wide:] if combine_kl_pd.shape[0] > jump_wide else combine_kl_pd

        # 返回的跳空特征键值对字典
        jump_dict = {}

        diff_down_days = 0
        jump_down_power = 0
        diff_up_days = 0
        jump_up_power = 0
        key_prefix = self.feature_prefix(buy_feature=buy_feature)
        # 通过特征周期时间序列jump_df做为参数，计算跳空，返回jumps对象为pd.DataFrame对象，详见ABuTLJump.calc_jump
        jumps = ABuTLJump.calc_jump(jump_df, show=False)
        """
                jump形式如下所示：jump代表跳空方向1为向上，－1为向下，jump_power代表对应的跳空能量，即每一行
                记录了一次跳空发生的时间，价格变化，跳空能量等信息
                            jump	jump_power	close	date	p_change	pre_close
                2014-08-11	 1.0	1.006085	259.32	20140811.0	4.51	248.13
                2014-10-10	-1.0	1.628481	236.91	20141010.0	-7.82	257.01
                2015-01-14	-1.0	1.325337	192.69	20150114.0	-5.66	204.25
                2015-02-12	-1.0	1.422285	202.88	20150212.0	-4.66	212.80
                .............
        """
        if not jumps.empty:
            # 筛选出所有向下跳空的
            down_jumps = jumps[(jumps.jump == -1)]
            if down_jumps.shape[0] > 0:
                # 筛选最后一次向下跳空的情况
                last_down_jump = down_jumps.iloc[-1:]
                # 跳空能力 * last_down_jump.jump.values[0] 转换成有方向的
                jump_down_power = last_down_jump.jump_power.values[0] * last_down_jump.jump.values[0]
                # 向下跳空距离买入或卖出的时间间隔
                diff_down_days = (kl_pd.iloc[day_ind: day_ind + 1].index.date - last_down_jump.index.date)[0].days
            # 筛选出所有向上跳空的
            up_jumps = jumps[(jumps.jump == 1)]
            if up_jumps.shape[0] > 0:
                # 筛选最后一次向上跳空的情况
                last_up_jump = up_jumps.iloc[-1:]
                # 跳空能力* last_up_jump.jump.values[0] 转换成有方向的
                jump_up_power = last_up_jump.jump_power.values[0] * last_up_jump.jump.values[0]
                # 向上跳空距离买入或卖出的时间间隔
                diff_up_days = (kl_pd.iloc[day_ind: day_ind + 1].index.date - last_up_jump.index.date)[0].days

        # 标准化跳空特征特征值
        jump_down_power = round(jump_down_power, 3)
        jump_up_power = round(jump_up_power, 3)

        # 跳空特征键值对字典添加跳空特征key和对应的跳空特征值
        jump_dict['{}jump_down_power'.format(key_prefix)] = jump_down_power
        jump_dict['{}diff_down_days'.format(key_prefix)] = diff_down_days

        jump_dict['{}jump_up_power'.format(key_prefix)] = jump_up_power
        jump_dict['{}diff_up_days'.format(key_prefix)] = diff_up_days

        return jump_dict


class AbuFeatureSnapshot(AbuFeatureBase, BuyFeatureMixin, SellFeatureMixin):
    """
        快照特征，支持买入，卖出 abupy.env.g_enable_take_kl_snapshot开关控制特征是否生成，
        生成的走势图在～/abu/data/save_png/今天的日期/目录下
    """

    def __init__(self):
        """
            默认60日价格做为快照，如外部修改，直接使用类似如下：
                abupy.feature.g_take_snap_shot_xd = 30
        """
        self.take_snap_shot_xd = g_take_snap_shot_xd
        self.snap_key = 'snap'

    def get_feature_keys(self, buy_feature):
        """
        返回对应的快照特征key值
        :param buy_feature: 是否是买入特征构造（bool）
        :return: 快照特征的键值对字典中的key序列
        """
        return ['{}{}'.format(self.feature_prefix(buy_feature=buy_feature), self.snap_key)]

    def calc_feature(self, kl_pd, combine_kl_pd, day_ind, buy_feature):
        """
        根据买入或者卖出时的金融时间序列，以及交易日信息构造快照特征
        :param kl_pd: 择时阶段金融时间序列
        :param combine_kl_pd: 合并择时阶段之前1年的金融时间序列
        :param day_ind: 交易发生的时间索引，即对应self.kl_pd.key
        :param buy_feature: 是否是买入特征构造（bool）
        :return: 构造快照特征的键值对字典
        """

        # 快照生成时间字符串
        tt = datetime.datetime.now().strftime("%Y_%m_%d_%H_%M_%S_%f")
        # 生成64位随机字符串
        rn = ABuStrUtil.create_random_with_alpha(64)
        # 快照文件名确定，由于会使用多任务并行，所以加入进程id，和64位随机数，避免产生文件名冲突
        snap_fn = '{}_{}_{}'.format(tt, os.getpid(), rn)

        if day_ind - self.take_snap_shot_xd >= 0:
            # 如果择时时间序列够提取特征，使用kl_pd截取特征交易周期收盘价格
            snap_window_pd = kl_pd[day_ind - self.take_snap_shot_xd + 1:day_ind + 1]
        else:
            # 如果择时时间序列不够提取特征，使用combine_kl_pd截取特征交易周期，首先截取直到day_ind的时间序列
            combine_kl_pd = combine_kl_pd.loc[:kl_pd.index[day_ind]]
            # 如combine_kl_pd长度大于特征周期长度－> 截取combine_kl_pd[-xd:].close，否则取combine_kl_pd所有交易收盘价格
            snap_window_pd = combine_kl_pd[-self.take_snap_shot_xd:] \
                if combine_kl_pd.shape[0] > self.take_snap_shot_xd else combine_kl_pd

        # 模块设置，绘制k线图只绘制价格曲线，不绘制成交量
        ABuMarketDrawing.g_only_draw_price = True
        # 通过特征周期时间序列snap_window_pd做为参数，绘制交易k线图快照，保存在本地，详ABuMarketDrawing.plot_candle_form_klpd
        ABuMarketDrawing.plot_candle_form_klpd(snap_window_pd, save=True, name=snap_fn)
        # 快照特征键值对字典添加快照特征key和对应的快照特征值
        snap_dict = {'{}{}'.format(self.feature_prefix(buy_feature=buy_feature),
                                   self.snap_key): ABuMarketDrawing.save_dir_name() + snap_fn + '.png'}
        return snap_dict


"""用户可扩展自定义特征"""
_g_extend_feature_list = list()


def append_user_feature(feature, check=True):
    """
    外部设置扩展feature接口
    :param feature: 可以是feature class类型，也可以是实例化后的feature object
    :param check: 是否检测feature是AbuFeatureBase实例
    :return:
    """

    if isinstance(feature, six.class_types):
        # 暂时认为所有feature的实例化不需要参数，如需要也可添加＊args
        feature_obj = feature()
    else:
        # 如果不是类直接赋值
        feature_obj = feature

    # check检测feature_obj是不是AbuFeatureBase的子类实例对象
    if check and not isinstance(feature_obj, AbuFeatureBase):
        raise TypeError('feature must a isinstance AbuFeatureBase!!!')

    # 添加到用户可扩展自定义特征序列中
    _g_extend_feature_list.append(feature_obj)


def clear_user_feature():
    """将用户可扩展自定义特征序列清空"""
    global _g_extend_feature_list
    _g_extend_feature_list = list()


class AbuMlFeature(object):
    """特征对外统一接口类，负责管理构建内部特征，用户扩展特征，提供买入卖出因子的交易特征生成，转换接口"""

    def __init__(self):
        """实例化 内置特征对象＋用户扩展自定义特征对象"""

        # 内置特征实例化
        self.features = [AbuFeatureDeg(), AbuFeaturePrice(), AbuFeatureWave(), AbuFeatureAtr(), AbuFeatureJump()]
        if ABuEnv.g_enable_take_kl_snapshot:
            # 快照特征比较特殊，默认不开启，因为大规模回测比较耗时等开销，故开关控制
            self.features.append(AbuFeatureSnapshot())

        # 用户扩展自定义特征对象extend到特征对象序列self.features中
        if len(_g_extend_feature_list) > 0:
            # 这里不再次check了，因为append_user_feature时已经做过，直接extend
            self.features.extend(_g_extend_feature_list)

    def make_feature_dict(self, kl_pd, combine_kl_pd, day_ind, buy_feature):
        """
        提供买入卖出因子构建交易特征的接口，使用见AbuFactorBuyBase.make_buy_order_ml_feature
        :param kl_pd: 择时阶段金融时间序列
        :param combine_kl_pd: 合并择时阶段之前1年的金融时间序列
        :param day_ind: 交易发生的时间索引，即对应self.kl_pd.key
        :param buy_feature: 是否是买入特征构造（bool）
        """

        if not ABuEnv.g_enable_ml_feature:
            # 需要env中开启特征生成，否则不生成交易特征
            return None
        ml_feature_dict = {}
        # 根据buy_feature，过滤出特征序列中支持的特征序列子集
        features = list(
            filter(lambda f: f.support_buy_feature() if buy_feature else f.support_sell_feature(), self.features))
        for feature in features:
            # 迭代特征序列对象，特征对象统一使用calc_feature接口生成自己的特征，结果特征update到总特征字典ml_feature_dict中
            ml_feature_dict.update(feature.calc_feature(kl_pd, combine_kl_pd, day_ind, buy_feature))
        return ml_feature_dict

    def _get_unzip_feature_keys(self, buy_feature):
        """
        根据buy_feature，过滤出特征序列中支持的特征序列子集，迭代特征序列对象，
        使用特征对象的get_feature_keys方法获取特征的key序列
        :param buy_feature: 是否是买入特征构造（bool）
        """
        feature_key_list = []
        # 根据buy_feature，过滤出特征序列中支持的特征序列子集
        features = list(
            filter(lambda f: f.support_buy_feature() if buy_feature else f.support_sell_feature(), self.features))

        # 迭代特征序列对象，使用特征对象的get_feature_keys方法获取特征的key序列
        for feature in features:
            feature_key_list.extend(feature.get_feature_keys(buy_feature))
        return feature_key_list

    def unzip_ml_feature(self, orders_pd):
        """
        ABuTradeExecute中make_orders_pd使用，将order中dict字典形式的特征解压拆解为独立的
        pd.DataFrame列，即一个特征key，对应一个列
        :param orders_pd: 回测结果生成的交易行为构成的pd.DataFrame对象
        :return:
        """
        if ABuEnv.g_enable_ml_feature:
            features_keys = list()
            # 收集买入特征keys
            features_keys.extend(self._get_unzip_feature_keys(True))
            # 收集卖出特征keys
            features_keys.extend(self._get_unzip_feature_keys(False))

            # from ..UtilBu.ABuDTUtil import except_debug
            # @except_debug
            def map_str_dict(order, key):
                if order.sell_type == 'keep' and key.startswith('sell_'):
                    # 针对卖出特征值，如果单子keep状态，即没有特征值
                    return np.nan

                if not isinstance(order.ml_features, dict):
                    # 低版本pandas dict对象取出来会成为str
                    map_ast = ast.literal_eval(order.ml_features)[key]
                else:
                    map_ast = order.ml_features[key]

                return map_ast

            for fk in features_keys:
                # 迭代所有key，fk做为pd.DataFrame对象orders_pd的新列名，
                orders_pd[fk] = orders_pd.apply(map_str_dict, axis=1, args=(fk,))


class AbuFeatureDegExtend(AbuFeatureBase, BuyFeatureMixin, SellFeatureMixin):
    """示例添加新的视角来录制比赛，角度特征，支持买入，卖出"""

    def __init__(self):
        """20, 40, 60, 90, 120日走势角度特征"""
        # frozenset包一下，一旦定下来就不能修改，否则特征对不上
        self.deg_keys = frozenset([10, 30, 50, 90, 120])

    def get_feature_keys(self, buy_feature):
        """
        迭代生成所有走势角度特征feature的列名称定, 使用feature_prefix区分买入，卖出前缀key
        :param buy_feature: 是否是买入特征构造（bool）
        :return: 角度特征的键值对字典中的key序列
        """
        return ['{}deg_ang{}'.format(self.feature_prefix(buy_feature=buy_feature), dk) for dk in self.deg_keys]

    def calc_feature(self, kl_pd, combine_kl_pd, day_ind, buy_feature):
        """
        根据买入或者卖出时的金融时间序列，以及交易日信息构造拟合角度特征
        :param kl_pd: 择时阶段金融时间序列
        :param combine_kl_pd: 合并择时阶段之前1年的金融时间序列
        :param day_ind: 交易发生的时间索引，即对应self.kl_pd.key
        :param buy_feature: 是否是买入特征构造（bool）
        :return: 构造角度特征的键值对字典
        """
        # 返回的角度特征键值对字典
        deg_dict = {}
        for dk in self.deg_keys:
            # 迭代预设角度周期，计算构建特征
            if day_ind - dk >= 0:
                # 如果择时时间序列够提取特征，使用kl_pd截取特征交易周期收盘价格
                deg_close = kl_pd[day_ind - dk + 1:day_ind + 1].close
            else:
                # 如果择时时间序列不够提取特征，使用combine_kl_pd截取特征交易周期，首先截取直到day_ind的时间序列
                combine_kl_pd = combine_kl_pd.loc[:kl_pd.index[day_ind]]
                # 如combine_kl_pd长度大于特征周期长度－> 截取combine_kl_pd[-dk:].close，否则取combine_kl_pd所有交易收盘价格
                deg_close = combine_kl_pd[-dk:].close if combine_kl_pd.shape[0] > dk else combine_kl_pd.close

            # 使用截取特征交易周期收盘价格deg_close做为参数，通过calc_regress_deg计算趋势拟合角度
            ang = ABuRegUtil.calc_regress_deg(deg_close, show=False)
            # 标准化拟合角度值
            ang = 0 if np.isnan(ang) else round(ang, 3)
            # 角度特征键值对字典添加拟合角度周期key和对应的拟合角度值
            deg_dict['{}deg_ang{}'.format(self.feature_prefix(buy_feature=buy_feature), dk)] = ang
        return deg_dict
