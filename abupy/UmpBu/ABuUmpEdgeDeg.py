# -*- encoding:utf-8 -*-
"""示例ump边裁特征走势拟合角度模块"""

from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

from ..TradeBu.ABuMLFeature import AbuFeatureDeg, AbuFeatureDegExtend
from ..MLBu.ABuMLPd import AbuMLPd
from .ABuUmpBase import ump_edge_make_xy, BuyUmpMixin
from .ABuUmpEdgeBase import AbuUmpEdgeBase

__author__ = '阿布'
__weixin__ = 'abu_quant'


class UmpDegFiter(AbuMLPd):
    """
        只需要实现make_xy，且使用ump_edge_make_xy装饰
        在边裁__init__中通过：
                self.fiter_cls = self.get_fiter_class()
                self.fiter = self.fiter_cls(orders_pd=orders_pd, **kwarg)
        构造裁判的filter以及重要的self.fiter.df，即pd.DataFrame对象特征
    """

    @ump_edge_make_xy
    def make_xy(self, **kwarg):
        """
         make_xy通过装饰器ump_edge_make_xy进行包装，调用make_xy前将有交易结果的单子进行筛选：
         order_has_ret = orders_pd[orders_pd['result'] != 0]，且赋予self.order_has_ret
         make_xy只需要使用filter选取需要的特征，即从self.order_has_ret中使用filter选取特征列形成df

        :param kwarg: ump_edge_make_xy装饰器中使用kwarg
                      kwargs['orders_pd'] 做为必须要有的关键字参数：交易训练集数据，pd.DataFrame对象
        :return: self.order_has_ret中使用filter选取特征列形成deg_df
                 ump_edge_make_xy装饰器在make_xy返回deg_df后做转换matrix，形成x，y等工作
        """

        filter_list = ['profit', 'profit_cg']
        cols = AbuFeatureDeg().get_feature_ump_keys(ump_cls=AbuUmpEdgeDeg)
        # filter_list: ['profit', 'profit_cg', 'buy_deg_ang42', 'buy_deg_ang252', 'buy_deg_ang60', 'buy_deg_ang21']
        filter_list.extend(cols)

        # noinspection PyUnresolvedReferences
        deg_df = self.order_has_ret.filter(filter_list)
        """
            eg: deg_df
                           profit  profit_cg  buy_deg_ang42  buy_deg_ang252  \
            2014-09-24  -22618.04    -0.0566          3.378           3.458
            2014-10-24  -29690.28    -0.0742          0.191           2.889
            2014-10-29   18959.19     0.0542         -2.026          16.689
            2014-10-29  148209.36     0.5022         -3.427         -11.956
            2014-10-29   24867.60     0.0952         -2.915          39.469
            2014-10-29   18959.19     0.0542         -2.026          16.689
            2014-11-03    1250.80     0.0045          0.103          39.202
            2014-11-11   59888.21     0.1857          8.341          -9.450
            2014-11-12   -3578.78    -0.0140          3.963           6.595
            2014-11-26  -29085.19    -0.0946         14.052           6.061
            ...               ...        ...            ...             ...
            2016-03-14   16220.57     0.0559          4.002         -10.559
            2016-03-14  -25328.12    -0.1218          0.129          -6.649
            2016-03-30  -29858.44    -0.0863         13.121          -8.461
            2016-04-04    5373.76     0.0244          4.409         -33.097
            2016-04-13  -28044.40    -0.1159          6.603         -31.459
            2016-04-14  -18645.93    -0.0467          4.611          18.428
            2016-04-15  -32484.79    -0.1149          4.238         -13.247
            2016-04-15  -32484.79    -0.1149          4.238         -13.247
            2016-04-29     290.96     0.0007          1.445          16.266
            2016-04-29     290.96     0.0007          1.445          16.266

                        buy_deg_ang60  buy_deg_ang21
            2014-09-24          3.458          1.818
            2014-10-24          2.809         -1.089
            2014-10-29         -0.761          1.980
            2014-10-29         -8.296          6.507
            2014-10-29         -6.043          7.046
            2014-10-29         -0.761          1.980
            2014-11-03         -4.614         10.125
            2014-11-11          0.730         12.397
            2014-11-12         -7.524          6.671
            2014-11-26          7.566         12.494
            ...                   ...            ...
            2016-03-14         -7.992          9.324
            2016-03-14        -10.880          5.201
            2016-03-30          4.498          4.070
            2016-04-04         -6.281          5.618
            2016-04-13          0.191          4.457
            2016-04-14          3.134          0.733
            2016-04-15          4.693          1.162
            2016-04-15          4.693          1.162
            2016-04-29          4.615         -1.115
            2016-04-29          4.615         -1.115
        """
        return deg_df


class AbuUmpEdgeDeg(AbuUmpEdgeBase, BuyUmpMixin):
    """边裁走势拟合角度特征类，AbuUmpEdgeBase子类，混入BuyUmpMixin，做为买入ump类"""

    def get_predict_col(self):
        """
        边裁走势拟合角度特征keys：['buy_deg_ang42', 'buy_deg_ang252', 'buy_deg_ang60', 'buy_deg_ang21']
        :return: ['buy_deg_ang42', 'buy_deg_ang252', 'buy_deg_ang60', 'buy_deg_ang21']
        """
        return AbuFeatureDeg().get_feature_ump_keys(ump_cls=AbuUmpEdgeDeg)

    def get_fiter_class(self):
        """
        边裁特征走势拟合角度返回的AbuMLPd子类：AbuUmpEdgeDeg.UmpDegFiter
        :return: AbuUmpEdgeDeg.UmpDegFiter
        """
        return UmpDegFiter

    @classmethod
    def class_unique_id(cls):
        """
        具体ump类关键字唯一名称，类方法：return 'deg_edge'
        主要针对外部user设置自定义ump使用, 需要user自己保证class_unique_id的唯一性，内部不做检测
        具体使用见ABuUmpManager中extend_ump_block方法
        """
        return 'deg_edge'


class UmpExtendEdgeFiter(AbuMLPd):
    """角度边裁扩展类make_xy"""
    @ump_edge_make_xy
    def make_xy(self, **kwarg):
        filter_list = ['profit', 'profit_cg']
        col = AbuFeatureDegExtend().get_feature_ump_keys(ump_cls=AbuUmpEegeDegExtend)
        filter_list.extend(col)
        mul_df = self.order_has_ret.filter(filter_list)
        return mul_df


class AbuUmpEegeDegExtend(AbuUmpEdgeBase, BuyUmpMixin):
    """边裁使用新的视角来决策交易，AbuUmpEdgeBase子类，混入BuyUmpMixin，做为买入ump类"""

    def get_predict_col(self):
        # 这里使用get_feature_ump_keys，只需要传递当前类名称即可，其根据是买入ump还是卖出ump返回对应特征列
        col = AbuFeatureDegExtend().get_feature_ump_keys(ump_cls=AbuUmpEegeDegExtend)
        return col

    def get_fiter_class(self):
        return UmpExtendEdgeFiter

    @classmethod
    def class_unique_id(cls):
        return 'extend_edge_deg'
