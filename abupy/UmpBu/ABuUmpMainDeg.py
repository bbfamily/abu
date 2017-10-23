# -*- encoding:utf-8 -*-
"""示例ump主裁特征走势拟合角度模块"""

from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

from ..TradeBu.ABuMLFeature import AbuFeatureDeg, AbuFeatureDegExtend
from ..MLBu.ABuMLPd import AbuMLPd
from .ABuUmpBase import ump_main_make_xy, BuyUmpMixin
from .ABuUmpMainBase import AbuUmpMainBase

__author__ = '阿布'
__weixin__ = 'abu_quant'


class UmpDegFiter(AbuMLPd):
    """
        内部类，只需要实现make_xy，且使用ump_main_make_xy装饰

        在__init__中通过：
                self.fiter_cls = self.get_fiter_class()
                self.fiter = self.fiter_cls(orders_pd=self.orders_pd, **kwarg)
        构造裁判的filter以及重要的self.fiter.df，即pd.DataFrame对象特征
    """

    @ump_main_make_xy
    def make_xy(self, **kwarg):
        """
        make_xy通过装饰器ump_main_make_xy进行二次包装
        这里只需要使用filter选取需要的特征，即从self.order_has_ret中使用filter选取特征列形成df
        :param kwarg: ump_main_make_xy装饰器中使用kwarg
                      kwargs['orders_pd'] 做为必须要有的关键字参数：交易训练集数据，pd.DataFrame对象
                      kwargs['scaler']    做为可选关键字参数：控制在make_xy中返回的特征矩阵数据是否进行标准化处理

        :return: self.order_has_ret中使用filter选取特征列形成deg_df
                 ump_main_make_xy装饰器在make_xy返回deg_df后做转换matrix，形成x，y等工作

            eg: deg_df

                            result  buy_deg_ang42  buy_deg_ang252  buy_deg_ang60  \
            2014-09-24       0          3.378           3.458          3.458
            2014-10-24       0          0.191           2.889          2.809
            2014-10-29       1         -2.026          16.689         -0.761
            2014-10-29       1         -3.427         -11.956         -8.296
            2014-10-29       1         -2.915          39.469         -6.043
            2014-10-29       1         -2.026          16.689         -0.761
            2014-11-03       1          0.103          39.202         -4.614
            2014-11-11       1          8.341          -9.450          0.730
            2014-11-12       0          3.963           6.595         -7.524
            2014-11-26       0         14.052           6.061          7.566
            ...            ...            ...             ...            ...
            2016-03-14       1          4.002         -10.559         -7.992
            2016-03-14       0          0.129          -6.649        -10.880
            2016-03-30       0         13.121          -8.461          4.498
            2016-04-04       1          4.409         -33.097         -6.281
            2016-04-13       0          6.603         -31.459          0.191
            2016-04-14       0          4.611          18.428          3.134
            2016-04-15       0          4.238         -13.247          4.693
            2016-04-15       0          4.238         -13.247          4.693
            2016-04-29       1          1.445          16.266          4.615
            2016-04-29       1          1.445          16.266          4.615

                        buy_deg_ang21
            2014-09-24          1.818
            2014-10-24         -1.089
            2014-10-29          1.980
            2014-10-29          6.507
            2014-10-29          7.046
            2014-10-29          1.980
            2014-11-03         10.125
            2014-11-11         12.397
            2014-11-12          6.671
            2014-11-26         12.494
            ...                   ...
            2016-03-14          9.324
            2016-03-14          5.201
            2016-03-30          4.070
            2016-04-04          5.618
            2016-04-13          4.457
            2016-04-14          0.733
            2016-04-15          1.162
            2016-04-15          1.162
            2016-04-29         -1.115
            2016-04-29         -1.115
        """
        # regex='result|buy_deg_ang42|buy_deg_ang252|buy_deg_ang60|buy_deg_ang21'
        regex = 'result|{}'.format(
            '|'.join(AbuFeatureDeg().get_feature_ump_keys(ump_cls=AbuUmpMainDeg)))
        # noinspection PyUnresolvedReferences
        deg_df = self.order_has_ret.filter(regex=regex)
        return deg_df


class AbuUmpMainDeg(AbuUmpMainBase, BuyUmpMixin):
    """主裁走势拟合角度特征类，AbuUmpMainBase子类，混入BuyUmpMixin，做为买入ump类"""

    def get_predict_col(self):
        """
        主裁走势拟合角度特征keys：['buy_deg_ang42', 'buy_deg_ang252', 'buy_deg_ang60', 'buy_deg_ang21']
        :return: ['buy_deg_ang42', 'buy_deg_ang252', 'buy_deg_ang60', 'buy_deg_ang21']
        """
        return AbuFeatureDeg().get_feature_ump_keys(ump_cls=AbuUmpMainDeg)

    def get_fiter_class(self):
        """
        主裁特征走势拟合角度返回的AbuMLPd子类：AbuUmpMainDeg.UmpDegFiter
        :return: AbuUmpMainDeg.UmpDegFiter
        """
        return UmpDegFiter

    @classmethod
    def class_unique_id(cls):
        """
        具体ump类关键字唯一名称，类方法：return 'deg_main'
        主要针对外部user设置自定义ump使用, 需要user自己保证class_unique_id的唯一性，内部不做检测
        具体使用见ABuUmpManager中extend_ump_block方法
        """
        return 'deg_main'


class UmpExtendFeatureFiter(AbuMLPd):
    """角度主裁扩展类make_xy"""
    @ump_main_make_xy
    def make_xy(self, **kwarg):
        # 这里使用get_feature_ump_keys，只需要传递当前类名称即可，其根据是买入ump还是卖出ump返回对应特征列
        col = AbuFeatureDegExtend().get_feature_ump_keys(ump_cls=AbuUmpMainDegExtend)
        regex = 'result|{}'.format('|'.join(col))
        extend_deg_df = self.order_has_ret.filter(regex=regex)
        return extend_deg_df


class AbuUmpMainDegExtend(AbuUmpMainBase, BuyUmpMixin):
    """主裁使用新的视角来决策交易，AbuUmpMainBase子类，混入BuyUmpMixin，做为买入ump类"""

    def get_predict_col(self):
        # 这里使用get_feature_ump_keys，只需要传递当前类名称即可，其根据是买入ump还是卖出ump返回对应特征列
        col = AbuFeatureDegExtend().get_feature_ump_keys(ump_cls=AbuUmpMainDegExtend)
        return col

    def get_fiter_class(self):
        return UmpExtendFeatureFiter

    @classmethod
    def class_unique_id(cls):
        return 'extend_main_deg'
