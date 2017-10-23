# -*- encoding:utf-8 -*-
"""示例ump边裁特征多混模块"""

from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

from ..TradeBu.ABuMLFeature import AbuFeatureDeg, AbuFeaturePrice, AbuFeatureWave, AbuFeatureAtr
from ..MLBu.ABuMLPd import AbuMLPd
from .ABuUmpBase import ump_edge_make_xy, BuyUmpMixin
from .ABuUmpEdgeBase import AbuUmpEdgeBase

__author__ = '阿布'
__weixin__ = 'abu_quant'


class UmpFullFiter(AbuMLPd):
    """
        内部类，只需要实现make_xy，且使用ump_edge_make_xy装饰

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
        :return: self.order_has_ret中使用filter选取特征列形成full_df
                 ump_edge_make_xy装饰器在make_xy返回full_df后做转换matrix，形成x，y等工作
        """

        filter_list = ['profit', 'profit_cg']
        cols = AbuFeatureDeg().get_feature_ump_keys(ump_cls=AbuUmpEdgeFull)
        cols.extend(AbuFeaturePrice().get_feature_ump_keys(ump_cls=AbuUmpEdgeFull))
        cols.extend(AbuFeatureWave().get_feature_ump_keys(ump_cls=AbuUmpEdgeFull))
        cols.extend(AbuFeatureAtr().get_feature_ump_keys(ump_cls=AbuUmpEdgeFull))

        filter_list.extend(cols)
        """
            filter_list:
            ['profit', 'profit_cg', 'buy_deg_ang42', 'buy_deg_ang252', 'buy_deg_ang60', 'buy_deg_ang21',
            'buy_price_rank120', 'buy_price_rank90', 'buy_price_rank60', 'buy_price_rank252', 'buy_wave_score1',
            'buy_wave_score2', 'buy_wave_score3', 'buy_atr_std']
        """
        # noinspection PyUnresolvedReferences
        full_df = self.order_has_ret.filter(filter_list)
        """
            eg: full_df
                           profit  profit_cg  buy_deg_ang42  buy_deg_ang252  \
            2014-09-24  -22618.04    -0.0566          3.378           3.458
            2014-10-24  -29690.28    -0.0742          0.191           2.889
            2014-10-29   18959.19     0.0542         -2.026          16.689
            2014-10-29  148209.36     0.5022         -3.427         -11.956
            2014-10-29   24867.60     0.0952         -2.915          39.469
            2014-10-29   18959.19     0.0542         -2.026          16.689
            ...               ...        ...            ...             ...
            2016-04-13  -28044.40    -0.1159          6.603         -31.459
            2016-04-14  -18645.93    -0.0467          4.611          18.428
            2016-04-15  -32484.79    -0.1149          4.238         -13.247
            2016-04-15  -32484.79    -0.1149          4.238         -13.247
            2016-04-29     290.96     0.0007          1.445          16.266
            2016-04-29     290.96     0.0007          1.445          16.266

                        buy_deg_ang60  buy_deg_ang21  buy_price_rank120  buy_price_rank90  \
            2014-09-24          3.458          1.818              1.000             1.000
            2014-10-24          2.809         -1.089              1.000             1.000
            2014-10-29         -0.761          1.980              1.000             1.000
            2014-10-29         -8.296          6.507              0.925             0.900
            2014-10-29         -6.043          7.046              0.962             0.950
            2014-10-29         -0.761          1.980              1.000             1.000
            ...                   ...            ...                ...               ...
            2016-04-13          0.191          4.457              0.567             0.722
            2016-04-14          3.134          0.733              0.875             0.878
            2016-04-15          4.693          1.162              0.775             0.733
            2016-04-15          4.693          1.162              0.775             0.733
            2016-04-29          4.615         -1.115              1.000             1.000
            2016-04-29          4.615         -1.115              1.000             1.000

                        buy_price_rank60  buy_price_rank252  buy_wave_score1  \
            2014-09-24             1.000              1.000            0.287
            2014-10-24             1.000              1.000            0.596
            2014-10-29             1.000              1.000            0.444
            2014-10-29             0.883              0.750           -0.173
            2014-10-29             0.925              0.982            0.031
            2014-10-29             1.000              1.000            0.444
            ...                      ...                ...              ...
            2016-04-13             1.000              0.270            0.271
            2016-04-14             0.967              0.940           -0.030
            2016-04-15             1.000              0.631            0.596
            2016-04-15             1.000              0.631            0.596
            2016-04-29             1.000              1.000            0.743
            2016-04-29             1.000              1.000            0.743

                        buy_wave_score2  buy_wave_score3  buy_atr_std
            2014-09-24            0.234            0.218        0.226
            2014-10-24            0.488            0.449        0.146
            2014-10-29            0.338            0.329        0.538
            2014-10-29           -0.202           -0.203        0.558
            2014-10-29           -0.128           -0.173        0.063
            2014-10-29            0.338            0.329        0.538
            ...                     ...              ...          ...
            2016-04-13            0.509            0.651        0.262
            2016-04-14            0.081            0.170        0.049
            2016-04-15            0.753            0.800        0.135
            2016-04-15            0.753            0.800        0.135
            2016-04-29            0.840            0.918        0.838
            2016-04-29            0.840            0.918        0.838
        """
        return full_df


class AbuUmpEdgeFull(AbuUmpEdgeBase, BuyUmpMixin):
    """边裁多混特征类，AbuUmpEdgeBase子类，混入BuyUmpMixin，做为买入ump类"""

    def get_predict_col(self):
        """
        边裁多混特征keys：
            ['buy_deg_ang42', 'buy_deg_ang252', 'buy_deg_ang60', 'buy_deg_ang21', 'buy_price_rank120',
            'buy_price_rank90', 'buy_price_rank60', 'buy_price_rank252',
            'buy_wave_score1', 'buy_wave_score2', 'buy_wave_score3', 'buy_atr_std',
            'buy_diff_down_days', 'buy_jump_up_power', 'buy_diff_up_days', 'buy_jump_down_power']

        :return: ['buy_deg_ang42', 'buy_deg_ang252', 'buy_deg_ang60', 'buy_deg_ang21', 'buy_price_rank120',
                'buy_price_rank90', 'buy_price_rank60', 'buy_price_rank252',
                'buy_wave_score1', 'buy_wave_score2', 'buy_wave_score3', 'buy_atr_std',
                'buy_diff_down_days', 'buy_jump_up_power', 'buy_diff_up_days', 'buy_jump_down_power']
        """

        cols = AbuFeatureDeg().get_feature_ump_keys(ump_cls=AbuUmpEdgeFull)
        cols.extend(AbuFeaturePrice().get_feature_ump_keys(ump_cls=AbuUmpEdgeFull))
        cols.extend(AbuFeatureWave().get_feature_ump_keys(ump_cls=AbuUmpEdgeFull))
        cols.extend(AbuFeatureAtr().get_feature_ump_keys(ump_cls=AbuUmpEdgeFull))
        return cols

    def get_fiter_class(self):
        """
        边裁多混特征返回的AbuMLPd子类：AbuUmpEdgeFull.UmpFullFiter
        :return: AbuUmpEdgeFull.UmpFullFiter
        """
        return UmpFullFiter

    @classmethod
    def class_unique_id(cls):
        """
        具体ump类关键字唯一名称，类方法：return 'full_edge'
        主要针对外部user设置自定义ump使用, 需要user自己保证class_unique_id的唯一性，内部不做检测
        具体使用见ABuUmpManager中extend_ump_block方法
        """
        return 'full_edge'
