# -*- encoding:utf-8 -*-
"""示例ump边裁特征单混模块"""

from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

from ..TradeBu.ABuMLFeature import AbuFeatureDeg, AbuFeaturePrice, AbuFeatureWave, AbuFeatureAtr
from ..MLBu.ABuMLPd import AbuMLPd
from .ABuUmpBase import ump_edge_make_xy, BuyUmpMixin
from .ABuUmpEdgeBase import AbuUmpEdgeBase

__author__ = '阿布'
__weixin__ = 'abu_quant'


class UmpMulFiter(AbuMLPd):
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
        :return: self.order_has_ret中使用filter选取特征列形成mul_df
                 ump_edge_make_xy装饰器在make_xy返回mul_df后做转换matrix，形成x，y等工作
        """

        filter_list = ['profit', 'profit_cg']
        # ['profit', 'profit_cg', 'buy_deg_ang21', 'buy_price_rank252', 'buy_wave_score3', 'buy_atr_std']
        filter_list.extend(
            [AbuFeatureDeg().get_feature_ump_keys(ump_cls=AbuUmpEdgeMul)[-1],
             AbuFeaturePrice().get_feature_ump_keys(ump_cls=AbuUmpEdgeMul)[-1],
             AbuFeatureWave().get_feature_ump_keys(ump_cls=AbuUmpEdgeMul)[-1],
             AbuFeatureAtr().get_feature_ump_keys(ump_cls=AbuUmpEdgeMul)[-1]])
        # noinspection PyUnresolvedReferences
        mul_df = self.order_has_ret.filter(filter_list)
        """
            eg: mul_df
                           profit  profit_cg  buy_deg_ang21  buy_price_rank252  \
            2014-09-24  -22618.04    -0.0566          1.818              1.000
            2014-10-24  -29690.28    -0.0742         -1.089              1.000
            2014-10-29   18959.19     0.0542          1.980              1.000
            2014-10-29  148209.36     0.5022          6.507              0.750
            2014-10-29   24867.60     0.0952          7.046              0.982
            2014-10-29   18959.19     0.0542          1.980              1.000
            2014-11-03    1250.80     0.0045         10.125              1.000
            2014-11-11   59888.21     0.1857         12.397              0.808
            2014-11-12   -3578.78    -0.0140          6.671              0.560
            2014-11-26  -29085.19    -0.0946         12.494              0.762
            ...               ...        ...            ...                ...
            2016-03-14   16220.57     0.0559          9.324              0.444
            2016-03-14  -25328.12    -0.1218          5.201              0.623
            2016-03-30  -29858.44    -0.0863          4.070              0.536
            2016-04-04    5373.76     0.0244          5.618              0.190
            2016-04-13  -28044.40    -0.1159          4.457              0.270
            2016-04-14  -18645.93    -0.0467          0.733              0.940
            2016-04-15  -32484.79    -0.1149          1.162              0.631
            2016-04-15  -32484.79    -0.1149          1.162              0.631
            2016-04-29     290.96     0.0007         -1.115              1.000
            2016-04-29     290.96     0.0007         -1.115              1.000

                        buy_wave_score3  buy_atr_std
            2014-09-24            0.218        0.226
            2014-10-24            0.449        0.146
            2014-10-29            0.329        0.538
            2014-10-29           -0.203        0.558
            2014-10-29           -0.173        0.063
            2014-10-29            0.329        0.538
            2014-11-03           -0.172       -0.002
            2014-11-11            0.001        0.171
            2014-11-12           -0.509        0.093
            2014-11-26            0.015        0.812
            ...                     ...          ...
            2016-03-14            0.948        0.126
            2016-03-14            0.788        0.663
            2016-03-30            0.702       -0.008
            2016-04-04            0.743        0.035
            2016-04-13            0.651        0.262
            2016-04-14            0.170        0.049
            2016-04-15            0.800        0.135
            2016-04-15            0.800        0.135
            2016-04-29            0.918        0.838
            2016-04-29            0.918        0.838
        """

        return mul_df


class AbuUmpEdgeMul(AbuUmpEdgeBase, BuyUmpMixin):
    """边裁单混特征类，AbuUmpEdgeBase子类，混入BuyUmpMixin，做为买入ump类"""

    def get_predict_col(self):
        """
        边裁单混特征keys：['buy_deg_ang21', 'buy_price_rank252', 'buy_wave_score3', 'buy_atr_std']
        :return: ['buy_deg_ang21', 'buy_price_rank252', 'buy_wave_score3', 'buy_atr_std']
        """

        return [AbuFeatureDeg().get_feature_ump_keys(ump_cls=AbuUmpEdgeMul)[-1],
                AbuFeaturePrice().get_feature_ump_keys(ump_cls=AbuUmpEdgeMul)[-1],
                AbuFeatureWave().get_feature_ump_keys(ump_cls=AbuUmpEdgeMul)[-1],
                AbuFeatureAtr().get_feature_ump_keys(ump_cls=AbuUmpEdgeMul)[-1]]

    def get_fiter_class(self):
        """
        边裁单混特征返回的AbuMLPd子类：AbuUmpEdgeMul.UmpMulFiter
        :return: AbuUmpEdgeMul.UmpMulFiter
        """
        return UmpMulFiter

    @classmethod
    def class_unique_id(cls):
        """
         具体ump类关键字唯一名称，类方法：return 'mul_edge'
         主要针对外部user设置自定义ump使用, 需要user自己保证class_unique_id的唯一性，内部不做检测
         具体使用见ABuUmpManager中extend_ump_block方法
        """
        return 'mul_edge'
