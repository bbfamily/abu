# -*- encoding:utf-8 -*-
"""示例ump边裁价格波动特征模块"""

from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

from ..TradeBu.ABuMLFeature import AbuFeatureWave
from ..MLBu.ABuMLPd import AbuMLPd
from .ABuUmpBase import ump_edge_make_xy, BuyUmpMixin
from .ABuUmpEdgeBase import AbuUmpEdgeBase

__author__ = '阿布'
__weixin__ = 'abu_quant'


class UmpWaveFiter(AbuMLPd):
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
        :return: self.order_has_ret中使用filter选取特征列形成wave_df
                 ump_edge_make_xy装饰器在make_xy返回wave_df后做转换matrix，形成x，y等工作
        """

        filter_list = ['profit', 'profit_cg']
        cols = AbuFeatureWave().get_feature_ump_keys(ump_cls=AbuUmpEdgeWave)
        # filter_list=['profit', 'profit_cg', 'buy_wave_score1', 'buy_wave_score2', 'buy_wave_score3']
        filter_list.extend(cols)
        # noinspection PyUnresolvedReferences
        wave_df = self.order_has_ret.filter(filter_list)
        """
            eg: wave_df
                           profit  profit_cg  buy_wave_score1  buy_wave_score2  \
            2014-09-24  -22618.04    -0.0566            0.287            0.234
            2014-10-24  -29690.28    -0.0742            0.596            0.488
            2014-10-29   18959.19     0.0542            0.444            0.338
            2014-10-29  148209.36     0.5022           -0.173           -0.202
            2014-10-29   24867.60     0.0952            0.031           -0.128
            2014-10-29   18959.19     0.0542            0.444            0.338
            2014-11-03    1250.80     0.0045            0.018           -0.128
            2014-11-11   59888.21     0.1857           -0.144           -0.060
            2014-11-12   -3578.78    -0.0140           -0.453           -0.505
            2014-11-26  -29085.19    -0.0946           -0.005           -0.007
            ...               ...        ...              ...              ...
            2016-03-14   16220.57     0.0559            0.928            0.941
            2016-03-14  -25328.12    -0.1218            1.209            0.891
            2016-03-30  -29858.44    -0.0863            0.470            0.630
            2016-04-04    5373.76     0.0244            0.363            0.608
            2016-04-13  -28044.40    -0.1159            0.271            0.509
            2016-04-14  -18645.93    -0.0467           -0.030            0.081
            2016-04-15  -32484.79    -0.1149            0.596            0.753
            2016-04-15  -32484.79    -0.1149            0.596            0.753
            2016-04-29     290.96     0.0007            0.743            0.840
            2016-04-29     290.96     0.0007            0.743            0.840

                        buy_wave_score3
            2014-09-24            0.218
            2014-10-24            0.449
            2014-10-29            0.329
            2014-10-29           -0.203
            2014-10-29           -0.173
            2014-10-29            0.329
            2014-11-03           -0.172
            2014-11-11            0.001
            2014-11-12           -0.509
            2014-11-26            0.015
            ...                     ...
            2016-03-14            0.948
            2016-03-14            0.788
            2016-03-30            0.702
            2016-04-04            0.743
            2016-04-13            0.651
            2016-04-14            0.170
            2016-04-15            0.800
            2016-04-15            0.800
            2016-04-29            0.918
            2016-04-29            0.918

        """
        return wave_df


class AbuUmpEdgeWave(AbuUmpEdgeBase, BuyUmpMixin):
    """边裁价格波动特征类，AbuUmpEdgeBase子类，混入BuyUmpMixin，做为买入ump类"""

    def get_predict_col(self):
        """
        边裁价格波动特征keys：['buy_wave_score1', 'buy_wave_score2', 'buy_wave_score3']
        :return: ['buy_wave_score1', 'buy_wave_score2', 'buy_wave_score3']
        """
        return AbuFeatureWave().get_feature_ump_keys(ump_cls=AbuUmpEdgeWave)

    def get_fiter_class(self):
        """
        边裁价格波动特征返回的AbuMLPd子类：AbuUmpEdgeWave.UmpWaveFiter
        :return: AbuUmpEdgeWave.UmpWaveFiter
        """
        return UmpWaveFiter

    @classmethod
    def class_unique_id(cls):
        """
         具体ump类关键字唯一名称，类方法：return 'wave_edge'
         主要针对外部user设置自定义ump使用, 需要user自己保证class_unique_id的唯一性，内部不做检测
         具体使用见ABuUmpManager中extend_ump_block方法
        """
        return 'wave_edge'
