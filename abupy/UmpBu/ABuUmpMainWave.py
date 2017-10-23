# -*- encoding:utf-8 -*-
"""示例ump价格波动特征模块"""

from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

from ..TradeBu.ABuMLFeature import AbuFeatureWave
from ..MLBu.ABuMLPd import AbuMLPd
from .ABuUmpBase import ump_main_make_xy, BuyUmpMixin
from .ABuUmpMainBase import AbuUmpMainBase

__author__ = '阿布'
__weixin__ = 'abu_quant'


class UmpWaveFiter(AbuMLPd):
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

        :return: self.order_has_ret中使用filter选取特征列形成wave_df
                 ump_main_make_xy装饰器在make_xy返回wave_df后做转换matrix，形成x，y等工作
        """

        # regex=result|buy_wave_score1|buy_wave_score2|buy_wave_score3'
        regex = 'result|{}'.format('|'.join(AbuFeatureWave().get_feature_ump_keys(ump_cls=AbuUmpMainWave)))
        # noinspection PyUnresolvedReferences
        wave_df = self.order_has_ret.filter(regex=regex)
        """
             eg: wave_df
                        result  buy_wave_score1  buy_wave_score2  buy_wave_score3
            2014-09-24       0            0.287            0.234            0.218
            2014-10-24       0            0.596            0.488            0.449
            2014-10-29       1            0.444            0.338            0.329
            2014-10-29       1           -0.173           -0.202           -0.203
            2014-10-29       1            0.031           -0.128           -0.173
            2014-10-29       1            0.444            0.338            0.329
            2014-11-03       1            0.018           -0.128           -0.172
            2014-11-11       1           -0.144           -0.060            0.001
            2014-11-12       0           -0.453           -0.505           -0.509
            2014-11-26       0           -0.005           -0.007            0.015
            ...            ...              ...              ...              ...
            2016-03-14       1            0.928            0.941            0.948
            2016-03-14       0            1.209            0.891            0.788
            2016-03-30       0            0.470            0.630            0.702
            2016-04-04       1            0.363            0.608            0.743
            2016-04-13       0            0.271            0.509            0.651
            2016-04-14       0           -0.030            0.081            0.170
            2016-04-15       0            0.596            0.753            0.800
            2016-04-15       0            0.596            0.753            0.800
            2016-04-29       1            0.743            0.840            0.918
            2016-04-29       1            0.743            0.840            0.918
        """

        return wave_df


class AbuUmpMainWave(AbuUmpMainBase, BuyUmpMixin):
    """主裁价格波动特征类，AbuUmpMainBase子类，混入BuyUmpMixin，做为买入ump类"""

    def get_predict_col(self):
        """
        主裁价格波动特征keys：['buy_wave_score1', 'buy_wave_score2', 'buy_wave_score3']
        :return: ['buy_wave_score1', 'buy_wave_score2', 'buy_wave_score3']
        """

        return AbuFeatureWave().get_feature_ump_keys(ump_cls=AbuUmpMainWave)

    def get_fiter_class(self):
        """
        主裁价格波动特征返回的AbuMLPd子类：AbuUmpMainWave.UmpWaveFiter
        :return: AbuUmpMainWave.UmpWaveFiter
        """
        return UmpWaveFiter

    @classmethod
    def class_unique_id(cls):
        """
        具体ump类关键字唯一名称，类方法：return 'wave_main'
        主要针对外部user设置自定义ump使用, 需要user自己保证class_unique_id的唯一性，内部不做检测
        具体使用见ABuUmpManager中extend_ump_block方法
        """
        return 'wave_main'
