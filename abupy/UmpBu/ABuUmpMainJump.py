# -*- encoding:utf-8 -*-
"""示例ump主裁特征跳空模块"""

from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

from ..TradeBu.ABuMLFeature import AbuFeatureJump
from ..MLBu.ABuMLPd import AbuMLPd
from .ABuUmpBase import ump_main_make_xy, BuyUmpMixin
from .ABuUmpMainBase import AbuUmpMainBase

__author__ = '阿布'
__weixin__ = 'abu_quant'


class UmpJumpFiter(AbuMLPd):
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

        :return: self.order_has_ret中使用filter选取特征列形成jump_df
                 ump_main_make_xy装饰器在make_xy返回jump_df后做转换matrix，形成x，y等工作
        """
        # 'regex=result|buy_diff_down_days|buy_diff_up_days|buy_jump_down_power|buy_jump_up_power'
        regex = 'result|{}'.format('|'.join(AbuFeatureJump().get_feature_ump_keys(ump_cls=AbuUmpMainJump)))
        # noinspection PyUnresolvedReferences
        jump_df = self.order_has_ret.filter(regex=regex)
        """
            eg: jump_df
                        result  buy_jump_down_power  buy_diff_down_days  \
            2014-09-24       0                0.000                   0
            2014-10-24       0                0.000                   0
            2014-10-29       1               -1.109                 278
            2014-10-29       1                0.000                   0
            2014-10-29       1               -1.522                  75
            2014-10-29       1               -1.109                 278
            2014-11-03       1               -1.451                  78
            2014-11-11       1                0.000                   0
            2014-11-12       0               -1.863                  82
            2014-11-26       0               -2.115                  96
            ...            ...                  ...                 ...
            2016-03-14       1               -2.965                  56
            2016-03-14       0               -2.323                  67
            2016-03-30       0               -1.744                  74
            2016-04-04       1               -2.085                  36
            2016-04-13       0               -2.940                  47
            2016-04-14       0               -1.764                 100
            2016-04-15       0               -1.455                 101
            2016-04-15       0               -1.455                 101
            2016-04-29       1               -1.178                  24
            2016-04-29       1               -1.178                  24

                        buy_jump_up_power  buy_diff_up_days
            2014-09-24              3.344                61
            2014-10-24              3.344                91
            2014-10-29              2.920                95
            2014-10-29              1.283               243
            2014-10-29              3.727               238
            2014-10-29              2.920                95
            2014-11-03              1.014               214
            2014-11-11              1.291               256
            2014-11-12              1.606               256
            2014-11-26              1.772               270
            ...                       ...               ...
            2016-03-14              2.682                14
            2016-03-14              2.478               246
            2016-03-30              2.621                32
            2016-04-04              1.507                45
            2016-04-13              1.678                56
            2016-04-14              1.158                71
            2016-04-15              1.075                58
            2016-04-15              1.075                58
            2016-04-29              3.259                 0
            2016-04-29              3.259                 0
        """
        return jump_df


class AbuUmpMainJump(AbuUmpMainBase, BuyUmpMixin):
    """主裁跳空特征类，AbuUmpMainBase子类，混入BuyUmpMixin，做为买入ump类"""

    def get_predict_col(self):
        """
        主裁跳空特征keys：['buy_diff_down_days', 'buy_diff_up_days', 'buy_jump_down_power', 'buy_jump_up_power']
        :return: ['buy_diff_down_days', 'buy_diff_up_days', 'buy_jump_down_power', 'buy_jump_up_power']
        """
        return AbuFeatureJump().get_feature_ump_keys(ump_cls=AbuUmpMainJump)

    def get_fiter_class(self):
        """
        主裁特征跳空返回的AbuMLPd子类：AbuUmpMainJump.UmpJumpFiter
        :return: AbuUmpMainJump.UmpJumpFiter
        """
        return UmpJumpFiter

    @classmethod
    def class_unique_id(cls):
        """
        具体ump类关键字唯一名称，类方法：return 'jump_main'
        主要针对外部user设置自定义ump使用, 需要user自己保证class_unique_id的唯一性，内部不做检测
        具体使用见ABuUmpManager中extend_ump_block方法
        """
        return 'jump_main'
