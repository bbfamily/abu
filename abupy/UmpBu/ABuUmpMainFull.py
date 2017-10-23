# -*- encoding:utf-8 -*-
"""示例ump多混特征模块"""

from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

from ..TradeBu.ABuMLFeature import AbuFeatureDeg, AbuFeaturePrice, AbuFeatureJump, AbuFeatureWave, AbuFeatureAtr
from ..MLBu.ABuMLPd import AbuMLPd
from .ABuUmpBase import ump_main_make_xy, BuyUmpMixin
from .ABuUmpMainBase import AbuUmpMainBase

__author__ = '阿布'
__weixin__ = 'abu_quant'


class UmpFullFiter(AbuMLPd):
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

        :return: self.order_has_ret中使用filter选取特征列形成full_df
                 ump_main_make_xy装饰器在make_xy返回full_df后做转换matrix，形成x，y等工作
        """

        regex = 'result|{}|{}|{}|{}|{}'.format(
            '|'.join(AbuFeatureDeg().get_feature_ump_keys(ump_cls=AbuUmpMainFull)),
            '|'.join(AbuFeaturePrice().get_feature_ump_keys(ump_cls=AbuUmpMainFull)),
            '|'.join(AbuFeatureWave().get_feature_ump_keys(ump_cls=AbuUmpMainFull)),
            '|'.join(AbuFeatureAtr().get_feature_ump_keys(ump_cls=AbuUmpMainFull)),
            '|'.join(AbuFeatureJump().get_feature_ump_keys(ump_cls=AbuUmpMainFull)))
        """
            regex = 'result|buy_deg_ang42|buy_deg_ang252|buy_deg_ang60|buy_deg_ang21|buy_price_rank120|
            buy_price_rank90|buy_price_rank60|buy_price_rank252|buy_wave_score1|buy_wave_score2|buy_wave_score3
            |buy_atr_std|buy_diff_down_days|buy_jump_up_power|buy_diff_up_days|buy_jump_down_power'
        """

        # noinspection PyUnresolvedReferences
        full_df = self.order_has_ret.filter(regex=regex)
        """
            eg: full_df
                        result  buy_deg_ang42  buy_deg_ang252  buy_deg_ang60  \
            2014-09-24       0          3.378           3.458          3.458
            2014-10-24       0          0.191           2.889          2.809
            2014-10-29       1         -2.026          16.689         -0.761
            2014-10-29       1         -3.427         -11.956         -8.296
            2014-10-29       1         -2.915          39.469         -6.043
            2014-10-29       1         -2.026          16.689         -0.761
            ...            ...            ...             ...            ...
            2016-04-14       0          4.611          18.428          3.134
            2016-04-15       0          4.238         -13.247          4.693
            2016-04-15       0          4.238         -13.247          4.693
            2016-04-29       1          1.445          16.266          4.615
            2016-04-29       1          1.445          16.266          4.615

                        buy_deg_ang21  buy_price_rank120  buy_price_rank90  \
            2014-09-24          1.818              1.000             1.000
            2014-10-24         -1.089              1.000             1.000
            2014-10-29          1.980              1.000             1.000
            2014-10-29          6.507              0.925             0.900
            2014-10-29          7.046              0.962             0.950
            2014-10-29          1.980              1.000             1.000
            ...                   ...                ...               ...
            2016-04-14          0.733              0.875             0.878
            2016-04-15          1.162              0.775             0.733
            2016-04-15          1.162              0.775             0.733
            2016-04-29         -1.115              1.000             1.000
            2016-04-29         -1.115              1.000             1.000

                        buy_price_rank60  buy_price_rank252  buy_wave_score1  \
            2014-09-24             1.000              1.000            0.287
            2014-10-24             1.000              1.000            0.596
            2014-10-29             1.000              1.000            0.444
            2014-10-29             0.883              0.750           -0.173
            2014-10-29             0.925              0.982            0.031
            2014-10-29             1.000              1.000            0.444
            ...                      ...                ...              ...
            2016-04-14             0.967              0.940           -0.030
            2016-04-15             1.000              0.631            0.596
            2016-04-15             1.000              0.631            0.596
            2016-04-29             1.000              1.000            0.743
            2016-04-29             1.000              1.000            0.743

                        buy_wave_score2  buy_wave_score3  buy_atr_std  \
            2014-09-24            0.234            0.218        0.226
            2014-10-24            0.488            0.449        0.146
            2014-10-29            0.338            0.329        0.538
            2014-10-29           -0.202           -0.203        0.558
            2014-10-29           -0.128           -0.173        0.063
            2014-10-29            0.338            0.329        0.538
            ...                     ...              ...          ...
            2016-04-14            0.081            0.170        0.049
            2016-04-15            0.753            0.800        0.135
            2016-04-15            0.753            0.800        0.135
            2016-04-29            0.840            0.918        0.838
            2016-04-29            0.840            0.918        0.838

                        buy_jump_down_power  buy_diff_down_days  buy_jump_up_power  \
            2014-09-24                0.000                   0              3.344
            2014-10-24                0.000                   0              3.344
            2014-10-29               -1.109                 278              2.920
            2014-10-29                0.000                   0              1.283
            2014-10-29               -1.522                  75              3.727
            2014-10-29               -1.109                 278              2.920
            ...                         ...                 ...                ...
            2016-04-14               -1.764                 100              1.158
            2016-04-15               -1.455                 101              1.075
            2016-04-15               -1.455                 101              1.075
            2016-04-29               -1.178                  24              3.259
            2016-04-29               -1.178                  24              3.259

                        buy_diff_up_days
            2014-09-24                61
            2014-10-24                91
            2014-10-29                95
            2014-10-29               243
            2014-10-29               238
            2014-10-29                95
            ...                      ...
            2016-04-14                71
            2016-04-15                58
            2016-04-15                58
            2016-04-29                 0
            2016-04-29                 0
        """
        return full_df


class AbuUmpMainFull(AbuUmpMainBase, BuyUmpMixin):
    """主裁多混特征类，AbuUmpMainBase子类，混入BuyUmpMixin，做为买入ump类"""

    def get_predict_col(self):
        """
        主裁多混特征keys：
            ['buy_deg_ang42', 'buy_deg_ang252', 'buy_deg_ang60', 'buy_deg_ang21', 'buy_price_rank120',
            'buy_price_rank90', 'buy_price_rank60', 'buy_price_rank252',
            'buy_wave_score1', 'buy_wave_score2', 'buy_wave_score3', 'buy_atr_std',
            'buy_diff_down_days', 'buy_jump_up_power', 'buy_diff_up_days', 'buy_jump_down_power']

        :return: ['buy_deg_ang42', 'buy_deg_ang252', 'buy_deg_ang60', 'buy_deg_ang21', 'buy_price_rank120',
                'buy_price_rank90', 'buy_price_rank60', 'buy_price_rank252',
                'buy_wave_score1', 'buy_wave_score2', 'buy_wave_score3', 'buy_atr_std',
                'buy_diff_down_days', 'buy_jump_up_power', 'buy_diff_up_days', 'buy_jump_down_power']
        """
        cols = AbuFeatureDeg().get_feature_ump_keys(ump_cls=AbuUmpMainFull)
        cols.extend(AbuFeaturePrice().get_feature_ump_keys(ump_cls=AbuUmpMainFull))
        cols.extend(AbuFeatureWave().get_feature_ump_keys(ump_cls=AbuUmpMainFull))
        cols.extend(AbuFeatureAtr().get_feature_ump_keys(ump_cls=AbuUmpMainFull))
        cols.extend(AbuFeatureJump().get_feature_ump_keys(ump_cls=AbuUmpMainFull))
        return cols

    def get_fiter_class(self):
        """
        主裁多混特征返回的AbuMLPd子类：AbuUmpMainFull.UmpFullFiter
        :return: AbuUmpMainFull.UmpFullFiter
        """
        return UmpFullFiter

    @classmethod
    def class_unique_id(cls):
        """
        具体ump类关键字唯一名称，类方法：return 'full_main'
        主要针对外部user设置自定义ump使用, 需要user自己保证class_unique_id的唯一性，内部不做检测
        具体使用见ABuUmpManager中extend_ump_block方法
        """
        return 'full_main'
