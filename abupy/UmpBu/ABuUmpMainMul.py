# -*- encoding:utf-8 -*-
"""示例ump单混特征模块"""

from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

from ..TradeBu.ABuMLFeature import AbuFeatureDeg, AbuFeaturePrice, AbuFeatureWave, AbuFeatureAtr
from ..MLBu.ABuMLPd import AbuMLPd
from .ABuUmpBase import ump_main_make_xy, BuyUmpMixin
from .ABuUmpMainBase import AbuUmpMainBase

__author__ = '阿布'
__weixin__ = 'abu_quant'


class UmpMulFiter(AbuMLPd):
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

        :return: self.order_has_ret中使用filter选取特征列形成mul_df
                 ump_main_make_xy装饰器在make_xy返回mul_df后做转换matrix，形成x，y等工作
        """

        # regex='result|buy_deg_ang21|buy_price_rank252|buy_wave_score3|buy_atr_std'
        regex = 'result|{}|{}|{}|{}'.format(AbuFeatureDeg().get_feature_ump_keys(ump_cls=AbuUmpMainMul)[-1],
                                            AbuFeaturePrice().get_feature_ump_keys(ump_cls=AbuUmpMainMul)[-1],
                                            AbuFeatureWave().get_feature_ump_keys(ump_cls=AbuUmpMainMul)[-1],
                                            AbuFeatureAtr().get_feature_ump_keys(ump_cls=AbuUmpMainMul)[-1])
        # noinspection PyUnresolvedReferences
        mul_df = self.order_has_ret.filter(regex=regex)
        """
            eg: mul_df
                        result  buy_deg_ang21  buy_price_rank252  buy_wave_score3  \
            2014-09-24       0          1.818              1.000            0.218
            2014-10-24       0         -1.089              1.000            0.449
            2014-10-29       1          1.980              1.000            0.329
            2014-10-29       1          6.507              0.750           -0.203
            2014-10-29       1          7.046              0.982           -0.173
            2014-10-29       1          1.980              1.000            0.329
            2014-11-03       1         10.125              1.000           -0.172
            2014-11-11       1         12.397              0.808            0.001
            2014-11-12       0          6.671              0.560           -0.509
            2014-11-26       0         12.494              0.762            0.015
            ...            ...            ...                ...              ...
            2016-03-14       1          9.324              0.444            0.948
            2016-03-14       0          5.201              0.623            0.788
            2016-03-30       0          4.070              0.536            0.702
            2016-04-04       1          5.618              0.190            0.743
            2016-04-13       0          4.457              0.270            0.651
            2016-04-14       0          0.733              0.940            0.170
            2016-04-15       0          1.162              0.631            0.800
            2016-04-15       0          1.162              0.631            0.800
            2016-04-29       1         -1.115              1.000            0.918
            2016-04-29       1         -1.115              1.000            0.918

                        buy_atr_std
            2014-09-24        0.226
            2014-10-24        0.146
            2014-10-29        0.538
            2014-10-29        0.558
            2014-10-29        0.063
            2014-10-29        0.538
            2014-11-03       -0.002
            2014-11-11        0.171
            2014-11-12        0.093
            2014-11-26        0.812
            ...                 ...
            2016-03-14        0.126
            2016-03-14        0.663
            2016-03-30       -0.008
            2016-04-04        0.035
            2016-04-13        0.262
            2016-04-14        0.049
            2016-04-15        0.135
            2016-04-15        0.135
            2016-04-29        0.838
            2016-04-29        0.838
        """
        return mul_df

class AbuUmpMainMul(AbuUmpMainBase, BuyUmpMixin):
    """主裁单混特征类，AbuUmpMainBase子类，混入BuyUmpMixin，做为买入ump类"""



    def get_predict_col(self):
        """
        主裁单混特征keys：['buy_deg_ang21', 'buy_price_rank252', 'buy_wave_score3', 'buy_atr_std']
        :return: ['buy_deg_ang21', 'buy_price_rank252', 'buy_wave_score3', 'buy_atr_std']
        """

        return [AbuFeatureDeg().get_feature_ump_keys(ump_cls=AbuUmpMainMul)[-1],
                AbuFeaturePrice().get_feature_ump_keys(ump_cls=AbuUmpMainMul)[-1],
                AbuFeatureWave().get_feature_ump_keys(ump_cls=AbuUmpMainMul)[-1],
                AbuFeatureAtr().get_feature_ump_keys(ump_cls=AbuUmpMainMul)[-1]]

    def get_fiter_class(self):
        """
        主裁单混特征返回的AbuMLPd子类：AbuUmpMainMul.UmpMulFiter
        :return: AbuUmpMainMul.UmpMulFiter
        """
        return UmpMulFiter

    @classmethod
    def class_unique_id(cls):
        """
        具体ump类关键字唯一名称，类方法：return 'mul_main'
        主要针对外部user设置自定义ump使用, 需要user自己保证class_unique_id的唯一性，内部不做检测
        具体使用见ABuUmpManager中extend_ump_block方法
        """
        return 'mul_main'
