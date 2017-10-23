# -*- encoding:utf-8 -*-
"""示例ump价格特征模块"""

from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

from ..TradeBu.ABuMLFeature import AbuFeaturePrice
from ..MLBu.ABuMLPd import AbuMLPd
from .ABuUmpBase import ump_main_make_xy, BuyUmpMixin
from .ABuUmpMainBase import AbuUmpMainBase

__author__ = '阿布'
__weixin__ = 'abu_quant'


class UmpPriceFiter(AbuMLPd):
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

        :return: self.order_has_ret中使用filter选取特征列形成price_df
                 ump_main_make_xy装饰器在make_xy返回price_df后做转换matrix，形成x，y等工作
        """

        # regex='result|buy_price_rank120|buy_price_rank90|buy_price_rank60|buy_price_rank252'
        regex = 'result|{}'.format('|'.join(AbuFeaturePrice().get_feature_ump_keys(ump_cls=AbuUmpMainPrice)))
        # noinspection PyUnresolvedReferences
        price_df = self.order_has_ret.filter(regex=regex)
        """
            eg: price_df
                        result  buy_price_rank120  buy_price_rank90  buy_price_rank60  \
            2014-09-24       0              1.000             1.000             1.000
            2014-10-24       0              1.000             1.000             1.000
            2014-10-29       1              1.000             1.000             1.000
            2014-10-29       1              0.925             0.900             0.883
            2014-10-29       1              0.962             0.950             0.925
            2014-10-29       1              1.000             1.000             1.000
            2014-11-03       1              1.000             1.000             1.000
            2014-11-11       1              0.954             0.939             0.992
            2014-11-12       0              0.475             0.522             0.783
            2014-11-26       0              0.642             0.733             1.000
            ...            ...                ...               ...               ...
            2016-03-14       1              0.617             0.500             0.750
            2016-03-14       0              0.683             0.589             0.850
            2016-03-30       0              0.658             0.667             1.000
            2016-04-04       1              0.400             0.511             0.767
            2016-04-13       0              0.567             0.722             1.000
            2016-04-14       0              0.875             0.878             0.967
            2016-04-15       0              0.775             0.733             1.000
            2016-04-15       0              0.775             0.733             1.000
            2016-04-29       1              1.000             1.000             1.000
            2016-04-29       1              1.000             1.000             1.000

                        buy_price_rank252
            2014-09-24              1.000
            2014-10-24              1.000
            2014-10-29              1.000
            2014-10-29              0.750
            2014-10-29              0.982
            2014-10-29              1.000
            2014-11-03              1.000
            2014-11-11              0.808
            2014-11-12              0.560
            2014-11-26              0.762
            ...                       ...
            2016-03-14              0.444
            2016-03-14              0.623
            2016-03-30              0.536
            2016-04-04              0.190
            2016-04-13              0.270
            2016-04-14              0.940
            2016-04-15              0.631
            2016-04-15              0.631
            2016-04-29              1.000
            2016-04-29              1.000
        """
        return price_df


class AbuUmpMainPrice(AbuUmpMainBase, BuyUmpMixin):
    """主裁价格特征类，AbuUmpMainBase子类，混入BuyUmpMixin，做为买入ump类"""

    def get_predict_col(self):
        """
        主裁价格特征keys：['buy_price_rank120', 'buy_price_rank90', 'buy_price_rank60', 'buy_price_rank252']
        :return: ['buy_price_rank120', 'buy_price_rank90', 'buy_price_rank60', 'buy_price_rank252']
        """
        return AbuFeaturePrice().get_feature_ump_keys(ump_cls=AbuUmpMainPrice)

    def get_fiter_class(self):
        """
        主裁价格特征返回的AbuMLPd子类：AbuUmpMainPrice.UmpPriceFiter
        :return: AbuUmpMainPrice.UmpPriceFiter
        """
        return UmpPriceFiter

    @classmethod
    def class_unique_id(cls):
        """
        具体ump类关键字唯一名称，类方法：return 'price_main'
        主要针对外部user设置自定义ump使用, 需要user自己保证class_unique_id的唯一性，内部不做检测
        具体使用见ABuUmpManager中extend_ump_block方法
        """
        return 'price_main'
