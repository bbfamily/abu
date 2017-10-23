# -*- encoding:utf-8 -*-
"""示例ump边裁特征价格模块"""

from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

from ..TradeBu.ABuMLFeature import AbuFeaturePrice
from ..MLBu.ABuMLPd import AbuMLPd
from .ABuUmpBase import ump_edge_make_xy, BuyUmpMixin
from .ABuUmpEdgeBase import AbuUmpEdgeBase

__author__ = '阿布'
__weixin__ = 'abu_quant'


class UmpPriceFiter(AbuMLPd):
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
        :return: self.order_has_ret中使用filter选取特征列形成price_df
                 ump_edge_make_xy装饰器在make_xy返回price_df后做转换matrix，形成x，y等工作
        """

        filter_list = ['profit', 'profit_cg']
        cols = AbuFeaturePrice().get_feature_ump_keys(ump_cls=AbuUmpEdgePrice)
        # ['profit', 'profit_cg', 'buy_price_rank120', 'buy_price_rank90', 'buy_price_rank60', 'buy_price_rank252']
        filter_list.extend(cols)

        # noinspection PyUnresolvedReferences
        price_df = self.order_has_ret.filter(filter_list)
        """
            eg: price_df
                           profit  profit_cg  buy_price_rank120  buy_price_rank90  \
            2014-09-24  -22618.04    -0.0566              1.000             1.000
            2014-10-24  -29690.28    -0.0742              1.000             1.000
            2014-10-29   18959.19     0.0542              1.000             1.000
            2014-10-29  148209.36     0.5022              0.925             0.900
            2014-10-29   24867.60     0.0952              0.962             0.950
            2014-10-29   18959.19     0.0542              1.000             1.000
            2014-11-03    1250.80     0.0045              1.000             1.000
            2014-11-11   59888.21     0.1857              0.954             0.939
            2014-11-12   -3578.78    -0.0140              0.475             0.522
            2014-11-26  -29085.19    -0.0946              0.642             0.733
            ...               ...        ...                ...               ...
            2016-03-14   16220.57     0.0559              0.617             0.500
            2016-03-14  -25328.12    -0.1218              0.683             0.589
            2016-03-30  -29858.44    -0.0863              0.658             0.667
            2016-04-04    5373.76     0.0244              0.400             0.511
            2016-04-13  -28044.40    -0.1159              0.567             0.722
            2016-04-14  -18645.93    -0.0467              0.875             0.878
            2016-04-15  -32484.79    -0.1149              0.775             0.733
            2016-04-15  -32484.79    -0.1149              0.775             0.733
            2016-04-29     290.96     0.0007              1.000             1.000
            2016-04-29     290.96     0.0007              1.000             1.000

                        buy_price_rank60  buy_price_rank252
            2014-09-24             1.000              1.000
            2014-10-24             1.000              1.000
            2014-10-29             1.000              1.000
            2014-10-29             0.883              0.750
            2014-10-29             0.925              0.982
            2014-10-29             1.000              1.000
            2014-11-03             1.000              1.000
            2014-11-11             0.992              0.808
            2014-11-12             0.783              0.560
            2014-11-26             1.000              0.762
            ...                      ...                ...
            2016-03-14             0.750              0.444
            2016-03-14             0.850              0.623
            2016-03-30             1.000              0.536
            2016-04-04             0.767              0.190
            2016-04-13             1.000              0.270
            2016-04-14             0.967              0.940
            2016-04-15             1.000              0.631
            2016-04-15             1.000              0.631
            2016-04-29             1.000              1.000
            2016-04-29             1.000              1.000
        """

        return price_df


class AbuUmpEdgePrice(AbuUmpEdgeBase, BuyUmpMixin):
    """边裁价格特征类，AbuUmpEdgeBase子类，混入BuyUmpMixin，做为买入ump类"""

    def get_predict_col(self):
        """
        边裁价格特征keys：['buy_price_rank120', 'buy_price_rank90', 'buy_price_rank60', 'buy_price_rank252']
        :return: ['buy_price_rank120', 'buy_price_rank90', 'buy_price_rank60', 'buy_price_rank252']
        """
        return AbuFeaturePrice().get_feature_ump_keys(ump_cls=AbuUmpEdgePrice)

    def get_fiter_class(self):
        """
        边裁价格特征返回的AbuMLPd子类：AbuUmpEdgePrice.UmpPriceFiter
        :return: AbuUmpEdgePrice.UmpPriceFiter
        """
        return UmpPriceFiter

    @classmethod
    def class_unique_id(cls):
        """
         具体ump类关键字唯一名称，类方法：return 'price_edge'
         主要针对外部user设置自定义ump使用, 需要user自己保证class_unique_id的唯一性，内部不做检测
         具体使用见ABuUmpManager中extend_ump_block方法
        """
        return 'price_edge'
