# -*- encoding:utf-8 -*-
"""封装AbuML为业务逻辑层进行规范模块"""

from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

from abc import ABCMeta, abstractmethod

import pandas as pd
import numpy as np

from .ABuML import AbuML
from ..CoreBu.ABuFixes import six
from ..CoreBu import ABuEnv
from ..MarketBu import ABuSymbolPd
from ..IndicatorBu import ABuNDMa
from ..UtilBu import ABuScalerUtil

__author__ = '阿布'
__weixin__ = 'abu_quant'


class AbuMLPd(six.with_metaclass(ABCMeta, object)):
    """封装AbuML的上层具体业务逻辑类"""

    def __init__(self, **kwarg):
        """
        从kwarg中输入数据或者，make_xy中本身生产数据，在做完
        make_xy之后，类变量中一定要有x，y和df，使用AbuML继续
        构造self.fiter
        :param kwarg: 直接透传给make_xy的关键子参数，没有必须的参数
        """
        self.make_xy(**kwarg)
        if not hasattr(self, 'x') or not hasattr(self, 'y') \
                or not hasattr(self, 'df'):
            raise ValueError('make_xy failed! x, y not exist!')
        # noinspection PyUnresolvedReferences
        self.fiter = AbuML(self.x, self.y, self.df)

    @abstractmethod
    def make_xy(self, **kwarg):
        """
        子类需要完成的abstractmethod方法，可以从**kwarg中得到数据
        或者make_xy中本身生产数据，但在make_xy之后，类变量中一定要有
        x，y和df
        """
        pass

    def __getattr__(self, item):
        """
        使用ABuML对象self.fiter做为方法代理:
            return getattr(self.fiter, item)
        即AbuMLPd中可以使用ABuML类对象中任何方法
        """
        if item.startswith('__'):
            # noinspection PyUnresolvedReferences
            return super().__getattr__(item)
        return getattr(self.fiter, item)

    def __call__(self):
        """
        方便外面直接call，不用每次去get
        :return: self.fiter
        """
        return self.fiter


# noinspection PyAttributeOutsideInit
class ClosePredict(AbuMLPd):
    """
        示例AbuMLPd基本使用:

        获取usTSLA的沙盒测试数据，将收盘价格做为y，
        开盘，最高，最低，昨收，周几组成x矩阵，通过
        训练，预测收盘价格
    """

    def make_xy(self, **kwarg):
        """
            make_xy中读取usTSLA金融时间序列数据，使用'open', 'high', 'low', 'pre_close', 'date_week'
            做为特征列x，close即收盘价格为y，更多AbuMLPd使用阅读AbuUmpMainDeg等ump类实行
        """

        # 从沙盒中读取测试数据
        ABuEnv.enable_example_env_ipython()
        tsla = ABuSymbolPd.make_kl_df('usTSLA')
        ABuEnv.disable_example_env_ipython()

        # 留五个做为测试，其它的都做训练
        train_df = tsla[:-5]
        # make_xy中需要确定self.df
        self.df = train_df.filter(['close', 'open', 'high', 'low', 'pre_close', 'date_week'])
        tsla_matrix = self.df.as_matrix()
        # close列做为y，make_xy中需要确定self.y
        self.y = tsla_matrix[:, 0]
        # 'open', 'high', 'low', 'pre_close', 'date_week'做为x
        self.x = tsla_matrix[:, 1:]

        # 最后5个交易日做为测试数据, 只做为AbuMLPd使用示例
        test_df = tsla[-5:]
        tsla_matrix = test_df.filter(['close', 'open', 'high', 'low', 'pre_close', 'date_week']).as_matrix()
        self.y_test = tsla_matrix[:, 0]
        self.x_test = tsla_matrix[:, 1:]


def test_close_predict():
    """
        示例通过ClosePredict以及AbuMLPd的使用:
        eg:
            from abupy.MLBu.ABuMLPd import test_close_predict
            test_close_predict()
    """

    close_predict = ClosePredict()
    # ClosePredict中的数据为连续数据，属于回归，ABuML中自动会使用回归器，使用adaboost_regressor_best
    close_predict.adaboost_regressor_best()
    """
        AdaBoostRegressor(base_estimator=DecisionTreeRegressor(criterion='mse', max_depth=None, max_features=None,
           max_leaf_nodes=None, min_impurity_split=1e-07,
           min_samples_leaf=1, min_samples_split=2,
           min_weight_fraction_leaf=0.0, presort=False, random_state=None,
           splitter='best'),
         learning_rate=1.0, loss='linear', n_estimators=450,
         random_state=None)
    """
    # 可以使用ABuML中的所有方法，因为AbuMLPd使用__getattr__做了方法代理
    close_predict.plot_decision_function()
    close_predict.plot_graphviz_tree()
    # 训练数据
    close_predict.fit()

    for test_ind in np.arange(0, 5):
        predict = close_predict.predict(close_predict.x_test[test_ind])
        print('predict close is {:.3f}, actual close is {:.3f}'.format(predict[0], close_predict.y_test[test_ind]))
        """
            predict close is 228.100, actual close is 228.360
            predict close is 223.430, actual close is 220.500
            predict close is 220.690, actual close is 222.270
            predict close is 228.920, actual close is 230.010
            predict close is 228.280, actual close is 225.930
        """

    close_predict.feature_selection()
    """
        eg:
            RFE selection
                   ranking support
        open             2   False
        high             1    True
        low              1    True
        pre_close        3   False
        date_week        4   False
    """


# noinspection PyAttributeOutsideInit,PyUnresolvedReferences,PyTypeChecker
class BtcBigWaveClf(AbuMLPd):
    """
        任何大的决策其实都是由很多看极起来极不起眼的小事组成的，如果我们是做比特币日内的交易者，首先你需要判断今天适不适合做交易，
        做出这个判断的依据里有一条即是今天的波动需要足够大
    """

    def __init__(self, **kwarg):
        """
            如果传递了btc数据，说明不是运行：
            12-机器学习与比特币示例(ABU量化使用文档)
        """
        self.btc = kwarg.pop('btc', None)
        super(BtcBigWaveClf, self).__init__(**kwarg)

    def make_xy(self, **kwarg):
        if self.btc is None:
            # 从沙盒中读取测试数据
            ABuEnv.enable_example_env_ipython()
            btc = ABuSymbolPd.make_kl_df('btc', start='2013-09-01', end='2017-07-26')
            ABuEnv.disable_example_env_ipython()
        else:
            btc = self.btc
        # .055的日震荡幅度可以成做大波动的交易对比特币来说，下面对数据添加新列big_wave
        btc['big_wave'] = (btc.high - btc.low) / btc.pre_close > 0.055
        btc['big_wave'] = btc['big_wave'].astype(int)

        if self.btc is None:
            # 如果是12-机器学习与比特币示例(ABU量化使用文档)，保留60天数据
            # 首先切割训练集和测试集，保留最后60天走势数据做为测试集数据
            btc_train_raw = btc[:-60]
            btc_test_raw = btc[-60:]
        else:
            btc_train_raw = btc
            btc_test_raw = None

        # 下面为训练集和测试集数据都加上5，10，21，60日均线特征
        def calc_ma(tc, p_ma):
            ma_key = 'p_ma{}'.format(p_ma)
            tc[ma_key] = ABuNDMa.calc_ma_from_prices(tc.close, p_ma, min_periods=1)

        for ma in [5, 10, 21, 60]:
            calc_ma(btc_train_raw, ma)
            if btc_test_raw is not None:
                calc_ma(btc_test_raw, ma)
        # 下面使用训练集数据btc_train_raw做为参数抽取组合特征，重新组合好的特征
        btc_train0 = self.btc_siblings_df(btc_train_raw)
        # 由于每3条连续交易日数据组合成一个特征，只要向前跳一条数据进行特征组合抽取即可以得到另一组新特征
        btc_train1 = self.btc_siblings_df(btc_train_raw[1:])
        btc_train2 = self.btc_siblings_df(btc_train_raw[2:])

        # 把周几这个特征使用pd.get_dummies进行离散化处理，使得所有特征值的范围都在0-1之间
        btc_train = pd.concat([btc_train0, btc_train1, btc_train2])
        btc_train.index = np.arange(0, btc_train.shape[0])
        dummies_one_week = pd.get_dummies(btc_train['one_date_week'], prefix='one_date_week')
        dummies_two_week = pd.get_dummies(btc_train['two_date_week'], prefix='two_date_week')
        dummies_today_week = pd.get_dummies(btc_train['today_date_week'], prefix='today_date_week')
        btc_train.drop(['one_date_week', 'two_date_week', 'today_date_week'], inplace=True, axis=1)
        # make_xy中需要确定self.df
        self.df = pd.concat([btc_train, dummies_one_week, dummies_two_week, dummies_today_week], axis=1)
        # make_xy中需要确定x, y
        train_matrix = self.df.as_matrix()
        self.y = train_matrix[:, 0]
        self.x = train_matrix[:, 1:]

        if btc_test_raw is not None:
            # 下面将前面保留切割的60条测试数据进行特征抽取组合，方式和抽取训练集时一样
            btc_test0 = self.btc_siblings_df(btc_test_raw)
            btc_test1 = self.btc_siblings_df(btc_test_raw[1:])
            btc_test2 = self.btc_siblings_df(btc_test_raw[2:])
            btc_test = pd.concat([btc_test0, btc_test1, btc_test2])
            btc_test.index = np.arange(0, btc_test.shape[0])
            dummies_one_week = pd.get_dummies(btc_test['one_date_week'], prefix='one_date_week')
            dummies_two_week = pd.get_dummies(btc_test['two_date_week'], prefix='two_date_week')
            dummies_today_week = pd.get_dummies(btc_test['today_date_week'], prefix='today_date_week')
            btc_test.drop(['one_date_week', 'two_date_week', 'today_date_week'], inplace=True, axis=1)
            self.btc_test = pd.concat([btc_test, dummies_one_week, dummies_two_week, dummies_today_week], axis=1)
            # 测试集数据构建
            matrix_test = self.btc_test.as_matrix()
            self.y_test = matrix_test[:, 0]
            self.x_test = matrix_test[:, 1:]

    # noinspection PyMethodMayBeStatic
    def btc_siblings_df(self, btc_raw):
        """
        * 首先将所有交易日以3个为一组，切割成多个子df，即每一个子df中有3个交易日的交易数据
        * 使用数据标准化将连续3天交易日中的连续数值特征进行标准化操作
        * 抽取第一天，第二天的大多数特征分别改名字以one，two为特征前缀，如：one_open，one_close，two_ma5，two_high.....,
        * 第三天的特征只使用'open', 'low', 'pre_close', 'date_week'，该名前缀today，如today_open，today_date_week
        * 第三天的抽取了'big_wave'，其将在之后做为y
        * 将抽取改名字后的特征连接起来组合成为一条新数据，即3天的交易数据特征－>1条新的数据

        :param btc_raw: btc走势数据，pd.DataFrame对象
        :return: 重新组合好的特征数据，pd.DataFrame对象
        """

        # 将所有交易日以3个为一组，切割成多个子df，即每一个子df中有3个交易日的交易数据
        btc_siblings = [btc_raw.iloc[sib_ind * 3:(sib_ind + 1) * 3, :]
                        for sib_ind in np.arange(0, int(btc_raw.shape[0] / 3))]

        btc_df = pd.DataFrame()
        for sib_btc in btc_siblings:
            # 使用数据标准化将连续3天交易日中的连续数值特征进行标准化操作
            sib_btc_scale = ABuScalerUtil.scaler_std(
                sib_btc.filter(['open', 'close', 'high', 'low', 'volume', 'pre_close',
                                'ma5', 'ma10', 'ma21', 'ma60', 'atr21', 'atr14']))
            # 把标准化后的和big_wave，date_week连接起来
            sib_btc_scale = pd.concat([sib_btc['big_wave'], sib_btc_scale, sib_btc['date_week']], axis=1)

            # 抽取第一天，第二天的大多数特征分别改名字以one，two为特征前缀，如：one_open，one_close，two_ma5，two_high.....
            a0 = sib_btc_scale.iloc[0].filter(['open', 'close', 'high', 'low', 'volume', 'pre_close',
                                               'ma5', 'ma10', 'ma21', 'ma60', 'atr21', 'atr14', 'date_week'])
            a0.rename(index={'open': 'one_open', 'close': 'one_close', 'high': 'one_high', 'low': 'one_low',
                             'volume': 'one_volume', 'pre_close': 'one_pre_close',
                             'ma5': 'one_ma5', 'ma10': 'one_ma10', 'ma21': 'one_ma21',
                             'ma60': 'one_ma60', 'atr21': 'one_atr21', 'atr14': 'one_atr14',
                             'date_week': 'one_date_week'}, inplace=True)

            a1 = sib_btc_scale.iloc[1].filter(['open', 'close', 'high', 'low', 'volume', 'pre_close',
                                               'ma5', 'ma10', 'ma21', 'ma60', 'atr21', 'atr14', 'date_week'])
            a1.rename(index={'open': 'two_open', 'close': 'two_close', 'high': 'two_high', 'low': 'two_low',
                             'volume': 'two_volume', 'pre_close': 'two_pre_close',
                             'ma5': 'two_ma5', 'ma10': 'two_ma10', 'ma21': 'two_ma21',
                             'ma60': 'two_ma60', 'atr21': 'two_atr21', 'atr14': 'two_atr14',
                             'date_week': 'two_date_week'}, inplace=True)
            # 第三天的特征只使用'open', 'low', 'pre_close', 'date_week'，该名前缀today，如today_open，today_date_week
            a2 = sib_btc_scale.iloc[2].filter(['big_wave', 'open', 'low', 'pre_close', 'date_week'])
            a2.rename(index={'open': 'today_open', 'low': 'today_low',
                             'pre_close': 'today_pre_close',
                             'date_week': 'today_date_week'}, inplace=True)
            # 将抽取改名字后的特征连接起来组合成为一条新数据，即3天的交易数据特征－>1条新的数据
            btc_df = btc_df.append(pd.concat([a0, a1, a2], axis=0), ignore_index=True)
        return btc_df
