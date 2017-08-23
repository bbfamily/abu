# coding=utf-8
"""
    数据源基础模块
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from abc import ABCMeta, abstractmethod

from ..MarketBu.ABuSymbol import Symbol
from ..CoreBu.ABuEnv import EMarketTargetType
from ..CoreBu.ABuFixes import six
from ..UtilBu import ABuDateUtil


# noinspection PyUnresolvedReferences
class SupportMixin(object):
    """混入类，声明数据源支持的市场，以及检测是否支持市场"""

    def _support_market(self):
        """声明数据源支持的市场，默认声明支持美股，港股，a股"""
        return [EMarketTargetType.E_MARKET_TARGET_US, EMarketTargetType.E_MARKET_TARGET_HK,
                EMarketTargetType.E_MARKET_TARGET_CN]

    def check_support(self, symbol=None, rs=True):
        """
        检测参数symbol对象或者内部self._symbol是否被数据源支持
        :param symbol: 外部可设置检测symbol对象，Symbol对象，EMarketTargetType对象或字符串对象
        :param rs: 如果数据源不支持，是否抛出异常，默认抛出
        :return: 返回是否支持 bool
        """
        if symbol is None:
            symbol = self._symbol

        if isinstance(symbol, six.string_types):
            # 如果是str，使用_support_market返回的value组成字符串数组，进行成员测试
            if symbol in [market.value for market in self._support_market()]:
                return True
        else:
            if isinstance(symbol, Symbol):
                # Symbol对象取market
                market = symbol.market
            elif isinstance(symbol, EMarketTargetType):
                market = symbol
            else:
                raise TypeError('symbol type is Symbol or str!!')
            # _support_market序列进行成员测试
            if market in self._support_market():
                return True

        if rs:
            #  根据rs设置，如果数据源不支持，抛出异常
            raise TypeError('{} don\'t support {}!'.format(self.__class__.__name__, symbol))
        return False


class BaseMarket(object):
    """数据源基础市场基类"""

    # 预先设置模拟手机请求的device
    K_DEV_MODE_LIST = ["A0001", "OPPOR9", "OPPOR9", "VIVOX5",
                       "VIVOX6", "VIVOX6PLUS", "VIVOX9", "VIVOX9PLUS"]
    # 预先设置模拟手机请求的os version
    K_OS_VERSION_LIST = ["4.3", "4.2.2", "4.4.2", "5.1.1"]
    # 预先设置模拟手机请求的屏幕大小
    K_PHONE_SCREEN = [[1080, 1920]]

    def __init__(self, symbol):
        """
        :param symbol: Symbol类型对象
        """
        if not isinstance(symbol, Symbol):
            raise TypeError('symbol is not type Symbol')

        self._symbol = symbol

    # noinspection PyMethodMayBeStatic
    def req_time(self):
        """请求时间seconds模拟"""
        tm = int(ABuDateUtil.time_seconds() * 1000)
        return tm

    @classmethod
    def _fix_kline_pd_se(cls, kl_df, n_folds, start=None, end=None):
        """
        删除多余请求数据，即重新根据start，end或n_folds参数进行金融时间序列切割
        :param kl_df: 金融时间序列切割pd.DataFrame对象
        :param n_folds: n_folds年的数据
        :return: 删除多余数据，规则后的pd.DataFrame对象
        """
        if kl_df is None:
            return kl_df

        # 从start和end中切片
        if start is not None:
            # 有start转换为int，使用kl_df中的date列进行筛选切割
            start = ABuDateUtil.date_str_to_int(start)
            kl_df = kl_df[kl_df.date >= start]
            if end is not None:
                # 有end转换为int，使用kl_df中的date列进行筛选切割
                end = ABuDateUtil.date_str_to_int(end)
                kl_df = kl_df[kl_df.date <= end]
        else:
            # 根据n_folds构造切片的start
            start = ABuDateUtil.begin_date(365 * n_folds)
            start = ABuDateUtil.date_str_to_int(start)
            # 使用kl_df中的date列进行筛选切割
            kl_df = kl_df[kl_df.date >= start]
        return kl_df

    @classmethod
    def _fix_kline_pd_zero(cls, kl_df):
        """
        修复金融时间序列切中的异常点，比如价格为0的点，注意只能用在确定会有异常点发生的市场，
        比如期权期货市场进入交割日，否则不能随意进行修复
        :param kl_df: 金融时间序列切割pd.DataFrame对象
        :return: 修复后的金融时间序列pd.DataFrame对象
        """

        if kl_df is None:
            return kl_df

        def fix_zero(trade_day):
            """对高开低收为0的价格认为是异常，使用昨天的价格替换"""
            if trade_day.close == 0:
                # 收盘异常，如果今天的low还是>0的使用low，否则使用昨天收盘价格
                trade_day.close = trade_day.low if trade_day.low > 0 else trade_day.pre_close

            # 由于已fix了trade_day.close，所以high，low，open使用trade_day.close fix
            if trade_day.high == 0:
                trade_day.high = trade_day.close
            if trade_day.low == 0:
                trade_day.low = trade_day.close
            if trade_day.open == 0:
                trade_day.open = trade_day.close
            return trade_day

        kl_df = kl_df.apply(fix_zero, axis=1)

        return kl_df


class StockBaseMarket(six.with_metaclass(ABCMeta, BaseMarket)):
    """基于股票类型的数据源抽象基类"""

    @abstractmethod
    def minute(self, *args, **kwargs):
        """分钟k线接口"""
        pass

    @abstractmethod
    def kline(self, n_folds=2, start=None, end=None):
        """日k线接口"""
        pass

    @classmethod
    def _fix_kline_pd(cls, kl_df, n_folds, start=None, end=None):
        """修复kline接口的返回金融时间序列"""
        return cls._fix_kline_pd_se(kl_df, n_folds, start=start, end=end)


class FuturesBaseMarket(six.with_metaclass(ABCMeta, BaseMarket)):
    """基于期货类型的数据源抽象基类"""

    @abstractmethod
    def kline(self, n_folds=2, start=None, end=None):
        """日k线接口"""
        pass

    @classmethod
    def _fix_kline_pd(cls, kl_df, n_folds, start=None, end=None):
        """修复kline接口的返回金融时间序列"""
        kl_df = cls._fix_kline_pd_se(kl_df, n_folds, start=start, end=end)
        # 期货数据要修复交割0 close的bar
        return cls._fix_kline_pd_zero(kl_df)


class TCBaseMarket(six.with_metaclass(ABCMeta, BaseMarket)):
    """基于比特币，莱特币等类型的数据源抽象基类"""

    @abstractmethod
    def kline(self, n_folds=2, start=None, end=None):
        """日k线接口"""
        pass

    @abstractmethod
    def minute(self, *args, **kwargs):
        """比特币量化日内短线频繁，需要定制自己的日内策略"""
        pass

    @classmethod
    def _fix_kline_pd(cls, kl_df, n_folds, start=None, end=None):
        """修复kline接口的返回金融时间序列"""
        return cls._fix_kline_pd_se(kl_df, n_folds, start=start, end=end)
