# coding=utf-8
"""
    数据源解析模块以及示例内置数据源的解析类实现
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import logging

import numpy as np
import pandas as pd

from .ABuSymbol import EMarketTargetType
from ..CoreBu.ABuFixes import six
from ..UtilBu import ABuDateUtil

__author__ = '阿布'
__weixin__ = 'abu_quant'


def del_columns(df, columns):
    """
    从df中删除参数columns指定的整列数据
    :param df: 金融时间序列切割pd.DataFrame对象
    :param columns: 可迭代的字符序列，代表需要删除的指定列
    :return:
    """
    old_c = df.columns.tolist()
    for col in filter(lambda x: x in old_c, columns):
        df.drop(col, axis=1, inplace=True)


class AbuDataParseWrap(object):
    """
        做为类装饰器封装替换解析数据统一操作，装饰替换init
    """

    def __call__(self, cls):
        """只做为数据源解析类的装饰器，统一封装通用的数据解析规范及流程"""
        if isinstance(cls, six.class_types):
            # 只做为类装饰器使用
            init = cls.__init__

            def wrapped(*args, **kwargs):
                try:
                    # 拿出被装饰的self对象
                    warp_self = args[0]
                    warp_self.df = None
                    # 调用原始init
                    init(*args, **kwargs)
                    symbol = args[1]
                    # 开始数据解析
                    self._gen_warp_df(warp_self, symbol)
                except Exception as e:
                    logging.exception(e)

            # 使用wrapped替换原始init
            cls.__init__ = wrapped

            wrapped.__name__ = '__init__'
            # 将原始的init赋予deprecated_original，必须要使用这个属性名字，在其它地方，如AbuParamBase会寻找原始方法找它
            wrapped.deprecated_original = init
            return cls
        else:
            raise TypeError('AbuDataParseWrap just for class warp')

    # noinspection PyMethodMayBeStatic
    def _gen_warp_df(self, warp_self, symbol):
        """
        封装通用的数据解析规范及流程
        :param warp_self: 被封装类init中使用的self对象
        :param symbol: 请求的symbol str对象
        :return:
        """

        # 规范原始init函数中必须为类添加了如下属性
        must_col = ['open', 'close', 'high', 'low', 'volume', 'date']
        # 检测所有的属性都有
        all_has = all([hasattr(warp_self, col) for col in must_col])
        # raise RuntimeError('df.columns must have |date|open|close|high|volume| ')
        if all_has:
            # 将时间序列转换为pd时间
            dates_pd = pd.to_datetime(warp_self.date)
            # 构建df，index使用dates_pd
            warp_self.df = pd.DataFrame(index=dates_pd)
            for col in must_col:
                # 所以必须有的类属性序列设置给df的列
                warp_self.df[col] = getattr(warp_self, col)

            # 从收盘价格序列shift出昨收价格序列
            warp_self.df['pre_close'] = warp_self.df['close'].shift(1)
            warp_self.df['pre_close'].fillna(warp_self.df['open'], axis=0, inplace=True)
            # 添加日期int列
            warp_self.df['date'] = warp_self.df['date'].apply(lambda x: ABuDateUtil.date_str_to_int(str(x)))
            # 添加周几列date_week，值为0-4，分别代表周一到周五
            warp_self.df['date_week'] = warp_self.df['date'].apply(
                lambda x: ABuDateUtil.week_of_date(str(x), '%Y%m%d'))

            # 类型转换
            warp_self.df['close'] = warp_self.df['close'].astype(float)
            warp_self.df['high'] = warp_self.df['high'].astype(float)
            warp_self.df['low'] = warp_self.df['low'].astype(float)
            warp_self.df['open'] = warp_self.df['open'].astype(float)
            warp_self.df['volume'] = warp_self.df['volume'].astype(float)
            warp_self.df['volume'] = warp_self.df['volume'].astype(np.int64)
            warp_self.df['date'] = warp_self.df['date'].astype(int)
            warp_self.df['pre_close'] = warp_self.df['pre_close'].astype(float)
            # 不使用df['close'].pct_change计算
            # noinspection PyTypeChecker
            warp_self.df['p_change'] = np.where(warp_self.df['pre_close'] == 0, 0,
                                                (warp_self.df['close'] - warp_self.df['pre_close']) / warp_self.df[
                                                    'pre_close'] * 100)

            warp_self.df['p_change'] = warp_self.df['p_change'].apply(lambda x: round(x, 3))
            # 给df加上name
            warp_self.df.name = symbol


@AbuDataParseWrap()
class TXParser(object):
    """tx数据源解析类，被类装饰器AbuDataParseWrap装饰"""

    def __init__(self, symbol, sub_market, json_dict):
        """
        :param symbol: 请求的symbol str对象
        :param sub_market: 子市场（交易所）类型
        :param json_dict: 请求返回的json数据
        """
        if json_dict['code'] == 0:
            if symbol.market == EMarketTargetType.E_MARKET_TARGET_US:
                data = json_dict['data'][symbol.value + sub_market]
            else:
                data = json_dict['data'][symbol.value]

            if 'qfqday' in data.keys():
                data = data['qfqday']
            else:
                data = data['day']

            # 为AbuDataParseWrap准备类必须的属性序列
            if len(data) > 0:
                # 时间日期序列，时间格式为2017-07-26格式字符串
                self.date = [item[0] for item in data]
                # 开盘价格序列
                self.open = [item[1] for item in data]
                # 收盘价格序列
                self.close = [item[2] for item in data]
                # 最高价格序列
                self.high = [item[3] for item in data]
                # 最低价格序列
                self.low = [item[4] for item in data]
                # 成交量序列
                self.volume = [item[5] for item in data]


@AbuDataParseWrap()
class NTParser(object):
    """nt数据源解析类，被类装饰器AbuDataParseWrap装饰"""

    # noinspection PyUnusedLocal
    def __init__(self, symbol, json_dict):
        """
        :param symbol: 请求的symbol str对象
        :param json_dict: 请求返回的json数据
        """
        data = json_dict['data']
        # 为AbuDataParseWrap准备类必须的属性序列
        if len(data) > 0:
            # 时间日期序列
            self.date = [item[0] for item in data]
            # 开盘价格序列
            self.open = [item[1] for item in data]
            # 收盘价格序列
            self.close = [item[2] for item in data]
            # 最高价格序列
            self.high = [item[3] for item in data]
            # 最低价格序列
            self.low = [item[4] for item in data]
            # 成交量序列
            self.volume = [item[5] for item in data]


@AbuDataParseWrap()
class SNUSParser(object):
    """snus数据源解析类，被类装饰器AbuDataParseWrap装饰"""

    # noinspection PyUnusedLocal
    def __init__(self, symbol, json_dict):
        """
        :param symbol: 请求的symbol str对象
        :param json_dict: 请求返回的json数据
        """
        data = json_dict
        # 为AbuDataParseWrap准备类必须的属性序列
        if len(data) > 0:
            # 时间日期序列
            self.date = [item['d'] for item in data]
            # 开盘价格序列
            self.open = [item['o'] for item in data]
            # 收盘价格序列
            self.close = [item['c'] for item in data]
            # 最高价格序列
            self.high = [item['h'] for item in data]
            # 最低价格序列
            self.low = [item['l'] for item in data]
            # 成交量序列
            self.volume = [item['v'] for item in data]


@AbuDataParseWrap()
class SNFuturesParser(object):
    """示例期货数据源解析类，被类装饰器AbuDataParseWrap装饰"""

    # noinspection PyUnusedLocal
    def __init__(self, symbol, json_dict):
        """
        :param symbol: 请求的symbol str对象
        :param json_dict: 请求返回的json数据
        """
        data = json_dict
        # 为AbuDataParseWrap准备类必须的属性序列
        if len(data) > 0:
            # 时间日期序列
            self.date = [item[0] for item in data]
            # 开盘价格序列
            self.open = [item[1] for item in data]
            # 最高价格序列
            self.high = [item[2] for item in data]
            # 最低价格序列
            self.low = [item[3] for item in data]
            # 收盘价格序列
            self.close = [item[4] for item in data]
            # 成交量序列
            self.volume = [item[5] for item in data]


@AbuDataParseWrap()
class SNFuturesGBParser(object):
    """示例国际期货数据源解析类，被类装饰器AbuDataParseWrap装饰"""
    # noinspection PyUnusedLocal
    def __init__(self, symbol, json_dict):
        """
        :param symbol: 请求的symbol str对象
        :param json_dict: 请求返回的json或者dict数据
        """
        data = json_dict
        # 为AbuDataParseWrap准备类必须的属性序列
        if len(data) > 0:
            # 时间日期序列
            self.date = [item['date'] for item in data]
            # 开盘价格序列
            self.open = [item['open'] for item in data]
            # 最高价格序列
            self.high = [item['high'] for item in data]
            # 最低价格序列
            self.low = [item['low'] for item in data]
            # 收盘价格序列
            self.close = [item['close'] for item in data]
            # 成交量序列
            self.volume = [item['volume'] for item in data]


@AbuDataParseWrap()
class HBTCParser(object):
    """示例币类市场数据源解析类，被类装饰器AbuDataParseWrap装饰"""

    # noinspection PyUnusedLocal
    def __init__(self, symbol, json_dict):
        """
        :param symbol: 请求的symbol str对象
        :param json_dict: 请求返回的json数据
        """

        data = json_dict
        # 为AbuDataParseWrap准备类必须的属性序列
        if len(data) > 0:
            # 时间日期序列
            self.date = [item[0] for item in data]
            # 开盘价格序列
            self.open = [item[1] for item in data]
            # 最高价格序列
            self.high = [item[2] for item in data]
            # 最低价格序列
            self.low = [item[3] for item in data]
            # 收盘价格序列
            self.close = [item[4] for item in data]
            # 成交量序列
            self.volume = [item[5] for item in data]

            # 时间日期进行格式转化，转化为如2017-07-26格式字符串
            self.date = list(map(lambda date: ABuDateUtil.fmt_date(date), self.date))


class BDParser(object):
    """bd数据源解析类"""

    data_keys = ['data', 'dataMash']
    s_calc_dm = True

    def __init__(self, symbol, json_dict):
        """
        没有使用AbuDataParseWrap装饰类，保留一个原始的解析流程类，
        其它的解析类都使用AbuDataParseWrap装饰类，解析过程不做多注解，
        详阅读AbuDataParseWrap的实现
        :param symbol: 请求的symbol str对象
        :param json_dict: 请求返回的json数据
        """
        try:
            if BDParser.data_keys[0] in json_dict.keys():
                self.data = json_dict[BDParser.data_keys[0]][::-1]
            elif BDParser.data_keys[1] in json_dict.keys():
                self.data = json_dict[BDParser.data_keys[1]][::-1]
            else:
                raise ValueError('content not json format')

            dates = [mash['date'] for mash in self.data]
            klines = [mash['kline'] for mash in self.data]

            self.df = None
            if len(klines) > 0 and len(dates) > 0:
                dates_fmt = list(map(lambda date: ABuDateUtil.fmt_date(date), dates))
                dates_pd = pd.to_datetime(dates_fmt)

                self.df = pd.DataFrame(klines, index=dates_pd)
                self.df['date'] = dates
                self.df['date_week'] = self.df['date'].apply(lambda x: ABuDateUtil.week_of_date(str(x), '%Y%m%d'))

                self.df['close'] = self.df['close'].astype(float)
                self.df['high'] = self.df['high'].astype(float)
                self.df['low'] = self.df['low'].astype(float)
                self.df['open'] = self.df['open'].astype(float)
                self.df['volume'] = self.df['volume'].astype(np.int64)
                self.df['date'] = self.df['date'].astype(int)
                self.df['netChangeRatio'] = self.df['netChangeRatio'].map(lambda x: x[:-1]).astype(float)
                self.df['preClose'] = self.df['preClose'].astype(float)
                self.df.rename(columns={'preClose': 'pre_close', 'netChangeRatio': 'p_change'}, inplace=True)
                del_columns(self.df, ['amount'])
                if BDParser.s_calc_dm:
                    self.df['pre_close'] = self.df['close'].shift(1)
                    self.df['pre_close'].fillna(self.df['open'], axis=0, inplace=True)
                    # 不使用df['close'].pct_change计算
                    # noinspection PyTypeChecker
                    self.df['p_change'] = np.where(self.df['pre_close'] == 0, 0,
                                                   (self.df['close'] - self.df['pre_close']) / self.df[
                                                       'pre_close'] * 100)
                    self.df['p_change'] = self.df['p_change'].apply(lambda x: round(x, 3))

                self.df.name = symbol

        except Exception as e:
            logging.exception(e)
