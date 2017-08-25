# coding=utf-8
"""
    数据源模块
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import logging

import numpy as np

from ..MarketBu.ABuDataBase import BaseMarket
from ..MarketBu.ABuDataFeed import BDApi, TXApi, NTApi, HBApi, SNUSApi, SNFuturesApi, SNFuturesGBApi
from .ABuSymbol import Symbol
from .ABuSymbol import code_to_symbol
from ..CoreBu import ABuEnv
from ..CoreBu.ABuFixes import six
from ..CoreBu.ABuEnv import EMarketDataSplitMode, EMarketDataFetchMode
from ..CoreBu.ABuEnv import EMarketSourceType
from ..MarketBu.ABuDataCache import load_kline_df, load_kline_df_net
from ..UtilBu import ABuDateUtil

try:
    from tables import HDF5ExtError
except ImportError:
    class HDF5ExtError(RuntimeError):
        """如果没有HDF5环境只能使用其它存贮模式"""
        pass


"""内置数据源source_dict"""
source_dict = {EMarketSourceType.E_MARKET_SOURCE_bd.value: BDApi,
               EMarketSourceType.E_MARKET_SOURCE_tx.value: TXApi,
               EMarketSourceType.E_MARKET_SOURCE_nt.value: NTApi,
               EMarketSourceType.E_MARKET_SOURCE_sn_us.value: SNUSApi,
               EMarketSourceType.E_MARKET_SOURCE_sn_futures.value: SNFuturesApi,
               EMarketSourceType.E_MARKET_SOURCE_sn_futures_gb.value: SNFuturesGBApi,
               EMarketSourceType.E_MARKET_SOURCE_hb_tc.value: HBApi}


def _calc_start_end_date(df, force_local, n_folds, start, end):
    """
    根据参数计算start，end
    :param df: 本地缓存的金融时间序列对象，pd.DataFrame对象
    :param force_local: 是否强制走本地数据
    :param n_folds: 需要几年的数据
    :param start: 开始的时间
    :param end: 结束的时间
    :return:
    """

    # 当前今天时间日期str对象，如果是强制本地，即缓存的最后一个交易日
    today = ABuDateUtil.timestamp_to_str(df.index[-1]) if force_local else ABuDateUtil.current_str_date()
    if end is None:
        # 没有end也没start，end＝today，否则使用n_folds计算end
        end = today if start is None else ABuDateUtil.begin_date(-365 * n_folds, date_str=start, fix=False)
    # int类型的end, today转换
    end_int = ABuDateUtil.date_str_to_int(end)
    today_int = ABuDateUtil.date_str_to_int(today)
    if end_int > today_int:
        end_int = today_int

    if start is None:
        if force_local:
            end_ss = df[df.date >= end_int]
            if end_ss is None or end_ss.empty:
                ind = 0
            else:
                # +1补上
                ind = end_ss.key.values[0] - (ABuEnv.g_market_trade_year * n_folds) + 1
            if ind < 0:
                ind = 0
            # 强制本地，计算从第几个ind开始取df数据
            start = ABuDateUtil.timestamp_to_str(df.index[ind])
        else:
            # 非强制本地，使用n_folds年数，向前推计算start
            start = ABuDateUtil.begin_date(365 * n_folds, date_str=end, fix=False)
    start_int = ABuDateUtil.date_str_to_int(start)

    df_end_int = 0
    # 给正无穷匹配之后的start_int >= df_start_int
    df_start_int = np.inf

    if df is not None and df.shape[0] > 0:
        # 获取本地缓存df的开始，结束，字符串对象以及int值
        try:
            df_end = ABuDateUtil.timestamp_to_str(df.index[-1])
            df_end_int = ABuDateUtil.date_str_to_int(df_end)

            df_start = ABuDateUtil.timestamp_to_str(df.index[0])
            df_start_int = ABuDateUtil.date_str_to_int(df_start)
        except Exception as e:
            logging.exception(e)

    return end, end_int, df_end_int, start, start_int, df_start_int


def kline_pd(symbol, data_mode, n_folds=2, start=None, end=None, save=True):
    """
    统一调度选择内部或者外部数据源，决策是否本地数据读取，还是网络数据读取，以及根据不
    同的数据获取模式，调整数据的选择范围

    eg: n_fold=2, start=None, end=None ，从今天起往前数两年
        n_fold=2, start='2015-02-14', end=None， 从2015-02-14到现在，n_fold无效
        n_fold=2, start=None, end='2016-02-14'，从2016-02-14起往前数两年
        n_fold=2, start='2015-02-14', end='2016-02-14'，从start到end

    :param data_mode: EMarketDataSplitMode enum对象
    :param symbol: string or Symbol对象
                   e.g. 'sz300104'
                   e.g. Symbol(MType.SZ, '300104')
    :param n_folds: 年, 如果start不为空，则n_fold失效
    :param start: 开始时间 start为None时，start会根据end和n_fold计算出来，str对象
    :param end: 结束时间，str对象
    :param save: 从网络下载后是否缓存到本地
    """
    try:
        if isinstance(symbol, Symbol):
            temp_symbol = symbol
        elif isinstance(symbol, six.string_types):
            # 如果是str对象，通过code_to_symbol转化为Symbol对象
            temp_symbol = code_to_symbol(symbol)
        else:
            raise TypeError('symbol must like as "usTSLA" or "TSLA" or Symbol(MType.US, "TSLA")')
        if ABuEnv.g_private_data_source is None:
            # 如果没有设置私有数据源，使用env中设置的内置示例测试源
            source = source_dict[ABuEnv.g_market_source.value]
        else:
            # 有设置私有数据源
            source = ABuEnv.g_private_data_source
            # 私有源首先设置的需要是class类型，然后判断是BaseMarket的子类
            if not isinstance(source, six.class_types):
                raise TypeError('g_private_data_source must be a class type!!!')
            if not issubclass(ABuEnv.g_private_data_source, BaseMarket):
                raise TypeError('g_private_data_source must be a subclass of BaseMarket!!!')
        temp_symbol.source = source
        # 如果外部负责保存，就需要save_kl_key中相关信息
        save_kl_key = (temp_symbol, None, None)

        # symbol本地的pd.DataFrame数据缓存
        df = None
        # 本地的pd.DataFrame金融时间序列的第一个日期 int类型
        df_req_start = 0
        # 本地的pd.DataFrame金融时间序列的最后一个个日期 int类型
        df_req_end = 0

        if ABuEnv.g_data_fetch_mode != EMarketDataFetchMode.E_DATA_FETCH_FORCE_NET:
            # 如果env中设置并非强制从网络获取数据，就从本地数据尝试读取df, df_req_start
            df, df_req_start, df_req_end = load_kline_df(temp_symbol.value)
        # 确定env中设置是否强制从本地缓存读取数据
        force_local = (ABuEnv.g_data_fetch_mode == EMarketDataFetchMode.E_DATA_FETCH_FORCE_LOCAL)

        if force_local and df is None:
            # 如果强制本地且df是空，直接返回
            return df, save_kl_key

        if data_mode == EMarketDataSplitMode.E_DATA_SPLIT_UNDO and force_local:
            # 如果强制本地且数据模式为E_DATA_SPLIT_UNDO，即不依据参数切割df，直接返回
            return df, save_kl_key

        # 标准化输入的start时间，eg 2016-7-26 －> 2016-07-26
        start = ABuDateUtil.fix_date(start)
        # 标准化输入的end时间，eg 2016-7-26 －> 2016-07-26
        end = ABuDateUtil.fix_date(end)
        # 根据n_folds，start，end计算需要请求的start，end
        end, end_int, df_end_int, start, start_int, df_start_int = _calc_start_end_date(df, force_local, n_folds, start,
                                                                                        end)
        save_kl_key = (temp_symbol, start_int, end_int)
        if ABuEnv.g_data_fetch_mode == EMarketDataFetchMode.E_DATA_FETCH_FORCE_NET:
            # 如果是强制走网络，直接请求使用load_kline_df_net
            return load_kline_df_net(source, temp_symbol, n_folds=n_folds, start=start, end=end, start_int=start_int,
                                     end_int=end_int, save=save), save_kl_key

        # 检测本地缓存数据是否满足需要，如果需要的数据在存储的数据之间，则可切片放回
        match = False
        if start_int >= df_start_int and end_int <= df_end_int:
            match = True
        elif start_int >= df_start_int and force_local:
            match = True
        elif start_int >= df_req_start and end_int <= df_req_end:
            match = True

        if match:
            if data_mode == EMarketDataSplitMode.E_DATA_SPLIT_SE:
                # 如果满足，且模式需要根据切割df的进行切割筛选
                df = df[(start_int <= df.date) & (df.date <= end_int)]
        elif not force_local:
            # 数据不满足，但非强制本地，走网络
            df = load_kline_df_net(source, temp_symbol, n_folds, start=start, end=end, start_int=start_int,
                                   end_int=end_int, save=save)
            if data_mode == EMarketDataSplitMode.E_DATA_SPLIT_UNDO:
                # SPLIT_UNDO需要读取所有本地数据不切割返回
                df, _, _ = load_kline_df(temp_symbol.value)
        return df, save_kl_key
    except HDF5ExtError:
        # hdf5 bug
        logging.debug('{} HDF5ExtError'.format(symbol))
    except Exception as e:
        logging.info('Exception kline_pd symbol:{} e:{}'.format(symbol, e))
    return None, None
