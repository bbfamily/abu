# coding=utf-8
"""
    数据对外接口模块，其它模块需要数据都只应该使用ABuSymbolPd, 不应涉及其它内部模块的使用
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import logging
from collections import Iterable

import pandas as pd

from .ABuDataSource import kline_pd
from ..MarketBu.ABuDataCache import save_kline_df, check_csv_local
from ..MarketBu.ABuSymbol import code_to_symbol
from .ABuSymbol import Symbol
from .ABuMarket import split_k_market
from ..CoreBu import ABuEnv
from ..CoreBu.ABuEnv import EMarketDataFetchMode, EDataCacheType
from ..CoreBu.ABuFixes import partial, ThreadPoolExecutor
# noinspection PyUnresolvedReferences
from ..CoreBu.ABuFixes import range
from ..CoreBu.ABuDeprecated import AbuDeprecated
from ..IndicatorBu import ABuNDAtr as Atr
from ..UtilBu import ABuDateUtil
from ..UtilBu.ABuFileUtil import batch_h5s
from ..UtilBu.ABuProgress import AbuMulPidProgress, do_clear_output
from ..CoreBu.ABuParallel import delayed, Parallel
from ..CoreBu.ABuFixes import six
# from ..UtilBu.ABuThreadPool import AbuThreadPoolExecutor

__author__ = '阿布'
__weixin__ = 'abu_quant'


def _benchmark(df, benchmark, symbol):
    """
    在内部使用kline_pd获取金融时间序列pd.DataFrame后，如果参数中
    基准benchmark（pd.DataFrame对象）存在，使用基准benchmark的
    时间范围切割kline_pd返回的金融时间序列
    :param df: 金融时间序列pd.DataFrame对象
    :param benchmark: 资金回测时间标尺，AbuBenchmark实例对象
    :param symbol: Symbol对象
    :return: 使用基准的时间范围切割返回的金融时间序列
    """
    if len(df.index & benchmark.kl_pd.index) <= 0:
        # 如果基准benchmark时间范围和输入的df没有交集，直接返回None
        return None

    # 两个金融时间序列通过loc寻找交集
    kl_pd = df.loc[benchmark.kl_pd.index]
    # nan的date个数即为不相交的个数
    nan_cnt = kl_pd['date'].isnull().value_counts()
    # 两个金融序列是否相同的结束日期
    same_end = df.index[-1] == benchmark.kl_pd.index[-1]
    # 两个金融序列是否相同的开始日期
    same_head = df.index[0] == benchmark.kl_pd.index[0]

    # 如果nan_cnt即不相交个数大于benchmark基准个数的1/3，放弃
    base_keep_div = 3
    if same_end or same_head:
        # 如果两个序列有相同的开始或者结束改为1/2，也就是如果数据头尾日起的标尺有一个对的上的话，放宽na数量
        base_keep_div = 2
    if same_end and same_head:
        # 如果两个序列同时有相同的开始和结束改为1，也就是如果数据头尾日起的标尺都对的上的话，na数量忽略不计
        base_keep_div = 1

    if symbol.is_a_stock():
        # 如果是A股市场的目标，由于停盘频率和周期都会长与其它市场所以再放宽一些
        base_keep_div *= 0.7

    if nan_cnt.index.shape[0] > 0 and nan_cnt.index.tolist().count(True) > 0 \
            and nan_cnt.loc[True] > benchmark.kl_pd.shape[0] / base_keep_div:
        # nan 个数 > 基准base_keep_div分之一放弃
        return None

    # 来到这里说明没有放弃，那么就填充nan
    # 首先nan的交易量是0
    kl_pd.volume.fillna(value=0, inplace=True)
    # nan的p_change是0
    kl_pd.p_change.fillna(value=0, inplace=True)
    # 先把close填充了，然后用close填充其它的
    kl_pd.close.fillna(method='pad', inplace=True)
    kl_pd.close.fillna(method='bfill', inplace=True)
    # 用close填充open
    kl_pd.open.fillna(value=kl_pd.close, inplace=True)
    # 用close填充high
    kl_pd.high.fillna(value=kl_pd.close, inplace=True)
    # 用close填充low
    kl_pd.low.fillna(value=kl_pd.close, inplace=True)
    # 用close填充pre_close
    kl_pd.pre_close.fillna(value=kl_pd.close, inplace=True)

    # 细节nan处理完成后，把剩下的nan都填充了
    kl_pd = kl_pd.fillna(method='pad')
    # bfill再来一遍只是为了填充最前面的nan
    kl_pd.fillna(method='bfill', inplace=True)

    # pad了数据所以，交易日期date的值需要根据time index重新来一遍
    kl_pd['date'] = [int(ts.date().strftime("%Y%m%d")) for ts in kl_pd.index]
    kl_pd['date_week'] = kl_pd['date'].apply(lambda x: ABuDateUtil.week_of_date(str(x), '%Y%m%d'))

    return kl_pd


def _make_kl_df(symbol, data_mode, n_folds, start, end, benchmark, save):
    """
    针对一个symbol进行数据获取，内部使用kline_pd从本地加载或者指定数据源进行网络请求
    :param symbol: str对象 or Symbol对象
    :param data_mode: EMarketDataSplitMode enum对象
    :param n_folds: 请求几年的历史回测数据int
    :param start: 请求的开始日期 str对象
    :param end: 请求的结束日期 str对象
    :param benchmark: 资金回测时间标尺，AbuBenchmark实例对象
    :param save: 是否进行网络获取数据后，直接进行本地保存
    :return: (df: 金融时间序列pd.DataFrame对象，save_kl_key: 提供外部进行保存)
    """
    df, save_kl_key = kline_pd(symbol, data_mode, n_folds=n_folds, start=start, end=end, save=save)
    if df is not None and df.shape[0] == 0:
        # 把行数＝0的归结为＝None, 方便后续统一处理
        df = None

    if benchmark is not None and df is not None:
        # 如果有标尺，进行标尺切割，进行标尺切割后也可能变成none
        temp_symbol = save_kl_key[0]
        df = _benchmark(df, benchmark, temp_symbol)

    if df is not None:
        # 规避重复交易日数据风险，subset只设置date做为滤除重复
        df.drop_duplicates(subset=['date'], inplace=True)
        # noinspection PyProtectedMember
        if not ABuEnv._g_enable_example_env_ipython or 'atr14' not in df.columns or 'atr21' not in df.columns:
            # 非沙盒环境计算, 或者是沙盒但数据本身没有atr14，atr21
            calc_atr(df)
        # 根据df长度重新进行key计算
        df['key'] = list(range(0, len(df)))
        temp_symbol = save_kl_key[0]
        df.name = temp_symbol.value
    return df, save_kl_key


def _kl_df_dict_parallel(choice_symbols, data_mode, n_folds, start, end, benchmark):
    """
    多进程或者多线程被委托的任务函数，多任务批量获取时间序列数据
    :param choice_symbols: symbol序列
    :param data_mode: EMarketDataSplitMode enum对象
    :param n_folds: 请求几年的历史回测数据int
    :param start: 请求的开始日期 str对象
    :param end: 请求的结束日期 str对象
    :param benchmark: 资金回测时间标尺，AbuBenchmark实例对象
    :return: df_dict字典中key=请求symbol的str对象，value＝(save_kl_key: 提供外部进行保存, df: 金融时间序列pd.DataFrame对象)
    """
    df_dict = {}
    # 注意save=False

    # 启动多进程进度条，如果是多线程暂时也启动了AbuMulPidProgress，需优化
    with AbuMulPidProgress(len(choice_symbols), 'kl_df parallel complete') as progress:
        for epoch, symbol in enumerate(choice_symbols):
            # 迭代choice_symbols进行_make_kl_df, 注意_make_kl_df的参数save=False，即并行获取，不在内部save，要在外部save
            df, key_tuple = _make_kl_df(symbol, data_mode=data_mode,
                                        n_folds=n_folds, start=start, end=end, benchmark=benchmark, save=False)
            if isinstance(key_tuple, tuple) and len(key_tuple) == 3:
                # key=请求symbol的str对象，value＝(save_kl_key: 提供外部进行保存, df: 金融时间序列pd.DataFrame对象)
                df_dict[key_tuple[0].value] = (key_tuple, df)
            progress.show(epoch + 1)

    return df_dict


def kl_df_dict_parallel(symbols, data_mode=ABuEnv.EMarketDataSplitMode.E_DATA_SPLIT_SE,
                        n_folds=2, start=None, end=None, benchmark=None, n_jobs=16, save=True, how='thread'):
    """
    多进程或者多线程对外执行函数，多任务批量获取时间序列数据
    :param symbols: symbol序列
    :param data_mode: EMarketDataSplitMode enum对象
    :param n_folds: 请求几年的历史回测数据int
    :param start: 请求的开始日期 str对象
    :param end: 请求的结束日期 str对象
    :param benchmark: 资金回测时间标尺，AbuBenchmark实例对象
    :param n_jobs: 并行的任务数，对于进程代表进程数，线程代表线程数
    :param save: 是否统一进行批量保存，即在批量获取金融时间序列后，统一进行批量保存，默认True
    :param how: process：多进程，thread：多线程，main：单进程单线程
    """

    # TODO Iterable和six.string_types的判断抽出来放在一个模块，做为Iterable的判断来使用
    if not isinstance(symbols, Iterable) or isinstance(symbols, six.string_types):
        # symbols必须是可迭代的序列对象
        raise TypeError('symbols must a Iterable obj!')
    # 可迭代的symbols序列分成n_jobs个子序列
    parallel_symbols = split_k_market(n_jobs, market_symbols=symbols)
    # 使用partial对并行函数_kl_df_dict_parallel进行委托
    parallel_func = partial(_kl_df_dict_parallel, data_mode=data_mode, n_folds=n_folds, start=start, end=end,
                            benchmark=benchmark)
    # 因为切割会有余数，所以将原始设置的进程数切换为分割好的个数, 即32 -> 33 16 -> 17
    n_jobs = len(parallel_symbols)
    if how == 'process':
        """
            mac os 10.9 以后的并行加上numpy不是crash就是进程卡死，不要用，用thread
        """
        if ABuEnv.g_is_mac_os:
            logging.info('mac os 10.9 parallel with numpy crash or dead!!')

        parallel = Parallel(
            n_jobs=n_jobs, verbose=0, pre_dispatch='2*n_jobs')
        df_dicts = parallel(delayed(parallel_func)(choice_symbols)
                            for choice_symbols in parallel_symbols)
    elif how == 'thread':
        # 通过ThreadPoolExecutor进行线程并行任务
        with ThreadPoolExecutor(max_workers=n_jobs) as pool:
            k_use_map = True
            if k_use_map:
                df_dicts = list(pool.map(parallel_func, parallel_symbols))
            else:
                futures = [pool.submit(parallel_func, symbols) for symbols in parallel_symbols]
                df_dicts = [future.result() for future in futures if future.exception() is None]
    elif how == 'main':
        # 单进程单线程
        df_dicts = [parallel_func(symbols) for symbols in parallel_symbols]
    else:
        raise TypeError('ONLY process OR thread!')

    if save:
        # 统一进行批量保存
        h5s_fn = ABuEnv.g_project_kl_df_data if ABuEnv.g_data_cache_type == EDataCacheType.E_DATA_CACHE_HDF5 else None

        @batch_h5s(h5s_fn)
        def _batch_save():
            for df_dict in df_dicts:
                # 每一个df_dict是一个并行的序列返回的数据
                for ind, (key_tuple, df) in enumerate(df_dict.values()):
                    # (key_tuple, df)是保存kl需要的数据, 迭代后直接使用save_kline_df
                    save_kline_df(df, *key_tuple)
                    if df is not None:
                        print("save kl {}_{}_{} {}/{}".format(key_tuple[0].value, key_tuple[1], key_tuple[2], ind,
                                                              df.shape[0]))
                # 完成一层循环一次，即批量保存完一个并行的序列返回的数据后，进行清屏
                do_clear_output()

        _batch_save()
    return df_dicts


# noinspection PyDeprecation
def make_kl_df(symbol, data_mode=ABuEnv.EMarketDataSplitMode.E_DATA_SPLIT_SE,
               n_folds=2, start=None, end=None, benchmark=None, show_progress=True, parallel=False, parallel_save=True):
    """
    外部获取金融时间序列接口
    eg: n_fold=2, start=None, end=None ，从今天起往前数两年
        n_fold=2, start='2015-02-14', end=None， 从2015-02-14到现在，n_fold无效
        n_fold=2, start=None, end='2016-02-14'，从2016-02-14起往前数两年
        n_fold=2, start='2015-02-14', end='2016-02-14'，从start到end

    :param data_mode: EMarketDataSplitMode对象
    :param symbol: list or Series or str or Symbol
                    e.g :['TSLA','SFUN'] or 'TSLA' or Symbol(MType.US,'TSLA')
    :param n_folds: 请求几年的历史回测数据int
    :param start: 请求的开始日期 str对象
    :param end: 请求的结束日期 str对象
    :param benchmark: 资金回测时间标尺，AbuBenchmark实例对象
    :param show_progress: 是否显示进度条
    :param parallel: 是否并行获取
    :param parallel_save: 是否并行后进行统一批量保存
    """

    if isinstance(symbol, (list, tuple, pd.Series, pd.Index)):
        # 如果symbol是可迭代的序列对象，最终返回三维面板数据pd.Panel
        panel = dict()
        if parallel:
            # 如果并行获取
            if ABuEnv.g_data_fetch_mode != EMarketDataFetchMode.E_DATA_FETCH_FORCE_NET \
                    and ABuEnv.g_data_cache_type == EDataCacheType.E_DATA_CACHE_HDF5:
                # 只能针对非hdf5存贮形式下或者针对force_net，因为hdf5多线程读容易卡死
                raise RuntimeError('make_kl_df just suit force net or not hdf5 store!')

            df_dicts = kl_df_dict_parallel(symbol, data_mode=data_mode, n_folds=n_folds, start=start, end=end,
                                           benchmark=benchmark, save=parallel_save, how='thread')
            for df_dict in df_dicts:
                for key_tuple, df in df_dict.values():
                    if df is None or df.shape[0] == 0:
                        continue
                    # 即丢弃原始df_dict保存金融时间序列时使用的save_kl_key，只保留df，赋予panel
                    panel[key_tuple[0].value] = df
        else:
            def _batch_make_kl_df():
                with AbuMulPidProgress(len(symbol), '_make_kl_df complete') as progress:
                    for pos, _symbol in enumerate(symbol):
                        _df, _ = _make_kl_df(_symbol, data_mode=data_mode,
                                             n_folds=n_folds, start=start, end=end, benchmark=benchmark, save=True)
                        if show_progress:
                            progress.show()
                        # TODO 做pd.Panel数据应该保证每一个元素的行数和列数都相等，不是简单的有数据就行
                        if _df is None or _df.shape[0] == 0:
                            continue

                        panel[symbol[pos]] = _df

            _batch_make_kl_df()
        # TODO pd.Panel过时
        return pd.Panel(panel)

    elif isinstance(symbol, Symbol) or isinstance(symbol, six.string_types):
        # 对单个symbol进行数据获取
        df, _ = _make_kl_df(symbol, data_mode=data_mode,
                            n_folds=n_folds, start=start, end=end, benchmark=benchmark, save=True)
        return df
    else:
        raise TypeError('symbol type is error')


def get_price(symbol, start_date=None, end_date=None):
    """
    通过make_kl_df获取金融时间序列后，只保留收盘价格，只是为了配合主流回测平台接口名称，适配使用
    :param symbol: str对象或Symbol对象
    :param start_date: 请求的开始日期 str对象
    :param end_date: 请求的结束日期 str对象
    :return: 金融时间序列pd.DataFrame对象只有一个price列
    """
    df = make_kl_df(symbol, start=start_date, end=end_date)
    if df is not None:
        df = df.filter(['close'])
        # 为了配合主流回测平台适配
        return df.rename(columns={'close': 'price'})


def check_symbol_in_local_csv(symbol):
    """
    通过传递symbol监测symbol对象是否存在csv缓存，不监测时间范围，只监测是否存在缓存
    :param symbol: str对象 or Symbol对象, 内部统一使用code_to_symbol变成Symbol对象
                   e.g : 'usTSLA' or Symbol(MType.US,'TSLA')
    :return: bool, symbol是否存在csv缓存
    """

    if isinstance(symbol, six.string_types):
        # 如果是str对象，通过code_to_symbol转化为Symbol对象
        symbol = code_to_symbol(symbol, rs=False)
    if symbol is None:
        # 主要针对code_to_symbol无规则进行转换的情况下只返回不存在缓存
        return False

    if not isinstance(symbol, Symbol):
        raise TypeError('symbol must like as "usTSLA" or "TSLA" or Symbol(MType.US, "TSLA")')

    return check_csv_local(symbol.value)


def combine_pre_kl_pd(kl_pd, n_folds=1):
    """
    通过传人一个kl_pd获取这个kl_pd之前n_folds年时间的kl，默认n_folds=1,
    eg. kl_pd 从2014-07-26至2016-07-26，首先get 2013-07-26至2014-07-25
    之后合并两段数据，最终返回的数据为2013-07-26至2016-07-26
    :param kl_pd: 金融时间序列pd.DataFrame对象
    :param n_folds: 获取之前n_folds年的数据
    :return: 结果是和输入kl_pd合并后的总kl
    """

    # 获取kl_pd的起始时间
    end = ABuDateUtil.timestamp_to_str(kl_pd.index[0])
    # kl_pd的起始时间做为end参数通过make_kl_df和n_folds参数获取之前的一段时间序列
    pre_kl_pd = make_kl_df(kl_pd.name, data_mode=ABuEnv.EMarketDataSplitMode.E_DATA_SPLIT_SE, n_folds=n_folds,
                           end=end)
    # 再合并两段时间序列，pre_kl_pd[:-1]跳过重复的end
    combine_kl = kl_pd if pre_kl_pd is None else pre_kl_pd[:-1].append(kl_pd)
    # 根据combine_kl长度重新进行key计算
    combine_kl['key'] = list(range(0, len(combine_kl)))
    return combine_kl


def calc_atr(kline_df):
    """
    为输入的kline_df金融时间序列计算atr21和atr14，计算结果直接加到kline_df的atr21列和atr14列中
    :param kline_df: 金融时间序列pd.DataFrame对象
    """
    kline_df['atr21'] = 0
    if kline_df.shape[0] > 21:
        # 大于21d计算atr21
        kline_df['atr21'] = Atr.atr21(kline_df['high'].values, kline_df['low'].values, kline_df['pre_close'].values)
        # 将前面的bfill
        kline_df['atr21'].fillna(method='bfill', inplace=True)
    kline_df['atr14'] = 0
    if kline_df.shape[0] > 14:
        # 大于14d计算atr14
        kline_df['atr14'] = Atr.atr14(kline_df['high'].values, kline_df['low'].values, kline_df['pre_close'].values)
        # 将前面的bfill
        kline_df['atr14'].fillna(method='bfill', inplace=True)


@AbuDeprecated('only for old abu!')
def get_n_year(kl_pd, from_year, get_year=1, how='ff'):
    """
    获取pd中第n年切片数据, Deprecated
    :param kl_pd: 金融时间序列pd.DataFrame对象
    :param from_year: form 1开始纠错0 to 1
    :param get_year: 要几年的数据1就是1年，0.5半年 默认1 year支持0.1 to inf
    :param how:='bf' 从后向前切 ='ff' 从前向后切
    """
    td = ABuEnv.g_market_trade_year
    n_n_year = int(kl_pd.shape[0] / td)
    if from_year == 0:
        from_year = 1
        logging.info('get_n_year form num 1 so you pass 0 covert to 1')
    if from_year > n_n_year:
        raise ValueError('get_n_year n_year > n_n_year!')

    get_days = int(get_year * td)

    if how == 'ff':
        # 从前向后切
        st = int((from_year - 1) * td)
        ed = st + get_days if st + get_days < kl_pd.shape[0] else kl_pd.shape[0]
    elif how == 'bf':
        ed = kl_pd.shape[0] - int((from_year - 1) * td)
        st = ed - get_days - 1 if ed - get_days > 0 else 0
    else:
        raise TypeError('error direction input!')

    ys = slice(st, ed)
    ret_pd = kl_pd[ys]
    if hasattr(kl_pd, 'name'):
        ret_pd.name = kl_pd.name
    return ret_pd
