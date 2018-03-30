# coding=utf-8
"""
    对数据采集进行存储，读取，以及数据更新merge策略等实现模块
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os

import pandas as pd

from ..CoreBu.ABuEnv import EDataCacheType, EMarketTargetType, EMarketSubType
from ..CoreBu import ABuEnv
from ..UtilBu.ABuFileUtil import load_df_csv, load_hdf5, ensure_dir, file_exist, del_file, dump_df_csv, \
    dump_del_hdf5
# noinspection PyUnresolvedReferences
from ..CoreBu.ABuFixes import xrange, range, filter
from ..UtilBu.ABuProgress import AbuProgress

try:
    from tables import HDF5ExtError
except ImportError:
    class HDF5ExtError(RuntimeError):
        """如果没有HDF5环境只能使用其它存贮模式"""
        pass

# 模块加载时统一确保文件夹存在，不在函数内部ensure_dir
ensure_dir(ABuEnv.g_project_kl_df_data)


def _kl_unique_key(symbol, start, end):
    """
    通过symbol以及start, end信息生成数据存储唯一id
    :param symbol: Symbol对象
    :param start: str日期对象 eg 2015-02-14
    :param end: str日期对象 eg 2017-02-14
    :return: e.g : 'usTSLA_2015-02-14_2017-02-14'
    """
    return "{}_{}_{}".format(symbol.value, start, end)


def rm_data_from_symbol(symbol):
    """
    删除特定symbol对应的本地缓存数据
    :param symbol: Symbol对象
    :return:
    """

    # TODO 只实现了针对hdf5的数据删除，添加其它存储模式的数据删除
    target_hdf5 = ABuEnv.g_project_kl_df_data
    with pd.HDFStore(target_hdf5) as h5s:

        if symbol in h5s:
            ind_key = h5s[symbol].values[0]
            if ind_key in h5s:
                # 删除缓存实体数据
                del h5s[ind_key]
            # 删除缓存数据index
            del h5s[symbol]


def load_all_kline(want_df=True, market=None, all_market=False):
    """
    只针对hdf5模式下生效，根据参数want_df决定读取hdf5中所有的index symbol数据或者实体pd.DataFrame数据
    :param want_df: 是要实体pd.DataFrame数据还是索引symbol数据
    :param market: 默认None，如None则服从ABuEnv.g_market_target市场设置
    :param all_market: 默认False, 如果True则不过滤市场，即忽略market参数指定的市场
    :return:
    """
    if ABuEnv.g_data_cache_type != EDataCacheType.E_DATA_CACHE_HDF5:
        raise RuntimeError('only support hdf5 cache mode!')

    # noinspection PyProtectedMember
    target_hdf5 = ABuEnv.g_project_kl_df_data
    # 如果是实体pd.DataFrame数据的key，key的长度最少要两个日期 8 ＊ 2 ＋ 市场前缀 ＋ 2  ＋ 最小symbol长 ＋ 2
    k_min_index_key_len = 20  # 8 * 2 + 2 + 2
    with pd.HDFStore(target_hdf5) as h5s:

        # 根据参数want_df使用k_min_index_key_len过滤出需要的key序列
        keys = list(filter(
            lambda p_key: len(p_key) >= k_min_index_key_len if want_df else len(p_key) < k_min_index_key_len,
            h5s.keys()))

        if not all_market:
            # 非所有市场，即需要根据market再次过滤
            if market is None:
                market = ABuEnv.g_market_target

            k_market_map = {EMarketTargetType.E_MARKET_TARGET_US: [EMarketTargetType.E_MARKET_TARGET_US.value],
                            EMarketTargetType.E_MARKET_TARGET_HK: [EMarketTargetType.E_MARKET_TARGET_HK.value],
                            EMarketTargetType.E_MARKET_TARGET_CN: [EMarketSubType.SZ.value, EMarketSubType.SH.value]}

            # 对应市场的head
            market_head_list = k_market_map[market]

            def filter_market_key(p_key):
                """检测p_key是否startswith对应市场market_head_list"""
                for mh in market_head_list:
                    # key[0] = '/'
                    if p_key[1:].startswith(mh):
                        return True
                return False
            # 筛选出指定市场的key
            keys = list(filter(lambda p_key: filter_market_key(p_key), keys))
        # 结果返回序列，序列元素由（(key, h5s[key]) 构成
        return [(key, h5s[key]) for key in keys]


def covert_hdf_to_csv():
    """转换hdf5下的所有cache缓存至csv文件存贮格式"""

    # 获取hdf5下所有数据
    dfs = load_all_kline(all_market=True)

    # 临时保存存贮模式
    tmp_cache = ABuEnv.g_data_cache_type
    ABuEnv.g_data_cache_type = EDataCacheType.E_DATA_CACHE_CSV
    with AbuProgress(len(dfs), 0, 'csv covert') as pg:
        for symbol, dump_df in dfs:
            pg.show()
            # eg: usTSLA
            symbol_key = symbol.split('_')[0][1:]
            # eg: usTSLA_20110808_20170808
            date_key = symbol[1:]
            # 将df转换为csv格式
            dump_kline_df(dump_df, symbol_key, date_key)
    # 还原之前的存贮模式
    ABuEnv.g_data_cache_type = tmp_cache


def load_kline_df(symbol_key):
    """
    封装不同存储模式，根据symbol_key读取对应的本地缓存金融时间序列对象数据
    :param symbol_key: str对象symbol
    :return: (金融时间序列pd.DataFrame对象，索引date_key中start请求日期int，索引date_key中end请求日期int)
    """

    """老版本默认的为hdf5，windows用户有hdf5环境问题，改为首先csv"""
    # 初始化默认读取日k数据使用_load_kline_csv方法
    load_kline_func = _load_kline_csv
    # 初始化默认读取日k数据key使用_load_csv_key方法
    load_kline_key = _load_csv_key
    # noinspection PyProtectedMember
    if ABuEnv.g_data_cache_type == EDataCacheType.E_DATA_CACHE_HDF5 and \
            not ABuEnv._g_enable_example_env_ipython:
        # 读取方式是HDF5，并且不是沙盒数据模式，切换load_kline_func，load_kline_key为HDF5读取函数
        load_kline_func = _load_kline_hdf5
        load_kline_key = _load_hdf5_key

    # noinspection PyUnusedLocal
    date_key = None
    try:
        # 首先通过symbol_key查询对应的金融时间序列是否存在索引date_key
        date_key = load_kline_key(symbol_key)
    except HDF5ExtError as e:
        # r_s = False的话，hdf5物理性错误就删除了，重来，所以重要的hdf5需要手动备份.
        r_s = True
        raise RuntimeError('hdf5 load error!! err={} '.format(e)) if r_s else os.remove(ABuEnv.g_project_kl_df_data)

    if date_key is not None:
        # 在索引date_key存在的情况下，继续查询实体金融时间序列对象
        df = load_kline_func(date_key[0])
        if df is not None:
            df['key'] = list(range(0, len(df)))
            # 索引date_key中转换df_req_start
            df_req_start = int(date_key[0][-17: -9])
            # 索引date_key中转换df_req_end
            df_req_end = int(date_key[0][-8:])
            return df, df_req_start, df_req_end
    return None, 0, 0


def _load_kline_csv(date_key):
    """
    针对csv存储模式，读取本地cache金融时间序列
    :param date_key: 金融时间序列索引key，针对对csv存储模式为目标csv的具体文件名
    """
    # noinspection PyProtectedMember
    csv_dir = ABuEnv.g_project_kl_df_data_example if ABuEnv._g_enable_example_env_ipython \
        else ABuEnv.g_project_kl_df_data_csv

    # 通过连接date_key和csv存储根目录，得到目标csv文件路径
    csv_fn = os.path.join(csv_dir, date_key)
    df = load_df_csv(csv_fn)
    # 这里要把类型转换为time
    df.index = pd.to_datetime(df.index)
    return df


def _load_kline_hdf5(date_key):
    """
    针对hdf5存储模式，读取本地cache金融时间序列
    :param date_key: 金融时间序列索引key，针对对hdf5存储模式为目标金融时间序列查询key
    """
    target_hdf5 = ABuEnv.g_project_kl_df_data
    return load_hdf5(target_hdf5, date_key)


def check_csv_local(symbol_key):
    """
    套结_load_csv_key，但不返回key具体值，只返回对应的symbol是否
    存在csv缓存
    :param symbol_key: str对象，eg. usTSLA
    :return: bool, symbol是否存在csv缓存
    """
    return _load_csv_key(symbol_key) is not None


def _load_csv_key(symbol_key):
    """
    针对csv存储模式，通过symbol_key字符串找到对应的csv具体文件名称，
    如从usTSLA->找到usTSLA_2014-7-26_2016_7_26这个具体csv文件路径
    :param symbol_key: str对象，eg. usTSLA
    """
    # noinspection PyProtectedMember
    csv_dir = ABuEnv.g_project_kl_df_data_example if ABuEnv._g_enable_example_env_ipython \
        else ABuEnv.g_project_kl_df_data_csv

    if file_exist(csv_dir):
        for name in os.listdir(csv_dir):
            # 从csv缓存文件夹下进行模糊查询通过fnmatch匹配具体csv文件路径，eg. usTSLA->usTSLA_2014-7-26_2016_7_26
            # if fnmatch(name, '{}*'.format(symbol_key)):
            """
                这里不能模糊匹配，否则会因为TSL匹配上TSLA导致删除原有的symbol
                而且必须要加'_'做为symbol结束匹配标记
            """
            if name.startswith(symbol_key + '_'):
                # []只是为了配合外面针对不同store统一使用key[0]
                return [name]
    return None


def _load_hdf5_key(symbol_key):
    """
    针对hdf5存储模式，通过symbol_key字符串找到对应的在hdf5中的实体金融时间
    序列实体的索引序列
    :param symbol_key: 金融时间序列索引key，针对对hdf5存储模式为目标金融时间序列查询key
    """
    # noinspection PyProtectedMember
    target_hdf5 = ABuEnv.g_project_kl_df_data
    return load_hdf5(target_hdf5, symbol_key)


def dump_kline_df(dump_df, symbol_key, date_key):
    """
    封装不同存储模式，根据symbol_key，date_key存储dump_df金融时间序列
    储存方法 symbol_key->date_key->dump_df

    eg  : usTSLA->usTSLA_20100214_20170214->tsla_df

    :param dump_df: 需要存储的金融时间序列实体pd.DataFrame对象
    :param symbol_key: str对象，eg. usTSLA
    :param date_key: str对象，eg. usTSLA_20100214_20170214 包含了df的时间开始时间与结束时间，便于计算需要的数据段是否在此之间

    """
    # 默认csv模式分配工作函数
    dump_kline_func = _dump_kline_csv
    load_kline_key = _load_csv_key
    load_kline_func = _load_kline_csv
    # hdf5模式分配工作函数
    if ABuEnv.g_data_cache_type == EDataCacheType.E_DATA_CACHE_HDF5:
        load_kline_key = _load_hdf5_key
        dump_kline_func = _dump_kline_hdf5
        load_kline_func = _load_kline_hdf5

    _start = int(date_key[-17: -9])
    _end = int(date_key[-8:])

    # 读取本地缓存symbol_key对应的索引对象df_date_key
    df_date_key = load_kline_key(symbol_key)
    if df_date_key is not None:
        # 即之前存在本地缓存，需要merge金融时间序列

        # 之前本地缓存的时间序列结束日期
        df_end = int(df_date_key[0][-8:])
        # 之前本地缓存的时间序列开始日期
        df_start = int(df_date_key[0][-17: -9])
        if _start <= df_start and df_end <= _end:
            """
                新请求回来的数据完全包裹了原来的数据，直接存储即可
                       _start o------------------o _end
                       df_start   o--------o df_end

            result:    _start o------------------o _end
            """
            dump_kline_func(symbol_key, date_key, dump_df, delete_key=df_date_key)
        elif _start < df_start and df_end >= _end:
            """
                新请求回来的数据开始时间比原来的要考前，结束时间没有原来的远
                       _start o------------------o _end
                       df_start   o-----------------o df_end

            result:    _start o---------------------o df_end
            """
            date_key = '{}_{}_{}'.format(symbol_key, _start, df_end)
            # 首先读取原来的金融时间序列
            h5_df = load_kline_func(df_date_key[0])
            # [_start, df_start)
            # 选取dump_df.date < df_start部分
            new_df = dump_df[dump_df.date < df_start]
            # concat连起来两部分
            new_df = pd.concat([new_df, h5_df])
            # 最终保存的为new_df
            dump_kline_func(symbol_key, date_key, new_df, delete_key=df_date_key)
        elif _start >= df_start and df_end < _end:
            """
                新请求回来的数据开始时间比原来的晚，但结束时间也比原来的晚
                            _start o------------------o _end
                       df_start o-------------o df_end

            result:    df_start o---------------------o _end
            """
            date_key = '{}_{}_{}'.format(symbol_key, df_start, _end)
            # 首先读取原来的金融时间序列
            h5_df = load_kline_func(df_date_key[0])
            # 选取dump_df.date > df_end部分
            new_df = dump_df[dump_df.date > df_end]
            # concat连起来两部分
            new_df = pd.concat([h5_df, new_df])
            # 最终保存的为new_df
            dump_kline_func(symbol_key, date_key, new_df, delete_key=df_date_key)
        else:
            # 完全包裹数据，但是更新最新下载下来的数据替换之前的本地数据，类似更新操作
            # 首先读取原来的金融时间序列
            # date_key 取和之前一摸一样的

            date_key = '{}_{}_{}'.format(symbol_key, df_start, df_end)
            local_df = load_kline_func(df_date_key[0])
            local_df_st = local_df[local_df.date < _start]
            local_df_ed = local_df[local_df.date > _end]
            # concat连起来三个部分
            new_df = pd.concat([local_df_st, dump_df, local_df_ed])
            # 最终保存的为new_df
            dump_kline_func(symbol_key, date_key, new_df, delete_key=df_date_key)
    else:
        # 即之前不存在本地缓存，直接存储在本地即可
        dump_kline_func(symbol_key, date_key, dump_df)


# noinspection PyUnusedLocal
def _dump_kline_csv(symbol_key, date_key, dump_df, delete_key=None):
    """
    针对csv存储模式，根据symbol_key，date_key存储dump_df金融时间序列
    :param symbol_key: str对象，eg. usTSLA，对于csv模式不需要，只为保持接口统一
    :param date_key: str对象，eg. usTSLA_20100214_20170214，csv模式下为对应的文件名
    :param dump_df: 需要存储的金融时间序列实体pd.DataFrame对象
    :param delete_key: 是否有需要删除的csv文件
    :return:
    """
    # 先删后后写入
    if delete_key is not None:
        delete_key = delete_key[0]
        del_fn = os.path.join(ABuEnv.g_project_kl_df_data_csv, delete_key)
        if file_exist(del_fn):
            del_file(del_fn)

    csv_fn = os.path.join(ABuEnv.g_project_kl_df_data_csv, date_key)
    dump_df_csv(csv_fn, dump_df)


def _dump_kline_hdf5(symbol_key, date_key, dump_df, delete_key=None):
    """
    针对hdf5存储模式，根据symbol_key，date_key存储dump_df金融时间序列
    :param symbol_key: str对象，eg. usTSLA，hdf5模式下为数据的索引key
    :param date_key: str对象，eg. usTSLA_20100214_20170214，hdf5模式下为金融时间序列实体key
    :param dump_df: 需要存储的金融时间序列实体pd.DataFrame对象
    :param delete_key: 是否有需要删除的csv文件
    :return:
    """

    # 需要dump的dict
    dump_dict = {symbol_key: pd.Series(date_key), date_key: dump_df}
    # 需要delete的dict
    del_array = [symbol_key]
    if delete_key is not None:
        del_array.append(delete_key)
    """
        dump中的target_hdf5只能是ABuEnv.g_project_kl_df_data，即使在沙盒测试数据下，
        在dump_del_hdf5中分别迭代dump_dict，del_array进行操作，详阅dump_del_hdf5
    """
    dump_del_hdf5(ABuEnv.g_project_kl_df_data, dump_dict, del_array)


def save_kline_df(df, temp_symbol, start_int, end_int):
    """
    独立对外的保存kl数据接口
    :param df: 需要存储的金融时间序列实体pd.DataFrame对象
    :param temp_symbol: Symbbol对象
    :param start_int: 请求的开始日期int
    :param end_int: 请求的结束日期int
    :return:
    """
    if df is not None:
        # 通过emp_symbol, start_int, end_int拼接唯一保存df_key
        df_key = _kl_unique_key(temp_symbol, start_int, end_int)
        dump_kline_df(df, temp_symbol.value, df_key)


def load_kline_df_net(source, temp_symbol, n_folds, start, end, start_int, end_int, save):
    """
    通过网络请求数据源，获取temp_symbol以及参数时间日期对应的金融时间序列pd.DataFrame对象
    :param source: 数据源BaseMarket的子类，非实例化对象
    :param temp_symbol: Symbol类对象
    :param n_folds: 需要获取几年的回测数据，int
    :param start: 开始回测日期，str对象
    :param end: 结束回测日期，str对象
    :param start_int: 开始回测日期，int
    :param end_int: 结束回测日期，int
    :param save: 是否从网络成功获取数据后进行数据的保存
    """
    df = None
    # 实例化数据源对象
    data_source = source(temp_symbol)

    if data_source.check_support():
        # 通过数据源混入的SupportMixin类检测数据源是否支持temp_symbol对应的市场数据
        df = data_source.kline(n_folds=n_folds, start=start, end=end)

    if df is not None and save:
        """
            这里的start_int， end_int会记作下次读取的df_req_start, df_req_end，即就是没有完整的数据返回，也可通过索引匹配上，
            即如果今天刚刚请求了直到今天为止的数据，但是数据源没有返回到今天的数据，今天的还没有，但是由于记录了end_int为今天，所以
            再次发起请求时不会走网络，会从本地获取数据
        """
        df_key = _kl_unique_key(temp_symbol, start_int, end_int)
        dump_kline_df(df, temp_symbol.value, df_key)
    return df
