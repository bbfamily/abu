# coding=utf-8
"""
    市场相关切割，选股，等操作模块
"""
from __future__ import print_function
from __future__ import absolute_import
from __future__ import division

import os
import datetime

import numpy as np
import pandas as pd

from ..UtilBu import ABuFileUtil
from ..CoreBu import ABuEnv
from ..CoreBu.ABuDeprecated import AbuDeprecated
from ..CoreBu.ABuEnv import EMarketTargetType, EMarketSubType
# noinspection PyUnresolvedReferences
from ..CoreBu.ABuFixes import filter, zip
from ..CoreBu.ABuFixes import KFold, six
from ..UtilBu.ABuLazyUtil import LazyFunc
from ..MarketBu.ABuSymbol import Symbol, code_to_symbol
from ..MarketBu.ABuSymbolFutures import AbuFuturesCn, AbuFuturesGB
from ..MarketBu.ABuSymbolStock import AbuSymbolCN, AbuSymbolUS, AbuSymbolHK

__author__ = '阿布'
__weixin__ = 'abu_quant'

# TODO 在全市场回测时设置g_use_env_market_set=True
"""默认False，如果进行全市场回测这里可以设置True, 配合LazyFunc提高效率"""
g_use_env_market_set = False

"""在market_train_test_split函数中，切割的测试集交易symbol，本地序列化存储路径的基础路径名"""
K_MARKET_TEST_FN_BASE = os.path.join(ABuEnv.g_project_cache_dir, 'market_test_symbols')
"""在market_train_test_split函数中，切割的训练集交易symbol，本地序列化存储路径的基础路径名"""
K_MARKET_TRAIN_FN_BASE = os.path.join(ABuEnv.g_project_cache_dir, 'market_train_symbols')

# TODO 从沙盒数据库里读取，否则之后有变动还需要跟着改
K_SAND_BOX_US = ['usTSLA', 'usNOAH', 'usSFUN', 'usBIDU', 'usAAPL', 'usGOOG', 'usWUBA', 'usVIPS']
K_SAND_BOX_CN = ['sz002230', 'sz300104', 'sz300059', 'sh601766', 'sh600085', 'sh600036',
                 'sh600809', 'sz000002', 'sz002594', 'sz002739']
K_SAND_BOX_HK = ['hk03333', 'hk00700', 'hk02333', 'hk01359', 'hk00656', 'hk03888', 'hk02318']


# noinspection PyUnresolvedReferences
class MarketMixin(object):
    """
        市场信息混入类，被混入类需要设置self.symbol_name，
        通过code_to_symbol将symbol转换为Symbol对象, 通过Symbol对象
        查询market和sub_market
    """

    @LazyFunc
    def _symbol(self):
        """通过code_to_symbol将symbol转换为Symbol对象 LazyFunc"""
        if not hasattr(self, 'symbol_name'):
            # 被混入类需要设置self.symbol_name
            raise NameError('must name symbol_name!!')
        # 通过code_to_symbol将symbol转换为Symbol对象
        return code_to_symbol(self.symbol_name)

    @LazyFunc
    def symbol_market(self):
        """查询self.symbol_name的市场类型 LazyFunc"""
        return self._symbol.market

    @LazyFunc
    def symbol_sub_market(self):
        """查询self.symbol_name的子市场类型，即交易所信息 LazyFunc"""
        return self._symbol.sub_market


def split_k_market(k_split, market_symbols=None, market=None):
    """
    将market_symbols序列切分成k_split个序列
    :param k_split: 切分成子序列个数int
    :param market_symbols: 待切割的原始symbols序列，如果none, 将取market参数中指定的市场所有symbol
    :param market: 默认None，如None则服从ABuEnv.g_market_target市场设置
    :return: list序列，序列中的每一个元素都是切割好的子symbol序列
    """
    if market is None:
        # 取env中的ABuEnv.g_market_target设置
        market = ABuEnv.g_market_target
    if market_symbols is None:
        # 取market参数中指定的市场所有symbol
        market_symbols = all_symbol(market=market)
    if len(market_symbols) < k_split:
        # 特殊情况，eg：k_split＝100，但是len(market_symbols)＝50，就不切割了，直接返回
        return [[symbol] for symbol in market_symbols]

    # 计算每一个子序列的承载的symbol个数，即eg：100 ／ 5 ＝ 20
    sub_symbols_cnt = int(len(market_symbols) / k_split)
    group_adjacent = lambda a, k: zip(*([iter(a)] * k))
    # 使用lambda函数group_adjacent将market_symbols切割子序列，每个子系列sub_symbols_cnt个
    symbols = list(group_adjacent(market_symbols, sub_symbols_cnt))
    # 将不能整除的余数symbol个再放进去
    residue_ind = -(len(market_symbols) % sub_symbols_cnt) if sub_symbols_cnt > 0 else 0
    if residue_ind < 0:
        # 所以如果不能除尽，最终切割的子序列数量为k_split+1, 外部如果需要进行多认为并行，可根据最终切割好的数量重分配任务数
        symbols.append(market_symbols[residue_ind:])
    return symbols


def choice_symbols(count, market_symbols=None, market=None):
    """
    在market_symbols中随机选择count个symbol，不放回随机的抽取方式
    :param count: 选择count个(int)
    :param market_symbols: 备选symbols序列，如果None, 则从参数market选择全市场symbol做为备选
    :param market: 默认None，如None则服从ABuEnv.g_market_target市场设置
    :return: 随机选择count个symbol结果序列
    """
    if market is None:
        # 如None则服从ABuEnv.g_market_target市场设置
        market = ABuEnv.g_market_target
    if market_symbols is None:
        # 从参数market选择全市场symbol做为备选
        market_symbols = all_symbol(market=market)
    # 使用不放回随机的抽取方式
    return np.random.choice(market_symbols, count, replace=False)


def choice_symbols_with_replace(count, market_symbols=None, market=None):
    """
    在market_symbols中随机选择count个symbol，有放回随机的抽取方式
    :param count: 选择count个(int)
    :param market_symbols: 备选symbols序列，如果None, 则从参数market选择全市场symbol做为备选
    :param market: 默认None，如None则服从ABuEnv.g_market_target市场设置
    :return: 随机选择count个symbol结果序列
    """

    if market is None:
        market = ABuEnv.g_market_target
    if market_symbols is None:
        market_symbols = all_symbol(market=market)
    # 使用有放回随机的抽取方式，即replace=True
    return np.random.choice(market_symbols, count, replace=True)


def _all_us_symbol(index=False):
    """
    通过AbuSymbolUS().all_symbol获取美股全市场股票代码
    :param index: 是否包含指数
    :return:
    """

    # noinspection PyProtectedMember
    if ABuEnv._g_enable_example_env_ipython:
        return K_SAND_BOX_US
    return AbuSymbolUS().all_symbol(index=index)


def _all_cn_symbol(index=False):
    """
    通过AbuSymbolCN().all_symbol获取A股全市场股票代码
    :param index: 是否包含指数
    :return:
    """
    # noinspection PyProtectedMember
    if ABuEnv._g_enable_example_env_ipython:
        return K_SAND_BOX_CN
    return AbuSymbolCN().all_symbol(index=index)


def _all_hk_symbol(index=False):
    """
    通过AbuSymbolHK().all_symbol获取A股全市场股票代码
    :param index: 是否包含指数
    :return:
    """
    # noinspection PyProtectedMember
    if ABuEnv._g_enable_example_env_ipython:
        return K_SAND_BOX_HK
    return AbuSymbolHK().all_symbol(index=index)


def _all_futures_cn():
    """
    AbuFuturesCn().symbol获取国内期货symbol代码，注意这里只取连续合约代码
    :return:
    """
    return AbuFuturesCn().symbol


def _all_futures_gb():
    """
    AbuFuturesGB().symbol获取国际期货symbol代码，LME，CBOT，COMEX
    :return:
    """
    return AbuFuturesGB().symbol


def _all_tc_symbol():
    """
    获取币类symbol，注意这里只取比特币与莱特币，可自行扩展其它币种
    :return:
    """
    return ['btc', 'ltc']


def all_symbol(market=None, ss=False, index=False, value=True):
    """
    根据传入的市场获取全市场代码
    :param market: 默认None，如None则服从ABuEnv.g_market_target市场设置
    :param ss: 是否将返回序列使用pd.Series包装
    :param index: 是否包含指数大盘symbol
    :param value: 返回字符串值，即如果序列中的元素是Symbol对象，Symbol转换字符串
    :return:
    """
    if market is None:
        # None则服从ABuEnv.g_market_target市场设置
        market = ABuEnv.g_market_target

    if market == EMarketTargetType.E_MARKET_TARGET_US:
        symbols = _all_us_symbol(index)
    elif market == EMarketTargetType.E_MARKET_TARGET_CN:
        symbols = _all_cn_symbol(index)
    elif market == EMarketTargetType.E_MARKET_TARGET_HK:
        symbols = _all_hk_symbol(index)
    elif market == EMarketTargetType.E_MARKET_TARGET_FUTURES_CN:
        symbols = _all_futures_cn()
    elif market == EMarketTargetType.E_MARKET_TARGET_FUTURES_GLOBAL:
        symbols = _all_futures_gb()
    elif market == EMarketTargetType.E_MARKET_TARGET_TC:
        symbols = _all_tc_symbol()
    else:
        raise TypeError('JUST SUPPORT EMarketTargetType!')

    # 在出口统一确保唯一性, 在每一个内部_all_xx_symbol中也要尽量保证唯一
    symbols = list(set(symbols))

    if value:
        """
            如果是Symbol类型的还原成字符串，尽量在上面返回的symbols序列是字符串类型
            不要在上面构造symbol，浪费效率，统一会在之后的逻辑中通过code_to_symbol
            进行转换
        """
        symbols = [sb.value if isinstance(sb, Symbol) else sb for sb in symbols]
    # 根据参数ss是否将返回序列使用pd.Series包装
    return pd.Series(symbols) if ss else symbols


def query_symbol_market(target_symbol):
    """
    查询target_symbol所对应的市场对象EMarketTargetType
    :param target_symbol: 支持Symbol对象类型和字符串对象类型
    :return: EMarketTargetType对象
    """
    symbol_obj = None
    if target_symbol is None:
        symbol_obj = None
    elif isinstance(target_symbol, Symbol):
        symbol_obj = target_symbol
    elif isinstance(target_symbol, six.string_types):
        try:
            # 如果字符串通过code_to_symbol转换为Symbol对象类型
            symbol_obj = code_to_symbol(target_symbol)
        except:
            return None
    if symbol_obj is not None:
        # 返回市场类型
        return symbol_obj.market
    return None


def market_train_test_split(n_folds, market_symbols, market=None):
    """
    切割训练集与测试集，本地训练化保存，只返回训练集
    :param n_folds: 切割比例，透传KFold中使用的参数
    :param market_symbols: 待切分的总market_symbols
    :param market: 待切分的市场，eg：EMarketTargetType.E_MARKET_TARGET_US
    :return: 返回训练集symbols数据
    """
    market_symbols, test_symbols = _do_market_train_test_split(n_folds=n_folds, market_symbols=market_symbols)

    if market is None:
        # None则服从ABuEnv.g_market_target市场设置
        market = ABuEnv.g_market_target
    market_name = market.value
    # 匹配对应的市场组成市场名称
    last_path_train = '{}_{}'.format(K_MARKET_TRAIN_FN_BASE, market_name)
    last_path_test = '{}_{}'.format(K_MARKET_TEST_FN_BASE, market_name)

    tt = datetime.datetime.now().strftime('%Y_%m_%d_%H_%M_%S_%f')
    if ABuFileUtil.file_exist(last_path_train):
        # 之前如果存在训练集切割数据，使用当前时间做记号，另保存起来
        store_train_name = '{}_{}'.format(last_path_train, tt)
        os.rename(last_path_train, store_train_name)
    if ABuFileUtil.file_exist(last_path_test):
        # 之前如果存在测试集切割数据，使用当前时间做记号，另保存起来
        store_test_name = '{}_{}'.format(last_path_test, tt)
        os.rename(last_path_test, store_test_name)

    # 本地序列化测试集数据
    ABuFileUtil.dump_pickle(test_symbols, last_path_test)
    # 本地序列化训练集数据
    ABuFileUtil.dump_pickle(market_symbols, last_path_train)
    # 只返回训练集symbols数据
    return market_symbols


def market_last_split_test(market=None):
    """
    使用最后一次切割好的测试集symbols数据
    :param market: 待获取测试集市场，eg：EMarketTargetType.E_MARKET_TARGET_US
    :return: 最后一次切割好的测试集symbols数据
    """
    if market is None:
        # None则服从ABuEnv.g_market_target市场设置
        market = ABuEnv.g_market_target
    market_name = market.value
    # 匹配对应的市场组成市场名称
    last_path_test = '{}_{}'.format(K_MARKET_TEST_FN_BASE, market_name)

    if not ABuFileUtil.file_exist(last_path_test):
        raise RuntimeError('g_enable_last_split_test not fileExist(fn)!')
    market_symbols = ABuFileUtil.load_pickle(last_path_test)
    return market_symbols


def market_last_split_train(market=None):
    """
    使用最后一次切割好的训练集symbols数据
    :param market: 待获取测试集市场，eg：EMarketTargetType.E_MARKET_TARGET_US
    :return: 最后一次切割好的训练集symbols数据
    """
    if market is None:
        # None则服从ABuEnv.g_market_target市场设置
        market = ABuEnv.g_market_target
    market_name = market.value
    # 匹配对应的市场组成市场名称
    last_path_train = '{}_{}'.format(K_MARKET_TRAIN_FN_BASE, market_name)

    if not ABuFileUtil.file_exist(last_path_train):
        raise RuntimeError('g_enable_last_split_train not ZCommonUtil.fileExist(fn)!')
    market_symbols = ABuFileUtil.load_pickle(last_path_train)
    return market_symbols


def _do_market_train_test_split(n_folds=10, market_symbols=None, shuffle=True, market=None):
    """
    分割市场训练集与测试集
    :param market_symbols，备选symbols序列，如果None, 则从参数market选择全市场symbol做为备选
    :param n_folds: 切割比例，KFold中使用的参数
    :param shuffle: 是否将原始序列打乱，默认True
    :param market: 默认None，如None则服从ABuEnv.g_market_target市场设置
    :return:
    """
    if market is None:
        # 如None则服从ABuEnv.g_market_target市场设置
        market = ABuEnv.g_market_target

    if market_symbols is None:
        # 如果None, 则从参数market选择全市场symbol做为备选
        market_symbols = all_symbol(market=market, ss=True)

    if not isinstance(market_symbols, np.ndarray):
        # 使用np.array包装
        market_symbols = np.array(market_symbols)

    n_folds = n_folds if len(market_symbols) > n_folds else len(market_symbols)
    # 使用KFold对market_symbols进行训练集，测试集拆分
    kf = KFold(len(market_symbols), n_folds=n_folds, shuffle=shuffle)
    train = None
    test = None
    for train_index, test_index in kf:
        train, test = market_symbols[train_index], market_symbols[test_index]
        # 暂时只保留一组就够了，多层级切割回测还未进行代码迁移
        break
    if train is not None and test is not None:
        # noinspection PyUnresolvedReferences
        return train.tolist(), test.tolist()
    return list(), list()


def is_in_sand_box(symbol):
    """判定symbol是否在沙盒数据支持里面"""
    cs = code_to_symbol(symbol, rs=False)
    if cs is None:
        return False
    if cs.is_futures() or cs.is_tc():
        # 沙盒数据支持完整期货和电子货币市场
        return True
    if symbol in K_SAND_BOX_CN \
            or symbol in K_SAND_BOX_US \
            or symbol in K_SAND_BOX_HK:
        # A股，美股，港股沙盒数据需要在沙盒数据序列中
        return True
    return False


"""＊＊＊＊＊＊＊＊＊＊＊＊＊＊＊＊＊＊＊＊＊＊＊Deprecated 旧数据格式文件＊＊＊＊＊＊＊＊＊＊＊＊＊＊＊＊＊＊＊＊＊＊＊＊＊＊＊＊"""
_rom_dir = ABuEnv.g_project_rom_data_dir
_stock_code_cn = os.path.join(_rom_dir, 'stock_code_cn.txt')
_stock_code_us = os.path.join(_rom_dir, 'stock_code_us.txt')
_stock_code_hk = os.path.join(_rom_dir, 'stock_code_hk.txt')


@AbuDeprecated()
def _parse_code(line, index):
    if ABuEnv.g_is_py3:
        line = line.decode()
    _slice = line.split('|')
    market = _slice[0]
    is_index = _slice[1] == 4
    code = _slice[2]
    if market == 'us':
        m_type = AbuSymbolUS().query_symbol_sub_market(code)
    else:
        m_type = EMarketSubType(market)
    if m_type is not None and not (not index and is_index):
        return Symbol(ABuEnv.g_market_target, m_type, code)


@AbuDeprecated('only read old symbol, miss update!!!')
def _all_us_symbol_deprecated(index=False):
    """
    获取美股全市场股票代码
    :param index: 是否包含指数
    :return:
    """
    with open(_stock_code_us, 'rb') as us_f:
        return list(filter(lambda x: x is not None, [_parse_code(line, index) for line in us_f.readlines()]))


@AbuDeprecated('only read old symbol, miss update!!!')
def _all_cn_symbol_deprecated(index=False):
    """
    获取A股全市场股票代码
    :param index: 是否包含指数
    :return:
    """
    with open(_stock_code_cn, 'rb') as cn_f:
        return list(filter(lambda x: x is not None, [_parse_code(line, index) for line in cn_f.readlines()]))


@AbuDeprecated('only read old symbol, miss update!!!')
def _all_hk_symbol_deprecated(index=False):
    """
    获取港股全市场股票代码
    :param index: 是否包含指数
    :return:
    """
    with open(_stock_code_hk, 'rb') as hk_f:
        return list(filter(lambda x: x is not None, [_parse_code(line, index) for line in hk_f.readlines()]))
