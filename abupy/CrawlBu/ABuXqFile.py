# -*- encoding:utf-8 -*-
from __future__ import print_function
from __future__ import absolute_import
from __future__ import division

from os import path
import os
from ..CoreBu.ABuEnv import g_project_data_dir
from ..CoreBu.ABuEnv import g_project_cache_dir
from ..CoreBu.ABuEnv import g_project_rom_data_dir
from ..UtilBu import ABuFileUtil
from ..UtilBu.ABuProgress import AbuProgress

from .ABuXqConsts import columns_map

import pandas as pd

__author__ = '小青蛙'
__weixin__ = 'abu_quant'

# 保存股票信息
STOCK_INFO_DIR = path.join(g_project_data_dir, 'stock_info')
# 保存爬取得股票信息
STOCK_INFO_CACHE_DIR = path.join(g_project_cache_dir, 'stock_info')
STOCK_INFO_ERROR_DIR = path.join(g_project_cache_dir, 'stock_info/error')

ABuFileUtil.ensure_dir(STOCK_INFO_DIR)
ABuFileUtil.ensure_dir(STOCK_INFO_CACHE_DIR)
ABuFileUtil.ensure_dir(STOCK_INFO_ERROR_DIR)


def map_stock_list_rom(market):
    return path.join(g_project_rom_data_dir, 'stock_code_{}.csv'.format(market))


def map_stock_list(market):
    return path.join(STOCK_INFO_DIR, 'stock_code_{}.csv'.format(market))


def map_cache_stock_info(market, symbol):
    """
    临时存取 stock info
    :param market:
    :param symbol:
    :return:
    """
    p = path.join(STOCK_INFO_CACHE_DIR, '{}_{}.txt'.format(market, symbol))
    return p


def save_cache_stock_info(stock_info, market, symbol):
    if stock_info is not None:
        df = pd.DataFrame.from_dict(stock_info, orient='index')
        df.to_csv(map_cache_stock_info(market, symbol))


def exist_stock_info(market, symbol):
    return ABuFileUtil.file_exist(map_cache_stock_info(market, symbol))


def read_stock_symbol_list(market):
    return pd.read_csv(map_stock_list(market))['symbol'].tolist()


def read_stock_symbol_name(market, symbol):
    df = pd.read_csv(map_stock_list(market))
    item = df.loc[df.symbol == symbol]
    return None if item is None else item.name.tolist()[0]


def error_stock_info(market, symbol, error_info):
    """
    抓取stock info 失败 保存到磁盘，以便，下次查看，由于多进程的原因，每个错占用一个空文件,避免多进程共享文件可能出错的问题
    :param error_info:
    :param market:
    :param symbol:
    :return:
    """
    p = path.join(STOCK_INFO_ERROR_DIR, '{}_{}'.format(market, symbol))
    ABuFileUtil.save_file(error_info, p)
    return p


def read_all_error_stock_info_symbol():
    """
    :return: 返回抓取stock info失败的symbol列表
    """
    error_symbols = os.listdir(STOCK_INFO_ERROR_DIR)
    return map(lambda error: error.split('_'), error_symbols)


def query_a_stock(market, symbol):
    stock_df = pd.read_csv(map_stock_list(market))
    return stock_df.loc[stock_df.symbol == symbol]


def _create_a_column(info, columns, size):
    """
    创建size大小的None数组
    :param info:
    :param columns:
    :param size:
    """
    for c in columns:
        if c not in info.keys():
            info[c] = [None] * size


def merge_stock_info_to_stock_list(market=('US', 'HK', 'CN')):
    for m in market:
        stock_df = pd.read_csv(map_stock_list(m), dtype=str)
        extra_info = {}
        # 08012
        progess = AbuProgress(stock_df.shape[0], 0, 'merging {}'.format(m))
        for i, symbol in enumerate(stock_df['symbol']):
            progess.show(i + 1)
            if ABuFileUtil.file_exist(map_cache_stock_info(m, symbol)):
                a_stock_info = pd.read_csv(map_cache_stock_info(m, symbol), dtype=str)

                if a_stock_info is not None and not a_stock_info.empty:
                    keys = a_stock_info.ix[:, 0].tolist()
                    _create_a_column(extra_info, keys, stock_df.shape[0])
                    values = a_stock_info.ix[:, 1].tolist()
                    for k, v in zip(keys, values):
                        extra_info[k][i] = v

        for key in extra_info:
            stock_df[key] = extra_info[key]

        stock_df.fillna('-', inplace=True)
        # 某些symbol的stockinfo为空，，stockinfp为空的原因是stockinfo页面404，因此可以丢弃
        valid_df = stock_df.loc[stock_df.symbol != '-']
        valid_df.to_csv(map_stock_list_rom(m), index=False, encoding='utf-8')


def del_columns(df, columns):
    old_c = df.columns.tolist()
    for col in filter(lambda x: x in old_c, columns):
        df.drop(col, axis=1, inplace=True)


def drop_nuisance(df):
    del_columns(df, ['percent', 'pettm', 'marketcapital', 'volume', 'low52w', 'low', 'high52w', 'high', 'hasexist',
                     'current', 'amount', 'change', 'pe_ttm', 'open', 'last_close', 'chg', 'market_capital',
                     'code'])
    return df


def fix_xq_columns_name():
    """
    雪球获取的数据的key都是中文，dataframe的columns不变与用中文
    """
    for m in ('US', 'CN', 'HK'):
        stock_df = pd.read_csv(map_stock_list_rom(m), dtype=str)
        unnecessary_columns = stock_df.columns.difference(columns_map.keys())
        columns_intersection = stock_df.columns & columns_map.keys()
        del_columns(stock_df, unnecessary_columns)

        stock_df.rename(columns={c: columns_map[c] for c in columns_intersection}, inplace=True)
        stock_df.to_csv(map_stock_list_rom(m), index=True, encoding='utf-8')
