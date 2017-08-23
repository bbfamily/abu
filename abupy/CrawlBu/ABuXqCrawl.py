# -*- encoding:utf-8 -*-
from __future__ import print_function
from __future__ import absolute_import
from __future__ import division

import pandas as pd

import math
import logging

from os import path

from abupy.CrawlBu import ABuXqFile
from ..CoreBu.ABuEnvProcess import add_process_env_sig
from ..CrawlBu.ABuXqFile import map_stock_list
from .ABuXqCrawlImp import StockInfoListBrower, StockListCrawlBrower
from .ABuXqCrawlImp import NavHQCrawlBrower

from ..CoreBu.ABuEnv import g_project_rom_data_dir
from ..CoreBu.ABuParallel import Parallel, delayed
from ..UtilBu import ABuDTUtil

# noinspection PyUnresolvedReferences
from ..CoreBu.ABuFixes import map, reduce, filter

__author__ = '小青蛙'
__weixin__ = 'abu_quant'

_NAV_PATH = path.join(g_project_rom_data_dir, 'hq_nav.txt')


@add_process_env_sig
def __crawl_stock_parallel(market, url):
    with StockListCrawlBrower(url) as crawler:
        name, symbol = crawler.crawl()
        df = pd.DataFrame({'name': name, 'symbol': symbol})
        if market == 'CN':
            # 确保都是A股正确的symbol
            df = df.loc[
                list(
                    map(lambda s: (s[:2] in ['SZ', 'SH', 'sz', 'sh']) and str(s[2:]).isdigit(), df['symbol'].tolist()))]
            df['market'] = df['symbol'].map(lambda s: s[:2])
        else:
            df['market'] = market

        df.to_csv(map_stock_list(market), index=False)


@add_process_env_sig
def __crawl_stock_info_parallel(market, symbols, process):
    with StockInfoListBrower(market, symbols) as stockInfo:
        rest = stockInfo.crawl(process=process)
        print(rest)


def hq_nav():
    """
    :return: dict
    一级菜单：类别，如。美股，港股，沪深，基金，债券等
    二级菜单：如，行业分类，美股一览等
    三级菜单：可能会有(,比如行业就需细分)
    """
    with NavHQCrawlBrower() as nav:
        return nav.crawl()


@ABuDTUtil.warnings_filter
def crawl_stock_code(markets=('CN', 'HK', 'US')):
    """
    从雪球获取市场股市代码
    :param markets: 市场类型
    """
    nav = hq_nav()
    urls = dict()
    for market_nav in nav:
        for first_menu in nav[market_nav]:
            if first_menu in [u'沪深一览'] and 'CN' in markets:
                urls['CN'] = nav[market_nav][first_menu]
            if first_menu in [u'港股一览'] and 'HK' in markets:
                urls['HK'] = nav[market_nav][first_menu]
            if first_menu in [u'美股一览'] and 'US' in markets:
                urls['US'] = nav[market_nav][first_menu]

    Parallel(n_jobs=-1)(delayed(__crawl_stock_parallel)(m, urls[m]) for m in urls)


def crawl_stock_info(markets):
    """
    获取股票信息，例如，股市简介，行业，市值等信息
    :param markets:
    """
    _markets = []
    _parts = []
    for m in markets:
        symbols = ABuXqFile.read_stock_symbol_list(m)
        interval = 1000

        # 将 symbols 划分成一组1000个
        for s in range(int(math.ceil(len(symbols) / interval))):
            st = s * interval
            en = (s + 1) * interval if (s + 1) * interval <= len(symbols) else len(symbols)
            _parts.append(symbols[st:en])
            _markets.append(m)
    # 开启len(_parts)个进程同时
    print('任务数{}'.format(len(_parts)))

    parallel = Parallel(n_jobs=-1)
    parallel(delayed(__crawl_stock_info_parallel)(m, p, i)
             for i, (m, p) in enumerate(zip(_markets, _parts)))


def ensure_symbol(symbol):
    """
    保证本地的股市代码库纯在该symbol才能查看stock info
    A 股 sh、sz开头 + 6位数字
    HK 股 5位数字
    UK 股 英文代码
    :param symbol:
    :return:
    """
    symbol = str(symbol).upper()
    if (len(symbol) == 6 and str(symbol).isdigit()) or (
                        len(symbol) == 8 and symbol[:2] in ['SZ', 'SH'] and str(symbol[2:]).isdigit()):
        market = 'CN'
    elif (len(symbol) == 5 and str(symbol).isdigit()) or (
                        len(symbol) == 7 and symbol[:2] in ['HK'] and str(symbol[2:]).isdigit()):
        market = 'HK'
    else:
        market = 'US'
    stock_list = ABuXqFile.read_stock_symbol_list(market)

    if market in ['CN']:
        # symbol自带 sh或sz
        if len(symbol) == 8:
            return market, symbol
        # 此刻 symbol是6位数字
        sh_code = 'SH{}'.format(symbol)
        sz_code = 'SZ{}'.format(symbol)
        is_sh = sh_code in stock_list
        is_sz = sz_code in stock_list

        if is_sz and is_sh:
            # 即使 上证又是深圳股
            sh_name = ABuXqFile.read_stock_symbol_name(market, sh_code)
            sz_name = ABuXqFile.read_stock_symbol_name(market, sz_code)
            logging.error(
                '找到两个关于{}的股票{}({}),{}({}),请改写{}成{}或{}'.format(
                    symbol, sh_code, sh_name, sz_code, sz_name, symbol, sz_code, sh_code))
            return market, None
        elif is_sh:
            return market, sh_code
        elif is_sz:
            return sz_code
        else:
            pass
    elif market in ['US', 'HK']:
        if symbol in stock_list:
            return market, symbol

    logging.error('没有找到{}相关的股票'.format(symbol))
    return market, None


def update_all(markets=('US', 'CN', 'HK')):
    crawl_stock_code(markets)
    crawl_stock_info(markets)
    ABuXqFile.merge_stock_info_to_stock_list(markets)
    ABuXqFile.fix_xq_columns_name()


def query_symbol_info(symbol):
    m, symbol = ensure_symbol(symbol)
    return None if symbol is None else ABuXqFile.query_a_stock(m, symbol)
