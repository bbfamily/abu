# -*- encoding:utf-8 -*-
"""
    行业分类模块，仅支持美股，a股，港股
"""

from __future__ import print_function
from __future__ import absolute_import
from __future__ import division

from collections import Iterable
import logging
from fnmatch import fnmatch

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from ..MarketBu import ABuSymbolPd
from ..UtilBu.ABuStrUtil import to_unicode
from ..UtilBu import ABuScalerUtil
from ..MarketBu.ABuSymbolStock import AbuSymbolUS, AbuSymbolCN, AbuSymbolHK
from ..CoreBu.ABuEnv import EMarketDataSplitMode, EMarketTargetType
from ..CoreBu import ABuEnv


def industries_df(target_symbol):
    """
    分别查询target_symbol是否在美股，a股，港股中有对应的行业分类，如果查询到
    返回查询的结果industries（pd.DataFrame对象）以及 所属symbol市场类，即
    AbuSymbolStockBase子类
    :param target_symbol: symbol str对象
    :return: （返回查询的结果industries: pd.DataFrame对象, 所属symbol市场类: AbuSymbolStockBase子类）
    """
    industries = AbuSymbolUS().query_industry_symbols(target_symbol)
    if industries is not None:
        # 美股查到了有行业分类结果
        return industries, AbuSymbolUS()

    industries = AbuSymbolCN().query_industry_symbols(target_symbol)
    if industries is not None:
        # a股查到了有行业分类结果
        return industries, AbuSymbolCN()

    industries = AbuSymbolHK().query_industry_symbols(target_symbol)
    if industries is not None:
        # 港股查到了有行业分类结果
        return industries, AbuSymbolHK()
    # 都没查到，返回None, None
    return None, None


def industries_factorize(market=None):
    """
    查询market所在市场的行业分类离散值，默认market=None，即使用ABuEnv.g_market_target
    :param market: 需要查询的市场，eg：EMarketTargetType.E_MARKET_TARGET_US
    :return: 对应market的行业分类简述和factorize值，pd.Series对象
            eg：
                1      中国食品、饮料与烟草
                2        油气/钻孔与探测
                3          出版业/报纸
                4            生物技术
                5          数据存储设备
                6            教育培训
                7              电视
                8          中国网络游戏
                9           中国新能源
                          ...
                249       食品多样化经营
                250       电影制作及影院
                251           农产品
                252       烟草制品及其它
                253             铝
                254       汽车配件批发商
                255         家具装饰店
                256          乳酪产品
                257          医药生物
                258        中国酒店餐饮
           eg：查询A股市场行业：
                input：
                        ABuIndustries.industries_factorize(market=EMarketTargetType.E_MARKET_TARGET_CN)
                output：
                        1     商业贸易
                        2     有色金属
                        3     电气设备
                        4     家用电器
                        5     建筑装饰
                        6      计算机
                        7     轻工制造
                        8     机械设备
                        9     医药生物
                              ...
                        25    国防军工
                        26    A股指数
                        27    非银金融
                        28    建筑建材
                        29      银行
                        30    信息设备
                        31      钢铁
                        32    交运设备
                        33    餐饮旅游
                        34    黑色金属
    """
    if market is None:
        # None则服从ABuEnv.g_market_target市场设置
        market = ABuEnv.g_market_target

    if market == EMarketTargetType.E_MARKET_TARGET_US:
        return AbuSymbolUS().industry_factorize_name_series
    elif market == EMarketTargetType.E_MARKET_TARGET_CN:
        return AbuSymbolCN().industry_factorize_name_series
    elif market == EMarketTargetType.E_MARKET_TARGET_HK:
        return AbuSymbolHK().industry_factorize_name_series
    # 仅支持美股，a股，港股
    raise TypeError('JUST SUPPORT US, CN, HK!')


def industries_market(market=None):
    """
    查询market所在市场的句柄对象，即：
        美股市场E_MARKET_TARGET_US：AbuSymbolUS对象
        a股市场E_MARKET_TARGET_CN：AbuSymbolCN对象
        港股市场E_MARKET_TARGET_HK：AbuSymbolHK对象
    :param market: 需要查询的市场，eg：EMarketTargetType.E_MARKET_TARGET_US
    :return: AbuSymbolUS对象 or AbuSymbolCN对象 or AbuSymbolHK对象
    """
    if market is None:
        # None则服从ABuEnv.g_market_target市场设置
        market = ABuEnv.g_market_target

    if market == EMarketTargetType.E_MARKET_TARGET_US:
        industries_market_op = AbuSymbolUS()
    elif market == EMarketTargetType.E_MARKET_TARGET_CN:
        industries_market_op = AbuSymbolCN()
    elif market == EMarketTargetType.E_MARKET_TARGET_HK:
        industries_market_op = AbuSymbolHK()
    else:
        raise TypeError('JUST SUPPORT US, CN, HK!')
    # 返回市场的句柄操作对象
    return industries_market_op


def match_industries_factorize(match, market=None):
    """
        通过模糊查询market所在市场中match关键字所示的所有行业factorize，
        获取factorize后可通过query_factorize_industry_df将factorize
        所示行业进行获取
        eg：
        input：
                ABuIndustries.match_industries_factorize('中国')
        output：
                [(1, '中国食品、饮料与烟草'),
                 (8, '中国网络游戏'),
                 (9, '中国新能源'),
                 (22, '中国汽车与汽车零部件'),
                 (31, '中国制药、生物科技和生命科学'),
                 (32, '中国金融'),
                 (33, '中国互联网软件服务'),
                 (41, '中国金属与采矿'),
                 (54, '中国建筑材料'),
                 (66, '中国硬件、半导体与设备'),
                 (79, '中国运输'),
                 (81, '中国化学制品'),
                 (114, '中国互联网信息服务'),
                 (169, '中国房地产'),
                 (195, '中国电子商务'),
                 (212, '中国耐用消费品与服装'),
                 (214, '中国一般制造业'),
                 (216, '中国媒体'),
                 (217, '中国日消品零售'),
                 (220, '中国软件与服务'),
                 (223, '中国传统能源'),
                 (224, '中国能源设备与服务'),
                 (228, '中国纸业与包装'),
                 (232, '中国商业与专业服务'),
                 (237, '中国教育培训'),
                 (238, '中国医疗保健设备与服务'),
                 (240, '中国非日消品零售'),
                 (258, '中国酒店餐饮')]
    :param match: 匹配的行业关键字，支持通配符，eg：'医药*'， '*互联网*', '*科技'
    :param market: 需要查询的市场，eg：EMarketTargetType.E_MARKET_TARGET_US
    :return: 匹配的list序列对象，序列中每一个元素为(factorize, 本地描述)，eg：(33, '中国互联网软件服务')
    """
    match = to_unicode(match)
    _industries_factorize = industries_factorize(market=market)
    if u'*' not in match:
        # 如果不带＊那就前后加＊进行match
        match = u'*{}*'.format(match)
    match_list = list()
    for factorize, industries in enumerate(_industries_factorize):
        if fnmatch(to_unicode(industries), match):
            # 使用fnmatch进行匹配
            match_list.append((factorize, industries))

    match = match.replace(u'*', u'')
    if len(match_list) == 0 and len(match) > 1:
        # 如果第一次模糊查询没有查到，开始一个字一个字的进行模糊查询，eg：*教育*－>*教*－>*育*
        for match_pos in np.arange(0, len(match)):
            # 伪递归，迭代递归方法里面由于match = match.replace('*', '')所以不满足迭代再次递归的条件：len(match) > 1
            match_list = match_industries_factorize(match[match_pos], market=market)
            if len(match_list) > 0:
                # 一旦其中一个字的子模糊查询查到了，就直接返回eg：*教*
                break

    # TODO: 如果还没有查到，添加按照拼音等模糊查询匹配方式
    return match_list


def query_match_industries_df(match, market=None):
    """
    通过模糊查询market所在市场中match关键字所示的所有行业信息组装成pd.DataFrame对象
    :param match: 匹配的行业关键字，支持通配符，eg：'医药*'， '*互联网*', '*科技'
    :param market: 需要查询的市场，eg：EMarketTargetType.E_MARKET_TARGET_US
    :return: 返回行业组合pd.DataFrame对象
    """
    # 获取模糊查询factorize序列
    match_list = match_industries_factorize(match, market=market)
    # 通过industries_market获取对应市场操作句柄industries_market_op
    industries_market_op = industries_market(market=market)

    industry_df = None
    for factorize, _ in match_list:
        # 将所有查询到的行业信息，pd.DtaFrame对象连接起来
        query_industry = industries_market_op.query_industry_factorize(factorize)
        industry_df = query_industry if industry_df is None else pd.concat([query_industry, industry_df])
    if industry_df is not None:
        # 去除重复的，比如a在行业b，又在行业c
        # noinspection PyUnresolvedReferences
        industry_df.drop_duplicates(inplace=True)

    return industry_df


def query_factorize_industry_df(factorize_arr, market=None):
    """
    使用match_industries_factorize可以查询到行业所对应的factorize序列，
    使用factorize序列即组成需要查询的行业组合，返回行业组合pd.DataFrame对象
    eg: 从美股所有行业中找到中国企业的行业
        input：ABuIndustries.match_industries_factorize('中国', market=EMarketTargetType.E_MARKET_TARGET_US)
        output：
            [(1, '中国食品、饮料与烟草'),
             (8, '中国网络游戏'),
             (9, '中国新能源'),
             (22, '中国汽车与汽车零部件'),
             (31, '中国制药、生物科技和生命科学'),
             (32, '中国金融'),
             (33, '中国互联网软件服务'),
             (41, '中国金属与采矿'),
             (54, '中国建筑材料'),
             (66, '中国硬件、半导体与设备'),
             (79, '中国运输'),
             (81, '中国化学制品'),
             (114, '中国互联网信息服务'),
             (169, '中国房地产'),
             (195, '中国电子商务'),
             (212, '中国耐用消费品与服装'),
             (214, '中国一般制造业'),
             (216, '中国媒体'),
             (217, '中国日消品零售'),
             (220, '中国软件与服务'),
             (223, '中国传统能源'),
             (224, '中国能源设备与服务'),
             (228, '中国纸业与包装'),
             (232, '中国商业与专业服务'),
             (237, '中国教育培训'),
             (238, '中国医疗保健设备与服务'),
             (240, '中国非日消品零售'),
             (258, '中国酒店餐饮')]

        然后使用ABuIndustries.query_factorize_industry_df((31, 32, 33))即可获取到
             (31, '中国制药、生物科技和生命科学'),
             (32, '中国金融'),
             (33, '中国互联网软件服务'),
        行业中的所有股票信息的pd.DataFrame对象

    :param factorize_arr: eg：(31, 32, 33) or [31, 32, 33] or 31
    :param market: 需要查询的市场，eg：EMarketTargetType.E_MARKET_TARGET_US
    :return: 返回行业组合pd.DataFrame对象
    """
    if not isinstance(factorize_arr, Iterable):
        # 如果不是可迭代的，即只是一个factorize序列，转换为序列，方便统一处理
        factorize_arr = [factorize_arr]
    # 通过industries_market获取对应市场操作句柄industries_market_op
    industries_market_op = industries_market(market=market)

    industry_df = None
    for ind in factorize_arr:
        query_industry = industries_market_op.query_industry_factorize(ind)
        # 将所有查询到的行业信息，pd.DtaFrame对象连接起来
        industry_df = query_industry if industry_df is None else pd.concat([query_industry, industry_df])
    if industry_df is not None:
        # 去除重复的，比如a在行业b，又在行业c
        # noinspection PyUnresolvedReferences
        industry_df.drop_duplicates(inplace=True)

    return industry_df


def query_factorize_industry_symbol(factorize, market=None):
    """
    套接query_factorize_industry_df方法，只返回df在的symbol序列：
        query_factorize_industry_df(factorize, market=market).symbol
    :param factorize: factorize_arr: eg：(31, 32, 33) or [31, 32, 33] or 31
    :param market: 需要查询的市场，eg：EMarketTargetType.E_MARKET_TARGET_US
    :return: pd.Series对象
            eg：
                1. 首先使用industries_factorize查询到outputa股行业分类
                input：
                    ABuIndustries.industries_factorize(market=EMarketTargetType.E_MARKET_TARGET_CN)
                output：
                1     商业贸易
                2     有色金属
                3     电气设备
                4     家用电器
                5     建筑装饰
                6      计算机
                7     轻工制造
                8     机械设备
                9     医药生物
                      ...
                29      银行
                30    信息设备
                31      钢铁
                32    交运设备
                33    餐饮旅游
                34    黑色金属

                2. 使用query_factorize_industry_symbol((6, 9, 29))，查询计算机，医药生物，银行行业symbool
                input：
                    ABuIndustries.query_factorize_industry_symbol((6, 9, 29),
                                  market=EMarketTargetType.E_MARKET_TARGET_CN)
                output：
                    ['sh601939',
                     'sh601398',
                     'sh601288',
                     'sh601328',
                     'sh601009',
                     ..........
                     ..........
                     'sz300513',
                     'sz002236',
                     'sz300044',
                     'sz300302',
                     'sh600756']
    """
    factorize_df = query_factorize_industry_df(factorize, market=market)
    if factorize_df is not None:
        # 通过industries_market获取对应市场操作句柄industries_market_op
        industries_market_op = industries_market(market=market)
        # 通过对应的市场对象op自己去从factorize_df中组装symbol序列
        return industries_market_op.symbol_func(factorize_df)
    # factorize没有匹配上直接返回空序列
    return list()


def query_match_industries_symbol(match, market=None):
    """
    套接query_match_industries_df方法，只返回df在的symbol序列：
        query_match_industries_df(factorize, market=market).symbol
    :param match: 匹配的行业关键字，支持通配符，eg：'医药*'， '*互联网*', '*科技'
    :param market: 需要查询的市场，eg：EMarketTargetType.E_MARKET_TARGET_US
    :return:
            eg：
                input：ABuIndustries.query_match_industries_symbol('医疗', market=EMarketTargetType.E_MARKET_TARGET_CN)
                output：
                        ['sz002826',
                         'sh600645',
                         'sz300534',
                         'sh600080',
                         'sz300595',
                         ..........
                         ..........
                         'sh600518',
                         'sh900904',
                         'sh600993',
                         'sh600332',
                         'sz002589']
    """
    match_df = query_match_industries_df(match, market=market)
    if match_df is not None:
        # 通过industries_market获取对应市场操作句柄industries_market_op
        industries_market_op = industries_market(market=market)
        # 通过对应的市场对象op自己去从factorize_df中组装symbol序列
        return industries_market_op.symbol_func(match_df)
    # match没有匹配上直接返回空序列
    return list()


def get_industries_panel_from_target(target_symbol, show=False, n_folds=2):
    """
    获取target_symbol所在的行业分类中所有的金融时间序列，组成三维pd.Panel对象
    :param target_symbol: symbol str对象
    :param show: 是否可视化行业分类中所有的金融时间序列
    :param n_folds: 获取n_folds年历史交易数据
    :return: (pd.Panel对象, p_date.swapaxes('items', 'minor'))
    """
    df, s_obj = industries_df(target_symbol)
    if df is not None:
        # 通过symbol_func转换为可直接使用ABuSymbolPd.make_kl_df请求的target_symbols序列
        target_symbols = s_obj.symbol_func(df)

        from ..TradeBu.ABuBenchmark import AbuBenchmark
        # 以target_symbol做为标尺实例化benchmark对象为make_kl_df做准备
        benchmark = AbuBenchmark(target_symbol, n_folds=n_folds)
        # 传递序列target_symbols，返回的p_date是pd.Panel对象
        p_date = ABuSymbolPd.make_kl_df(target_symbols, data_mode=EMarketDataSplitMode.E_DATA_SPLIT_UNDO,
                                        benchmark=benchmark, n_folds=n_folds)
        # 内部做个转轴处理，方便外部使用
        p_data_it = p_date.swapaxes('items', 'minor')
        if show:
            # 可视化行业分类中所有的金融时间序列
            data = p_data_it['close']
            data = ABuScalerUtil.scaler_std(data)
            data.plot()
            plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
            plt.ylabel('Price')
            plt.xlabel('Time')
        return p_date, p_data_it
    else:
        logging.info('Industries targetSymbols len = 0')
        return None, None
