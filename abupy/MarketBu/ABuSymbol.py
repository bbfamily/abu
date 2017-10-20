# coding=utf-8
"""
    symbol模块
"""
from __future__ import print_function
from __future__ import absolute_import
from __future__ import division

from fnmatch import fnmatch

import numpy as np

from ..CoreBu.ABuEnv import EMarketTargetType, EMarketSubType
from ..CoreBu.ABuFixes import six
from ..UtilBu.ABuStrUtil import to_unicode
from ..UtilBu.ABuLazyUtil import LazyFunc


# noinspection PyProtectedMember
def code_to_symbol(code, rs=True):
    """
    解析code成Symbol,如果code中带有市场编码直接用该市场，否则进行查询所属市场信息，
    如果最后也没有发现symbol所在的市场，会向外raise ValueError
    :param code: str对象，代码 如：300104，sz300104，usTSLA
    :param rs: 没有匹配上是否对外抛异常，默认True
    :return: Symbol对象
    """
    from ..MarketBu.ABuSymbolFutures import AbuFuturesCn, AbuFuturesGB
    from ..MarketBu.ABuSymbolStock import AbuSymbolCN, AbuSymbolUS
    from ..MarketBu.ABuMarket import all_symbol

    if isinstance(code, Symbol):
        # code本身时symbol对象直接返回
        return code
    if not isinstance(code, six.string_types):
        # code必须是string_types
        raise TypeError('code must be string_types!!!，{} : type is {}'.format(code, type(code)))

    sub_market = None
    market = None
    # 尝试获取市场信息
    head = code[:2].lower()
    if head in [EMarketSubType.SH.value, EMarketSubType.SZ.value] and code[2:].isdigit():
        # 市场信息匹配沪深股，查询子市场
        sub_market = EMarketSubType(head)
        market = EMarketTargetType.E_MARKET_TARGET_CN
    elif head == EMarketTargetType.E_MARKET_TARGET_CN.value and code[2:].isdigit():
        # 没指名A股子市场，使用如hs000001这种形式，需求获取symbol后查询子市场使用AbuSymbolCN().query_symbol_sub_market
        sub_market = EMarketSubType(AbuSymbolCN().query_symbol_sub_market(code[2:]))
        market = EMarketTargetType.E_MARKET_TARGET_CN
    elif head == EMarketSubType.HK.value and (code[2:].isdigit() or code[2:] in Symbol.HK_INDEX):
        # 市场信息匹配港股，查询子市场
        sub_market = EMarketSubType.HK
        market = EMarketTargetType.E_MARKET_TARGET_HK
    elif head == EMarketTargetType.E_MARKET_TARGET_US.value:
        # 市场信息匹配美股，查询子市场，AbuSymbolUS().query_symbol_sub_market
        sub_market = EMarketSubType(AbuSymbolUS().query_symbol_sub_market(code[2:]))
        market = EMarketTargetType.E_MARKET_TARGET_US

    if market is not None and sub_market is not None:
        # 通过stock_code, market, sub_market构建Symbol对象
        stock_code = code[2:].upper()
        return Symbol(market, sub_market, stock_code)

    if code.isdigit():
        if len(code) == 6:
            # 6位全数字，匹配查询a股子市场
            market = EMarketTargetType.E_MARKET_TARGET_CN
            sub_market = EMarketSubType(AbuSymbolCN().query_symbol_sub_market(code))
            return Symbol(market, sub_market, code)
        elif len(code) == 5:
            # 5位全数字，匹配查询港股子市场
            market = EMarketTargetType.E_MARKET_TARGET_HK
            sub_market = EMarketSubType.HK
            return Symbol(market, sub_market, code)
        if rs:
            raise TypeError('cn symbol len = 6, hk symbol len = 5')
    elif code.isalpha() and code in Symbol.HK_INDEX:
        # 全字母且匹配港股大盘'HSI', 'HSCEI', 'HSCCI'
        market = EMarketTargetType.E_MARKET_TARGET_HK
        sub_market = EMarketSubType.HK
        return Symbol(market, sub_market, code)
    elif code.isalpha() and code in all_symbol(EMarketTargetType.E_MARKET_TARGET_TC):
        # 全字母且匹配币类市场''btc', 'ltc'
        market = EMarketTargetType.E_MARKET_TARGET_TC
        sub_market = EMarketSubType.COIN
        return Symbol(market, sub_market, code)
    elif code.isalpha() and code.upper() in all_symbol(EMarketTargetType.E_MARKET_TARGET_FUTURES_GLOBAL):
        # 全字母且匹配国际期货市场
        futures_gb_code = code.upper()
        q_df = AbuFuturesGB().query_symbol(futures_gb_code)
        sub_market = EMarketSubType(q_df.exchange.values[0])
        market = EMarketTargetType.E_MARKET_TARGET_FUTURES_GLOBAL
        return Symbol(market, sub_market, futures_gb_code)
    elif code.isalpha() or (code.startswith('.') and code[1:].isalpha()):
        # 全字母进行美股大盘匹配或者美股股票匹配
        stock_code = code.upper()

        if stock_code in Symbol.K_US_INDEX_FIXES.keys():
            stock_code = Symbol.K_US_INDEX_FIXES[stock_code]

        sub_market = EMarketSubType(AbuSymbolUS().query_symbol_sub_market(code))
        market = EMarketTargetType.E_MARKET_TARGET_US
        return Symbol(market, sub_market, stock_code)
    elif (code[0].isalpha() and code[1:].isdigit()) or (code[:2].isalpha() and code[2:].isdigit()):
        # 匹配国内期货市场symbol
        futures_code = code.upper()
        q_df = AbuFuturesCn().query_symbol(futures_code)
        if q_df is not None:
            sub_market = EMarketSubType(q_df.exchange.values[0])
            market = EMarketTargetType.E_MARKET_TARGET_FUTURES_CN
            return Symbol(market, sub_market, futures_code)

    if rs:
        raise ValueError('arg code :{} format dt support'.format(code))


def __search(market_df, search_match, search_code, search_result, match_key='co_name'):
    """具体搜索执行接口"""

    def __search_whole_code(_match_code):
        _sc_df = market_df[market_df.symbol == _match_code]
        if not _sc_df.empty:
            search_result[_sc_df.symbol.values[0]] = _sc_df[match_key].values[0]
            return True
        return False

    def __search_pinyin_code(_match_code):
        from ..MarketBu.ABuDataFeed import query_symbol_from_pinyin
        # 使用query_symbol_from_pinyin对模糊拼音进行查询
        pinyin_symbol = query_symbol_from_pinyin(_match_code)
        if pinyin_symbol is not None:
            # 需要把拼音code标准化为可查询的code
            search_symbol = code_to_symbol(pinyin_symbol, rs=False)
            if search_symbol is not None:
                _search_code = search_symbol.symbol_code
                sc_df = market_df[market_df.symbol == _search_code]
                if not sc_df.empty:
                    search_result[sc_df.symbol.values[0]] = sc_df[match_key].values[0]

    def __search_fnmatch_info(_search_match):
        # 模糊匹配公司名称信息或者交易产品信息
        mc_df = market_df[market_df[match_key].apply(lambda name:
                                                     fnmatch(to_unicode(name),
                                                             _search_match))]
        if not mc_df.empty:
            for ind in np.arange(0, len(mc_df)):
                mcs = mc_df.iloc[ind]
                search_result[mcs.symbol] = mcs[match_key]

    # 首先全匹配search_code
    if not __search_whole_code(search_code):
        # 如果search_code没有能全匹配成功，使用拼音进行匹配一次
        __search_pinyin_code(search_code)
    # 模糊匹配公司名称或者产品等信息symbol
    __search_fnmatch_info(search_match)


def _us_search(search_match, search_code, search_result):
    """美股市场symbol关键字搜索"""
    from ..MarketBu.ABuSymbolStock import AbuSymbolUS
    __search(AbuSymbolUS().df, search_match, search_code, search_result)


def _cn_search(search_match, search_code, search_result):
    """a股市场symbol关键字搜索"""
    from ..MarketBu.ABuSymbolStock import AbuSymbolCN
    __search(AbuSymbolCN().df, search_match, search_code, search_result)


def _hk_search(search_match, search_code, search_result):
    """港股市场symbol关键字搜索"""
    from ..MarketBu.ABuSymbolStock import AbuSymbolHK
    __search(AbuSymbolHK().df, search_match, search_code, search_result)


def _fcn_search(search_match, search_code, search_result):
    """国内期货symbol关键字搜索"""
    from ..MarketBu.ABuSymbolFutures import AbuFuturesCn
    __search(AbuFuturesCn().futures_cn_df,
             search_match, search_code, search_result, match_key='product')


def _fgb_search(search_match, search_code, search_result):
    """国际期货symbol关键字搜索"""
    from ..MarketBu.ABuSymbolFutures import AbuFuturesGB
    __search(AbuFuturesGB().futures_gb_df,
             search_match, search_code, search_result, match_key='product')


# TODO 币类匹配统一标准规范
def _tc_search(search_match, search_code, search_result):
    if fnmatch(u'比特币', search_match) or 'btc' == search_code:
        search_result['btc'] = u'比特币'
    if fnmatch(u'莱特币', search_match) or 'ltc' == search_code:
        search_result['ltc'] = u'莱特币'


def search_to_symbol_dict(search, fast_mode=False):
    """
    symbol搜索对外接口，全匹配symbol code，拼音匹配symbol，别名匹配，模糊匹配公司名称，产品名称等信息
    eg：
        in：
        search_to_symbol_dict('黄金')
        out：
        {'002155': '湖南黄金',
         '600489': '中金黄金',
         '600547': '山东黄金',
         '600766': '园城黄金',
         '600988': '赤峰黄金',
         'ABX': '巴里克黄金',
         'AU0': '黄金',
         'DGL': '黄金基金-PowerShares',
         'DGLD': '黄金3X做空-VelocityShares',
         'DGP': '黄金2X做多-DB',
         'DGZ': '黄金做空-PowerShares',
         'DZZ': '黄金2X做空-DB',
         'EGO': '埃尔拉多黄金公司',
         'GC': '纽约黄金',
         'GEUR': 'Gartman欧元黄金ETF-AdvisorShares ',
         'GLD': '黄金ETF-SPDR',
         'GLL': '黄金2X做空-ProShares',
         'GYEN': 'Gartman日元黄金ETF-AdvisorShares',
         'HMY': '哈莫尼黄金',
         'IAU': '黄金ETF-iShares',
         'KGC': '金罗斯黄金',
         'LIHR': '利希尔黄金',
         'PRME': '全球黄金地段房地产ETF-First Trust Heitman',
         'RGLD': '皇家黄金',
         'UGL': '黄金2x做多-ProShares',
         'UGLD': '黄金3X做多-VelocityShares'}
    :param search: eg：'黄金'， '58'
    :param fast_mode: 是否尽快匹配，速度优先模式
    :return: symbol dict
    """
    search_symbol_dict = {}
    search = search.lower()
    while len(search_symbol_dict) == 0 and len(search) > 0:
        # 构建模糊匹配进行匹配带通配符的字符串
        search_match = u'*{}*'.format(search)
        # 构建精确匹配或拼音模糊匹配的symbol
        search_symbol = code_to_symbol(search, rs=False)
        search_code = ''
        if search_symbol is not None:
            search_code = search_symbol.symbol_code
        # 对search的内容进行递减匹配
        search = search[:-1]
        # 依次对各个市场进行搜索匹配操作
        _tc_search(search_match, search_code, search_symbol_dict)
        _cn_search(search_match, search_code, search_symbol_dict)
        _us_search(search_match, search_code, search_symbol_dict)
        _hk_search(search_match, search_code, search_symbol_dict)
        _fcn_search(search_match, search_code, search_symbol_dict)
        _fgb_search(search_match, search_code, search_symbol_dict)
        if fast_mode:
            break
    return search_symbol_dict


class Symbol(object):
    """统一所有市场的symbol，统一对外接口对象"""

    # 定义使用的美股大盘
    US_INDEX = ['.DJI', '.IXIC', '.INX']
    # 定义使用的sh大盘
    SH_INDEX = ['000001', '000300']
    # 定义使用的sz大盘
    SZ_INDEX = ['399001', '399006']
    # 定义使用的港股大盘
    HK_INDEX = ['HSI', 'HSCEI', 'HSCCI']

    # 修复美股其它大盘写法匹配，主要为外部user使用code_to_symbol进行标准化使用
    K_US_INDEX_FIXES = {'DJI': '.DJI', 'IXIC': '.IXIC', 'INX': '.INX'}

    def __init__(self, market, sub_market, symbol_code):
        """
        :param market: EMarketTargetType enum对象
        :param sub_market: EMarketSubType enum对象
        :param symbol_code: str对象，不包含市场信息的code
        """
        if isinstance(market, EMarketTargetType) and isinstance(sub_market, EMarketSubType):
            self.market = market
            self.sub_market = sub_market
            self.symbol_code = symbol_code
            self.source = None
        else:
            raise TypeError('market type error')

    def __str__(self):
        """打印对象显示：market， sub_market， symbol_code"""
        return '{}_{}:{}'.format(self.market.value, self.sub_market.value, self.symbol_code)

    __repr__ = __str__

    def __len__(self):
        """对象长度：拼接市场＋子市场＋code的字符串长度"""
        m_symbol = '{}_{}:{}'.format(self.market.value, self.sub_market.value, self.symbol_code)
        return len(m_symbol)

    @LazyFunc
    def value(self):
        """不同市场返回ABuSymbolPd.make_kl_df使用的symbol LazyFunc"""
        if self.market == EMarketTargetType.E_MARKET_TARGET_HK or self.market == EMarketTargetType.E_MARKET_TARGET_US:
            # hk,us eg: usTSLA, hk00836
            return '{}{}'.format(self.market.value, self.symbol_code)
        elif self.market == EMarketTargetType.E_MARKET_TARGET_CN:
            # cn eg: sh000001
            return '{}{}'.format(self.sub_market.value, self.symbol_code)
        # 其它市场直接返回symbol_code
        return self.symbol_code

    def is_a_stock(self):
        """判定是否a股symbol"""
        return self.market == EMarketTargetType.E_MARKET_TARGET_CN

    def is_sh_stock(self):
        """判定是否a股sh symbol"""
        return self.sub_market == EMarketSubType.SH

    def is_sz_stock(self):
        """判定是否a股sz symbol"""
        return self.sub_market == EMarketSubType.SZ

    def is_us_stock(self):
        """判定是否美股 symbol"""
        return self.market == EMarketTargetType.E_MARKET_TARGET_US

    def is_us_n_stock(self):
        """判定是否美股纽约交易所 symbol"""
        return self.sub_market == EMarketSubType.US_N

    def is_us_oq_stock(self):
        """判定是否美股纳斯达克交易所 symbol"""
        return self.sub_market == EMarketSubType.US_OQ

    def is_hk_stock(self):
        """判定是否港股 symbol"""
        return self.market == EMarketTargetType.E_MARKET_TARGET_HK

    def is_sh_index(self):
        """判定是否a股sh 大盘"""
        return self.is_sh_stock() and self.symbol_code in Symbol.SH_INDEX

    def is_sz_index(self):
        """判定是否a股sz 大盘"""
        return self.is_sz_stock() and self.symbol_code in Symbol.SZ_INDEX

    def is_a_index(self):
        """判定是否a股 大盘"""
        return self.is_sh_index() or self.is_sz_index()

    def is_us_index(self):
        """判定是否美股 大盘"""
        return self.is_us_stock() and self.symbol_code in Symbol.US_INDEX

    def is_hk_index(self):
        """判定是否港股 大盘"""
        return self.is_hk_stock() and self.symbol_code in Symbol.HK_INDEX

    def is_index(self):
        """判定是否大盘"""
        return self.is_us_index() or self.is_hk_index() or self.is_a_index()

    def is_futures(self):
        """判定是否期货symbol"""
        return self.market == EMarketTargetType.E_MARKET_TARGET_FUTURES_CN \
            or self.market == EMarketTargetType.E_MARKET_TARGET_FUTURES_GLOBAL

    def is_tc(self):
        """判定是否电子货币symbol"""
        return self.market == EMarketTargetType.E_MARKET_TARGET_TC


class IndexSymbol(object):
    """定义IndexSymbol类，设定大盘指数Symbol对象的规范"""

    # 美股大盘DJI Symbol对象
    DJI = Symbol(EMarketTargetType.E_MARKET_TARGET_US, EMarketSubType.US_N, '.DJI')
    # 美股大盘IXIC Symbol对象
    IXIC = Symbol(EMarketTargetType.E_MARKET_TARGET_US, EMarketSubType.US_N, '.IXIC')
    # 美股大盘INX Symbol对象
    INX = Symbol(EMarketTargetType.E_MARKET_TARGET_US, EMarketSubType.US_N, '.INX')

    # a股sh大盘Symbol对象
    SH = Symbol(EMarketTargetType.E_MARKET_TARGET_CN, EMarketSubType.SH, '000001')
    # a股sz大盘Symbol对象
    SZ = Symbol(EMarketTargetType.E_MARKET_TARGET_CN, EMarketSubType.SZ, '399001')
    # a股sz Growth大盘Symbol对象
    Growth = Symbol(EMarketTargetType.E_MARKET_TARGET_CN, EMarketSubType.SZ, '399006')
    # a股sh SH300大盘Symbol对象
    SH300 = Symbol(EMarketTargetType.E_MARKET_TARGET_CN, EMarketSubType.SH, '000300')

    # 港股sh大盘HSI对象
    HSI = Symbol(EMarketTargetType.E_MARKET_TARGET_HK, EMarketSubType.HK, 'HSI')
    # 港股sh大盘HSCEI对象
    HSCEI = Symbol(EMarketTargetType.E_MARKET_TARGET_HK, EMarketSubType.HK, 'HSCEI')
    # 港股sh大盘HSCCI对象
    HSCCI = Symbol(EMarketTargetType.E_MARKET_TARGET_HK, EMarketSubType.HK, 'HSCCI')

    # 国内期货只是使用黄金做为时间标尺，不具备对比大盘作用
    BM_FUTURES_CN = Symbol(EMarketTargetType.E_MARKET_TARGET_FUTURES_CN, EMarketSubType.SHFE, 'AU0')
    # 国际期货只是使用纽约黄金做为时间标尺，不具备对比大盘作用
    BM_FUTURES_GB = Symbol(EMarketTargetType.E_MARKET_TARGET_FUTURES_GLOBAL, EMarketSubType.NYMEX, 'GC')

    # 币类只是使用btc做为时间标尺，不具备对比大盘作用
    TC_INX = Symbol(EMarketTargetType.E_MARKET_TARGET_TC, EMarketSubType.COIN, 'btc')
