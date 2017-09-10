# coding=utf-8
"""
    期货symbol数据模块
"""

from __future__ import print_function
from __future__ import absolute_import
from __future__ import division

import os

import pandas as pd

from ..CoreBu.ABuBase import FreezeAttrMixin
from ..CoreBu import ABuEnv
from ..UtilBu.ABuLazyUtil import LazyFunc
from ..UtilBu.ABuDTUtil import singleton
from ..MarketBu.ABuSymbol import Symbol

__author__ = '阿布'
__weixin__ = 'abu_quant'

_rom_dir = ABuEnv.g_project_rom_data_dir
"""国内期货symbol文件，文件定期重新爬取，更新"""
_stock_code_futures_cn = os.path.join(_rom_dir, 'futures_cn.csv')

"""国际期货symbol文件，文件定期重新爬取，更新"""
_stock_code_futures_gb = os.path.join(_rom_dir, 'futures_gb.csv')


@singleton
class AbuFuturesCn(FreezeAttrMixin):
    """国内期货symbol数据，AbuFuturesCn单例，混入FreezeAttrMixin在__init__中冻结了接口，外部只可以读取"""

    def __init__(self):
        """
            self.futures_cn_df表结构如下所示：列分别为
            symbol product	min_deposit	min_unit	commission	exchange

                0	V0	PVC	0.07	5	2.0	DCE
                1	P0	棕榈	0.07	10	2.5	DCE
                2	B0	豆二	0.05	10	2.0	DCE
                3	M0	豆粕	0.07	10	1.5	DCE
                4	I0	铁矿石	0.10	100	8.0	DCE
                5	JD0 	鸡蛋	0.08	5	6.0	DCE
                6	L0	塑料	0.07	5	2.0	DCE
                7	PP0	PP	0.07	5	4.0	DCE
                8	FB0	纤维板	0.20	500	10.0	DCE
                9	BB0	胶合板	0.20	500	10.0	DCE
                10	Y0	豆油	0.07	10	2.5	DCE
            如第5行：symbol＝JD0，product＝鸡蛋，min_deposit(最少保证金比例)＝0.08
                    min_unit(每手最少交易单位)=5, commission(每一手手续费)=6.0,
                    exchange(交易所)=DCE
        """
        # 读取本地csv到内存
        self.futures_cn_df = pd.read_csv(_stock_code_futures_cn, index_col=0)
        # __init__中使用FreezeAttrMixin._freeze冻结了接口
        self._freeze()

    def __str__(self):
        """打印对象显示：futures_cn_df.info， futures_cn_df.describe"""
        return 'info:\n{}\ndescribe:\n{}'.format(self.futures_cn_df.info(),
                                                 self.futures_cn_df.describe())

    __repr__ = __str__

    def __len__(self):
        """对象长度：futures_cn_df.shape[0]，即行数"""
        return self.futures_cn_df.shape[0]

    def __contains__(self, item):
        """成员测试：item in self.futures_cn_df.columns，即item在不在futures_cn_df的列中"""
        return item in self.futures_cn_df.columns

    def __getitem__(self, key):
        """索引获取：套接self.futures_cn_df[key]"""
        if key in self:
            return self.futures_cn_df[key]
        # 不在的话，返回整个表格futures_cn_df

        symbol_df = self.futures_cn_df[self.futures_cn_df.symbol == key]
        if not symbol_df.empty:
            return symbol_df
        return self.futures_cn_df

    def __setitem__(self, key, value):
        """索引设置：对外抛出错误， 即不准许外部设置"""
        raise AttributeError("AbuFuturesCn set value!!!")

    def query_symbol(self, symbol):
        """
        对外查询接口
        :param symbol: 可以是Symbol对象，也可以是symbol字符串对象
        """
        if isinstance(symbol, Symbol):
            symbol = symbol.value

        if symbol[0].isalpha() and symbol[1:].isdigit():
            head = symbol[0]
        elif symbol[:2].isalpha() and symbol[2:].isdigit():
            head = symbol[:2]
        else:
            return None

        for fs in self.symbol:
            if head == fs[:len(head)]:
                return self.futures_cn_df[self.futures_cn_df.symbol == fs]
        return None

    def query_min_unit(self, symbol):
        """
        对query_symbol的查询一手单位的近一部函数封装
        :param symbol: 可以是Symbol对象，也可以是symbol字符串对象
        """
        min_cnt = 10
        # 查询最少一手单位
        q_df = self.query_symbol(symbol)
        if q_df is not None:
            min_cnt = q_df.min_unit.values[0]
        return min_cnt

    @LazyFunc
    def symbol(self):
        """代理获取futures_cn_df.symbol，LazyFunc"""
        return self.futures_cn_df.symbol

    @LazyFunc
    def product(self):
        """代理获取futures_cn_df.product，LazyFunc"""
        return self.futures_cn_df.product

    @LazyFunc
    def min_deposit(self):
        """代理获取futures_cn_df.min_deposit，LazyFunc"""
        return self.futures_cn_df.min_deposit

    @LazyFunc
    def min_unit(self):
        """代理获取futures_cn_df.min_unit，LazyFunc"""
        return self.futures_cn_df.min_unit

    @LazyFunc
    def commission(self):
        """代理获取futures_cn_df.commission，LazyFunc"""
        return self.futures_cn_df.commission

    @LazyFunc
    def exchange(self):
        """代理获取futures_cn_df.exchange，LazyFunc"""
        return self.futures_cn_df.exchange


# TODO 提取AbuFutures基类，删除重复代码
@singleton
class AbuFuturesGB(FreezeAttrMixin):
    """国际期货数据，AbuFuturesGB单例，混入FreezeAttrMixin在__init__中冻结了接口，外部只可以读取"""

    def __init__(self):
        """
            self.futures_gb_df表结构如下所示：列分别为

            symbol	product	min_deposit	min_unit	exchange
            0	NID	伦敦镍	0.07	6	LME
            1	PBD	伦敦铅	0.10	25	LME
            2	SND	伦敦锡	0.05	5	LME
            3	ZSD	伦敦锌	0.10	25	LME
            4	AHD	伦敦铝	0.08	25	LME
            5	CAD	伦敦铜	0.08	25	LME
        """
        # 读取本地csv到内存
        self.futures_gb_df = pd.read_csv(_stock_code_futures_gb, index_col=0)
        # __init__中使用FreezeAttrMixin._freeze冻结了接口
        self._freeze()

    def __str__(self):
        """打印对象显示：futures_cn_df.info， futures_cn_df.describe"""
        return 'info:\n{}\ndescribe:\n{}'.format(self.futures_gb_df.info(),
                                                 self.futures_gb_df.describe())

    __repr__ = __str__

    def __len__(self):
        """对象长度：futures_cn_df.shape[0]，即行数"""
        return self.futures_gb_df.shape[0]

    def __contains__(self, item):
        """成员测试：item in self.futures_cn_df.columns，即item在不在futures_cn_df的列中"""
        return item in self.futures_gb_df.columns

    def __getitem__(self, key):
        """索引获取：套接self.futures_cn_df[key]"""
        if key in self:
            return self.futures_gb_df[key]

        symbol_df = self.futures_gb_df[self.futures_gb_df.symbol == key]
        if not symbol_df.empty:
            return symbol_df

        # 不在的话，返回整个表格futures_cn_df
        return self.futures_gb_df

    def __setitem__(self, key, value):
        """索引设置：对外抛出错误， 即不准许外部设置"""
        raise AttributeError("AbuFuturesGB set value!!!")

    def query_symbol(self, symbol):
        """
        对外查询接口
        :param symbol: 可以是Symbol对象，也可以是symbol字符串对象
        """
        if isinstance(symbol, Symbol):
            symbol = symbol.value
        if symbol in self.symbol.values:
            return self.futures_gb_df[self.futures_gb_df.symbol.values == symbol]
        return None

    def query_min_unit(self, symbol):
        """
        对query_symbol的查询一手单位的近一部函数封装
        :param symbol: 可以是Symbol对象，也可以是symbol字符串对象
        """
        min_cnt = 10
        # 查询最少一手单位
        q_df = self.query_symbol(symbol)
        if q_df is not None:
            min_cnt = q_df.min_unit.values[0]
        return min_cnt

    @LazyFunc
    def symbol(self):
        """代理获取futures_gb_df.symbol，LazyFunc"""
        return self.futures_gb_df.symbol

    @LazyFunc
    def product(self):
        """代理获取futures_gb_df.product，LazyFunc"""
        return self.futures_gb_df.product

    @LazyFunc
    def min_deposit(self):
        """代理获取futures_gb_df.min_deposit，LazyFunc"""
        return self.futures_gb_df.min_deposit

    @LazyFunc
    def min_unit(self):
        """代理获取futures_gb_df.min_unit，LazyFunc"""
        return self.futures_gb_df.min_unit

    @LazyFunc
    def exchange(self):
        """代理获取futures_gb_df.exchange，LazyFunc"""
        return self.futures_gb_df.exchange

"""由于封装对外所以不使用模块单例"""
# """模块单例"""
# futures_cn = AbuFuturesCn()
