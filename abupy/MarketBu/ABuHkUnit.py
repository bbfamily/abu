# coding=utf-8
"""
    港股每一手交易数量模块
"""

from __future__ import print_function
from __future__ import absolute_import
from __future__ import division

import os

import pandas as pd

from ..CoreBu.ABuBase import FreezeAttrMixin
from ..CoreBu import ABuEnv
from ..CoreBu.ABuFixes import six
from ..UtilBu.ABuDTUtil import singleton
from ..MarketBu.ABuSymbol import Symbol

__author__ = '阿布'
__weixin__ = 'abu_quant'

_rom_dir = ABuEnv.g_project_rom_data_dir
"""文件定期重新爬取，更新"""
_hk_unit_csv = os.path.join(_rom_dir, 'hk_unit.csv')
"""默认每一手股数1000"""
K_DEFAULT_UNIT = 1000


@singleton
class AbuHkUnit(FreezeAttrMixin):
    """AbuHkUnit单例，混入FreezeAttrMixin在__init__中冻结了接口，外部只可以读取"""

    def __init__(self):
        """
            self.hk_unit_df表结构如下所示：只有一个列unit代表每一手股数，行代表港股symbol
                    unit
            hk02011	2000
            hk01396	2000
            hk08112	4800
            hk08198	4000
            hk01143	4000
            ............
        """
        # 读取本地csv到内存，由于AbuHkUnit单例只进行一次
        self.hk_unit_df = pd.read_csv(_hk_unit_csv, index_col=0)
        # __init__中使用FreezeAttrMixin._freeze冻结了接口
        self._freeze()

    def query_unit(self, symbol):
        """
        对外查询接口，查询对应symbol每一手交易数量
        :param symbol: 可以是Symbol对象，也可以是symbol字符串对象
        """
        if isinstance(symbol, Symbol):
            # Symbol对象进行转换
            symbol = symbol.value
        elif isinstance(symbol, six.string_types) and symbol.isdigit():
            # symbol字符串, 但是没有hk，则加上
            symbol = 'hk{}'.format(symbol)

        # noinspection PyBroadException
        try:
            unit = self.hk_unit_df.loc[symbol].values[0]
        except:
            # 查询失败赋予默认值
            unit = K_DEFAULT_UNIT
        return unit

    def __str__(self):
        """打印对象显示：hk_unit_df.info， hk_unit_df.describe"""
        return 'info:\n{}\ndescribe:\n{}'.format(self.hk_unit_df.info(),
                                                 self.hk_unit_df.describe())

    __repr__ = __str__

    def __len__(self):
        """对象长度：hk_unit_df.shape[0]，即行数"""
        return self.hk_unit_df.shape[0]

    def __contains__(self, item):
        """
        成员测试：标准化item后检测item是否在self.hk_unit_df.index中
        :param item: 可以是Symbol对象，也可以是symbol字符串对象
        """
        if isinstance(item, Symbol):
            item = item.value
        elif isinstance(item, six.string_types) and item.isdigit():
            item = 'hk{}'.format(item)

        return item in self.hk_unit_df.index

    def __getitem__(self, key):
        """索引获取：套接self.query_unit(key)"""
        return self.query_unit(key)

    def __setitem__(self, key, value):
        """索引设置：对外抛出错误， 即不准许外部设置"""
        raise AttributeError("AbuHkUnit set value!!!")


"""由于封装对外所以不使用模块单例"""
# """模块单例"""
# single_hk_unit = AbuHkUnit()
