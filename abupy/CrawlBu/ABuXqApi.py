# -*- encoding:utf-8 -*-
from __future__ import print_function
from __future__ import absolute_import
from __future__ import division

import time

from ..CoreBu import ABuFixes

__author__ = '小青蛙'
__weixin__ = 'abu_quant'

BASE_XQ_URL = 'https://xueqiu.com'

BASE_XQ_HQ_URL = BASE_XQ_URL + '/hq'

BASE_XQ_STOCK_INFO = BASE_XQ_URL + '/S/'

BASE_XQ_HQ_EXCHANGE_URL = BASE_XQ_HQ_URL + '#exchange={}'

CN_INDUSTRY_LIST = 'https://xueqiu.com/industry/quote_order.json?' \
                   'size=100&order=asc&orderBy=code&exchange=CN&'
HK_INDUSTRY_LIST = 'https://xueqiu.com/stock/cata/stocklist.json?' \
                   'size=100&order=asc&orderby=code&exchange=HK&'
US_INDUSTRY_LIST = 'https://xueqiu.com/stock/cata/stocklist.json?' \
                   'size=100&order=asc&orderby=code&exchange=US&'

CN_STOCK_LIST = 'https://xueqiu.com/stock/cata/stocklist.json?' \
                'size=100&order=asc&orderby=code&type=11%2C12&'
HK_STOCK_LIST = 'https://xueqiu.com/stock/cata/stocklist.json?' \
                'size=100&order=asc&orderby=code&type=30%2C31&'
US_STOCK_LIST = 'https://xueqiu.com/stock/cata/stocklist.json?' \
                'size=100&order=asc&orderby=code&type=0%2C1%2C2%2C3&isdelay=1&'

LOGIN_URL = 'https://xueqiu.com/user/login'

# 雪球url中type值
TYPE_INFO = {
    0: 'us_stock',  # 美股(不包含中国区)
    1: 'us_a_stock_NASDAQ',  # 在拉斯达克上市的中国股票
    2: 'us_a_stock_NYSE',  # 在纽交所上市的中国股票
    3: 'us_index',  # 美国指数
    11: 'a_stock',  # A股
    12: 'a_index',  # A股指数
    30: 'hk_stock',  # 港股
    31: 'hk_index',  # 港股指数
}


class IndustryUrl(object):
    def __init__(self, market, **kwargs):
        self.market = market
        self.kwargs = kwargs
        self._base_url = None
        self.init_base_url()

    def init_base_url(self):
        if 'CN' == self.market:
            self._base_url = CN_INDUSTRY_LIST
        elif 'HK' == self.market:
            self._base_url = HK_INDUSTRY_LIST
        elif 'US' == self.market:
            self._base_url = US_INDUSTRY_LIST
        else:
            raise RuntimeError('only support {}'.format(['US', 'CN', 'HK']))

    @property
    def url(self):
        param = dict(self.kwargs)
        param['_'] = time.time()

        return self._base_url + ABuFixes.urlencode(param)


class StockListUrl(IndustryUrl):
    def init_base_url(self):
        if 'CN' == self.market:
            self._base_url = CN_STOCK_LIST
        elif 'HK' == self.market:
            self._base_url = HK_STOCK_LIST
        elif 'US' == self.market:
            self._base_url = US_STOCK_LIST
        else:
            raise RuntimeError('only support {}'.format(['US', 'CN', 'HK']))


def test():
    print(IndustryUrl('CN', plate='保险业', page=1, level2code='J68').url)
    print(IndustryUrl('US', plate='金融', page=1).url)
    print(IndustryUrl('HK', plate='保险', page=1).url)
