# -*- encoding:utf-8 -*-
from __future__ import print_function
from __future__ import absolute_import
from __future__ import division

__author__ = '小青蛙'
__weixin__ = 'abu_quant'

"""
s:static
d:dynamic
co: company
pb: price book ratio
pe: price earning ratio
ps: price sales ratio
asset : net_asset_value_per_share
oo:organization_ownership
cc:circulation_capital
mv:market_value
sc:short_covering
"""
columns_map = {
    "symbol": 'symbol',
    "market": 'market',
    "exchange": 'exchange',

    "industry": 'industry',
    "name": 'co_name',
    "公司网站:": 'co_site',
    "公司电话：": 'co_tel',
    "公司地址：": 'co_addr',
    "业务:": 'co_business',
    "简介:": 'co_intro',

    "市盈率(静)/(动)：": 'pe_s_d',
    "市净率(动)：": 'pb_d',
    "市净率MRQ：": 'pb_MRQ',
    "市销率(动)：": 'ps_d',
    "市销率：": 'ps',
    "市盈率(静)：": 'pe_s',
    "换手率：": 'turnover',

    "每股净资产：": 'asset',
    "流通股本：": 'cc',
    "振幅：": 'amplitude',
    "空头补回天数：": 'sc',
    "招股说明书：": 'prospectus',
    "总股本：": 'equity',
    "总市值：": 'mv',
    "机构持股：": 'oo',
    "港股股本：": 'hk_equity'
}
