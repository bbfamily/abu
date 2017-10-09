# -*- encoding:utf-8 -*-
"""股票池选股ui界面"""

from __future__ import print_function
from __future__ import absolute_import
from __future__ import division

import ipywidgets as widgets

from ..CoreBu.ABuEnv import EMarketTargetType
from ..WidgetBu.ABuWGBase import WidgetBase, WidgetSearchBox
from ..MarketBu import ABuMarket
from ..CoreBu import ABuEnv
from ..UtilBu.ABuStrUtil import to_unicode
from ..MarketBu.ABuSymbolStock import AbuSymbolCN, AbuSymbolUS, AbuSymbolHK
from ..MarketBu.ABuSymbolFutures import AbuFuturesCn, AbuFuturesGB

__author__ = '阿布'
__weixin__ = 'abu_quant'


class WidgetSymbolChoice(WidgetBase):
    """股票池选股ui界面"""

    # noinspection PyProtectedMember
    def __init__(self):
        """构建股票池选股ui界面"""
        label_layout = widgets.Layout(width='300px', align_items='stretch', justify_content='space-between')
        self.cs_tip = widgets.Label(value=u'如果股池为空，回测将使用大盘市场中所有股票', layout=label_layout)
        # 股票池多选框
        self.choice_symbols = widgets.SelectMultiple(
            description=u'股池:',
            disabled=False,
            layout=widgets.Layout(width='300px', align_items='stretch', justify_content='space-between')
        )
        self.choice_symbols.observe(self.choice_symbols_select, names='value')

        # 构建所有沙盒中的数据序列
        market_title = [u'美股', u'A股', u'港股', u'国内', u'国际', u'币类']
        us_seed_symbol = [to_unicode('{}:{}'.format(AbuSymbolUS()[symbol].co_name.values[0], symbol))
                          for symbol in ABuMarket.K_SAND_BOX_US]
        cn_seed_symbol = [to_unicode('{}:{}'.format(AbuSymbolCN()[symbol].co_name.values[0], symbol))
                          for symbol in ABuMarket.K_SAND_BOX_CN]
        hk_seed_symbol = [to_unicode('{}:{}'.format(AbuSymbolHK()[symbol].co_name.values[0], symbol))
                          for symbol in ABuMarket.K_SAND_BOX_HK]
        fcn_seed_symbol = [to_unicode('{}:{}'.format(AbuFuturesCn()[symbol]['product'].values[0], symbol))
                           for symbol in AbuFuturesCn().symbol]
        fgb_seed_symbol = [to_unicode('{}:{}'.format(AbuFuturesGB()[symbol]['product'].values[0], symbol))
                           for symbol in AbuFuturesGB().symbol]
        # 沙盒中的数据序列构建数据字典
        self.market_dict = {u'美股': us_seed_symbol,
                            u'A股': cn_seed_symbol,
                            u'港股': hk_seed_symbol,
                            u'国内': fcn_seed_symbol,
                            u'国际': fgb_seed_symbol,
                            u'币类': [u'比特币:btc', u'莱特币:ltc']}

        # 一个市场一个tab，tab中的symbol为沙盒中的symbol
        self.market_widget_tab = widgets.Tab()
        self.market_symbol_widget = []
        for title in market_title:
            market_symbol = widgets.SelectMultiple(
                options=self.market_dict[title],
                description=title,
                disabled=False
            )
            market_symbol.observe(self.on_already_select, names='value')
            self.market_symbol_widget.append(market_symbol)
        self.market_widget_tab.children = self.market_symbol_widget
        for ind, name in enumerate(market_title):
            self.market_widget_tab.set_title(ind, name)

        self.sc_box = WidgetSearchBox(self.on_already_select)()

        # 下拉选择标尺大盘
        self.market = widgets.Dropdown(
            options={u'美股': EMarketTargetType.E_MARKET_TARGET_US.value,
                     u'A股': EMarketTargetType.E_MARKET_TARGET_CN.value,
                     u'港股': EMarketTargetType.E_MARKET_TARGET_HK.value,
                     u'国内期货': EMarketTargetType.E_MARKET_TARGET_FUTURES_CN.value,
                     u'国际期货': EMarketTargetType.E_MARKET_TARGET_FUTURES_GLOBAL.value,
                     u'数字货币': EMarketTargetType.E_MARKET_TARGET_TC.value},
            value=ABuEnv.g_market_target.value,
            description=u'大盘市场:',
        )
        self.market.observe(self.on_market_change, names='value')

        market_tip = widgets.Label(value=u'大盘市场设置只影响收益对比标尺', layout=label_layout)
        market_box = widgets.VBox([self.market, market_tip])

        self.widget = widgets.VBox([self.cs_tip, self.choice_symbols, self.market_widget_tab,
                                    self.sc_box, market_box])

    def on_market_change(self, change):
        """切换大盘市场"""
        ABuEnv.g_market_target = EMarketTargetType(change['new'])

    def on_already_select(self, select):
        """搜索框或者内置沙盒symbol中点击放入股票池"""
        st_symbol = [symbol.split(':')[1] if symbol.find(':') > 0
                     else symbol for symbol in list(select['new'])]
        # 更新股票池中原有的symbol序列
        self.choice_symbols.options = list(set(st_symbol + list(self.choice_symbols.options)))

    def choice_symbols_select(self, select):
        """股票池中点击删除股票池中对应的symbol"""
        # print(select)
        # FIXME BUG 低版本ipywidgets下删除的不对
        self.choice_symbols.options = list(set(self.choice_symbols.options) - set(select['new']))
