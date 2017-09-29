# -*- encoding:utf-8 -*-
"""股票基本信息图形可视化"""

from __future__ import print_function
from __future__ import absolute_import
from __future__ import division

import logging

import ipywidgets as widgets

from ..MarketBu.ABuSymbolStock import query_stock_info
from ..MarketBu import ABuIndustries
from ..MarketBu.ABuSymbol import code_to_symbol
from ..UtilBu.ABuDTUtil import catch_error
from ..CoreBu import ABuEnv
from ..CoreBu.ABuFixes import six
from ..UtilBu.ABuStrUtil import to_unicode
from ..WidgetBu.ABuWGBase import WidgetBase, WidgetSearchBox

__author__ = '阿布'
__weixin__ = 'abu_quant'


class WidgetSearchStockInfo(WidgetBase):
    def __init__(self):
        self.stock_info = None
        self.widget = WidgetSearchBox(self.on_search_result_click)()

    def on_search_result_click(self, select):
        """选中搜索结果中的一个进行点击"""

        st_symbol = [symbol.split(':')[1] if symbol.find(':') > 0
                     else symbol for symbol in list(select['new'])]
        if len(st_symbol) == 0:
            return

        # result是多选框，值拿最后一个选中的
        symbol = st_symbol[-1]
        if self.stock_info is not None and self.stock_info() is not None:
            self.stock_info().close()
        self.stock_info = WidgetStockInfo(symbol)
        self.stock_info.display()


class WidgetStockInfo(WidgetBase):
    """股票基本信息界面组件类"""

    def __init__(self, symbol_code):
        """
        构建股票基本信息
        :param symbol_code: 字符串symbol eg usTSLA 或者 Symbol对象
        """
        if isinstance(symbol_code, six.string_types):
            symbol_code = code_to_symbol(symbol_code)

        # 类中的symbol对象为Symbol类对象，即包含市场，子市场等信息的对象
        self.symbol = symbol_code
        # 默认规则所有组件最终ui成品为widget，暂时未使用基类来约束
        self.widget = self.stock_base_info(self.symbol)

    def other_industries_symbol(self, stock_info):
        """从股票信息查询其对应行业的其它股票，构建形势为按钮序列，暂时只取出10个相关的股票"""
        other_co_symbol = ABuIndustries.query_factorize_industry_symbol(
            stock_info.industry_factorize.values[0], market=self.symbol.market)
        other_co_bs = []
        # TODO 加载更多或者分段加载方式，暂时只取出10个相关的股票
        other_co_symbol = other_co_symbol[:10] if len(other_co_symbol) > 10 else other_co_symbol
        for symbol in other_co_symbol:
            # 通过symbol查询公司名称等信息
            stock_info = query_stock_info(symbol)
            if stock_info is None or stock_info.empty:
                continue

            # 构建button上显示的文字
            co_name_str = self._combine_stock_name(stock_info, only_name=True)
            button = widgets.Button(description=co_name_str, disabled=False)
            # 添加一个新属性symbol在button对象里，on_button_clicked使用
            button.symbol = symbol

            def on_button_clicked(bt):
                # 关闭当前整个大的widget界面，重新构建一个全新的界面
                self.widget.close()
                symbol_code = code_to_symbol(bt.symbol)
                self.symbol = symbol_code
                # 重新赋予self.widget值，即一个新的widget
                self.widget = self.stock_base_info(self.symbol)
                # 重新显示新的界面
                self.display()

            button.on_click(on_button_clicked)
            other_co_bs.append(button)

        # 将symbol button一行显示两个，2个为一组，组装子symbol button序列,
        other_co_bs = self._sub_split(other_co_bs, len(other_co_bs) / 2)
        # 将每一组加到一个行box里面
        tbs_boxs = [widgets.HBox(tbs) for tbs in other_co_bs]
        self.other_co_box = widgets.VBox(tbs_boxs)
        return self.other_co_box

    def _sub_split(self, n_buttons, n_split):
        """将symbol button，每n_split个为一组，组装子symbol button序列"""
        sub_bt_cnt = int(len(n_buttons) / n_split)
        if sub_bt_cnt == 0:
            sub_bt_cnt = 1
        group_adjacent = lambda a, k: zip(*([iter(a)] * k))
        bts_group = list(group_adjacent(n_buttons, sub_bt_cnt))
        residue_ind = -(len(n_buttons) % sub_bt_cnt) if sub_bt_cnt > 0 else 0
        if residue_ind < 0:
            bts_group.append(n_buttons[residue_ind:])
        return bts_group

    @catch_error(return_val='')
    def _combine_stock_name(self, stock_info, only_name=False):
        """通过stock_info中的公司信息构建相关字符串名称"""
        if only_name:
            co_name_str = to_unicode(stock_info.co_name.values[0])
        else:
            # eg: 特斯拉电动车（US.NASDQ:TSLA）
            co_name_str = u'{}({}.{}:{})'.format(to_unicode(stock_info.co_name.values[0]),
                                                 to_unicode(stock_info.market.values[0]),
                                                 to_unicode(stock_info.exchange.values[0]),
                                                 to_unicode(stock_info.symbol.values[0]))
        return co_name_str

    def stock_base_info(self, symbol_code):
        """构建股票基本信息：公司简介，业务，市值，市盈率，市净率，每股净资产，流通股本，总股本，机构持股等信息"""
        if not ABuEnv.g_is_ipython:
            logging.info('widget op only support ipython env!')
            return

        stock_info = query_stock_info(symbol_code)
        if stock_info is None or stock_info.empty:
            logging.info('stock_info is None or stock_info.empty!')
            return

        # 公司名称
        co_name_str = self._combine_stock_name(stock_info)
        co_name = widgets.Text(
            value=co_name_str,
            description=u'公司名称:',
            disabled=False
        )

        # 公司简介
        co_intro = None
        if 'co_intro' in stock_info:
            co_intro = widgets.Textarea(
                value=to_unicode(stock_info.co_intro.values[0]),
                description=u'公司简介:',
                disabled=False,
                layout=widgets.Layout(height='226px')
            )

        co_site = None
        if 'co_site' in stock_info:
            site = to_unicode(stock_info.co_site.values[0])
            co_site_str = u'<p><a target="_blank" a href="{}">公司网站: {}</a></p>'.format(
                site, site)
            co_site = widgets.HTML(value=co_site_str)

        pv_dict = {
            'pe_s_d': u"市盈率(静)/(动):",
            'pb_d': u"市净率(动):",
            'pb_MRQ': u"市净率MRQ:",
            'ps_d': u"市销率(动):",
            'ps': u"市销率:",
            'pe_s': u"市盈率(静):"}

        p_widget_tab = self.make_sub_tab_widget(stock_info, pv_dict)

        asset_dict = {
            'mv': u"总市值:",
            'asset': u"每股净资产：",
            'cc': u"流通股本："
        }

        asset_widget_tab = self.make_sub_tab_widget(stock_info, asset_dict)

        equity_dict = {
            'equity': u"总股本:",
            'hk_equity': u"港股股本:",
            'oo': u"机构持股:"
        }
        equity_widget_tab = self.make_sub_tab_widget(stock_info, equity_dict)
        accordion = widgets.Accordion(children=[self.other_industries_symbol(stock_info)])
        industry_str = to_unicode(stock_info.industry.values[0])
        industry = u'行业：{}'.format(industry_str)
        accordion.set_title(0, industry)
        base_info_widgets = list(filter(lambda widget: widget is not None,
                                        [co_name, co_intro, co_site, p_widget_tab, asset_widget_tab, equity_widget_tab,
                                         accordion]))
        base_info = widgets.VBox(base_info_widgets)
        return base_info

    def make_sub_tab_widget(self, stock_info, sub_dict):
        """用于构建：股本/港股股本/机构持股子tab，市盈率/市净率/市销率子tab, 总市值/每股净资产/流通股本子tab"""
        sub_widget_array = []
        sub_widget_table_name = []
        for sc in sub_dict:
            if sc in stock_info.columns:
                sub_name = to_unicode(sub_dict[sc])
                sub_widget = widgets.Text(
                    value=to_unicode(stock_info[sc].values[0]),
                    description=sub_name,
                    disabled=False
                )
                sub_widget_array.append(sub_widget)
                sub_widget_table_name.append(sub_name)

        sub_widget_tab = widgets.Tab()
        sub_widget_tab.children = sub_widget_array
        for ind, name in enumerate(sub_widget_table_name):
            sub_widget_tab.set_title(ind, name)
        return sub_widget_tab
