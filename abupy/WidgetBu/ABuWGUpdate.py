# -*- encoding:utf-8 -*-
"""数据下载图形可视化"""

from __future__ import print_function
from __future__ import absolute_import
from __future__ import division
from ..WidgetBu.ABuWGBRunBase import WidgetTimeModeMixin

import logging
from collections import OrderedDict

import ipywidgets as widgets
from ..CoreBu.ABu import run_kl_update
from ..CoreBu import ABuEnv
from ..UtilBu import ABuProgress
from ..WidgetBu.ABuWGBase import WidgetBase
from ..MarketBu.ABuDataCheck import browser_down_csv_zip
from ..CoreBu.ABuEnv import EMarketTargetType, EMarketSourceType

__author__ = '阿布'
__weixin__ = 'abu_quant'


class WidgetUpdate(WidgetBase, WidgetTimeModeMixin):
    """数据下载图形可视化类"""

    def __init__(self):
        tm_box = self.init_time_mode_ui()
        # 修改为默认使用开始结束日期
        self.time_mode.value = 1
        # 修改开始结束日期时间字符串
        self.start.value = '2011-08-08'
        self.end.value = '2017-08-08'

        self.market = widgets.Dropdown(
            options=OrderedDict({u'美股': EMarketTargetType.E_MARKET_TARGET_US.value,
                                 u'A股': EMarketTargetType.E_MARKET_TARGET_CN.value,
                                 u'港股': EMarketTargetType.E_MARKET_TARGET_HK.value,
                                 u'国内期货': EMarketTargetType.E_MARKET_TARGET_FUTURES_CN.value,
                                 u'国际期货': EMarketTargetType.E_MARKET_TARGET_FUTURES_GLOBAL.value,
                                 u'数字货币': EMarketTargetType.E_MARKET_TARGET_TC.value}),
            value=ABuEnv.g_market_target.value,
            description=u'下载更新市场:',
        )
        self.market.observe(self.on_market_change, names='value')

        """数据源进行切换"""
        self.data_source_accordion = widgets.Accordion()
        date_source_dict_us = OrderedDict({
            u'腾讯数据源(美股，A股，港股)': EMarketSourceType.E_MARKET_SOURCE_tx.value,
            u'百度数据源(美股，A股，港股)': EMarketSourceType.E_MARKET_SOURCE_bd.value,
            u'新浪美股(美股)': EMarketSourceType.E_MARKET_SOURCE_sn_us.value,
            u'网易数据源(美股，A股，港股)': EMarketSourceType.E_MARKET_SOURCE_nt.value,
        })

        date_source_dict_cn = OrderedDict({
            u'百度数据源(美股，A股，港股)': EMarketSourceType.E_MARKET_SOURCE_bd.value,
            u'腾讯数据源(美股，A股，港股)': EMarketSourceType.E_MARKET_SOURCE_tx.value,
            u'网易数据源(美股，A股，港股)': EMarketSourceType.E_MARKET_SOURCE_nt.value
        })

        date_source_dict_hk = OrderedDict({
            u'网易数据源(美股，A股，港股)': EMarketSourceType.E_MARKET_SOURCE_nt.value,
            u'腾讯数据源(美股，A股，港股)': EMarketSourceType.E_MARKET_SOURCE_tx.value,
            u'百度数据源(美股，A股，港股)': EMarketSourceType.E_MARKET_SOURCE_bd.value
        })

        date_source_dict_futures_cn = {u'新浪国内期货(国内期货)': EMarketSourceType.E_MARKET_SOURCE_sn_futures.value}
        date_source_dict_futures_gb = {u'新浪国际期货(国际期货)': EMarketSourceType.E_MARKET_SOURCE_sn_futures_gb.value}
        date_source_dict_futures_tc = {u'火币网：比特币，莱特币': EMarketSourceType.E_MARKET_SOURCE_hb_tc.value}

        self.date_source_market_map = {
            EMarketTargetType.E_MARKET_TARGET_US.value: date_source_dict_us,
            EMarketTargetType.E_MARKET_TARGET_CN.value: date_source_dict_cn,
            EMarketTargetType.E_MARKET_TARGET_HK.value: date_source_dict_hk,
            EMarketTargetType.E_MARKET_TARGET_FUTURES_CN.value: date_source_dict_futures_cn,
            EMarketTargetType.E_MARKET_TARGET_FUTURES_GLOBAL.value: date_source_dict_futures_gb,
            EMarketTargetType.E_MARKET_TARGET_TC.value: date_source_dict_futures_tc}

        self.current_date_source_dict = self.date_source_market_map[self.market.value]
        self.date_source = widgets.RadioButtons(
            options=list(self.current_date_source_dict.keys()),
            value=list(self.current_date_source_dict.keys())[0],
            description=u'数据源:',
            disabled=False
        )
        self.date_source.observe(self.on_date_source_change, names='value')

        self.yun_down_bt = widgets.Button(description=u'美股|A股|港股|币类|期货6年日k', layout=widgets.Layout(width='50%'),
                                          button_style='info')
        self.yun_down_bt.on_click(self.run_yun_down)

        self.run_kl_update_bt = widgets.Button(description=u'从数据源下载更新', layout=widgets.Layout(width='50%'),
                                               button_style='danger')
        self.run_kl_update_bt.on_click(self.run_kl_update)
        description = widgets.Textarea(
            value=u'非沙盒数据，特别是回测交易多的情况下，比如全市场测试，回测前需要先将数据进行更新。\n'
                  u'建议直接从云盘下载入库完毕的数据库，不需要从各个数据源再一个一个的下载数据进行入库。\n'
                  u'abupy内置数据源都只是为用户学习使用，并不能保证数据一直通畅，而且如果用户很在乎数据质量，'
                  u'比如有些数据源会有前复权数据错误问题，有些数据源成交量不准确等问题，那么就需要接入用户自己的数据源。\n'
                  u'接入用户的数据源请阅读教程第19节中相关内容',
            disabled=False,
            layout=widgets.Layout(height='200px')
        )
        self.widget = widgets.VBox(
            [description, self.market, tm_box, self.date_source, self.yun_down_bt, self.run_kl_update_bt])

    # noinspection PyUnusedLocal
    def run_yun_down(self, bt):
        """打开浏览器csv zip地址准备开始下载"""
        browser_down_csv_zip(open_browser=True)

    # noinspection PyUnusedLocal
    def run_kl_update(self, bt):
        """数据下载更新主接口"""
        n_folds = 2
        start = None
        end = None
        if not self.run_years.disabled:
            # 如果使用年回测模式
            n_folds = self.run_years.value
        if not self.start.disabled:
            # 使用开始回测日期
            start = self.start.value
        if not self.end.disabled:
            # 使用结束回测日期
            end = self.end.value
        market = ABuEnv.g_market_target

        logging.info(u'开始下载更新{}市场{}-{}:{}年, 数据源{}'.format(
            market.value, '' if start is None else start, '' if end is None else end, n_folds,
            ABuEnv.g_market_source.value))
        run_kl_update(start=start, end=end, n_folds=n_folds, market=market, n_jobs=10)
        logging.info(u'下载更新完成')

    def time_mode_str(self):
        """实现混入WidgetTimeModeMixin，声明时间模块代表下载更新"""
        return u'下载更新'

    def on_date_source_change(self, change):
        """数据源界面操作进行改变"""
        # 改变设置值
        ABuEnv.g_market_source = EMarketSourceType(self.current_date_source_dict[change['new']])

    def on_market_change(self, change):
        """切换大盘市场"""
        ABuEnv.g_market_target = EMarketTargetType(change['new'])

        self.current_date_source_dict = self.date_source_market_map[ABuEnv.g_market_target.value]

        options = list(self.current_date_source_dict.keys())
        self.date_source.options = options
        self.date_source.value = options[0]
