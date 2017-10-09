# -*- encoding:utf-8 -*-
"""股票基本信息图形可视化"""

from __future__ import print_function
from __future__ import absolute_import
from __future__ import division

import os
import ipywidgets as widgets

from ..CoreBu import ABuEnv
from ..CoreBu.ABuEnv import EDataCacheType, EMarketDataFetchMode, EMarketSourceType
from ..WidgetBu.ABuWGBase import WidgetBase, accordion_shut
from ..UtilBu import ABuFileUtil

__author__ = '阿布'
__weixin__ = 'abu_quant'


class WidgetEnvSetMixin(object):
    """
        使用混入而不要做为上层widget拥有的模式，可为多个上层使用
        便于上层widgte使用self去获取设置，统一上层使用
        混入类：基础env设置：
        1. 沙盒模式与实时
        2. csv模式与hdf5模式
        3. 数据获取模式
        4. 数据源切换
    """

    # noinspection PyProtectedMember
    def init_env_set_ui(self):
        """构建基础env widget ui return widgets.VBox"""
        """沙盒数据与开放数据模式切换"""
        self.date_mode = widgets.RadioButtons(
            options=[u'沙盒数据模式', u'开放数据模式'],
            value=u'沙盒数据模式' if ABuEnv._g_enable_example_env_ipython else u'开放数据模式',
            description=u'数据模式:',
            disabled=False
        )
        self.date_mode.observe(self.on_data_mode_change, names='value')

        set_mode_label_tip = widgets.Label(u'缓存模式|联网模式|数据源只在开放数据模式下生效：',
                                           layout=widgets.Layout(width='300px', align_items='stretch'))

        """csv模式与hdf5模式模式切换"""
        self.store_mode_dict = {EDataCacheType.E_DATA_CACHE_CSV.value: u'csv模式(推荐)',
                                EDataCacheType.E_DATA_CACHE_HDF5.value: u'hdf5模式'}
        self.store_mode = widgets.RadioButtons(
            options=[u'csv模式(推荐)', u'hdf5模式'],
            value=self.store_mode_dict[ABuEnv.g_data_cache_type.value],
            description=u'缓存模式:',
            disabled=False
        )
        self.store_mode.observe(self.on_data_store_change, names='value')

        """数据获取模式模式切换"""
        self.fetch_mode_dict = {EMarketDataFetchMode.E_DATA_FETCH_FORCE_LOCAL.value: u'本地数据模式(推荐)',
                                EMarketDataFetchMode.E_DATA_FETCH_NORMAL.value: u'本地网络结合模式',
                                EMarketDataFetchMode.E_DATA_FETCH_FORCE_NET.value: u'强制全部使用网络'}
        self.fetch_mode = widgets.RadioButtons(
            options=[u'本地数据模式(推荐)', u'本地网络结合模式', u'强制全部使用网络'],
            value=self.fetch_mode_dict[ABuEnv.g_data_fetch_mode.value],
            description=u'联网模式:',
            disabled=False
        )
        self.fetch_mode.observe(self.on_fetch_mode_change, names='value')

        """数据源进行切换"""
        self.data_source_accordion = widgets.Accordion()
        self.date_source_dict = {EMarketSourceType.E_MARKET_SOURCE_bd.value: u'百度数据源(美股，A股，港股)',
                                 EMarketSourceType.E_MARKET_SOURCE_tx.value: u'腾讯数据源(美股，A股，港股)',
                                 EMarketSourceType.E_MARKET_SOURCE_nt.value: u'网易数据源(美股，A股，港股)',
                                 EMarketSourceType.E_MARKET_SOURCE_sn_us.value: u'新浪美股(美股)',
                                 EMarketSourceType.E_MARKET_SOURCE_sn_futures.value: u'新浪国内期货(国内期货)',
                                 EMarketSourceType.E_MARKET_SOURCE_sn_futures_gb.value: u'新浪国际期货(国际期货)',
                                 EMarketSourceType.E_MARKET_SOURCE_hb_tc.value: u'比特币，莱特币'}
        self.date_source = widgets.RadioButtons(
            options=[u'百度数据源(美股，A股，港股)', u'腾讯数据源(美股，A股，港股)', u'网易数据源(美股，A股，港股)',
                     u'新浪美股(美股)', u'新浪国内期货(国内期货)', u'新浪国际期货(国际期货)',
                     u'比特币，莱特币'],
            value=self.date_source_dict[ABuEnv.g_market_source.value],
            description=u'数据源:',
            disabled=False
        )
        self.date_source.observe(self.on_date_source_change, names='value')
        source_label_tip1 = widgets.Label(u'内置的数据源仅供学习使用',
                                          layout=widgets.Layout(width='300px', align_items='stretch'))
        source_label_tip2 = widgets.Label(u'abupy提供了接入外部数据源的接口和规范',
                                          layout=widgets.Layout(width='300px', align_items='stretch'))
        source_label_tip3 = widgets.Label(u'详阅读github上教程第19节的示例',
                                          layout=widgets.Layout(width='300px', align_items='stretch'))

        other_data_set_box = widgets.VBox([self.fetch_mode, source_label_tip1, self.date_source, source_label_tip2,
                                           source_label_tip3, self.store_mode])

        self.data_source_accordion.children = [other_data_set_box]
        self.data_source_accordion.set_title(0, u'缓存模式|联网模式|数据源')
        accordion_shut(self.data_source_accordion)

        mdm_box = widgets.VBox([self.date_mode, set_mode_label_tip, self.data_source_accordion])

        if ABuEnv._g_enable_example_env_ipython:
            # 非沙盒数据下数据存贮以及数据获取模式切换才生效
            self.store_mode.disabled = True
            self.fetch_mode.disabled = True

        return mdm_box

    def on_data_mode_change(self, change):
        """沙盒与非沙盒数据界面操作转换"""
        if change['new'] == u'沙盒数据模式':
            ABuEnv.enable_example_env_ipython(show_log=False)
            self.store_mode.disabled = True
            self.fetch_mode.disabled = True
            accordion_shut(self.data_source_accordion)
        else:
            if ABuFileUtil.file_exist(ABuEnv.g_project_kl_df_data_csv) and \
                            len(os.listdir(ABuEnv.g_project_kl_df_data_csv)) > 5000:
                # 如果有很多缓存数据，从沙盒改变依然网络模式是本地模式
                ABuEnv._g_enable_example_env_ipython = False
            else:
                # 如果没有很多缓存从沙盒改到开放一起改变网络模式本地网络结合
                ABuEnv.disable_example_env_ipython(show_log=False)

            self.store_mode.disabled = False
            self.fetch_mode.disabled = False
            self.data_source_accordion.selected_index = 0
        # 需要数据获取模式界面进行同步更新
        self.fetch_mode.value = self.fetch_mode_dict[ABuEnv.g_data_fetch_mode.value]

    def on_data_store_change(self, change):
        """数据存储模式界面操作进行改变"""
        store_mode_dict = {self.store_mode_dict[sk]: EDataCacheType(sk) for sk in self.store_mode_dict}
        # 改变设置值
        ABuEnv.g_data_cache_type = store_mode_dict[change['new']]

    def on_fetch_mode_change(self, change):
        """数据获取模式界面操作进行改变"""
        fetch_mode_dict = {self.fetch_mode_dict[fm]: EMarketDataFetchMode(fm) for fm in self.fetch_mode_dict}
        # 改变设置值
        ABuEnv.g_data_fetch_mode = fetch_mode_dict[change['new']]

    def on_date_source_change(self, change):
        """数据源界面操作进行改变"""
        date_source_dict = {self.date_source_dict[ds]: EMarketSourceType(ds) for ds in self.date_source_dict}
        # 改变设置值
        ABuEnv.g_market_source = date_source_dict[change['new']]


class WidgetTimeModeMixin(object):
    """
        使用混入而不要做为上层widget拥有的模式，可为多个上层使用
        便于上层widgte使用self去获取设置，统一上层使用
        混入类：基础时间模式设置：
        1. 年数模式
        2. 开始结束模式
    """

    # noinspection PyProtectedMember
    def init_time_mode_ui(self):
        """构建基础env widget ui return widgets.VBox"""
        # 回测时间模式
        self.time_mode = widgets.RadioButtons(
            options={u'使用{}年数'.format(self.time_mode_str()): 0,
                     u'使用{}开始结束日期'.format(self.time_mode_str()): 1},
            value=0,
            description=u'时间模式:',
            disabled=False
        )
        self.time_mode.observe(self.on_time_mode_change, names='value')

        # 年数模式
        self.run_years = widgets.BoundedIntText(
            value=2,
            min=1,
            max=6,
            step=1,
            description=u'{}年数:'.format(self.time_mode_str()),
            disabled=False
        )
        # 开始结束模式
        self.start = widgets.Text(
            value='2014-07-26',
            placeholder=u'年-月-日',
            description=u'开始日期:',
            disabled=False
        )
        self.end = widgets.Text(
            value='2016-07-26',
            placeholder=u'年-月-日',
            description=u'结束日期:',
            disabled=False
        )
        self.run_years.disabled = False
        self.start.disabled = True
        self.end.disabled = True

        return widgets.VBox([self.time_mode, self.run_years, self.start, self.end])

    def on_time_mode_change(self, change):
        """切换使用年数还是起始，结束时间做为回测参数"""
        if change['new'] == 0:
            self.run_years.disabled = False
            self.start.disabled = True
            self.end.disabled = True
        else:
            self.run_years.disabled = True
            self.start.disabled = False
            self.end.disabled = False

    def time_mode_str(self):
        """子类实现返回一个字符串代表时间设置的意义eg：回测，分析"""
        raise NotImplementedError('NotImplementedError time_mode_str!')


class WidgetMetricsSet(object):
    """
        使用混入而不要做为上层widget拥有的模式，可为多个上层使用
        便于上层widgte使用self去获取设置，统一上层使用
        混入类：回测输出设置：
        1. 输出模式：
            1. order_cmp + only_show_returns
            2. returns_cmp + only_info
        2. 输出度量对象：
            1. 只输出交易单：orders_pd
            2. 只输出行为单：action_pd
            3. 只输出资金单：capital_pd
            4. 同时输出交易单，行为单，资金单(orders_pd, action_pd, capital_pd)
        3. 输出交易单最大行列显示设置：
            1. 默认最大行显示50
            2. 默认最大列显示50
        4. 是否将交易单，行为单，资金单保存在本地output文件中
    """

    # noinspection PyProtectedMember
    def init_metrics_ui(self):
        """构建基础env widget ui return widgets.VBox"""
        # 回测时间模式
        self.metrics_mode = widgets.RadioButtons(
            options={u'考虑初始资金＋标尺大盘对比': 0,
                     u'不考虑初始资金＋不对比标尺': 1},
            value=0,
            description=u'度量模式:',
            disabled=False
        )

        self.metrics_out_put = widgets.RadioButtons(
            options={u'只输出交易单：orders_pd': 0,
                     u'只输出行为单：action_pd': 1,
                     u'只输出资金单：capital_pd': 2,
                     u'输出交易单，行为单，资金单': 3},
            value=0,
            description=u'输出对象:',
            disabled=False
        )

        out_put_display_max_label1 = widgets.Label(u'输出显示最大行列数，最大100行，100列',
                                                   layout=widgets.Layout(width='300px', align_items='stretch'))
        out_put_display_max_label2 = widgets.Label(u'如需查看更多输出表单，请选择保存输出至文件',
                                                   layout=widgets.Layout(width='300px', align_items='stretch'))
        self.out_put_display_max_rows = widgets.IntSlider(
            value=50,
            min=1,
            max=100,
            step=1,
            description=u'行数',
            disabled=False,
            orientation='horizontal',
            readout=True,
            readout_format='d'
        )

        self.out_put_display_max_columns = widgets.IntSlider(
            value=50,
            min=1,
            max=100,
            step=1,
            description=u'列数',
            disabled=False,
            orientation='horizontal',
            readout=True,
            readout_format='d'
        )
        out_put_display = widgets.VBox([out_put_display_max_label1,
                                        out_put_display_max_label2,
                                        self.out_put_display_max_rows,
                                        self.out_put_display_max_columns])

        save_out_put_lable = widgets.Label(u'是否保存交易单，行为单，资金单到文件',
                                           layout=widgets.Layout(width='300px', align_items='stretch'))
        save_out_put_lable2 = widgets.Label(u'路径:{}'.format(os.path.join(ABuEnv.g_project_data_dir, 'out_put')),
                                            layout=widgets.Layout(width='300px', align_items='stretch'))
        self.save_out_put = widgets.Checkbox(
            value=False,
            description=u'保存输出',
            disabled=False,
        )
        save_out_put = widgets.VBox([save_out_put_lable,
                                     save_out_put_lable2,
                                     self.save_out_put])

        accordion = widgets.Accordion()
        accordion.children = [widgets.VBox([self.metrics_mode, self.metrics_out_put, out_put_display, save_out_put])]
        accordion.set_title(0, u'回测度量结果设置')
        accordion_shut(accordion)

        return accordion


class WidgetRunTT(WidgetBase, WidgetEnvSetMixin, WidgetTimeModeMixin, WidgetMetricsSet):
    """基础设置界面：初始资金，回测开始，结束周期，参考大盘等"""

    def __init__(self):
        """初始化基础回测设置界面"""
        # 初始资金
        self.cash = widgets.BoundedIntText(
            value=1000000,
            min=10000,
            max=999999999,
            step=1,
            description=u'初始资金:',
            disabled=False
        )
        tm_box = self.init_time_mode_ui()
        mdm_box = self.init_env_set_ui()
        metrics_box = self.init_metrics_ui()

        self.widget = widgets.VBox([self.cash, tm_box, mdm_box, metrics_box])

    def time_mode_str(self):
        """实现混入WidgetTimeModeMixin，声明时间模块代表回测"""
        return u'回测'
