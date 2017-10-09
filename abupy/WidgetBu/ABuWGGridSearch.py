# -*- encoding:utf-8 -*-
"""策略最优参数grid search图形可视化"""

from __future__ import print_function
from __future__ import absolute_import
from __future__ import division

import os
import logging
import math

import ipywidgets as widgets

from ..UtilBu import ABuFileUtil
from ..WidgetBu.ABuWGBase import WidgetBase, show_msg_toast_func, browser_down_csv_zip
from ..WidgetBu.ABuWGBSymbol import WidgetSymbolChoice
from ..WidgetBu.ABuWGBFBase import BuyFactorWGManager
from ..WidgetBu.ABuWGSFBase import SellFactorWGManager
from ..CoreBu import ABuEnv
from ..CoreBu.ABuEnv import EDataCacheType
# noinspection PyUnresolvedReferences
from ..CoreBu.ABuFixes import filter
from ..MarketBu.ABuMarket import is_in_sand_box
from ..MarketBu.ABuSymbolPd import check_symbol_in_local_csv
from ..MetricsBu.ABuGridSearch import GridSearch

__author__ = '阿布'
__weixin__ = 'abu_quant'


# noinspection PyProtectedMember
class WidgetGridSearch(WidgetBase):
    """策略最优参数grid search"""

    # noinspection PyProtectedMember
    def __init__(self):
        """构建回测需要的各个组件形成tab"""

        tip_label1 = widgets.Label(u'最优参数grid search暂不支持实时网络数据模式', layout=widgets.Layout(width='300px'))
        tip_label2 = widgets.Label(u'非沙盒模式需先用\'数据下载界面操作\'进行下载', layout=widgets.Layout(width='300px'))
        """沙盒数据与开放数据模式切换"""
        self.date_mode = widgets.RadioButtons(
            options=[u'沙盒数据模式', u'开放数据模式'],
            value=u'沙盒数据模式' if ABuEnv._g_enable_example_env_ipython else u'开放数据模式',
            description=u'数据模式:',
            disabled=False
        )
        self.date_mode.observe(self.on_data_mode_change, names='value')
        date_mode_box = widgets.VBox([tip_label1, tip_label2, self.date_mode])

        self.sc = WidgetSymbolChoice()
        self.bf = BuyFactorWGManager(add_button_style='grid')
        self.sf = SellFactorWGManager(show_add_buy=False, add_button_style='grid')

        sub_widget_tab = widgets.Tab()
        sub_widget_tab.children = [self.bf.widget, self.sf.widget, self.sc.widget]
        for ind, name in enumerate([u'买策', u'卖策', u'股池']):
            sub_widget_tab.set_title(ind, name)

        self.begin_grid_search = widgets.Button(description=u'开始寻找策略最优参数组合', layout=widgets.Layout(width='98%'),
                                                button_style='danger')
        self.begin_grid_search.on_click(self.run_grid_search)

        self.widget = widgets.VBox([date_mode_box, sub_widget_tab, self.begin_grid_search])

    def on_data_mode_change(self, change):
        """沙盒与非沙盒数据界面操作转换"""
        if change['new'] == u'沙盒数据模式':
            ABuEnv.enable_example_env_ipython(show_log=False)
        else:
            # 从沙盒改变依然网络模式是本地模式
            ABuEnv._g_enable_example_env_ipython = False

    # noinspection PyUnusedLocal
    def run_grid_search(self, bt):
        """运行回测所对应的button按钮"""
        # 清理之前的输出结果
        # ABuProgress.clear_output()

        choice_symbols = self.sc.choice_symbols.options
        if choice_symbols is None or len(choice_symbols) == 0:
            # 如果一个symbol都没有设置None， gird search不支持全市场的最优参数寻找，
            show_msg_toast_func(u'请最少在\'股池\'中选择一个symbol！')
            return

        if not self.check_symbol_data(choice_symbols):
            # 监测是否本地缓存数据存在或者沙盒数据不匹配
            return

        # 买入策略构成序列
        buy_factors = list(self.bf.factor_dict.values())
        if len(buy_factors) == 0:
            show_msg_toast_func(u'请最少选择一个买入策略')
            return
        # 合并买入因子不同的class factor到符合grid search格式的因子参数组合
        # print(buy_factors)
        buy_factors = GridSearch.combine_same_factor_class(buy_factors)

        # 卖出策略可以一个也没有
        sell_factors = list(self.sf.factor_dict.values())
        # 合并卖出因子不同的class factor到符合grid search格式的因子参数组合
        sell_factors = GridSearch.combine_same_factor_class(sell_factors)

        cash = 10000000
        scores, score_tuple_array = GridSearch.grid_search(cash, choice_symbols, buy_factors, sell_factors)

    def check_symbol_data(self, choice_symbols):
        """检测是否需要提示下载csv数据或者使用数据下载界面进行操作"""

        if ABuEnv._g_enable_example_env_ipython and choice_symbols is not None:
            # 沙盒模式下 and choice_symbols不是none
            not_in_sb_list = list(filter(lambda symbol: not is_in_sand_box(symbol), choice_symbols))
            if len(not_in_sb_list) > 0:
                logging.info(
                    u'当前数据模式为\'沙盒模式\'无{}数据，'
                    u'请在\'设置\'中切换数据模式并确认数据在本地存在！'
                    u'最优参数grid search暂不支持实时网络数据模式！'
                    u'所以非沙盒模式需要先用\'数据下载界面操作\'进行数据下载'.format(not_in_sb_list))
                browser_down_csv_zip()
                return False
        else:
            # 非沙盒数据模式下
            if ABuEnv.g_data_cache_type == EDataCacheType.E_DATA_CACHE_CSV:
                # csv模式下，一个csv数据都没有
                if not ABuFileUtil.file_exist(ABuEnv.g_project_kl_df_data_csv):
                    # 股票类型全市场回测，但没有数据
                    logging.info(
                        u'未发现本地缓存数据，最优参数grid search暂不支持实时网络数据模式！'
                        u'所以非沙盒模式需要先用\'数据下载界面操作\'进行数据下载')
                    browser_down_csv_zip()
                    return False
                elif len(os.listdir(ABuEnv.g_project_kl_df_data_csv)) < 100:
                    # 未下载云盘上的csv为前提条件
                    not_in_local_csv = list(filter(lambda symbol:
                                                   not check_symbol_in_local_csv(symbol), choice_symbols))
                    # 需要grid search的symbol中有30%以上不在本地缓存中提示下载数据
                    if not_in_local_csv > math.ceil(len(choice_symbols) * 0.3):
                        logging.info(
                            u'{}未发现本地缓存数据，最优参数grid search暂不支持实时网络数据模式！'
                            u'需要先用\'数据下载界面操作\'进行数据下载'.format(not_in_local_csv))
                        browser_down_csv_zip()
                        return False

            elif ABuEnv.g_data_cache_type == EDataCacheType.E_DATA_CACHE_HDF5 \
                    and not ABuFileUtil.file_exist(ABuEnv.g_project_kl_df_data):
                # hdf5模式下文件不存在
                logging.info(
                    u'未发现本地缓存数据，最优参数grid search暂不支持实时网络数据模式！'
                    u'所以非沙盒模式需要先用\'数据下载界面操作\'进行数据下载')
                browser_down_csv_zip()
                return False

        return True
