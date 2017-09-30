# -*- encoding:utf-8 -*-
"""上层回测图形可视化"""

from __future__ import print_function
from __future__ import absolute_import
from __future__ import division

import os
import logging

import pandas as pd
from IPython.display import display
import ipywidgets as widgets

from ..UtilBu import ABuProgress, ABuFileUtil
from ..WidgetBu.ABuWGBase import WidgetBase, show_msg_func, browser_down_csv_zip
from ..WidgetBu.ABuWGBRunBase import WidgetRunTT
from ..WidgetBu.ABuWGBSymbol import WidgetSymbolChoice
from ..WidgetBu.ABuWGBFBase import BuyFactorWGManager
from ..WidgetBu.ABuWGSFBase import SellFactorWGManager
from ..WidgetBu.ABuWGPSBase import PickStockWGManager

from ..CoreBu.ABu import run_loop_back
from ..CoreBu.ABuStore import store_abu_result_out_put
from ..CoreBu import ABuEnv
from ..CoreBu.ABuEnv import EDataCacheType, EMarketDataFetchMode, EMarketTargetType
# noinspection PyUnresolvedReferences
from ..CoreBu.ABuFixes import filter
from ..MarketBu.ABuMarket import is_in_sand_box
from ..BetaBu import ABuAtrPosition
from ..AlphaBu import ABuPickTimeExecute
from ..TradeBu.ABuBenchmark import AbuBenchmark
from ..TradeBu.ABuCapital import AbuCapital
from ..MetricsBu.ABuMetricsBase import AbuMetricsBase
from ..CoreBu.ABuStore import AbuResultTuple

__author__ = '阿布'
__weixin__ = 'abu_quant'


# noinspection PyProtectedMember
class WidgetRunLoopBack(WidgetBase):
    """基础界面可以化：初始资金，回测开始，结束周期，参考大盘等"""

    # noinspection PyProtectedMember
    def __init__(self):
        """构建回测需要的各个组件形成tab"""
        self.tt = WidgetRunTT()
        self.sc = WidgetSymbolChoice()
        self.bf = BuyFactorWGManager()
        self.sf = SellFactorWGManager()
        # 卖出策略管理注册买入策略接收改变
        self.sf.register(self.bf)
        self.ps = PickStockWGManager()
        # 选股策略管理注册买入策略接收改变
        self.ps.register(self.bf)

        sub_widget_tab = widgets.Tab()
        sub_widget_tab.children = [self.tt.widget, self.sc.widget, self.bf.widget, self.sf.widget, self.ps.widget]
        for ind, name in enumerate([u'基本', u'股池', u'买策', u'卖策', u'选股']):
            sub_widget_tab.set_title(ind, name)

        self.run_loop_bt = widgets.Button(description=u'开始回测', layout=widgets.Layout(width='98%'),
                                          button_style='danger')
        self.run_loop_bt.on_click(self.run_loop_back)
        self.widget = widgets.VBox([sub_widget_tab, self.run_loop_bt])

    def check_symbol_data_mode(self, choice_symbols):
        """检测是否需要提示下载csv数据或者使用数据下载界面进行操作"""
        if ABuEnv._g_enable_example_env_ipython and choice_symbols is not None:
            # 沙盒模式下 and choice_symbols不是none
            not_in_sb_list = list(filter(lambda symbol: not is_in_sand_box(symbol), choice_symbols))
            if len(not_in_sb_list) > 0:
                logging.info(
                    u'当前数据模式为\'沙盒模式\'无{}数据，'
                    u'请在\'分析设置\'中切换数据模式并确认数据可获取！'
                    u'非沙盒模式建议先用\'数据下载界面操作\'进行数据下载'
                    u'之后设置数据模式为\'开放数据模式\'，联网模式使用\'本地数据模式\''.format(not_in_sb_list))
                browser_down_csv_zip()
                return False

        is_stock_market = \
            ABuEnv.g_market_target == EMarketTargetType.E_MARKET_TARGET_FUTURES_CN or \
            ABuEnv.g_market_target == EMarketTargetType.E_MARKET_TARGET_US or \
            ABuEnv.g_market_target == EMarketTargetType.E_MARKET_TARGET_HK

        if is_stock_market and not ABuEnv._g_enable_example_env_ipython and choice_symbols is None:
            # 非沙盒模式下要做全股票市场全市场回测
            if ABuEnv.g_data_fetch_mode != EMarketDataFetchMode.E_DATA_FETCH_FORCE_LOCAL:
                logging.info(
                    u'未选择任何回测目标且在非沙盒数据模式下，判定为进行全市场回测'
                    u'为了提高运行效率，请将联网模式修改为\'本地数据模式\'，如需要进行数据更新，'
                    u'请先使用\'数据下载界面操作\'进行数据更新！')
                browser_down_csv_zip()
                return False
            else:
                if ABuEnv.g_data_cache_type == EDataCacheType.E_DATA_CACHE_CSV:
                    # csv模式下
                    if not ABuFileUtil.file_exist(ABuEnv.g_project_kl_df_data_csv):
                        # 股票类型全市场回测，但没有数据
                        logging.info(
                            u'未选择任何回测目标且在非沙盒数据模式下，判定为进行全市场回测'
                            u'为了提高运行效率, 只使用\'本地数据模式\'进行回测，但未发现本地缓存数据，'
                            u'如需要进行数据更新'
                            u'请先使用\'数据下载界面操作\'进行数据更新！')
                        browser_down_csv_zip()
                        return False
                    elif len(os.listdir(ABuEnv.g_project_kl_df_data_csv)) < 100:
                        # 股票类型全市场回测，但数据不足
                        logging.info(
                            u'未选择任何回测目标且在非沙盒数据模式下，判定为进行全市场回测'
                            u'为了提高运行效率, 只使用\'本地数据模式\'进行回测，发现本地缓存数据不足，'
                            u'只有{}支股票历史数据信息'
                            u'如需要进行数据更新'
                            u'请先使用\'数据下载界面操作\'进行数据更新！'.format(
                                len(os.listdir(ABuEnv.g_project_kl_df_data_csv))))
                        browser_down_csv_zip()
                        return False
                elif ABuEnv.g_data_cache_type == EDataCacheType.E_DATA_CACHE_HDF5 \
                        and not ABuFileUtil.file_exist(ABuEnv.g_project_kl_df_data):
                    # hdf5模式下文件不存在
                    logging.info(
                        u'未选择任何回测目标且在非沙盒数据模式下，判定为进行全市场回测'
                        u'为了提高运行效率, 只使用\'本地数据模式\'进行回测'
                        u'hdf5模式下未发现hdf5本地缓存数据，'
                        u'如需要进行数据更新'
                        u'请先使用\'数据下载界面操作\'进行数据更新！')
                    browser_down_csv_zip()
                    return False
        return True

    def _metrics_out_put(self, metrics, abu_result_tuple):
        """针对输出结果和界面中的设置进行输出操作"""
        if metrics is None:
            return

        if self.tt.metrics_mode.value == 0:
            metrics.plot_returns_cmp(only_show_returns=True)
        else:
            metrics.plot_order_returns_cmp(only_info=True)

        pd.options.display.max_rows = self.tt.out_put_display_max_rows.value
        pd.options.display.max_columns = self.tt.out_put_display_max_columns.value

        """
            options={u'只输出交易单：orders_pd': 0,
                     u'只输出行为单：action_pd': 1,
                     u'只输出资金单：capital_pd': 2,
                     u'同时输出交易单，行为单，资金单':3
        """
        if self.tt.metrics_out_put.value == 0 or self.tt.metrics_out_put.value == 3:
            show_msg_func(u'交易买卖详情单：')
            display(abu_result_tuple.orders_pd)
        if self.tt.metrics_out_put.value == 1 or self.tt.metrics_out_put.value == 3:
            show_msg_func(u'交易行为详情单：')
            display(abu_result_tuple.action_pd)
        if self.tt.metrics_out_put.value == 2 or self.tt.metrics_out_put.value == 3:
            show_msg_func(u'交易资金详细单：')
            display(abu_result_tuple.capital.capital_pd)
            show_msg_func(u'交易手续费详单：')
            display(abu_result_tuple.capital.commission.commission_df)

        if self.tt.save_out_put.value is True:
            # 本地保存各个交易单到文件
            store_abu_result_out_put(abu_result_tuple)

    # noinspection PyUnusedLocal
    def run_loop_back(self, bt):
        """运行回测所对应的button按钮"""
        # 清理之前的输出结果
        ABuProgress.clear_output()

        base_run = self.tt
        # 初始资金
        cash = base_run.cash.value
        n_folds = 2
        start = None
        end = None
        if not base_run.run_years.disabled:
            # 如果使用年回测模式
            n_folds = base_run.run_years.value
        if not base_run.start.disabled:
            # 使用开始回测日期
            start = base_run.start.value
        if not base_run.end.disabled:
            # 使用结束回测日期
            end = base_run.end.value

        choice_symbols = self.sc.choice_symbols.options
        if choice_symbols is not None and len(choice_symbols) == 0:
            # 如果一个symbol都没有设置None， 将使用选择的市场进行全市场回测
            choice_symbols = None

        if not self.check_symbol_data_mode(choice_symbols):
            return

        # 买入策略构成序列
        buy_factors = list(self.bf.factor_dict.values())
        if len(buy_factors) == 0:
            msg = u'没有添加任何一个买入策略！'
            show_msg_func(msg)
            return

        # 卖出策略可以一个也没有
        sell_factors = list(self.sf.factor_dict.values())

        if choice_symbols is not None and len(choice_symbols) == 1:
            # 如果只有1支股票回测，直接使用这个股票做为做为对比基准
            benchmark = AbuBenchmark(choice_symbols[0])
            capital = AbuCapital(cash, benchmark)
            # 如果只有1支股票回测，持仓比例调高
            ABuAtrPosition.g_atr_pos_base = 0.5
            # 就一只股票的情况下也不运行选股策略
            orders_pd, action_pd, _ = ABuPickTimeExecute.do_symbols_with_same_factors(choice_symbols,
                                                                                      benchmark,
                                                                                      buy_factors,
                                                                                      sell_factors,
                                                                                      capital, show=True)
            abu_result_tuple = AbuResultTuple(orders_pd, action_pd, capital, benchmark)
            metrics = AbuMetricsBase(orders_pd, action_pd, capital, benchmark)
        else:
            # 针对选股策略中需要choice_symbols的情况进行选股策略choice_symbols更新
            self.ps.seed_choice_symbol_update(choice_symbols)
            # 多只的情况下使用选股策略
            stock_picks = list(self.ps.factor_dict.values())
            if len(stock_picks) == 0:
                stock_picks = None

            # 多只股票使用run_loop_back
            abu_result_tuple, _ = run_loop_back(cash,
                                                buy_factors,
                                                sell_factors,
                                                stock_picks,
                                                choice_symbols=choice_symbols,
                                                start=start,
                                                end=end,
                                                n_folds=n_folds)
            if abu_result_tuple is None:
                return
            ABuProgress.clear_output()
            metrics = AbuMetricsBase(*abu_result_tuple)
        metrics.fit_metrics()
        self._metrics_out_put(metrics, abu_result_tuple)
