# -*- encoding:utf-8 -*-
"""上层回测图形可视化"""

from __future__ import print_function
from __future__ import absolute_import
from __future__ import division

import pandas as pd
from IPython.display import display
import ipywidgets as widgets

from ..UtilBu import ABuProgress
from ..WidgetBu.ABuWGBase import WidgetBase, show_msg_func, show_msg_toast_func
from ..WidgetBu.ABuWGBRunBase import WidgetRunTT
from ..WidgetBu.ABuWGBSymbol import WidgetSymbolChoice
from ..WidgetBu.ABuWGBFBase import BuyFactorWGManager
from ..WidgetBu.ABuWGSFBase import SellFactorWGManager
from ..WidgetBu.ABuWGPSBase import PickStockWGManager
from ..WidgetBu.ABuWGPosBase import PosWGManager
from ..WidgetBu.ABuWGUmp import WidgetUmp

from ..CoreBu.ABu import run_loop_back
from ..CoreBu.ABuStore import store_abu_result_out_put
# noinspection PyUnresolvedReferences
from ..CoreBu.ABuFixes import filter
from ..MarketBu.ABuDataCheck import check_symbol_data_mode
from ..BetaBu import ABuAtrPosition, ABuPositionBase
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

        self.pos = PosWGManager()
        # 资金管理注册买入策略接收改变
        self.pos.register(self.bf)

        # 构造裁判界面
        self.ump = WidgetUmp()

        sub_widget_tab = widgets.Tab()
        sub_widget_tab.children = [self.tt.widget, self.sc.widget, self.bf.widget, self.sf.widget, self.ps.widget,
                                   self.pos.widget, self.ump.widget]
        for ind, name in enumerate([u'基本', u'股池', u'买策', u'卖策', u'选股', u'资管', u'裁判']):
            sub_widget_tab.set_title(ind, name)

        self.run_loop_bt = widgets.Button(description=u'开始回测', layout=widgets.Layout(width='98%'),
                                          button_style='danger')
        self.run_loop_bt.on_click(self.run_loop_back)
        self.widget = widgets.VBox([sub_widget_tab, self.run_loop_bt])

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
        # ABuProgress.clear_output()

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

        if not check_symbol_data_mode(choice_symbols):
            return

        # 买入策略构成序列
        buy_factors = list(self.bf.factor_dict.values())
        if len(buy_factors) == 0:
            msg = u'没有添加任何一个买入策略！'
            show_msg_toast_func(msg)
            return

        # 卖出策略可以一个也没有
        sell_factors = list(self.sf.factor_dict.values())

        pos_class_list = list(self.pos.factor_dict.values())
        if len(pos_class_list) == 1:
            # 资金仓位管理全局策略设置, [0]全局仓位管理策略只能是一个且是唯一
            ABuPositionBase.g_default_pos_class = pos_class_list[0]
        # 裁判根据工作模式进行回测前设置
        self.ump.run_before()
        if choice_symbols is not None and len(choice_symbols) == 1:
            # 如果只有1支股票回测，直接使用这个股票做为做为对比基准
            benchmark = AbuBenchmark(choice_symbols[0])
            capital = AbuCapital(cash, benchmark)
            if len(pos_class_list) == 0:
                # 如果只有1支股票回测，且没有修改过资金管理设置，持仓比例调高
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

        # ump收尾工作
        self.ump.run_end(abu_result_tuple, choice_symbols, list(self.bf.factor_dict.keys()),
                         list(self.sf.factor_dict.keys()), list(self.ps.factor_dict.keys()))
