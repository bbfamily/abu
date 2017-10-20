# -*- encoding:utf-8 -*-
"""量化相关分析工具图形可视化"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from contextlib import contextmanager

import ipywidgets as widgets
from IPython.display import display

from ..WidgetBu.ABuWGToolBase import WidgetToolBase, multi_fetch_symbol_df_analyse
from ..UtilBu import ABuProgress
from ..MarketBu.ABuSymbol import code_to_symbol
from ..SimilarBu.ABuCorrcoef import ECoreCorrType, corr_matrix
from ..SimilarBu.ABuSimilar import find_similar_with_se, find_similar_with_folds
from ..TLineBu.ABuTLSimilar import calc_similar, coint_similar
from ..MarketBu.ABuDataCheck import all_market_env_check
from ..CoreBu.ABuEnv import EMarketTargetType
from ..CoreBu import ABuEnv
from ..UtilBu.ABuStatsUtil import cosine_distance_matrix, manhattan_distance_matrix, euclidean_distance_matrix

__author__ = '阿布'
__weixin__ = 'abu_quant'


# noinspection PyUnusedLocal,PyProtectedMember
class WidgetSMTool(WidgetToolBase):
    """数据分析界面"""

    def __init__(self, tool_set):
        """初始化相关分析界面"""
        super(WidgetSMTool, self).__init__(tool_set)

        corr = self.init_corr_ui()
        distances = self.init_distances_ui()
        market_corr = self.init_market_corr_ui()
        relative_corr = self.init_relative_corr_ui()
        coint_corr = self.init_coint_corr_ui()

        children = [corr, distances, market_corr, relative_corr, coint_corr]
        if self.scroll_factor_box:
            sm_box = widgets.Box(children,
                                 layout=self.scroll_widget_layout)
            # 需要再套一层VBox，不然外部的tab显示有问题
            self.widget = widgets.VBox([sm_box])
        else:
            # 一行显示两个，2个为一组，组装sub_children_group序列,
            sub_children_group = self._sub_children(children, len(children) / self._sub_children_group_cnt)
            sub_children_box = [widgets.HBox(sub_children) for sub_children in sub_children_group]
            self.widget = widgets.VBox(sub_children_box)

    @contextmanager
    def _init_widget_list_action(self, callback_analyse, analyse_name, n_target):
        """上下文管理器，上文封装tip label，下文封装action按钮，widget_list做连接"""
        if not callable(callback_analyse):
            raise TabError('callback_analyse must callable!')
        widget_list = []
        tip_label = widgets.Label(self.map_tip_target_label(n_target=n_target), layout=self.label_layout)
        widget_list.append(tip_label)

        yield widget_list

        analyse_bt = widgets.Button(description=analyse_name, layout=widgets.Layout(width='98%'),
                                    button_style='info')
        analyse_bt.on_click(callback_analyse)
        widget_list.append(analyse_bt)

    def init_coint_corr_ui(self):
        """全市场协整相关分析ui"""
        with self._init_widget_list_action(self.coint_corr_market_analyse, u'全市场协整相关分析', 1) as widget_list:
            coint_similar_description = widgets.Textarea(
                value=u'全市场协整相关分析: \n'
                      u'综合利用相关和协整的特性返回查询的股票是否有统计套利的交易机会\n'
                      u'1. 通过相关性分析筛选出与查询股票最相关的前100支股票作为种子\n'
                      u'2. 从种子中通过计算协整程度来度量查询股票是否存在统计套利机会\n'
                      u'3. 可视化整个过程\n',
                disabled=False,
                layout=self.description_layout
            )
            widget_list.append(coint_similar_description)

            tip_label1 = widgets.Label(u'全市场相对相关分析不支持实时网络数据模式', layout=self.label_layout)
            tip_label2 = widgets.Label(u'非沙盒模式需先用\'数据下载界面操作\'进行下载', layout=self.label_layout)
            self.coint_corr_data_mode = widgets.RadioButtons(
                options={u'沙盒数据模式': True, u'本地数据模式': False},
                value=True,
                description=u'数据模式:',
                disabled=False
            )
            corr_market_box = widgets.VBox([tip_label1, tip_label2, self.coint_corr_data_mode])
            widget_list.append(corr_market_box)
        return widgets.VBox(widget_list,  # border='solid 1px',
                            layout=self.tool_layout)

    def init_relative_corr_ui(self):
        """全市场相对相关分析ui"""

        with self._init_widget_list_action(self.corr_relative_market_analyse, u'全市场相对相关分析', 2) as widget_list:
            relative_description = widgets.Textarea(
                value=u'全市场相对相关分析: \n'
                      u'度量的是两目标（a,b）相对整个市场的相关性评级，它不关心某一个股票具体相关性的数值的大小\n'
                      u'1. 计算a与市场中所有股票的相关性\n'
                      u'2. 将所有相关性进行rank排序\n'
                      u'3. 查询股票b在rank序列中的位置，此位置值即为结果\n',
                disabled=False,
                layout=self.description_layout
            )
            widget_list.append(relative_description)

            tip_label1 = widgets.Label(u'全市场相对相关分析不支持实时网络数据模式', layout=self.label_layout)
            tip_label2 = widgets.Label(u'非沙盒模式需先用\'数据下载界面操作\'进行下载', layout=self.label_layout)
            self.relative_corr_data_mode = widgets.RadioButtons(
                options={u'沙盒数据模式': True, u'本地数据模式': False},
                value=True,
                description=u'数据模式:',
                disabled=False
            )
            corr_market_box = widgets.VBox([tip_label1, tip_label2, self.relative_corr_data_mode])
            widget_list.append(corr_market_box)

        return widgets.VBox(widget_list,  # border='solid 1px',
                            layout=self.tool_layout)

    def init_market_corr_ui(self):
        """全市场相关分析ui"""

        with self._init_widget_list_action(self.corr_market_analyse, u'全市场相关分析', 1) as widget_list:
            self.corr_market = widgets.Dropdown(
                options={u'美股': EMarketTargetType.E_MARKET_TARGET_US.value,
                         u'A股': EMarketTargetType.E_MARKET_TARGET_CN.value,
                         u'港股': EMarketTargetType.E_MARKET_TARGET_HK.value,
                         u'国内期货': EMarketTargetType.E_MARKET_TARGET_FUTURES_CN.value,
                         u'国际期货': EMarketTargetType.E_MARKET_TARGET_FUTURES_GLOBAL.value,
                         u'数字货币': EMarketTargetType.E_MARKET_TARGET_TC.value},
                value=ABuEnv.g_market_target.value,
                description=u'分析市场:',
            )
            market_tip1 = widgets.Label(value=u'分析市场的选择可以和分析目标不在同一市场', layout=self.label_layout)
            market_tip2 = widgets.Label(value=u'如分析目标为美股股票，分析市场也可选A股', layout=self.label_layout)
            market_box = widgets.VBox([market_tip1, market_tip2, self.corr_market])
            widget_list.append(market_box)

            tip_label1 = widgets.Label(u'全市场相关分析不支持开放数据模式', layout=self.label_layout)
            tip_label2 = widgets.Label(u'非沙盒模式需先用\'数据下载界面操作\'进行下载', layout=self.label_layout)
            self.corr_market_data_mode = widgets.RadioButtons(
                options={u'沙盒数据模式': True, u'本地数据模式': False},
                value=True,
                description=u'数据模式:',
                disabled=False
            )
            corr_market_box = widgets.VBox([tip_label1, tip_label2, self.corr_market_data_mode])
            widget_list.append(corr_market_box)

            self.corr_market_mode = widgets.RadioButtons(
                options={u'皮尔逊相关系数计算': 'pears', u'斯皮尔曼相关系数计算': 'sperm', u'基于＋－符号相关系数': 'sign',
                         u'移动时间加权相关系数': 'rolling'},
                value='pears',
                description=u'相关模式:',
                disabled=False
            )
            widget_list.append(self.corr_market_mode)

        return widgets.VBox(widget_list,  # border='solid 1px',
                            layout=self.tool_layout)

    def init_corr_ui(self):
        """相关分析ui"""

        with self._init_widget_list_action(self.corr_analyse, u'相关分析', -1) as widget_list:
            self.corr_mode = widgets.RadioButtons(
                options={u'皮尔逊相关系数计算': 'pears', u'斯皮尔曼相关系数计算': 'sperm', u'基于＋－符号相关系数': 'sign',
                         u'移动时间加权相关系数': 'rolling'},
                value='pears',
                description=u'相关模式:',
                disabled=False
            )
            widget_list.append(self.corr_mode)

        return widgets.VBox(widget_list,  # border='solid 1px',
                            layout=self.tool_layout)

    def init_distances_ui(self):
        """距离分析ui"""

        with self._init_widget_list_action(self.distances_analyse, u'距离分析', -1) as widget_list:
            self.distances_mode = widgets.RadioButtons(
                options={u'曼哈顿距离(L1范数)': 0, u'欧式距离(L2范数)': 1, u'余弦距离': 2},
                value=0,
                description=u'距离模式:',
                disabled=False
            )
            widget_list.append(self.distances_mode)

            scale_end_label = widgets.Label(u'对结果矩阵进行标准化处理', layout=self.label_layout)
            self.scale_end = widgets.Checkbox(
                value=True,
                description=u'标准化',
                disabled=False
            )
            scale_end_box = widgets.VBox([scale_end_label, self.scale_end])
            widget_list.append(scale_end_box)

            similar_tip_label = widgets.Label(u'对结果矩阵进行转换相关性', layout=self.label_layout)
            self.to_similar = widgets.Checkbox(
                value=False,
                description=u'转换相关',
                disabled=False
            )
            to_similar_box = widgets.VBox([similar_tip_label, self.to_similar])
            widget_list.append(to_similar_box)

        return widgets.VBox(widget_list,  # border='solid 1px',
                            layout=self.tool_layout)

    @multi_fetch_symbol_df_analyse('p_change')
    def corr_analyse(self, cg_df, bt):
        """通过corr_matrix进行矩阵相关分析action"""
        display(corr_matrix(cg_df, similar_type=ECoreCorrType(self.corr_mode.value)))
        cg_df.cumsum().plot()

    @multi_fetch_symbol_df_analyse('p_change')
    def distances_analyse(self, cg_df, bt):
        """通过l1, l2, 余弦距离进行距离分析action"""
        if self.distances_mode.value == 0:
            distance = manhattan_distance_matrix(cg_df, scale_end=self.scale_end.value,
                                                 to_similar=self.to_similar.value)
        elif self.distances_mode.value == 1:
            distance = euclidean_distance_matrix(cg_df, scale_end=self.scale_end.value,
                                                 to_similar=self.to_similar.value)
        else:
            distance = cosine_distance_matrix(cg_df, scale_end=self.scale_end.value,
                                              to_similar=self.to_similar.value)
        display(distance)
        cg_df.cumsum().plot()

    def coint_corr_market_analyse(self, bt):
        """全市场协整相关分析"""
        with self.data_mode_recover(self.coint_corr_data_mode.value):
            ABuProgress.clear_output()
            if not all_market_env_check():
                return

            symbol = self._choice_symbol_single(default='usAAPL')

            mk = '{}_{}_sum_rank'.format(code_to_symbol(symbol).market.value, ABuEnv._g_enable_example_env_ipython)
            sum_rank = getattr(self, mk, None)
            _, sum_rank = coint_similar(symbol, sum_rank=sum_rank, show=True)
            # 缓存sum_rank
            setattr(self, mk, sum_rank)

    def corr_relative_market_analyse(self, bt):
        """全市场相对相关分析action"""
        with self.data_mode_recover(self.relative_corr_data_mode.value):
            ABuProgress.clear_output()
            if not all_market_env_check():
                return

            symbol1, symbol2 = self._choice_symbol_pair(default=['sh600036', 'sh601766'])
            mk = '{}_{}_sum_rank'.format(code_to_symbol(symbol1).market.value, ABuEnv._g_enable_example_env_ipython)
            sum_rank = getattr(self, mk, None)
            _, sum_rank = calc_similar(symbol1, symbol2, sum_rank=sum_rank, show=True)
            # 缓存sum_rank
            setattr(self, mk, sum_rank)

    def corr_market_analyse(self, bt):
        """全市场相关分析"""
        with self.data_mode_recover(self.corr_market_data_mode.value):
            ABuProgress.clear_output()

            if not all_market_env_check():
                return

            # 不做全局设置，做为后还需要恢复
            tmp_market = ABuEnv.g_market_target
            ABuEnv.g_market_target = EMarketTargetType(self.corr_market.value)

            symbol = self._choice_symbol_single(default='usAAPL')
            start, end, n_folds = self._start_end_n_fold()
            corr_type = ECoreCorrType(self.corr_market_mode.value)
            if start is not None and end is not None:
                find_similar_with_se(symbol, start=start, end=end, corr_type=corr_type)
            else:
                find_similar_with_folds(symbol, n_folds=n_folds, corr_type=corr_type)

            ABuEnv.g_market_target = tmp_market
