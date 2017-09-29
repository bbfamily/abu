# -*- encoding:utf-8 -*-
"""量化振幅分析工具图形可视化"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import ipywidgets as widgets
from IPython.display import display

from ..UtilBu import ABuKLUtil
from ..WidgetBu.ABuWGToolBase import WidgetToolBase, multi_fetch_symbol_analyse

__author__ = '阿布'
__weixin__ = 'abu_quant'


# noinspection PyUnusedLocal
class WidgetDATool(WidgetToolBase):
    """振幅分析界面"""

    def __init__(self, tool_set):
        """初始化数据分析界面"""
        super(WidgetDATool, self).__init__(tool_set)

        da_list = []
        tip_label1 = widgets.Label(u'分析目标需要在\'分析设置\'中选择', layout=self.label_layout)
        tip_label2 = widgets.Label(u'需要设置多个分析目标进行对比', layout=self.label_layout)
        da_list.append(tip_label1)
        da_list.append(tip_label2)

        date_week_wave_bt = widgets.Button(description=u'交易日震幅对比分析', layout=widgets.Layout(width='98%'),
                                           button_style='info')
        date_week_wave_bt.on_click(self.date_week_wave)
        da_list.append(date_week_wave_bt)

        p_change_stats_bt = widgets.Button(description=u'交易日涨跌对比分析', layout=widgets.Layout(width='98%'),
                                           button_style='info')
        p_change_stats_bt.on_click(self.p_change_stats)
        da_list.append(p_change_stats_bt)

        wave_change_rate_bt = widgets.Button(description=u'振幅统计套利条件分析', layout=widgets.Layout(width='98%'),
                                             button_style='info')
        wave_change_rate_bt.on_click(self.wave_change_rate)
        da_list.append(wave_change_rate_bt)

        date_week_win_bt = widgets.Button(description=u'交易日涨跌概率分析', layout=widgets.Layout(width='98%'),
                                          button_style='info')
        date_week_win_bt.on_click(self.date_week_win)
        da_list.append(date_week_win_bt)

        bcut_change_vc_bt = widgets.Button(description=u'交易日涨跌区间分析(预定区间)', layout=widgets.Layout(width='98%'),
                                           button_style='info')
        bcut_change_vc_bt.on_click(self.bcut_change_vc)
        da_list.append(bcut_change_vc_bt)

        qcut_change_vc_bt = widgets.Button(description=u'交易日涨跌区间分析(不定区间)', layout=widgets.Layout(width='98%'),
                                           button_style='info')
        qcut_change_vc_bt.on_click(self.qcut_change_vc)
        da_list.append(qcut_change_vc_bt)

        self.widget = widgets.VBox(da_list, layout=widgets.Layout(width='58%'))

    @multi_fetch_symbol_analyse
    def qcut_change_vc(self, kl_dict, bt):
        """交易日涨跌区间分析(不定区间)action"""
        display(ABuKLUtil.qcut_change_vc(kl_dict))

    @multi_fetch_symbol_analyse
    def bcut_change_vc(self, kl_dict, bt):
        """交易日涨跌区间分析(预定区间)action"""
        display(ABuKLUtil.bcut_change_vc(kl_dict))

    @multi_fetch_symbol_analyse
    def date_week_win(self, kl_dict, bt):
        """交易日涨跌概率分析action"""
        display(ABuKLUtil.date_week_win(kl_dict))

    @multi_fetch_symbol_analyse
    def wave_change_rate(self, kl_dict, bt):
        """振幅统计套利条件分析action"""
        ABuKLUtil.wave_change_rate(kl_dict)

    @multi_fetch_symbol_analyse
    def date_week_wave(self, kl_dict, bt):
        """交易日震幅对比分析action"""
        display(ABuKLUtil.date_week_wave(kl_dict))

    @multi_fetch_symbol_analyse
    def p_change_stats(self, kl_dict, bt):
        """交易日涨跌对比分析action"""
        ABuKLUtil.p_change_stats(kl_dict)
