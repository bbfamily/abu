# -*- encoding:utf-8 -*-
"""策略验证工具图形可视化"""

from __future__ import print_function
from __future__ import absolute_import
from __future__ import division

import ipywidgets as widgets

from ..WidgetBu.ABuWGBase import WidgetBase
from ..WidgetBu.ABuWGGridSearch import WidgetGridSearch
from ..WidgetBu.ABuWGCrossVal import WidgetCrossVal


__author__ = '阿布'
__weixin__ = 'abu_quant'


class WidgetVerifyTool(WidgetBase):
    """策略验证工具主界面类实现"""

    def __init__(self):
        self.grid_search = WidgetGridSearch()
        self.cross_val = WidgetCrossVal()
        sub_widget_tab = widgets.Tab()
        sub_widget_tab.children = [self.grid_search.widget, self.cross_val.widget]
        for ind, name in enumerate([u'最优参数', u'交叉验证']):
            sub_widget_tab.set_title(ind, name)
        self.widget = widgets.VBox([sub_widget_tab])
