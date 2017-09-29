# -*- encoding:utf-8 -*-
"""量化技术分析工具图形可视化"""

from __future__ import print_function
from __future__ import absolute_import
from __future__ import division

import ipywidgets as widgets

from ..WidgetBu.ABuWGBase import WidgetBase
from ..WidgetBu.ABuWGToolBase import WidgetToolSet
from ..WidgetBu.ABuWGTLTool import WidgetTLTool
from ..WidgetBu.ABuWGDATool import WidgetDATool
from ..WidgetBu.ABuWGSMTool import WidgetSMTool

__author__ = '阿布'
__weixin__ = 'abu_quant'


class WidgetQuantTool(WidgetBase):
    """量化分析工具主界面"""

    def __init__(self):
        self.ts = WidgetToolSet()
        self.da = WidgetDATool(self.ts)
        self.tl = WidgetTLTool(self.ts)
        self.sm = WidgetSMTool(self.ts)
        sub_widget_tab = widgets.Tab()
        sub_widget_tab.children = [self.tl.widget, self.sm.widget, self.da.widget, self.ts.widget]
        for ind, name in enumerate([u'技术分析', u'相关分析', u'振幅分析', u'分析设置']):
            sub_widget_tab.set_title(ind, name)
        self.widget = widgets.VBox([sub_widget_tab])
