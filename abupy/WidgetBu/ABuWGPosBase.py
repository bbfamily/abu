# -*- encoding:utf-8 -*-
"""仓位资金管理参数以及选择图形可视化"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from abc import abstractmethod

import ipywidgets as widgets

from ..WidgetBu.ABuWGBase import WidgetFactorBase, WidgetFactorManagerBase, accordion_shut
from ..WidgetBu.ABuWGBFBase import BFSubscriberMixin

__author__ = '阿布'
__weixin__ = 'abu_quant'


class PosWGManager(WidgetFactorManagerBase):
    """仓位资金管理组织类"""

    def __init__(self):
        super(PosWGManager, self).__init__()

        position_label = widgets.Label(u'无选定时默认资管为：atr资金仓位管理策略',
                                       layout=widgets.Layout(width='300px'))

        self.widget = widgets.VBox([self.factor_box, position_label, self.selected_factors])

    def _init_widget(self):
        """构建内置的仓位资金管理可视化组件，构造出self.factor_box"""

        from ..WidgetBu.ABuWGPosition import AtrPosWidget, KellyPosWidget, PtPosition
        self.pos_array = []
        self.pos_array.append(AtrPosWidget(self))
        self.pos_array.append(KellyPosWidget(self))
        self.pos_array.append(PtPosition(self))

        #  ps() call用widget组list
        children = [pos() for pos in self.pos_array]
        if self.scroll_factor_box:
            self.factor_box = widgets.Box(children=children,
                                          layout=self.factor_layout)
        else:
            # 一行显示两个，n个为一组，组装sub_children_group序列,
            sub_children_group = self._sub_children(children, len(children) / self._sub_children_group_cnt)
            sub_children_box = [widgets.HBox(sub_children) for sub_children in sub_children_group]
            self.factor_box = widgets.VBox(sub_children_box)
        # 买入因子是特殊的存在，都需要买入因子的全局数据
        self.buy_factor_manger = None

    def register(self, buy_factor_manger):
        """选股manager内部因子注册接收买入因子添加的改变"""
        self.buy_factor_manger = buy_factor_manger
        for ps in self.pos_array:
            self.buy_factor_manger.register_subscriber(ps)


class WidgetPositionBase(WidgetFactorBase, BFSubscriberMixin):
    """仓位资金管理可视化基础类"""

    def __init__(self, wg_manager):
        super(WidgetPositionBase, self).__init__(wg_manager)
        self.add = widgets.Button(description=u'选定为全局资金管理策略', layout=widgets.Layout(width='98%'),
                                  button_style='info')
        # 选定全局资金管理略指令按钮
        self.add.on_click(self.add_position)
        # 运行混入的BFSubscriberMixin中ui初始化
        self.subscriber_ui([u'点击\'已添加的买入策略\'框中的买入策略', u'资金管理做为买入策略的资金管理策略'])
        # 买入策略框点击行为：将本卖出策略加到对应的买入策略做为附属
        self.buy_factors.observe(self.add_position_to_buy_factor, names='value')
        self.accordion.set_title(0, u'添加为指定买入因子的资金管理策略')
        accordion_shut(self.accordion)
        self.add_box = widgets.VBox([self.add, self.accordion])

        # 具体子策略构建
        self._init_widget()

    @abstractmethod
    def _init_widget(self):
        """子类因子界面设置初始化"""
        pass

    @abstractmethod
    def make_position_unique(self):
        """
            子类因子构建唯一描述以及因子字典
            返回值两个：
            1. 因子构建字典对象
            2. 因子唯一描述
        """
        pass

    # noinspection PyUnusedLocal
    def add_position(self, bt):
        """对应按钮选定为全局仓位资金管理策略"""
        # 构建策略字典对象factor_dict以及唯一策略描述字符串factor_desc_key
        factor_dict, factor_desc_key = self.make_position_unique()
        self.wg_manager.add_factor(factor_dict, factor_desc_key, only_one=True)

    def add_position_to_buy_factor(self, select):
        """对应按钮添加策略到指定买入策略中"""
        self.add_to_buy_factor(select, self.make_position_unique, 'position', only_one=True)
