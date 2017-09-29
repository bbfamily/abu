# -*- encoding:utf-8 -*-
"""卖出因子参数以及选择图形可视化"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from abc import abstractmethod

import ipywidgets as widgets

from ..WidgetBu.ABuWGBase import WidgetFactorBase, WidgetFactorManagerBase
from ..WidgetBu.ABuWGBFBase import BFSubscriberMixin

__author__ = '阿布'
__weixin__ = 'abu_quant'


class SellFactorWGManager(WidgetFactorManagerBase):
    """卖出策略组织类"""

    def _init_widget(self):
        """构建内置的卖出策略可视化组件，构造出self.factor_box"""

        from ..WidgetBu.ABuWGSellFactor import SellXDWidget, SellAtrNStopWidget, SellCloseAtrNWidget
        from ..WidgetBu.ABuWGSellFactor import SellPreAtrNWidget, SellDMWidget, SellNDWidget

        self.sf_array = []
        self.sf_array.append(SellAtrNStopWidget(self))
        self.sf_array.append(SellCloseAtrNWidget(self))
        self.sf_array.append(SellPreAtrNWidget(self))
        self.sf_array.append(SellXDWidget(self))
        self.sf_array.append(SellDMWidget(self))
        self.sf_array.append(SellNDWidget(self))

        # sf() call用widget组list
        children = [sf() for sf in self.sf_array]
        if self.scroll_factor_box:
            self.factor_box = widgets.Box(children=children,
                                          layout=self.factor_layout)
        else:
            # 一行显示两个，2个为一组，组装sub_children_group序列,
            sub_children_group = self._sub_children(children, len(children) / 2)
            sub_children_box = [widgets.HBox(sub_children) for sub_children in sub_children_group]
            self.factor_box = widgets.VBox(sub_children_box)

        # 买入因子是特殊的存在，都需要买入因子的全局数据
        self.buy_factor_manger = None

    def register(self, buy_factor_manger):
        """卖出manager内部因子注册接收买入因子添加的改变"""
        self.buy_factor_manger = buy_factor_manger
        for sf in self.sf_array:
            self.buy_factor_manger.register_subscriber(sf)


class WidgetFactorSellBase(WidgetFactorBase, BFSubscriberMixin):
    """卖出策略可视化基础类"""

    def __init__(self, wg_manager):
        super(WidgetFactorSellBase, self).__init__(wg_manager)
        self.add = widgets.Button(description=u'添加为全局卖出策略', layout=widgets.Layout(width='98%'),
                                  button_style='info')
        # 添加全局卖出策略指令按钮
        self.add.on_click(self.add_sell_factor)
        # 运行混入的BFSubscriberMixin中ui初始化
        self.subscriber_ui([u'点击\'已添加的买入策略\'框中的买入策略', u'将卖出策略做为买入策略的附属卖出策略'])
        # 买入策略框点击行为：将本卖出策略加到对应的买入策略做为附属
        self.buy_factors.observe(self.add_sell_factor_to_buy_factor, names='value')
        self.accordion.set_title(0, u'添加为指定买入因子的卖出策略')
        self.accordion.selected_index = -1
        self.add_box = widgets.VBox([self.add, self.accordion])

        self._init_widget()

    @abstractmethod
    def _init_widget(self):
        """子类因子界面设置初始化"""
        pass

    @abstractmethod
    def make_sell_factor_unique(self):
        """
            子类因子构建唯一描述以及因子字典
            返回值两个：
            1. 因子构建字典对象
            2. 因子唯一描述
        """
        pass

    # noinspection PyUnusedLocal
    def add_sell_factor(self, bt):
        """对应按钮添加全局策略"""
        # 构建策略字典对象factor_dict以及唯一策略描述字符串factor_desc_key
        factor_dict, factor_desc_key = self.make_sell_factor_unique()
        self.wg_manager.add_factor(factor_dict, factor_desc_key)

    def add_sell_factor_to_buy_factor(self, select):
        """对应按钮添加策略到指定买入策略中"""
        self.add_to_buy_factor(select, self.make_sell_factor_unique, 'sell_factors')

