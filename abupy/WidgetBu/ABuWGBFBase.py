# -*- encoding:utf-8 -*-
"""买入因子参数以及选择图形可视化"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from abc import abstractmethod

import ipywidgets as widgets

from ..WidgetBu.ABuWGBase import WidgetFactorBase, WidgetFactorManagerBase

__author__ = '阿布'
__weixin__ = 'abu_quant'


# noinspection PyUnresolvedReferences
class BFSubscriberMixin(object):
    """
        混入类：订阅买入策略更新通知，以及构建添加基础附属于买入策略的策略
        如：依附于买入策略的卖出策略，依赖于买入策略的选股策略
    """

    def subscriber_ui(self, labels):
        """
        构建订阅的已添加的买入策略ui初始化
        :param labels: list序列内部对象str用来描述解释
        """
        # 添加针对指定买入策略的卖出策略
        self.accordion = widgets.Accordion()
        buy_factors_child = []
        for label in labels:
            buy_factors_child.append(widgets.Label(label,
                                                   layout=widgets.Layout(width='300px', align_items='stretch')))
        self.buy_factors = widgets.SelectMultiple(
            options=[],
            description=u'已添加的买入策略:',
            disabled=False,
            layout=widgets.Layout(width='100%', align_items='stretch')
        )
        buy_factors_child.append(self.buy_factors)
        buy_factors_box = widgets.VBox(buy_factors_child)
        self.accordion.children = [buy_factors_box]

    def notify_subscriber(self):
        """已添加的买入策略框接收买入tab已添加的买入策略信息改变ui同步"""
        self.buy_factors.options = self.wg_manager.buy_factor_manger.selected_factors.options

    def add_to_buy_factor(self, select, factor_unique_callable, factor_key):
        """对应按钮添加策略到指定买入策略中的基础方法，具体策略中外层需要再套一层"""
        if not callable(factor_unique_callable):
            raise TypeError('factor_unique_callable must callable!')

        if self.wg_manager.buy_factor_manger is not None and len(list(select['new'])) > 0:
            #  由于是多选框，简单处理，只能最后一个
            buy_factor_desc_key = list(select['new'])[-1]

            # 构建策略字典对象factor_dict以及唯一策略描述字符串factor_desc_key
            factor_dict, factor_desc_key = factor_unique_callable()

            desc_key_list = buy_factor_desc_key.split('+')
            if factor_desc_key in desc_key_list:
                # 已添加过相同描述的策略在买入因子中，返回不加了
                return

            # 从买入manager字典中出栈点击的这个描述买入字典对象
            buy_factor_dict = self.wg_manager.buy_factor_manger.factor_dict.pop(buy_factor_desc_key)
            # 买入字典对象中如果有独立因子策略序列，弹出
            factors = buy_factor_dict.pop(factor_key, [])

            # 买入独有因子序列中加入factor_dict
            factors.append(factor_dict)
            # 将因子序列放回到买入字典对象中
            buy_factor_dict[factor_key] = factors
            # 买入策略因子描述＋独有因子描述＝唯一策略描述
            combine_factor_desc_key = u'{}+{}'.format(buy_factor_desc_key, factor_desc_key)
            # 买入策略因子中放回组合好的买入策略
            self.wg_manager.buy_factor_manger.factor_dict[combine_factor_desc_key] = buy_factor_dict
            # 通知ui更新
            self.wg_manager.buy_factor_manger.refresh_factor()


class BuyFactorWGManager(WidgetFactorManagerBase):
    """买入策略组织类"""

    def _init_widget(self):
        """构建内置的买入策略可视化组件，构造出self.factor_box"""

        from ..WidgetBu.ABuWGBuyFactor import BuyDMWidget, BuyXDWidget, BuyWDWidget, BuySDWidget, BuyWMWidget
        self.bf_array = []
        self.bf_array.append(BuyDMWidget(self))
        self.bf_array.append(BuyXDWidget(self))
        self.bf_array.append(BuyWDWidget(self))
        self.bf_array.append(BuySDWidget(self))
        self.bf_array.append(BuyWMWidget(self))

        # bf() call用widget组list
        children = [bf() for bf in self.bf_array]

        if self.scroll_factor_box:
            self.factor_box = widgets.Box(children=children,
                                          layout=self.factor_layout)
        else:
            # 一行显示两个，2个为一组，组装sub_children_group序列,
            sub_children_group = self._sub_children(children, len(children) / 2)
            sub_children_box = [widgets.HBox(sub_children) for sub_children in sub_children_group]
            self.factor_box = widgets.VBox(sub_children_box)


class WidgetFactorBuyBase(WidgetFactorBase):
    """买入策略可视化基础类"""

    def __init__(self, wg_manager):
        super(WidgetFactorBuyBase, self).__init__(wg_manager)
        self.add = widgets.Button(description=u'添加为全局买入策略', layout=widgets.Layout(width='98%'),
                                  button_style='info')
        self.add.on_click(self.add_buy_factor)
        self._init_widget()

    @abstractmethod
    def _init_widget(self):
        """子类因子界面设置初始化"""
        pass

    @abstractmethod
    def make_buy_factor_unique(self):
        """
            子类因子构建唯一描述以及因子字典
            返回值两个：
            1. 因子构建字典对象
            2. 因子唯一描述
        """
        pass

    # noinspection PyUnusedLocal
    def add_buy_factor(self, bt):
        """对应按钮添加策略，构建策略字典对象factor_dict以及唯一策略描述字符串factor_desc_key"""
        factor_dict, factor_desc_key = self.make_buy_factor_unique()
        self.wg_manager.add_factor(factor_dict, factor_desc_key)
