# -*- encoding:utf-8 -*-
"""选股因子参数以及选择图形可视化"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from abc import abstractmethod

import ipywidgets as widgets

from ..WidgetBu.ABuWGBase import WidgetFactorBase, WidgetFactorManagerBase, accordion_shut
from ..WidgetBu.ABuWGBFBase import BFSubscriberMixin

__author__ = '阿布'
__weixin__ = 'abu_quant'


class PickStockWGManager(WidgetFactorManagerBase):
    """选股策略组织类"""

    def _init_widget(self):
        """构建内置的卖出策略可视化组件，构造出self.factor_box"""

        from ..WidgetBu.ABuWGPickStock import PSPriceWidget, PSRegressAngWidget
        from ..WidgetBu.ABuWGPickStock import PSShiftDistanceWidget, PSNTopWidget
        self.ps_array = []
        self.ps_array.append(PSPriceWidget(self))
        self.ps_array.append(PSRegressAngWidget(self))
        self.ps_array.append(PSShiftDistanceWidget(self))
        self.ps_array.append(PSNTopWidget(self))

        #  ps() call用widget组list
        children = [ps() for ps in self.ps_array]
        if self.scroll_factor_box:
            self.factor_box = widgets.Box(children=children,
                                          layout=self.factor_layout)
        else:
            # 一行显示两个，3个为一组，组装sub_children_group序列,
            sub_children_group = self._sub_children(children, len(children) / self._sub_children_group_cnt)
            sub_children_box = [widgets.HBox(sub_children) for sub_children in sub_children_group]
            self.factor_box = widgets.VBox(sub_children_box)
        # 买入因子是特殊的存在，都需要买入因子的全局数据
        self.buy_factor_manger = None

    def seed_choice_symbol_update(self, seed_choice_symbol):
        """更新股池中的种子symbol到需要种子symbol的策略中"""
        if seed_choice_symbol is None or len(seed_choice_symbol) == 0:
            return

        for ps in self.ps_array:
            if hasattr(ps, 'seed_choice_symbol_key'):
                # 策略中要更新种子symbol的需要定义函数, 定义一个key进行更新
                cs_key = ps.seed_choice_symbol_key()

                for factor in self.factor_dict.values():
                    # 从全局选股策略中寻找这个策略
                    if factor['class'] == ps.delegate_class():
                        # 如果找到将种子symbol进行更新
                        factor.update({cs_key: seed_choice_symbol})

                for factor in self.buy_factor_manger.factor_dict.values():
                    # 从附属于买入因子的选股策略中进行搜索更新
                    if 'stock_pickers' in factor:
                        for ps_factor in factor['stock_pickers']:
                            if ps_factor['class'] == ps.delegate_class():
                                ps_factor.update({cs_key: seed_choice_symbol})
                                # from ..UtilBu.ABuOsUtil import show_msg
                                # show_msg(cs_key, str(list(factor.keys())))

    def register(self, buy_factor_manger):
        """选股manager内部因子注册接收买入因子添加的改变"""
        self.buy_factor_manger = buy_factor_manger
        for ps in self.ps_array:
            self.buy_factor_manger.register_subscriber(ps)


class WidgetPickStockBase(WidgetFactorBase, BFSubscriberMixin):
    """选股策略可视化基础类"""

    def _pick_stock_base_ui(self):
        """选股策略中通用ui: xd, reversed初始构建"""
        xd_tip = widgets.Label(u'设置选股策略生效周期，默认252天',
                               layout=widgets.Layout(width='300px', align_items='stretch'))
        self.xd = widgets.IntSlider(
            value=252,
            min=1,
            max=252,
            step=1,
            description=u'周期',
            disabled=False,
            orientation='horizontal',
            readout=True,
            readout_format='d'
        )
        self.xd_box = widgets.VBox([xd_tip, self.xd])

        reversed_tip = widgets.Label(u'反转选股结果，默认不反转',
                                     layout=widgets.Layout(width='300px', align_items='stretch'))
        self.reversed = widgets.Checkbox(
            value=False,
            description=u'反转结果',
            disabled=False,
        )
        self.reversed_box = widgets.VBox([reversed_tip, self.reversed])

    def __init__(self, wg_manager):
        super(WidgetPickStockBase, self).__init__(wg_manager)
        self.add = widgets.Button(description=u'添加为全局选股策略', layout=widgets.Layout(width='98%'),
                                  button_style='info')
        # 添加全局选股策略指令按钮
        self.add.on_click(self.add_pick_stock)
        # 运行混入的BFSubscriberMixin中ui初始化
        self.subscriber_ui([u'点击\'已添加的买入策略\'框中的买入策略', u'将选股策略做为买入策略的附属选股策略'])
        # 买入策略框点击行为：将本卖出策略加到对应的买入策略做为附属
        self.buy_factors.observe(self.add_pick_stock_to_buy_factor, names='value')
        self.accordion.set_title(0, u'添加为指定买入因子的选股策略')
        accordion_shut(self.accordion)
        self.add_box = widgets.VBox([self.add, self.accordion])
        # 构建选股策略独有基础通用ui
        self._pick_stock_base_ui()
        # 具体子策略构建
        self._init_widget()

    @abstractmethod
    def _init_widget(self):
        """子类因子界面设置初始化"""
        pass

    @abstractmethod
    def make_pick_stock_unique(self):
        """
            子类因子构建唯一描述以及因子字典
            返回值两个：
            1. 因子构建字典对象
            2. 因子唯一描述
        """
        pass

    # noinspection PyUnusedLocal
    def add_pick_stock(self, bt):
        """对应按钮添加全局策略"""
        # 构建策略字典对象factor_dict以及唯一策略描述字符串factor_desc_key
        factor_dict, factor_desc_key = self.make_pick_stock_unique()
        self.wg_manager.add_factor(factor_dict, factor_desc_key)

    def add_pick_stock_to_buy_factor(self, select):
        """对应按钮添加策略到指定买入策略中"""
        self.add_to_buy_factor(select, self.make_pick_stock_unique, 'stock_pickers')
