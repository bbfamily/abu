# -*- encoding:utf-8 -*-
"""选股因子参数以及选择图形可视化"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import ipywidgets as widgets
import numpy as np

from ..PickStockBu.ABuPickRegressAngMinMax import AbuPickRegressAngMinMax
from ..PickStockBu.ABuPickStockDemo import AbuPickStockShiftDistance, AbuPickStockNTop
from ..PickStockBu.ABuPickStockPriceMinMax import AbuPickStockPriceMinMax
from ..WidgetBu.ABuWGPSBase import WidgetPickStockBase

__author__ = '阿布'
__weixin__ = 'abu_quant'


class PSPriceWidget(WidgetPickStockBase):
    """对应AbuPickStockPriceMinMax策略widget"""

    def _init_widget(self):
        """构建AbuPickStockPriceMinMax策略参数界面"""

        self.description = widgets.Textarea(
            value=u'价格选股因子策略：\n'
                  u'根据交易目标的一段时间内收盘价格的最大，最小值进行选股，选中规则：\n'
                  u'1. 交易目标最小价格 > 最小价格阀值\n'
                  u'2. 交易目标最大价格 < 最大价格阀值\n',
            description=u'价格选股',
            disabled=False,
            layout=self.description_layout
        )

        self.price_min_label = widgets.Label(u'设定选股价格最小阀值，默认15', layout=self.label_layout)
        self.price_min_float = widgets.FloatText(
            value=15,
            description=u'最小:',
            disabled=False
        )
        self.price_min_ck = widgets.Checkbox(
            value=True,
            description=u'使用最小阀值',
            disabled=False
        )

        def price_min_ck_change(change):
            self.price_min_float.disabled = not change['new']

        self.price_min_ck.observe(price_min_ck_change, names='value')
        self.price_min_box = widgets.VBox([self.price_min_label, self.price_min_ck, self.price_min_float])

        self.price_max_label = widgets.Label(u'设定选股价格最大阀值，默认50', layout=self.label_layout)
        self.price_max_float = widgets.FloatText(
            value=50,
            description=u'最大:',
            disabled=False
        )
        self.price_max_ck = widgets.Checkbox(
            value=True,
            description=u'使用最大阀值',
            disabled=False
        )

        def price_max_ck_change(change):
            self.price_max_float.disabled = not change['new']

        self.price_max_ck.observe(price_max_ck_change, names='value')
        self.price_max_box = widgets.VBox([self.price_max_label, self.price_max_ck, self.price_max_float])
        self.widget = widgets.VBox([self.description, self.price_min_box, self.price_max_box,
                                    self.xd_box, self.reversed_box, self.add_box],
                                   # border='solid 1px',
                                   layout=self.widget_layout)

    def make_pick_stock_unique(self):
        """对应按钮添加AbuPickStockPriceMinMax策略，构建策略字典对象factor_dict以及唯一策略描述字符串factor_desc_key"""
        price_min = self.price_min_float.value if self.price_min_ck.value else -np.inf
        price_max = self.price_max_float.value if self.price_max_ck.value else np.inf

        factor_dict = {'class': AbuPickStockPriceMinMax,
                       'xd': self.xd.value,
                       'reversed': self.reversed.value,
                       'threshold_price_min': price_min,
                       'threshold_price_max': price_max}

        factor_desc_key = u'价格选股最大:{}最小:{},周期:{},反转:{}'.format(
            price_max, price_min, self.xd.value, self.reversed.value)
        return factor_dict, factor_desc_key

    def delegate_class(self):
        """子类因子所委托的具体因子类AbuPickStockPriceMinMax"""
        return AbuPickStockPriceMinMax


class PSRegressAngWidget(WidgetPickStockBase):
    """对应AbuPickRegressAngMinMax策略widget"""

    def _init_widget(self):
        """构建AbuPickRegressAngMinMax策略参数界面"""

        self.description = widgets.Textarea(
            value=u'拟合角度选股因子策略：\n'
                  u'将交易目标前期走势进行线性拟合计算一个角度，选中规则：\n'
                  u'1. 交易目标前期走势拟合角度 > 最小拟合角度\n'
                  u'2. 交易目标前期走势拟合角度 < 最大拟合角度\n',
            description=u'角度选股',
            disabled=False,
            layout=self.description_layout
        )

        self.ang_min_label = widgets.Label(u'设定选股角度最小阀值，默认-5', layout=self.label_layout)
        self.ang_min_float = widgets.IntText(
            value=-5,
            description=u'最小:',
            disabled=False
        )
        self.ang_min_ck = widgets.Checkbox(
            value=True,
            description=u'使用最小阀值',
            disabled=False
        )

        def ang_min_ck_change(change):
            self.ang_min_float.disabled = not change['new']

        self.ang_min_ck.observe(ang_min_ck_change, names='value')
        self.ang_min_box = widgets.VBox([self.ang_min_label, self.ang_min_ck, self.ang_min_float])

        self.ang_max_label = widgets.Label(u'设定选股角度最大阀值，默认5', layout=self.label_layout)
        self.ang_max_float = widgets.IntText(
            value=5,
            description=u'最大:',
            disabled=False
        )
        self.ang_max_ck = widgets.Checkbox(
            value=True,
            description=u'使用最大阀值',
            disabled=False
        )

        def ang_max_ck_change(change):
            self.ang_max_float.disabled = not change['new']

        self.ang_max_ck.observe(ang_max_ck_change, names='value')
        self.ang_max_box = widgets.VBox([self.ang_max_label, self.ang_max_ck, self.ang_max_float])

        self.widget = widgets.VBox([self.description, self.ang_min_box, self.ang_max_box,
                                    self.xd_box, self.reversed_box, self.add_box],
                                   # border='solid 1px',
                                   layout=self.widget_layout)

    def make_pick_stock_unique(self):
        """对应按钮添加AbuPickRegressAngMinMax策略，构建策略字典对象factor_dict以及唯一策略描述字符串factor_desc_key"""
        ang_min = self.ang_min_float.value if self.ang_min_ck.value else -np.inf
        ang_max = self.ang_max_float.value if self.ang_max_ck.value else np.inf
        factor_dict = {'class': AbuPickRegressAngMinMax,
                       'xd': self.xd.value,
                       'reversed': self.reversed.value,
                       'threshold_ang_min': ang_min,
                       'threshold_ang_max': ang_max}

        factor_desc_key = u'角度选股最大:{}最小:{},周期:{},反转:{}'.format(
            ang_max, ang_min, self.xd.value, self.reversed.value)
        return factor_dict, factor_desc_key

    def delegate_class(self):
        """子类因子所委托的具体因子类AbuPickRegressAngMinMax"""
        return AbuPickRegressAngMinMax


class PSShiftDistanceWidget(WidgetPickStockBase):
    """对应AbuPickStockShiftDistance策略widget"""

    def _init_widget(self):
        """构建AbuPickStockShiftDistance策略参数界面"""

        self.description = widgets.Textarea(
            value=u'位移路程比选股因子策略：\n'
                  u'将交易目标走势每月计算价格位移路程比，根据比值进行选股，选取波动程度不能太大，也不太小的目标：\n'
                  u'1. 定义位移路程比大于参数阀值的月份为大波动月\n'
                  u'2. 一年中大波动月数量 < 最大波动月个数\n'
                  u'3. 一年中大波动月数量 > 最小波动月个数\n',
            description=u'位移路程',
            disabled=False,
            layout=self.description_layout
        )

        threshold_sd_label1 = widgets.Label(u'设大波动位移路程比阀值，期货市场建议2.0及以上', layout=self.label_layout)
        threshold_sd_label2 = widgets.Label(u'设大波动位移路程比阀值，股票市场建议3.0及以上', layout=self.label_layout)
        self.threshold_sd = widgets.FloatSlider(
            value=2.0,
            min=1.0,
            max=6.0,
            step=0.1,
            description=u'阀值',
            disabled=False,
            orientation='horizontal',
            readout=True,
            readout_format='.1f',
        )
        self.threshold_sd_box = widgets.VBox([threshold_sd_label1, threshold_sd_label2, self.threshold_sd])

        max_cnt_label = widgets.Label(u'选取大波动月数量 < 下面设定的最大波动月个数', layout=self.label_layout)
        min_cnt_label = widgets.Label(u'选取大波动月数量 > 下面设定的最小波动月个数', layout=self.label_layout)
        self.min_max_range = widgets.IntRangeSlider(
            value=[1, 4],
            min=0,
            max=10,
            step=1,
            description=u'范围',
            disabled=False,
            continuous_update=False,
            orientation='horizontal',
            readout=True,
            readout_format='d',
        )
        self.min_max_box = widgets.VBox([max_cnt_label, min_cnt_label, self.min_max_range])
        # 这个策略不要可自定义xd，限定选股周期为1年
        self.widget = widgets.VBox([self.description, self.threshold_sd_box,
                                    self.min_max_range, self.reversed_box, self.add_box],
                                   # border='solid 1px',
                                   layout=self.widget_layout)

    def make_pick_stock_unique(self):
        """对应按钮添加AbuPickStockShiftDistance策略，构建策略字典对象factor_dict以及唯一策略描述字符串factor_desc_key"""

        factor_dict = {'class': AbuPickStockShiftDistance,
                       'threshold_sd': self.threshold_sd.value,
                       'reversed': self.reversed.value,
                       'threshold_max_cnt': self.min_max_range.value[1],
                       'threshold_min_cnt': self.min_max_range.value[0]}

        factor_desc_key = u'位移路程选股大波动:{}最大:{}最小:{},反转:{}'.format(
            self.threshold_sd.value, self.min_max_range.value[1], self.min_max_range.value[0], self.reversed.value)
        return factor_dict, factor_desc_key

    def delegate_class(self):
        """子类因子所委托的具体因子类AbuPickStockShiftDistance"""
        return AbuPickStockShiftDistance


class PSNTopWidget(WidgetPickStockBase):
    """对应AbuPickStockNTop策略widget"""

    def _init_widget(self):
        """构建AbuPickStockNTop策略参数界面"""
        self.description = widgets.Textarea(
            value=u'涨跌幅top N选股因子策略：\n'
                  u'选股周期上对多只股票涨跌幅进行排序，选取top n个股票做为交易目标：\n'
                  u'(只对在股池中选定的symbol序列生效，对全市场回测暂时不生效)\n',
            description=u'top N涨跌',
            disabled=False,
            layout=self.description_layout
        )

        n_top_label = widgets.Label(u'设定选取top个交易目标数量，默认3', layout=self.label_layout)
        self.n_top = widgets.IntText(
            value=3,
            description=u'TOP N',
            disabled=False
        )
        self.n_top_box = widgets.VBox([n_top_label, self.n_top])

        direction_top_label1 = widgets.Label(u'direction_top参数的意义为选取方向：', layout=self.label_layout)
        direction_top_label2 = widgets.Label(u'默认值为正：即选取涨幅最高的n_top个股票', layout=self.label_layout)
        direction_top_label3 = widgets.Label(u'可设置为负：即选取跌幅最高的n_top个股票', layout=self.label_layout)
        self.direction_top = widgets.Dropdown(
            options={u'正(涨幅)': 1, u'负(跌幅)': -1},
            value=1,
            description=u'选取方向:',
        )
        self.direction_top_box = widgets.VBox([direction_top_label1, direction_top_label2, direction_top_label3,
                                               self.direction_top])

        self.widget = widgets.VBox([self.description, self.n_top_box, self.direction_top_box,
                                    self.xd_box, self.reversed_box, self.add_box],
                                   # border='solid 1px',
                                   layout=self.widget_layout)

    def seed_choice_symbol_key(self):
        """返回更新股池中的种子symbol到具体策略中需要的策略关键字定义"""
        return 'symbol_pool'

    def make_pick_stock_unique(self):
        """对应按钮添加AbuPickStockNTop策略，构建策略字典对象factor_dict以及唯一策略描述字符串factor_desc_key"""
        factor_dict = {'class': AbuPickStockNTop,
                       'n_top': self.n_top.value,
                       'direction_top': self.direction_top.value,
                       'reversed': self.reversed.value,
                       'xd': self.xd.value}

        factor_desc_key = u'涨跌幅选股n_top:{},方向:{},xd:{},反转:{}'.format(
            self.n_top.value, self.direction_top.value, self.xd.value, self.reversed.value)
        return factor_dict, factor_desc_key

    def delegate_class(self):
        """子类因子所委托的具体因子类AbuPickStockNTop"""
        return AbuPickStockNTop
