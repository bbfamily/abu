# -*- encoding:utf-8 -*-
"""卖出因子参数以及选择图形可视化"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import ipywidgets as widgets

from ..FactorSellBu.ABuFactorAtrNStop import AbuFactorAtrNStop
from ..FactorSellBu.ABuFactorCloseAtrNStop import AbuFactorCloseAtrNStop
from ..FactorSellBu.ABuFactorPreAtrNStop import AbuFactorPreAtrNStop
from ..FactorSellBu.ABuFactorSellBreak import AbuFactorSellBreak
from ..FactorSellBu.ABuFactorSellDM import AbuDoubleMaSell
from ..FactorSellBu.ABuFactorSellNDay import AbuFactorSellNDay
from ..WidgetBu.ABuWGSFBase import WidgetFactorSellBase

__author__ = '阿布'
__weixin__ = 'abu_quant'


class SellDMWidget(WidgetFactorSellBase):
    """对应AbuDoubleMaSell策略widget"""

    def _init_widget(self):
        """构建AbuDoubleMaSell策略参数界面"""

        self.description = widgets.Textarea(
            value=u'双均线卖出策略：\n'
                  u'双均线策略是量化策略中经典的策略之一，其属于趋势跟踪策略: \n'
                  u'1. 预设两条均线：如一个ma=5，一个ma=60, 5的均线被称作快线，60的均线被称作慢线\n'
                  u'2. 择时卖出策略中当快线下穿慢线（ma5下穿ma60）称为形成死叉卖点信号，卖出股票\n',
            description=u'双均线卖',
            disabled=False,
            layout=self.description_layout
        )

        self.slow_label = widgets.Label(u'默认慢线ma=60:当快线下穿慢线称为形成死叉', layout=self.label_layout)
        self.slow_int = widgets.IntSlider(
            value=60,
            min=10,
            max=120,
            step=1,
            description=u'慢线',
            disabled=False,
            orientation='horizontal',
            readout=True,
            readout_format='d'
        )
        self.slow_box = widgets.VBox([self.slow_label, self.slow_int])

        self.fast_label = widgets.Label(u'默认快线ma=5:当快线下穿慢线称为形成死叉', layout=self.label_layout)
        self.fast_int = widgets.IntSlider(
            value=5,
            min=1,
            max=90,
            step=1,
            description=u'快线',
            disabled=False,
            orientation='horizontal',
            readout=True,
            readout_format='d'
        )
        self.fast_box = widgets.VBox([self.fast_label, self.fast_int])
        self.widget = widgets.VBox([self.description, self.slow_box, self.fast_box, self.add_box],
                                   # border='solid 1px',
                                   layout=self.widget_layout)

    def make_sell_factor_unique(self):
        """对应按钮添加AbuDoubleMaSell策略，构建策略字典对象factor_dict以及唯一策略描述字符串factor_desc_key"""
        factor_dict = {'class': AbuDoubleMaSell, 'slow': self.slow_int.value, 'fast': self.fast_int.value}
        factor_desc_key = u'动态双均慢{}快{}卖出'.format(self.slow_int.value, self.fast_int.value)
        return factor_dict, factor_desc_key

    def delegate_class(self):
        """子类因子所委托的具体因子类AbuDoubleMaSell"""
        return AbuDoubleMaSell


class SellAtrNStopWidget(WidgetFactorSellBase):
    """对应AbuFactorAtrNStop策略widget"""

    def _init_widget(self):
        """构建AbuFactorAtrNStop策略参数界面"""

        self.description = widgets.Textarea(
            value=u'止盈策略 & 止损策略：\n'
                  u'1. 真实波幅atr作为最大止盈和最大止损的常数值\n'
                  u'2. 当stop_loss_n 乘以 当日atr > 买入价格 － 当日收盘价格->止损卖出\n'
                  u'3. 当stop_win_n 乘以 当日atr < 当日收盘价格 －买入价格->止盈卖出',
            description=u'止盈止损',
            disabled=False,
            layout=self.description_layout
        )
        self.stop_loss_n_label = widgets.Label(u'stop_loss_n乘以当日atr大于买价减close->止损',
                                               layout=self.label_layout)
        self.stop_loss_n = widgets.FloatSlider(
            value=1.0,
            min=0.10,
            max=10.0,
            step=0.1,
            description='stop_loss_n',
            disabled=False,
            orientation='horizontal',
            readout=True,
            readout_format='.1f',
        )
        self.stop_loss_n_box = widgets.VBox([self.stop_loss_n_label, self.stop_loss_n])
        self.stop_win_n_label = widgets.Label(u'stop_win_n乘以当日atr小于close减买价->止盈',
                                              layout=self.label_layout)
        self.stop_win_n = widgets.FloatSlider(
            value=3.0,
            min=0.10,
            max=10.0,
            step=0.10,
            description='stop_win_n',
            disabled=False,
            orientation='horizontal',
            readout=True,
            readout_format='.1f',
        )
        self.stop_win_n_box = widgets.VBox([self.stop_win_n_label, self.stop_win_n])
        self.widget = widgets.VBox([self.description, self.stop_loss_n_box,
                                    self.stop_win_n_box, self.add_box],  # border='solid 1px',
                                   layout=self.widget_layout)

    def make_sell_factor_unique(self):
        """对应按钮添加AbuFactorAtrNStop策略，构建策略字典对象factor_dict以及唯一策略描述字符串factor_desc_key"""
        factor_dict = {'class': AbuFactorAtrNStop,
                       'stop_win_n': self.stop_win_n.value,
                       'stop_loss_n': self.stop_loss_n.value}
        factor_desc_key = u'n atr止盈{}止损{}'.format(self.stop_win_n.value, self.stop_loss_n.value)
        return factor_dict, factor_desc_key

    def delegate_class(self):
        """子类因子所委托的具体因子类AbuFactorAtrNStop"""
        return AbuFactorAtrNStop


class SellCloseAtrNWidget(WidgetFactorSellBase):
    """对应AbuFactorCloseAtrNStop策略widget"""

    def _init_widget(self):
        """构建AbuFactorCloseAtrNStop策略参数界面"""
        self.description = widgets.Textarea(
            value=u'利润保护止盈策略：\n'
                  u'1. 买入后最大收益价格 - 今日价格 > 一定收益\n'
                  u'2. 买入后最大收益价格 - 今日价格 < close_atr_n * 当日atr\n'
                  u'3. 当买入有一定收益后，如果下跌幅度超过close_atr_n乘以当日atr->保护止盈卖出',
            description=u'保护止盈',
            disabled=False,
            layout=self.description_layout
        )

        self.close_atr_n_label = widgets.Label(u'收益下跌超过close_atr_n乘以当日atr->保护止盈',
                                               layout=self.label_layout)
        self.close_atr_n = widgets.FloatSlider(
            value=1.5,
            min=0.10,
            max=10.0,
            step=0.1,
            description='close_atr_n',
            disabled=False,
            orientation='horizontal',
            readout=True,
            readout_format='.1f',
        )
        self.close_atr_n_box = widgets.VBox([self.close_atr_n_label, self.close_atr_n])
        self.widget = widgets.VBox([self.description, self.close_atr_n_box, self.add_box],  # border='solid 1px',
                                   layout=self.widget_layout)

    def make_sell_factor_unique(self):
        """对应按钮添加AbuFactorCloseAtrNStop策略，构建策略字典对象factor_dict以及唯一策略描述字符串factor_desc_key"""
        factor_dict = {'class': AbuFactorCloseAtrNStop,
                       'close_atr_n': self.close_atr_n.value}
        factor_desc_key = u'利润保护止盈n={}'.format(self.close_atr_n.value)
        return factor_dict, factor_desc_key

    def delegate_class(self):
        """子类因子所委托的具体因子类AbuFactorCloseAtrNStop"""
        return AbuFactorCloseAtrNStop


class SellPreAtrNWidget(WidgetFactorSellBase):
    """对应AbuFactorPreAtrNStop策略widget"""

    def _init_widget(self):
        """构建AbuFactorPreAtrNStop策略参数界面"""
        self.description = widgets.Textarea(
            value=u'风险控制止损策略：\n'
                  u'1. 单日最大跌幅n倍atr止损\n'
                  u'2. 当今日价格下跌幅度 > 当日atr 乘以 pre_atr_n（下跌止损倍数）卖出操作',
            description=u'风险止损',
            disabled=False,
            layout=self.description_layout
        )

        self.pre_atr_n_label = widgets.Label(u'当今天价格开始剧烈下跌，采取果断平仓措施',
                                             layout=self.label_layout)
        self.pre_atr_n = widgets.FloatSlider(
            value=1.5,
            min=0.10,
            max=10.0,
            step=0.1,
            description='pre_atr_n',
            disabled=False,
            orientation='horizontal',
            readout=True,
            readout_format='.1f',
        )
        self.pre_atr_n_box = widgets.VBox([self.pre_atr_n_label, self.pre_atr_n])
        self.widget = widgets.VBox([self.description, self.pre_atr_n_box, self.add_box],  # border='solid 1px',
                                   layout=self.widget_layout)

    def make_sell_factor_unique(self):
        """对应按钮添加AbuFactorPreAtrNStop策略，构建策略字典对象factor_dict以及唯一策略描述字符串factor_desc_key"""
        factor_dict = {'class': AbuFactorPreAtrNStop,
                       'pre_atr_n': self.pre_atr_n.value}
        factor_desc_key = u'风险控制止损n={}'.format(self.pre_atr_n.value)
        return factor_dict, factor_desc_key

    def delegate_class(self):
        """子类因子所委托的具体因子类AbuFactorPreAtrNStop"""
        return AbuFactorPreAtrNStop


class SellXDWidget(WidgetFactorSellBase):
    """对应AbuFactorSellBreak策略widget"""

    def _init_widget(self):
        """构建AbuFactorSellBreak策略参数界面"""
        self.description = widgets.Textarea(
            value=u'海龟向下趋势突破卖出策略：\n'
                  u'趋势突破定义为当天收盘价格低于N天内的最低价，作为卖出信号，卖出操作',
            description=u'海龟卖出',
            disabled=False,
            layout=self.description_layout
        )
        self.xd_label = widgets.Label(u'突破周期参数：比如21，30，42天....突破',
                                      layout=self.label_layout)
        self.xd = widgets.IntSlider(
            value=10,
            min=3,
            max=120,
            step=1,
            description=u'周期',
            disabled=False,
            orientation='horizontal',
            readout=True,
            readout_format='d'
        )
        self.xd_box = widgets.VBox([self.xd_label, self.xd])
        self.widget = widgets.VBox([self.description, self.xd_box, self.add_box],  # border='solid 1px',
                                   layout=self.widget_layout)

    def make_sell_factor_unique(self):
        """对应按钮添加AbuFactorSellBreak策略，构建策略字典对象factor_dict以及唯一策略描述字符串factor_desc_key"""
        factor_dict = {'class': AbuFactorSellBreak, 'xd': self.xd.value}
        factor_desc_key = u'海龟{}天趋势突破卖出'.format(self.xd.value)
        return factor_dict, factor_desc_key

    def delegate_class(self):
        """子类因子所委托的具体因子类AbuFactorSellBreak"""
        return AbuFactorSellBreak


class SellNDWidget(WidgetFactorSellBase):
    """对应AbuFactorSellNDay策略widget"""

    def _init_widget(self):
        """构建AbuFactorSellNDay策略参数界面"""
        self.description = widgets.Textarea(
            value=u'持有N天后卖出策略：\n'
                  u'卖出策略，不管交易现在什么结果，买入后只持有N天\n'
                  u'需要与特定\'买入策略\'形成配合\n，'
                  u'单独使用N天卖出策略意义不大',
            description=u'N天卖出',
            disabled=False,
            layout=self.description_layout
        )
        sell_n_label = widgets.Label(u'设定买入后只持有天数，默认1', layout=self.label_layout)
        self.sell_n = widgets.IntText(
            value=1,
            description=u'N天',
            disabled=False
        )
        sell_n_box = widgets.VBox([sell_n_label, self.sell_n])

        is_sell_today_label = widgets.Label(u'设定买入n天后，当天还是隔天卖出', layout=self.label_layout)
        self.is_sell_today = widgets.Dropdown(
            options={u'N天后隔天卖出': False, u'N天后当天卖出': True},
            value=False,
            description=u'当天隔天:',
        )
        is_sell_today_box = widgets.VBox([is_sell_today_label, self.is_sell_today])

        self.widget = widgets.VBox([self.description, sell_n_box, is_sell_today_box, self.add_box],
                                   # border='solid 1px',
                                   layout=self.widget_layout)

    def make_sell_factor_unique(self):
        """对应按钮添加AbuFactorSellNDay策略，构建策略字典对象factor_dict以及唯一策略描述字符串factor_desc_key"""
        factor_dict = {'class': AbuFactorSellNDay, 'sell_n': self.sell_n.value,
                       'is_sell_today': self.is_sell_today.value}
        factor_desc_key = u'持有{}天{}卖出'.format(self.sell_n.value, u'当天' if self.is_sell_today.value else u'隔天')
        return factor_dict, factor_desc_key

    def delegate_class(self):
        """子类因子所委托的具体因子类AbuFactorSellNDay"""
        return AbuFactorSellNDay
