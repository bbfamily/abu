# -*- encoding:utf-8 -*-
"""买入因子参数以及选择图形可视化"""

from __future__ import print_function
from __future__ import absolute_import
from __future__ import division

import ipywidgets as widgets

from ..WidgetBu.ABuWGBFBase import WidgetFactorBuyBase
from ..FactorBuyBu.ABuFactorBuyBreak import AbuFactorBuyBreak
from ..FactorBuyBu.ABuFactorBuyDM import AbuDoubleMaBuy
from ..FactorBuyBu.ABuFactorBuyWD import AbuFactorBuyWD
from ..FactorBuyBu.ABuFactorBuyDemo import AbuSDBreak, AbuWeekMonthBuy
from ..FactorBuyBu.ABuFactorBuyTrend import AbuDownUpTrend

__author__ = '阿布'
__weixin__ = 'abu_quant'


class BuyDMWidget(WidgetFactorBuyBase):
    """对应AbuDoubleMaBuy策略widget"""

    def _init_widget(self):
        """构建AbuDoubleMaBuy策略参数界面"""

        self.description = widgets.Textarea(
            value=u'动态自适应双均线买入策略：\n'
                  u'双均线策略是量化策略中经典的策略之一，其属于趋势跟踪策略: \n'
                  u'1. 预设两条均线：如一个ma=5，一个ma=60, 5的均线被称作快线，60的均线被称作慢线\n'
                  u'2. 择时买入策略中当快线上穿慢线（ma5上穿ma60）称为形成金叉买点信号，买入股票\n'
                  u'3. 自适应动态慢线，不需要输入慢线值，根据走势震荡套利空间，寻找合适的ma慢线\n'
                  u'4. 自适应动态快线，不需要输入快线值，根据慢线以及大盘走势，寻找合适的ma快线',
            description=u'双均线买',
            disabled=False,
            layout=self.description_layout
        )

        self.slow_label = widgets.Label(u'默认使用动态慢线，可手动固定慢线值', layout=self.label_layout)
        self.slow_int = widgets.IntSlider(
            value=60,
            min=10,
            max=120,
            step=1,
            description=u'手动',
            disabled=False,
            orientation='horizontal',
            readout=True,
            readout_format='d'
        )
        self.auto_slow = widgets.Checkbox(
            value=True,
            description=u'动态慢线',
            disabled=False
        )

        def slow_change(change):
            self.slow_int.disabled = change['new']

        self.auto_slow.observe(slow_change, names='value')
        self.slow = widgets.VBox([self.auto_slow, self.slow_int])
        self.slow_box = widgets.VBox([self.slow_label, self.slow])

        self.fast_label = widgets.Label(u'默认使用动态快线，可手动固定快线值', layout=self.label_layout)
        self.fast_int = widgets.IntSlider(
            value=5,
            min=1,
            max=90,
            step=1,
            description=u'手动',
            disabled=False,
            orientation='horizontal',
            readout=True,
            readout_format='d'
        )
        self.auto_fast = widgets.Checkbox(
            value=True,
            description=u'动态快线',
            disabled=False,
        )

        def fast_change(change):
            self.fast_int.disabled = change['new']

        self.auto_fast.observe(fast_change, names='value')
        self.fast = widgets.VBox([self.auto_fast, self.fast_int])
        self.fast_box = widgets.VBox([self.fast_label, self.fast])

        self.widget = widgets.VBox([self.description, self.slow_box, self.fast_box, self.add],  # border='solid 1px',
                                   layout=self.widget_layout)

    def make_buy_factor_unique(self):
        """对应按钮添加AbuDoubleMaBuy策略，构建策略字典对象factor_dict以及唯一策略描述字符串factor_desc_key"""
        slow_int = -1 if self.auto_slow.value else self.slow_int.value
        fast_int = -1 if self.auto_fast.value else self.fast_int.value

        factor_dict = {'class': AbuDoubleMaBuy, 'slow': slow_int, 'fast': fast_int}
        factor_desc_key = u'动态双均慢{}快{}买入'.format(
            u'动态' if slow_int == -1 else slow_int, u'动态' if fast_int == -1 else fast_int)
        return factor_dict, factor_desc_key

    def delegate_class(self):
        """子类因子所委托的具体因子类AbuDoubleMaBuy"""
        return AbuDoubleMaBuy


class BuySDWidget(WidgetFactorBuyBase):
    """对应AbuSDBreak策略widget"""

    def _init_widget(self):
        """构建AbuSDBreak策略参数界面"""

        self.description = widgets.Textarea(
            value=u'参照大盘走势向上趋势突破买入策略：\n'
                  u'在海龟突破基础上，参照大盘走势，进行降低交易频率，提高系统的稳定性处理，当大盘走势震荡时封锁交易，'
                  u'当大盘走势平稳时再次打开交易，每一个月计算一次大盘走势是否平稳',
            description=u'平稳突破',
            disabled=False,
            layout=self.description_layout
        )
        self.poly_label = widgets.Label(u'大盘走势拟合次数阀值，poly大于阀值＝震荡',
                                        layout=self.label_layout)
        self.poly = widgets.IntSlider(
            value=2,
            min=1,
            max=5,
            step=1,
            description=u'拟合',
            disabled=False,
            orientation='horizontal',
            readout=True,
            readout_format='d'
        )
        self.poly_box = widgets.VBox([self.poly_label, self.poly])
        self.xd_label = widgets.Label(u'突破周期参数：比如21，30，42天....突破',
                                      layout=self.label_layout)
        self.xd = widgets.IntSlider(
            value=21,
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
        self.widget = widgets.VBox([self.description, self.poly_box,
                                    self.xd_box, self.add],  # border='solid 1px',
                                   layout=self.widget_layout)

    def make_buy_factor_unique(self):
        """对应按钮添加AbuSDBreak策略，构建策略字典对象factor_dict以及唯一策略描述字符串factor_desc_key"""
        factor_dict = {'class': AbuSDBreak, 'xd': self.xd.value, 'poly': self.poly.value}
        factor_desc_key = u'{}拟合{}天趋势突破参照大盘'.format(self.poly.value, self.xd.value)
        return factor_dict, factor_desc_key

    def delegate_class(self):
        """子类因子所委托的具体因子类AbuSDBreak"""
        return AbuSDBreak


class BuyWDWidget(WidgetFactorBuyBase):
    """对应AbuFactorBuyWD策略widget"""

    def _init_widget(self):
        """构建AbuFactorBuyWD策略参数界面"""

        self.description = widgets.Textarea(
            value=u'日胜率均值回复策略：\n'
                  u'1. 默认以40天为周期(8周)结合涨跌阀值计算周几适合买入\n'
                  u'2. 回测运行中每一月重新计算一次上述的周几适合买入\n'
                  u'3. 在策略日任务中买入信号为：昨天下跌，今天开盘也下跌，且明天是计算出来的上涨概率大的\'周几\'',
            description=u'周涨胜率',
            disabled=False,
            layout=self.description_layout
        )

        self.buy_dw_label = widgets.Label(u'代表周期胜率阀值，默认0.55即55%的胜率',
                                          layout=self.label_layout)
        self.buy_dw = widgets.FloatSlider(
            value=0.55,
            min=0.50,
            max=0.99,
            step=0.01,
            description=u'胜率',
            disabled=False,
            orientation='horizontal',
            readout=True,
            readout_format='.2f',
        )
        self.buy_dw_box = widgets.VBox([self.buy_dw_label, self.buy_dw])

        self.buy_dwm_label = widgets.Label(u'代表涨幅比例阀值系数，默认0.618',
                                           layout=self.label_layout)
        self.buy_dwm = widgets.FloatSlider(
            value=0.618,
            min=0.50,
            max=1.0,
            step=0.01,
            description=u'系数',
            disabled=False,
            orientation='horizontal',
            readout=True,
            readout_format='.3f'
        )
        self.buy_dwm_box = widgets.VBox([self.buy_dwm_label, self.buy_dwm])

        self.dw_period_label = widgets.Label(u'代表分所使用的交易周期，默认40天周期(8周)',
                                             layout=self.label_layout)
        self.dw_period = widgets.IntSlider(
            value=40,
            min=20,
            max=120,
            step=1,
            description=u'周期',
            disabled=False,
            orientation='horizontal',
            readout=True,
            readout_format='d'
        )
        self.dw_period_box = widgets.VBox([self.dw_period_label, self.dw_period])
        self.widget = widgets.VBox([self.description, self.buy_dw_box,
                                    self.buy_dwm_box, self.dw_period_box, self.add],  # border='solid 1px',
                                   layout=self.widget_layout)

    def make_buy_factor_unique(self):
        """对应按钮添加AbuFactorBuyWD策略，构建策略字典对象factor_dict以及唯一策略描述字符串factor_desc_key"""
        factor_dict = {'class': AbuFactorBuyWD, 'buy_dw': self.buy_dw.value,
                       'buy_dwm': self.buy_dwm.value, 'dw_period': self.dw_period.value}
        factor_desc_key = u'日胜率{},{},{}均值回复买入'.format(
            self.buy_dw.value, self.buy_dwm.value, self.dw_period.value)
        return factor_dict, factor_desc_key

    def delegate_class(self):
        """子类因子所委托的具体因子类AbuFactorBuyWD"""
        return AbuFactorBuyWD


class BuyXDWidget(WidgetFactorBuyBase):
    """对应AbuFactorBuyBreak策略widget"""

    def _init_widget(self):
        """构建AbuFactorBuyBreak策略参数界面"""

        self.description = widgets.Textarea(
            value=u'海龟向上趋势突破买入策略：\n'
                  u'趋势突破定义为当天收盘价格超过N天内的最高价，超过最高价格作为买入信号买入股票持有',
            description=u'海龟买入',
            disabled=False,
            layout=self.description_layout
        )
        self.xd_label = widgets.Label(u'突破周期参数：比如21，30，42天....突破', layout=self.label_layout)
        self.xd = widgets.IntSlider(
            value=21,
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
        self.widget = widgets.VBox([self.description, self.xd_box, self.add],  # border='solid 1px',
                                   layout=self.widget_layout)

    def make_buy_factor_unique(self):
        """对应按钮添加AbuFactorBuyBreak策略，构建策略字典对象factor_dict以及唯一策略描述字符串factor_desc_key"""
        factor_dict = {'class': AbuFactorBuyBreak, 'xd': self.xd.value}
        factor_desc_key = u'海龟{}天趋势突破买入'.format(self.xd.value)
        return factor_dict, factor_desc_key

    def delegate_class(self):
        """子类因子所委托的具体因子类AbuFactorBuyBreak"""
        return AbuFactorBuyBreak


class BuyWMWidget(WidgetFactorBuyBase):
    """对应AbuWeekMonthBuy策略widget"""

    def _init_widget(self):
        """构建AbuWeekMonthBuy策略参数界面"""

        self.description = widgets.Textarea(
            value=u'固定周期买入策略：\n'
                  u'根据参数每周买入一次或者每一个月买入一次\n'
                  u'需要与特定\'选股策略\'和\'卖出策略\'形成配合\n，'
                  u'单独使用固定周期买入策略意义不大',
            description=u'定期买入',
            disabled=False,
            layout=self.description_layout
        )

        is_buy_month_label = widgets.Label(u'可更改买入定期，默认定期一个月', layout=self.label_layout)
        self.is_buy_month = widgets.Dropdown(
            options={u'定期一个月': True, u'定期一个周': False},
            value=True,
            description=u'定期时长:',
        )
        is_buy_month_box = widgets.VBox([is_buy_month_label, self.is_buy_month])

        self.widget = widgets.VBox([self.description, is_buy_month_box, self.add],  # border='solid 1px',
                                   layout=self.widget_layout)

    def make_buy_factor_unique(self):
        """对应按钮添加AbuWeekMonthBuy策略，构建策略字典对象factor_dict以及唯一策略描述字符串factor_desc_key"""
        factor_dict = {'class': AbuWeekMonthBuy, 'is_buy_month': self.is_buy_month.value}
        factor_desc_key = u'{}买入一次'.format(u'每一月' if self.is_buy_month.value else u'每一周')
        return factor_dict, factor_desc_key

    def delegate_class(self):
        """子类因子所委托的具体因子类AbuWeekMonthBuy"""
        return AbuWeekMonthBuy


class BuyDUWidget(WidgetFactorBuyBase):
    """对应AbuDownUpTrend策略widget"""

    def _init_widget(self):
        """构建AbuDownUpTrend策略参数界面"""

        self.description = widgets.Textarea(
            value=u'整个择时周期分成两部分，长的为长线择时，短的为短线择时：\n'
                  u'1. 寻找长线下跌的股票，比如一个季度(4个月)整体趋势为下跌趋势\n'
                  u'2. 短线走势上涨的股票，比如一个月整体趋势为上涨趋势\n，'
                  u'3. 最后使用海龟突破的N日突破策略作为策略最终买入信号',
            description=u'长跌短涨',
            disabled=False,
            layout=self.description_layout
        )

        xd_label = widgets.Label(u'短线周期：比如20，30，40天,短线以及突破参数',
                                 layout=self.label_layout)
        self.xd = widgets.IntSlider(
            value=20,
            min=5,
            max=120,
            step=5,
            description=u'xd',
            disabled=False,
            orientation='horizontal',
            readout=True,
            readout_format='d'
        )
        xd_box = widgets.VBox([xd_label, self.xd])

        past_factor_label = widgets.Label(u'长线乘数：短线基础 x 长线乘数 = 长线周期',
                                          layout=self.label_layout)
        self.past_factor = widgets.IntSlider(
            value=4,
            min=1,
            max=10,
            step=1,
            description=u'长线乘数',
            disabled=False,
            orientation='horizontal',
            readout=True,
            readout_format='d'
        )
        past_factor_box = widgets.VBox([past_factor_label, self.past_factor])

        down_deg_threshold_label = widgets.Label(u'拟合趋势角度阀值：如-2,-3,-4',
                                                 layout=self.label_layout)
        self.down_deg_threshold = widgets.IntSlider(
            value=-3,
            min=-10,
            max=0,
            step=1,
            description=u'角度阀值',
            disabled=False,
            orientation='horizontal',
            readout=True,
            readout_format='d'
        )
        down_deg_threshold_box = widgets.VBox([down_deg_threshold_label, self.down_deg_threshold])

        self.widget = widgets.VBox([self.description, xd_box, past_factor_box, down_deg_threshold_box, self.add],
                                   layout=self.widget_layout)

    def make_buy_factor_unique(self):
        """对应按钮添加AbuDownUpTrend策略，构建策略字典对象factor_dict以及唯一策略描述字符串factor_desc_key"""
        factor_dict = {'class': AbuDownUpTrend, 'xd': self.xd.value,
                       'past_factor': self.past_factor.value, 'down_deg_threshold': self.down_deg_threshold.value}
        factor_desc_key = u'长线{}下跌短线{}上涨角度{}'.format(
            self.xd.value * self.past_factor.value, self.xd.value, self.down_deg_threshold.value)
        return factor_dict, factor_desc_key

    def delegate_class(self):
        """子类因子所委托的具体因子类AbuDownUpTrend"""
        return AbuDownUpTrend
