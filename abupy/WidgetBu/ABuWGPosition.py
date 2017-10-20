# -*- encoding:utf-8 -*-
"""资金仓位管理策略图形可视化"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import ipywidgets as widgets

from ..BetaBu.ABuAtrPosition import AbuAtrPosition
from ..BetaBu.ABuKellyPosition import AbuKellyPosition
from ..BetaBu.ABuPtPosition import AbuPtPosition
from ..WidgetBu.ABuWGPosBase import WidgetPositionBase

__author__ = '阿布'
__weixin__ = 'abu_quant'


class AtrPosWidget(WidgetPositionBase):
    """对应AbuAtrPosition策略widget"""

    def _init_widget(self):
        """构建AbuAtrPosition策略参数界面"""

        description = widgets.Textarea(
            value=u'atr资金仓位管理策略：\n'
                  u'默认的仓位资金管理全局策略\n'
                  u'根据决策买入当天的价格波动决策资金仓位配比\n'
                  u'注意不同于卖策，选股，一个买入因子只能有唯一个资金仓位管理策略',
            description=u'atr资管',
            disabled=False,
            layout=self.description_layout
        )

        atr_pos_base_label = widgets.Label(u'仓位基础配比：默认0.1即资金10%为仓位基数',
                                           layout=self.label_layout)
        # 需要精确到小数点后5位
        self.atr_pos_base = widgets.FloatSlider(
            value=0.10,
            min=0.00001,
            max=1.0,
            step=0.00001,
            description=u'基配',
            disabled=False,
            orientation='horizontal',
            readout=True,
            readout_format='.5f'
        )
        atr_pos_base_box = widgets.VBox([atr_pos_base_label, self.atr_pos_base])

        atr_base_price_label = widgets.Label(u'常数价格设定：默认15，建议在12-20之间',
                                             layout=self.label_layout)
        self.atr_base_price = widgets.IntSlider(
            value=15,
            min=12,
            max=20,
            step=1,
            description=u'常价',
            disabled=False,
            orientation='horizontal',
            readout=True,
            readout_format='d'
        )
        atr_base_price_box = widgets.VBox([atr_base_price_label, self.atr_base_price])

        # TODO AbuAtrPosition策略中std_atr_threshold的设置

        self.widget = widgets.VBox([description, atr_pos_base_box, atr_base_price_box,
                                    self.add_box], layout=self.widget_layout)

    def make_position_unique(self):
        """对应按钮添加AbuAtrPosition策略，构建策略字典对象factor_dict以及唯一策略描述字符串factor_desc_key"""
        factor_dict = {'class': AbuAtrPosition,
                       'atr_pos_base': self.atr_pos_base.value,
                       'atr_base_price': self.atr_base_price.value}

        factor_desc_key = u'atr资管仓位基数:{}常数价格:{}'.format(
            self.atr_pos_base.value, self.atr_base_price.value)
        return factor_dict, factor_desc_key

    def delegate_class(self):
        """子类因子所委托的具体因子类AbuAtrPosition"""
        return AbuAtrPosition


class KellyPosWidget(WidgetPositionBase):
    """对应AbuKellyPosition策略widget"""

    def _init_widget(self):
        """构建AbuKellyPosition策略参数界面"""

        description = widgets.Textarea(
            value=u'kelly资金仓位管理策略：\n'
                  u'根据策略历史胜率期望，盈利期望，亏损期望决策资金仓位配比\n'
                  u'仓位资金配比 = 胜率 - 败率/(盈利期望/亏损期望)\n'
                  u'注意不同于卖策，选股，一个买入因子只能有唯一个资金仓位管理策略',
            description=u'kelly资管',
            disabled=False,
            layout=self.description_layout
        )

        win_rate_label = widgets.Label(u'策略历史胜率期望，默认0.5即50%胜率',
                                       layout=self.label_layout)
        # 需要精确到小数点后5位
        self.win_rate = widgets.FloatSlider(
            value=0.50,
            min=0.01,
            max=1.00,
            step=0.0001,
            description=u'胜率期望',
            disabled=False,
            orientation='horizontal',
            readout=True,
            readout_format='.4f'
        )
        win_rate_box = widgets.VBox([win_rate_label, self.win_rate])

        gains_mean_label = widgets.Label(u'策略历史盈利期望，默认0.1即10%',
                                         layout=self.label_layout)
        self.gains_mean = widgets.FloatSlider(
            value=0.10,
            min=0.01,
            max=100.00,
            step=0.0001,
            description=u'盈利期望',
            disabled=False,
            orientation='horizontal',
            readout=True,
            readout_format='.4f'
        )
        gains_mean_box = widgets.VBox([gains_mean_label, self.gains_mean])

        losses_mean_label = widgets.Label(u'策略历史亏损期望，默认0.05即5%',
                                          layout=self.label_layout)
        self.losses_mean = widgets.FloatSlider(
            value=0.05,
            min=0.01,
            max=100.00,
            step=0.0001,
            description=u'亏损期望',
            disabled=False,
            orientation='horizontal',
            readout=True,
            readout_format='.4f'
        )
        losses_mean_box = widgets.VBox([losses_mean_label, self.losses_mean])

        self.widget = widgets.VBox([description, win_rate_box, gains_mean_box,
                                    losses_mean_box, self.add_box], layout=self.widget_layout)

    def make_position_unique(self):
        """对应按钮添加AbuKellyPosition策略，构建策略字典对象factor_dict以及唯一策略描述字符串factor_desc_key"""
        factor_dict = {'class': AbuKellyPosition,
                       'win_rate': self.win_rate.value,
                       'gains_mean': self.gains_mean.value,
                       'losses_mean': self.losses_mean.value}

        factor_desc_key = u'kelly资管仓位胜率:{}盈期:{}亏期:{}'.format(
            self.win_rate.value, self.gains_mean.value, self.losses_mean.value)
        return factor_dict, factor_desc_key

    def delegate_class(self):
        """子类因子所委托的具体因子类AbuKellyPosition"""
        return AbuKellyPosition


class PtPosition(WidgetPositionBase):
    """对应AbuPtPosition策略widget"""

    def _init_widget(self):
        """构建AbuPtPosition策略参数界面"""

        description = widgets.Textarea(
            value=u'价格位置仓位管理策略：\n'
                  u'针对均值回复类型策略的仓位管理策略\n'
                  u'根据买入价格在之前一段时间的价格位置来决策仓位大小\n'
                  u'假设过去一段时间的价格为[10, 9, 8, 7, 6, 5, 4, 3, 2, 1]\n'
                  u'如果当前买入价格为2元：则买入仓位配比很高(认为均值回复有很大向上空间)\n'
                  u'如果当前买入价格为9元：则买入仓位配比很低(认为均值回复向上空间比较小)',
            description=u'价格位置',
            disabled=False,
            layout=self.description_layout
        )

        pos_base_label = widgets.Label(u'仓位基础配比：默认0.1即资金10%为仓位基数',
                                       layout=self.label_layout)
        # 需要精确到小数点后5位
        self.pos_base = widgets.FloatSlider(
            value=0.10,
            min=0.00001,
            max=1.0,
            step=0.00001,
            description=u'基配',
            disabled=False,
            orientation='horizontal',
            readout=True,
            readout_format='.5f'
        )
        pos_base_box = widgets.VBox([pos_base_label, self.pos_base])
        past_day_cnt_label = widgets.Label(u'根据过去多长一段时间的价格趋势做为参考，默认20',
                                           layout=self.label_layout)
        self.past_day_cnt = widgets.IntSlider(
            value=20,
            min=5,
            max=250,
            step=1,
            description=u'参考天数',
            disabled=False,
            orientation='horizontal',
            readout=True,
            readout_format='d'
        )
        past_day_cnt_box = widgets.VBox([past_day_cnt_label, self.past_day_cnt])

        self.widget = widgets.VBox([description, pos_base_box,
                                    past_day_cnt_box, self.add_box], layout=self.widget_layout)

    def make_position_unique(self):
        """对应按钮添加AbuPtPosition策略，构建策略字典对象factor_dict以及唯一策略描述字符串factor_desc_key"""
        factor_dict = {'class': AbuPtPosition,
                       'pos_base': self.pos_base.value,
                       'past_day_cnt': self.past_day_cnt.value}
        factor_desc_key = u'价格位置基仓比例:{} 参考天数:{}'.format(
            self.pos_base.value, self.past_day_cnt.value)
        return factor_dict, factor_desc_key

    def delegate_class(self):
        """子类因子所委托的具体因子类AbuPtPosition"""
        return AbuPtPosition
