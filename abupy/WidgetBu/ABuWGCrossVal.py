# -*- encoding:utf-8 -*-
"""策略相关性交叉验证图形可视化"""

from __future__ import print_function
from __future__ import absolute_import
from __future__ import division

import ipywidgets as widgets

from ..WidgetBu.ABuWGBase import WidgetBase, show_msg_toast_func
from ..WidgetBu.ABuWGBFBase import BuyFactorWGManager
from ..WidgetBu.ABuWGSFBase import SellFactorWGManager
# noinspection PyUnresolvedReferences
from ..CoreBu.ABuFixes import filter
from ..MetricsBu.ABuCrossVal import AbuCrossVal
from ..CoreBu import ABuEnv
from ..CoreBu.ABuEnv import EMarketTargetType

__author__ = '阿布'
__weixin__ = 'abu_quant'


# noinspection PyProtectedMember
class WidgetCrossVal(WidgetBase):
    """策略相关性交叉验证ui类"""

    # noinspection PyProtectedMember
    def __init__(self):
        """构建回测需要的各个组件形成tab"""

        tip_label1 = widgets.Label(u'策略相关性交叉验证暂不支持实时网络数据模式', layout=widgets.Layout(width='300px'))
        tip_label2 = widgets.Label(u'需先用\'数据下载界面操作\'进行下载', layout=widgets.Layout(width='300px'))

        self.bf = BuyFactorWGManager()
        self.sf = SellFactorWGManager(show_add_buy=True)

        sub_widget_tab = widgets.Tab()
        sub_widget_tab.children = [self.bf.widget, self.sf.widget]
        for ind, name in enumerate([u'买策', u'卖策']):
            sub_widget_tab.set_title(ind, name)

        self.begin_cross_val = widgets.Button(description=u'开始交叉相关性验证策略有效性',
                                              layout=widgets.Layout(width='98%'),
                                              button_style='danger')
        self.begin_cross_val.on_click(self.run_cross_val)

        self.market = widgets.Dropdown(
            options={u'美股': EMarketTargetType.E_MARKET_TARGET_US.value,
                     u'A股': EMarketTargetType.E_MARKET_TARGET_CN.value,
                     u'港股': EMarketTargetType.E_MARKET_TARGET_HK.value,
                     u'国内期货': EMarketTargetType.E_MARKET_TARGET_FUTURES_CN.value,
                     u'国际期货': EMarketTargetType.E_MARKET_TARGET_FUTURES_GLOBAL.value,
                     u'数字货币': EMarketTargetType.E_MARKET_TARGET_TC.value},
            value=ABuEnv.g_market_target.value,
            description=u'验证市场:',
        )

        cv_label1 = widgets.Label(u'交叉验证的数量级：默认10', layout=widgets.Layout(width='300px'))
        cv_label2 = widgets.Label(u'cv次相关度范围随机抽取cv个symbol进行回测', layout=widgets.Layout(width='300px'))
        self.cv = widgets.IntSlider(
            value=10,
            min=4,
            max=50,
            step=1,
            description=u'cv',
            disabled=False,
            orientation='horizontal',
            readout=True,
            readout_format='d'
        )
        cv_box = widgets.VBox([cv_label1, cv_label2, self.cv])

        self.widget = widgets.VBox([tip_label1, tip_label2, sub_widget_tab, self.market, cv_box,
                                    self.begin_cross_val])

        # 初始化就new处理，每次运行都使用它，可以缓存similar数据
        self.cross_val = AbuCrossVal()

    # noinspection PyUnusedLocal
    def run_cross_val(self, bt):
        """交叉相关性验证策略有效性的button按钮"""

        # 买入策略构成序列
        buy_factors = list(self.bf.factor_dict.values())
        if len(buy_factors) == 0:
            show_msg_toast_func(u'请最少选择一个买入策略')
            return

        # 卖出策略可以一个也没有
        sell_factors = list(self.sf.factor_dict.values())

        market = EMarketTargetType(self.market.value)
        self.cross_val.fit(buy_factors, sell_factors, cv=self.cv.value, market=market, )
