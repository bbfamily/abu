# -*- encoding:utf-8 -*-
"""期货度量模块"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from ..CoreBu import ABuEnv
from ..ExtBu.empyrical import stats
from ..MetricsBu.ABuMetricsBase import AbuMetricsBase, valid_check
from ..UtilBu.ABuDTUtil import warnings_filter

__author__ = '阿布'
__weixin__ = 'abu_quant'


class AbuMetricsFutures(AbuMetricsBase):
    """期货度量类，主要区别在于不涉及benchmark"""

    def _metrics_base_stats(self):
        """度量真实成交了的capital_pd，即涉及资金的度量，期货相关不涉及benchmark"""
        # 平均资金利用率
        self.cash_utilization = 1 - (self.capital.capital_pd.cash_blance /
                                     self.capital.capital_pd.capital_blance).mean()
        self.algorithm_returns = np.round(self.capital.capital_pd['capital_blance'].pct_change(), 3)

        # 收益cum数据
        # noinspection PyTypeChecker
        self.algorithm_cum_returns = stats.cum_returns(self.algorithm_returns)

        # 最后一日的cum return
        self.algorithm_period_returns = self.algorithm_cum_returns[-1]

        # 交易天数
        self.num_trading_days = len(self.algorithm_cum_returns)

        # 年化收益
        self.algorithm_annualized_returns = \
            (ABuEnv.g_market_trade_year / self.num_trading_days) * self.algorithm_period_returns

        # noinspection PyUnresolvedReferences
        self.mean_algorithm_returns = self.algorithm_returns.cumsum() / np.arange(1, self.num_trading_days + 1,
                                                                                  dtype=np.float64)
        # noinspection PyTypeChecker
        self.algorithm_volatility = stats.annual_volatility(self.algorithm_returns)
        # noinspection PyTypeChecker
        self.algorithm_sharpe = stats.sharpe_ratio(self.algorithm_returns)
        # 最大回撤
        # noinspection PyUnresolvedReferences
        self.max_drawdown = stats.max_drawdown(self.algorithm_returns.values)

    @valid_check
    @warnings_filter  # skip: statsmodels / nonparametric / kdetools.py:20
    def plot_returns_cmp(self, only_show_returns=False, only_info=False):
        """考虑资金情况下的度量，进行与benchmark的收益度量对比，收益趋势，资金变动可视化，以及其它度量信息，不涉及benchmark"""

        self.log_func('买入后卖出的交易数量:{}'.format(self.order_has_ret.shape[0]))
        self.log_func('胜率:{:.4f}%'.format(self.win_rate * 100))

        self.log_func('平均获利期望:{:.4f}%'.format(self.gains_mean * 100))
        self.log_func('平均亏损期望:{:.4f}%'.format(self.losses_mean * 100))

        self.log_func('盈亏比:{:.4f}'.format(self.win_loss_profit_rate))

        self.log_func('策略收益: {:.4f}%'.format(self.algorithm_period_returns * 100))
        self.log_func('策略年化收益: {:.4f}%'.format(self.algorithm_annualized_returns * 100))

        self.log_func('策略买入成交比例:{:.4f}%'.format(self.buy_deal_rate * 100))
        self.log_func('策略资金利用率比例:{:.4f}%'.format(self.cash_utilization * 100))
        self.log_func('策略共执行{}个交易日'.format(self.num_trading_days))

        if only_info:
            return

        self.algorithm_cum_returns.plot()
        plt.legend(['algorithm returns'], loc='best')
        plt.show()

        if only_show_returns:
            return
        sns.regplot(x=np.arange(0, len(self.algorithm_cum_returns)), y=self.algorithm_cum_returns.values)
        plt.show()
        sns.distplot(self.capital.capital_pd['capital_blance'], kde_kws={"lw": 3, "label": "capital blance kde"})
        plt.show()

    @valid_check
    def plot_sharp_volatility_cmp(self, only_info=False):
        """sharp，volatility信息输出"""

        self.log_func('策略Sharpe夏普比率: {:.4f}'.format(self.algorithm_sharpe))
        self.log_func('策略波动率Volatility: {:.4f}'.format(self.algorithm_volatility))
