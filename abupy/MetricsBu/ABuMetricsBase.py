# -*- encoding:utf-8 -*-
"""度量模块基础"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import functools
import logging

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from ..ExtBu.empyrical import stats
from ..CoreBu import ABuEnv
from ..CoreBu.ABuEnv import EMarketDataFetchMode
from ..CoreBu.ABuFixes import six
from ..UtilBu import ABuDateUtil
from ..UtilBu import ABuStatsUtil, ABuScalerUtil
from ..UtilBu.ABuDTUtil import warnings_filter
from ..TradeBu.ABuKLManager import AbuKLManager
from ..TradeBu.ABuCapital import AbuCapital
from ..TradeBu import ABuTradeExecute


__author__ = '阿布'
__weixin__ = 'abu_quant'


def valid_check(func):
    """检测度量的输入是否正常，非正常显示info，正常继续执行被装饰方法"""

    @functools.wraps(func)
    def wrapper(self, *args, **kwargs):
        if self.valid:
            return func(self, *args, **kwargs)
        else:
            logging.info('metrics input is invalid or zero order gen!')

    return wrapper


class AbuMetricsBase(object):
    """主要适配股票类型交易对象的回测结果度量"""

    @classmethod
    def show_general(cls, orders_pd, action_pd, capital, benchmark, returns_cmp=False,
                     only_info=False, only_show_returns=False, enable_stocks_full_rate_factor=False):
        """
        类方法，针对输入执行度量后执行主要度量可视化及度量结果信息输出
        :param orders_pd: 回测结果生成的交易订单构成的pd.DataFrame对象
        :param action_pd: 回测结果生成的交易行为构成的pd.DataFrame对象
        :param capital: 资金类AbuCapital实例化对象
        :param benchmark: 交易基准对象，AbuBenchmark实例对象
        :param returns_cmp: 是否只度量无资金管理的情况下总体情况
        :param only_info: 是否只显示文字度量结果，不显示图像
        :param only_show_returns: 透传plot_returns_cmp，默认False, True则只显示收益对比不显示其它可视化
        :param enable_stocks_full_rate_factor: 是否开启满仓乘数
        :return AbuMetricsBase实例化类型对象
        """
        metrics = cls(orders_pd, action_pd, capital, benchmark,
                      enable_stocks_full_rate_factor=enable_stocks_full_rate_factor)
        metrics.fit_metrics()
        if returns_cmp:
            metrics.plot_order_returns_cmp(only_info=only_info)
        else:
            metrics.plot_returns_cmp(only_info=only_info, only_show_returns=only_show_returns)
            if not only_show_returns:
                metrics.plot_sharp_volatility_cmp(only_info=only_info)
        return metrics

    def __init__(self, orders_pd, action_pd, capital, benchmark, enable_stocks_full_rate_factor=False):
        """
        :param orders_pd: 回测结果生成的交易订单构成的pd.DataFrame对象
        :param action_pd: 回测结果生成的交易行为构成的pd.DataFrame对象
        :param capital: 资金类AbuCapital实例化对象
        :param benchmark: 交易基准对象，AbuBenchmark实例对象
        :param enable_stocks_full_rate_factor: 是否开启满仓乘数
        """
        self.capital = capital
        self.orders_pd = orders_pd
        self.action_pd = action_pd
        self.benchmark = benchmark
        """
            满仓乘数，如果设置为True, 针对度量信息如收益等需要除self.stocks_full_rate_factor
        """
        self.enable_stocks_full_rate_factor = enable_stocks_full_rate_factor
        # 验证输入的回测数据是否可度量，便于valid_check装饰器工作
        self.valid = False
        if self.orders_pd is not None and self.capital is not None and 'capital_blance' in self.capital.capital_pd:
            self.valid = True
        # ipython notebook下使用logging.info
        self.log_func = logging.info if ABuEnv.g_is_ipython else print

    @valid_check
    def fit_metrics(self):
        """执行所有度量函数"""
        # TODO 根据ORDER数量大于一定阀值启动进度条
        # with AbuProgress(100, 0, label='metrics progress...') as pg:
        # pg.show(5)
        self._metrics_base_stats()
        # pg.show(50)
        self._metrics_sell_stats()
        # pg.show(80)
        self._metrics_action_stats()
        # pg.show(95)
        self._metrics_extend_stats()

    def fit_metrics_order(self):
        """对外接口，并非度量真实成交了的结果，只度量orders_pd，即不涉及资金的度量"""
        self._metrics_sell_stats()

    def _metrics_base_stats(self):
        """度量真实成交了的capital_pd，即涉及资金的度量"""
        # 平均资金利用率
        self.cash_utilization = 1 - (self.capital.capital_pd.cash_blance /
                                     self.capital.capital_pd.capital_blance).mean()

        # 默认不使用满仓乘数即stocks_full_rate_factor＝1
        self.stocks_full_rate_factor = 1
        if self.enable_stocks_full_rate_factor:
            # 计算满仓比例
            stocks_full_rate = (self.capital.capital_pd.stocks_blance / self.capital.capital_pd.capital_blance)
            # 避免除0
            stocks_full_rate[stocks_full_rate == 0] = 1
            # 倒数得到满仓乘数
            self.stocks_full_rate_factor = (1 / stocks_full_rate)

        # 收益数据
        self.benchmark_returns = np.round(self.benchmark.kl_pd.close.pct_change(), 3)
        # 如果enable_stocks_full_rate_factor 则 * self.stocks_full_rate_factor的意义为随时都是满仓
        self.algorithm_returns = np.round(self.capital.capital_pd['capital_blance'].pct_change(),
                                          3) * self.stocks_full_rate_factor

        # 收益cum数据
        # noinspection PyTypeChecker
        self.algorithm_cum_returns = stats.cum_returns(self.algorithm_returns)
        self.benchmark_cum_returns = stats.cum_returns(self.benchmark_returns)

        # 最后一日的cum return
        self.benchmark_period_returns = self.benchmark_cum_returns[-1]
        self.algorithm_period_returns = self.algorithm_cum_returns[-1]

        # 交易天数
        self.num_trading_days = len(self.benchmark_returns)

        # 年化收益
        self.algorithm_annualized_returns = \
            (ABuEnv.g_market_trade_year / self.num_trading_days) * self.algorithm_period_returns
        self.benchmark_annualized_returns = \
            (ABuEnv.g_market_trade_year / self.num_trading_days) * self.benchmark_period_returns

        # 策略平均收益
        # noinspection PyUnresolvedReferences
        self.mean_algorithm_returns = self.algorithm_returns.cumsum() / np.arange(1, self.num_trading_days + 1,
                                                                                  dtype=np.float64)
        # 波动率
        self.benchmark_volatility = stats.annual_volatility(self.benchmark_returns)
        # noinspection PyTypeChecker
        self.algorithm_volatility = stats.annual_volatility(self.algorithm_returns)

        # 夏普比率
        self.benchmark_sharpe = stats.sharpe_ratio(self.benchmark_returns)
        # noinspection PyTypeChecker
        self.algorithm_sharpe = stats.sharpe_ratio(self.algorithm_returns)

        # 信息比率
        # noinspection PyUnresolvedReferences
        self.information = stats.information_ratio(self.algorithm_returns.values, self.benchmark_returns.values)

        # 阿尔法, 贝塔
        # noinspection PyUnresolvedReferences
        self.alpha, self.beta = stats.alpha_beta_aligned(self.algorithm_returns.values, self.benchmark_returns.values)

        # 最大回撤
        # noinspection PyUnresolvedReferences
        self.max_drawdown = stats.max_drawdown(self.algorithm_returns.values)

    def _metrics_sell_stats(self):
        """并非度量真实成交了的结果，只度量orders_pd，即认为没有仓位管理和资金量限制前提下的表现"""

        # 根据order中的数据，计算盈利比例
        self.orders_pd['profit_cg'] = self.orders_pd['profit'] / (
            self.orders_pd['buy_price'] * self.orders_pd['buy_cnt'])
        # 为了显示方便及明显
        self.orders_pd['profit_cg_hunder'] = self.orders_pd['profit_cg'] * 100
        # 成交了的pd isin win or loss
        deal_pd = self.orders_pd[self.orders_pd['sell_type'].isin(['win', 'loss'])]
        # 卖出原因get_dummies进行离散化
        dumm_sell = pd.get_dummies(deal_pd.sell_type_extra)
        dumm_sell_t = dumm_sell.T
        # 为plot_sell_factors函数生成卖出生效因子分布
        self.dumm_sell_t_sum = dumm_sell_t.sum(axis=1)

        # 买入因子唯一名称get_dummies进行离散化
        dumm_buy = pd.get_dummies(deal_pd.buy_factor)
        dumm_buy = dumm_buy.T
        # 为plot_buy_factors函数生成卖出生效因子分布
        self.dumm_buy_t_sum = dumm_buy.sum(axis=1)

        self.orders_pd['buy_date'] = self.orders_pd['buy_date'].astype(int)
        self.orders_pd[self.orders_pd['result'] != 0]['sell_date'].astype(int, copy=False)
        # 因子的单子的持股时间长度计算
        self.orders_pd['keep_days'] = self.orders_pd.apply(lambda x:
                                                           ABuDateUtil.diff(x['buy_date'],
                                                                            ABuDateUtil.current_date_int()
                                                                            if x['result'] == 0 else x[
                                                                                'sell_date']),
                                                           axis=1)
        # 筛出已经成交了的单子
        self.order_has_ret = self.orders_pd[self.orders_pd['result'] != 0]

        # 筛出未成交的单子
        self.order_keep = self.orders_pd[self.orders_pd['result'] == 0]

        xt = self.order_has_ret.result.value_counts()
        # 计算胜率
        if xt.shape[0] == 2:
            win_rate = xt[1] / xt.sum()
        elif xt.shape[0] == 1:
            win_rate = xt.index[0]
        else:
            win_rate = 0
        self.win_rate = win_rate
        # 策略持股天数平均值
        self.keep_days_mean = self.orders_pd['keep_days'].mean()
        # 策略持股天数中位数
        self.keep_days_median = self.orders_pd['keep_days'].median()

        # 策略期望收益
        self.gains_mean = self.order_has_ret[self.order_has_ret['profit_cg'] > 0].profit_cg.mean()
        if np.isnan(self.gains_mean):
            self.gains_mean = 0.0
        # 策略期望亏损
        self.losses_mean = self.order_has_ret[self.order_has_ret['profit_cg'] < 0].profit_cg.mean()
        if np.isnan(self.losses_mean):
            self.losses_mean = 0.0

        # 忽略仓位控的前提下，即假设每一笔交易使用相同的资金，策略的总获利交易获利比例和
        profit_cg_win_sum = self.order_has_ret[self.order_has_ret['profit_cg'] > 0].profit.sum()
        # 忽略仓位控的前提下，即假设每一笔交易使用相同的资金，策略的总亏损交易亏损比例和
        profit_cg_loss_sum = self.order_has_ret[self.order_has_ret['profit_cg'] < 0].profit.sum()

        if profit_cg_win_sum * profit_cg_loss_sum == 0 and profit_cg_win_sum + profit_cg_loss_sum > 0:
            # 其中有一个是0的，要转换成一个最小统计单位计算盈亏比，否则不需要
            if profit_cg_win_sum == 0:
                profit_cg_win_sum = 0.01
            if profit_cg_loss_sum == 0:
                profit_cg_win_sum = 0.01

        #  忽略仓位控的前提下，计算盈亏比
        self.win_loss_profit_rate = 0 if profit_cg_loss_sum == 0 else -round(profit_cg_win_sum / profit_cg_loss_sum, 4)
        #  忽略仓位控的前提下，计算所有交易单的盈亏总会
        self.all_profit = self.order_has_ret['profit'].sum()

    def _metrics_action_stats(self):
        """度量真实成交了的action_pd 计算买入资金的分布平均性，及是否有良好的分布"""

        action_pd = self.action_pd
        # 只选生效的, 由于忽略非交易日, 大概有多出0.6的误差
        self.act_buy = action_pd[action_pd.action.isin(['buy']) & action_pd.deal.isin([True])]
        # drop重复的日期上的行为，只保留一个，cp_date形如下所示
        cp_date = self.act_buy['Date'].drop_duplicates()
        """
            cp_date
            0      20141024
            2      20141029
            20     20150127
            21     20150205
            23     20150213
            25     20150218
            31     20150310
            34     20150401
            36     20150409
            39     20150422
            41     20150423
            44     20150428
            58     20150609
            59     20150610
            63     20150624
            66     20150715
            67     20150717
        """
        dt_fmt = cp_date.apply(lambda order: ABuDateUtil.str_to_datetime(str(order), '%Y%m%d'))
        dt_fmt = dt_fmt.apply(lambda order: (order - dt_fmt.iloc[0]).days)
        # 前后两两生效交易时间相减
        self.diff_dt = dt_fmt - dt_fmt.shift(1)
        # 计算平均生效间隔时间
        self.effect_mean_day = self.diff_dt.mean()

        if self.act_buy.empty:
            self.act_buy['cost'] = 0
            self.cost_stats = 0
            self.buy_deal_rate = 0
        else:
            self.act_buy['cost'] = self.act_buy.apply(lambda order: order.Price * order.Cnt, axis=1)
            # 计算cost各种统计度量值
            self.cost_stats = ABuStatsUtil.stats_namedtuple(self.act_buy['cost'])

            buy_action_pd = action_pd[action_pd['action'] == 'buy']
            buy_action_pd_deal = buy_action_pd['deal']
            # 计算资金对应的成交比例
            self.buy_deal_rate = buy_action_pd_deal.sum() / buy_action_pd_deal.count()

    def _metrics_extend_stats(self):
        """子类可扩展的metrics方法，子类在此方法中可定义自己需要度量的值"""
        pass

    @valid_check
    @warnings_filter  # skip: statsmodels / nonparametric / kdetools.py:20
    def plot_order_returns_cmp(self, only_info=True):
        """非真实成交的度量，认为资金无限，无资金管理的情况下总体情况"""

        self.log_func('买入后卖出的交易数量:{}'.format(self.order_has_ret.shape[0]))
        self.log_func('买入后尚未卖出的交易数量:{}'.format(self.order_keep.shape[0]))
        self.log_func('胜率:{:.4f}%'.format(self.win_rate * 100))
        self.log_func('平均获利期望:{:.4f}%'.format(self.gains_mean * 100))
        self.log_func('平均亏损期望:{:.4f}%'.format(self.losses_mean * 100))
        self.log_func('盈亏比:{:.4f}'.format(self.win_loss_profit_rate))
        self.log_func('所有交易收益比例和:{:.4f} '.format(self.order_has_ret.profit_cg.sum()))
        self.log_func('所有交易总盈亏和:{:.4f} '.format(self.all_profit))

        if only_info:
            return
        # 无法与基准对比，只能表示取向
        self.order_has_ret.sort_values('buy_date')['profit_cg'].cumsum().plot(grid=True, title='profit_cg cumsum')
        plt.show()

    @valid_check
    def plot_returns_cmp(self, only_show_returns=False, only_info=False):
        """考虑资金情况下的度量，进行与benchmark的收益度量对比，收益趋势，资金变动可视化，以及其它度量信息"""

        self.log_func('买入后卖出的交易数量:{}'.format(self.order_has_ret.shape[0]))
        self.log_func('买入后尚未卖出的交易数量:{}'.format(self.order_keep.shape[0]))

        self.log_func('胜率:{:.4f}%'.format(self.win_rate * 100))

        self.log_func('平均获利期望:{:.4f}%'.format(self.gains_mean * 100))
        self.log_func('平均亏损期望:{:.4f}%'.format(self.losses_mean * 100))

        self.log_func('盈亏比:{:.4f}'.format(self.win_loss_profit_rate))

        self.log_func('策略收益: {:.4f}%'.format(self.algorithm_period_returns * 100))
        self.log_func('基准收益: {:.4f}%'.format(self.benchmark_period_returns * 100))
        self.log_func('策略年化收益: {:.4f}%'.format(self.algorithm_annualized_returns * 100))
        self.log_func('基准年化收益: {:.4f}%'.format(self.benchmark_annualized_returns * 100))

        self.log_func('策略买入成交比例:{:.4f}%'.format(self.buy_deal_rate * 100))
        self.log_func('策略资金利用率比例:{:.4f}%'.format(self.cash_utilization * 100))
        self.log_func('策略共执行{}个交易日'.format(self.num_trading_days))

        if only_info:
            return

        self.benchmark_cum_returns.plot()
        self.algorithm_cum_returns.plot()
        plt.legend(['benchmark returns', 'algorithm returns'], loc='best')
        plt.show()

        if only_show_returns:
            return
        sns.regplot(x=np.arange(0, len(self.algorithm_cum_returns)), y=self.algorithm_cum_returns.values)
        plt.show()
        sns.distplot(self.capital.capital_pd['capital_blance'], kde_kws={"lw": 3, "label": "capital blance kde"})
        plt.show()

    @valid_check
    def plot_sharp_volatility_cmp(self, only_info=False):
        """sharp，volatility的策略与基准对比可视化，以及alpha阿尔法，beta贝塔，Information信息比率等信息输出"""

        self.log_func('alpha阿尔法:{:.4f}'.format(self.alpha))
        self.log_func('beta贝塔:{:.4f}'.format(self.beta))
        self.log_func('Information信息比率:{:.4f}'.format(self.information))

        self.log_func('策略Sharpe夏普比率: {:.4f}'.format(self.algorithm_sharpe))
        self.log_func('基准Sharpe夏普比率: {:.4f}'.format(self.benchmark_sharpe))

        self.log_func('策略波动率Volatility: {:.4f}'.format(self.algorithm_volatility))
        self.log_func('基准波动率Volatility: {:.4f}'.format(self.benchmark_volatility))

        if only_info:
            return

        sharp_volatility = pd.DataFrame([[self.algorithm_sharpe, self.benchmark_sharpe],
                                         [self.algorithm_volatility, self.benchmark_volatility]])
        sharp_volatility.columns = ['algorithm', 'benchmark']
        sharp_volatility.index = ['sharpe', 'volatility']
        sharp_volatility.plot(kind='bar', alpha=0.5)
        _ = plt.setp(plt.gca().get_xticklabels(), rotation=30)

    @valid_check
    def plot_effect_mean_day(self):
        """可视化因子平均生效间隔时间"""

        self.log_func('因子平均生效间隔:{}'.format(self.effect_mean_day))

        ddvc = self.diff_dt.value_counts()
        ddvc_rt = ddvc / ddvc.sum()
        plt.figure(figsize=(6, 6))
        plt.axes([0.025, 0.025, 0.95, 0.95])
        x = ddvc_rt.values
        labels = ddvc_rt.index
        plt.pie(x, labels=labels, explode=x * 0.1)
        plt.title('factor diff effect day')
        plt.show()

    @valid_check
    def plot_action_buy_cost(self):
        """可视化开仓花费情况"""

        self.log_func('开仓花费情况: ')
        self.log_func(self.cost_stats)

        plt.title('action buy cost')
        bins = int(len(self.act_buy['cost']) / 10)
        bins = bins if bins > 0 else 10
        self.act_buy['cost'].plot(kind='hist', bins=bins)
        plt.show()

    @valid_check
    def plot_sell_factors(self):
        """可视化卖出生效因子分布"""
        self.log_func('卖出择时生效因子分布：')
        self.log_func(self.dumm_sell_t_sum)
        if self.dumm_sell_t_sum.shape[0] > 1:
            self.dumm_sell_t_sum.plot(kind='barh')
            plt.title('sell factors barh')
            plt.show()

    @valid_check
    def plot_buy_factors(self):
        """可视化买入生效因子分布"""
        self.log_func('买入择时生效因子分布：')
        self.log_func(self.dumm_buy_t_sum)

        if self.dumm_buy_t_sum.shape[0] > 1:
            self.dumm_buy_t_sum.plot(kind='barh')
            plt.title('buy factors barh')
            plt.show()

    @valid_check
    def plot_keep_days(self):
        """可视化策略持股天数"""

        self.log_func('策略持股天数平均数: {:.3f}'.format(self.keep_days_mean))
        self.log_func('策略持股天数中位数: {:.3f}'.format(self.keep_days_median))
        bins = int(self.orders_pd['keep_days'].shape[0] / 5)
        bins = bins if bins > 0 else 5
        self.orders_pd['keep_days'].plot(kind='hist', bins=bins)
        plt.show()

    @valid_check
    def plot_max_draw_down(self):
        """可视化最大回撤"""

        cb_earn = self.capital.capital_pd['capital_blance'] - self.capital.read_cash
        shift = cb_earn.shape[0]
        max_draw_down = {-1: -1}
        cap_pd_index = cb_earn.index.tolist()

        for sf in np.arange(1, shift):
            sub_val = cb_earn.iloc[sf]
            sf_val = cb_earn[:sf]
            sf_val = sf_val.drop_duplicates(keep='last')

            diff = sf_val.values - sub_val

            if diff.max() > list(six.itervalues(max_draw_down))[0]:
                st_ind = diff.argmax()
                st_ind = sf_val.index[st_ind]
                end_ind = cap_pd_index[sf]
                max_draw_down = {(st_ind, end_ind): diff.max()}

        down_rate = list(six.itervalues(max_draw_down))[0] / self.capital.capital_pd['capital_blance'].loc[
            list(six.iterkeys(max_draw_down))[0][0]]
        """
            截取开始交易部分
        """
        cb_earn = cb_earn.loc[cb_earn[cb_earn != 0].index[0]:]
        cb_earn.plot()
        plt.plot(list(six.iterkeys(max_draw_down))[0][0], cb_earn.loc[list(six.iterkeys(max_draw_down))[0][0]],
                 'ro', markersize=12,
                 markeredgewidth=1.5,
                 markerfacecolor='None', markeredgecolor='green')

        plt.plot(list(six.iterkeys(max_draw_down))[0][1], cb_earn.loc[list(six.iterkeys(max_draw_down))[0][1]],
                 'ro', markersize=12,
                 markeredgewidth=1.5,
                 markerfacecolor='None', markeredgecolor='red')

        plt.plot([list(six.iterkeys(max_draw_down))[0][0], list(six.iterkeys(max_draw_down))[0][1]],
                 [cb_earn.loc[list(six.iterkeys(max_draw_down))[0][0]],
                  cb_earn.loc[list(six.iterkeys(max_draw_down))[0][1]]], 'o-')
        plt.grid(True)
        plt.show()

        self.log_func('最大回撤: {:5f}'.format(down_rate))
        self.log_func('最大回测启始时间:{}, 结束时间{}, 共回测{:3f}'.format(
            ABuDateUtil.timestamp_to_str(list(six.iterkeys(max_draw_down))[0][0]),
            ABuDateUtil.timestamp_to_str(list(six.iterkeys(max_draw_down))[0][1]),
            list(six.itervalues(max_draw_down))[0]))

    @valid_check
    def transform_to_full_rate_factor(self, read_cash=-1, kl_pd_manager=None, n_process_kl=ABuEnv.g_cpu_cnt,
                                      show=True):
        if ABuEnv.g_data_fetch_mode != EMarketDataFetchMode.E_DATA_FETCH_FORCE_LOCAL:
            self.log_func('transform_to_full_rate_factor func must in E_DATA_FETCH_FORCE_LOCAL env!')
            return

        if not hasattr(self, 'full_rate_metrics'):
            if read_cash == -1:
                # 如果外部不设置资金数，设置一个亿为大资金数
                read_cash = 100000000

            target_symbols = list(set(self.orders_pd.symbol))
            # 重新以很大的资金初始化AbuCapital
            capital = AbuCapital(read_cash, self.benchmark,
                                 user_commission_dict=self.capital.commission.commission_dict)
            if kl_pd_manager is None:
                kl_pd_manager = AbuKLManager(self.benchmark, capital)
                # 一次性在主进程中执行多进程获取k线数据，全部放入kl_pd_manager中，内部启动n_process_kl个进程执行
                kl_pd_manager.batch_get_pick_time_kl_pd(target_symbols, n_process=n_process_kl)

            # noinspection PyUnresolvedReferences
            action_pd = self.action_pd.sort_values(['Date', 'action'])
            action_pd.index = np.arange(0, action_pd.shape[0])
            # 最后将所有的action作用在资金上，生成资金时序，及判断是否能买入
            ABuTradeExecute.apply_action_to_capital(capital, action_pd, kl_pd_manager)
            # 最终创建一个子AbuMetricsBase对象在内部，action_pd, capital使用新计算出来的，满仓乘数参数设置为True
            # noinspection PyAttributeOutsideInit
            self.full_rate_metrics = AbuMetricsBase(self.orders_pd, action_pd, capital, self.benchmark,
                                                    enable_stocks_full_rate_factor=True)
            self.full_rate_metrics.fit_metrics()
        if show:
            self.full_rate_metrics.plot_returns_cmp(only_show_returns=True)
        return self.full_rate_metrics


class MetricsDemo(AbuMetricsBase):
    """
        扩展自定义度量类示例

        eg:
            metrics = MetricsDemo(*abu_result_tuple)
            metrics.fit_metrics()
            metrics.plot_commission()
    """

    def _metrics_extend_stats(self):
        """
            子类可扩展的metrics方法，子类在此方法中可定义自己需要度量的值:
            本demo示例交易手续费和策略收益之间的度量对比
        """
        commission_df = self.capital.commission.commission_df
        commission_df['commission'] = commission_df.commission.astype(float)
        commission_df['cumsum'] = commission_df.commission.cumsum()
        """
            eg:
                type	date	symbol	commission	cumsum
            0	buy	20141024	usAAPL	19.04	19.04
            0	buy	20141024	usAAPL	19.04	38.08
            0	buy	20141029	usNOAH	92.17	130.25
            0	buy	20141029	usBIDU	7.81	138.06
            0	buy	20141029	usBIDU	7.81	145.87
            0	buy	20141029	usVIPS	60.95	206.82
        """
        # 讲date转换为index
        dates_pd = pd.to_datetime(commission_df.date)
        commission = pd.DataFrame(index=dates_pd)
        """
            eg: commission
            2014-10-24	19.04
            2014-10-24	38.08
            2014-10-29	130.25
            2014-10-29	138.06
            2014-10-29	145.87
            2014-10-29	206.82
            2014-11-03	265.82
            2014-11-11	360.73
        """
        commission['cum'] = commission_df['cumsum'].values
        self.commission_cum = commission['cum']
        self.commission_sum = self.commission_cum[-1]

    def plot_commission(self):
        """
            使用计算好的首先费cumsum序列和策略收益cumsum序列进行可视化对比
            可视化收益曲线和手续费曲线之前的关系
        """
        print('回测周期内手续费共: {:.2f}'.format(self.commission_sum))
        # 使用缩放scaler_xy将两条曲线缩放到同一个级别
        x, y = ABuScalerUtil.scaler_xy(self.commission_cum, self.algorithm_cum_returns, type_look='look_max',
                                       mean_how=True)
        x.plot(label='commission')
        y.plot(label='algorithm returns')
        plt.legend(loc=2)
        plt.show()
