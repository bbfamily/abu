# -*- encoding:utf-8 -*-
from __future__ import print_function
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
import warnings

# noinspection PyUnresolvedReferences
import abu_local_env

import abupy
from abupy import AbuFactorBuyBreak
from abupy import AbuFactorSellBreak
from abupy import AbuFactorAtrNStop
from abupy import AbuFactorPreAtrNStop
from abupy import AbuFactorCloseAtrNStop
from abupy import AbuBenchmark
from abupy import AbuPickTimeWorker
from abupy import AbuCapital
from abupy import AbuKLManager
from abupy import ABuTradeProxy
from abupy import ABuTradeExecute
from abupy import ABuPickTimeExecute
from abupy import AbuMetricsBase
from abupy import ABuMarket
from abupy import AbuPickTimeMaster
from abupy import ABuRegUtil
from abupy import AbuPickRegressAngMinMax
from abupy import AbuPickStockWorker
from abupy import ABuPickStockExecute
from abupy import AbuPickStockPriceMinMax
from abupy import AbuPickStockMaster

warnings.filterwarnings('ignore')
sns.set_context(rc={'figure.figsize': (14, 7)})
# 使用沙盒数据，目的是和书中一样的数据环境
abupy.env.enable_example_env_ipython()


"""
    第八章 量化系统——开发

    abu量化系统github地址：https://github.com/bbfamily/abu (您的star是我的动力！)
    abu量化文档教程ipython notebook：https://github.com/bbfamily/abu/tree/master/abupy_lecture
"""


def sample_811():
    """
    8.1.1 买入因子的实现
    :return:
    """
    # buy_factors 60日向上突破，42日向上突破两个因子
    buy_factors = [{'xd': 60, 'class': AbuFactorBuyBreak},
                   {'xd': 42, 'class': AbuFactorBuyBreak}]
    benchmark = AbuBenchmark()
    capital = AbuCapital(1000000, benchmark)
    kl_pd_manager = AbuKLManager(benchmark, capital)
    # 获取TSLA的交易数据
    kl_pd = kl_pd_manager.get_pick_time_kl_pd('usTSLA')
    abu_worker = AbuPickTimeWorker(capital, kl_pd, benchmark, buy_factors, None)
    abu_worker.fit()

    orders_pd, action_pd, _ = ABuTradeProxy.trade_summary(abu_worker.orders, kl_pd, draw=True)

    ABuTradeExecute.apply_action_to_capital(capital, action_pd, kl_pd_manager)
    capital.capital_pd.capital_blance.plot()
    plt.show()


def sample_812():
    """
    8.1.2 卖出因子的实现
    :return:
    """
    # 120天向下突破为卖出信号
    sell_factor1 = {'xd': 120, 'class': AbuFactorSellBreak}
    # 趋势跟踪策略止盈要大于止损设置值，这里0.5，3.0
    sell_factor2 = {'stop_loss_n': 0.5, 'stop_win_n': 3.0, 'class': AbuFactorAtrNStop}
    # 暴跌止损卖出因子形成dict
    sell_factor3 = {'class': AbuFactorPreAtrNStop, 'pre_atr_n': 1.0}
    # 保护止盈卖出因子组成dict
    sell_factor4 = {'class': AbuFactorCloseAtrNStop, 'close_atr_n': 1.5}
    # 四个卖出因子同时生效，组成sell_factors
    sell_factors = [sell_factor1, sell_factor2, sell_factor3, sell_factor4]
    # buy_factors 60日向上突破，42日向上突破两个因子
    buy_factors = [{'xd': 60, 'class': AbuFactorBuyBreak},
                   {'xd': 42, 'class': AbuFactorBuyBreak}]
    benchmark = AbuBenchmark()

    capital = AbuCapital(1000000, benchmark)
    orders_pd, action_pd, _ = ABuPickTimeExecute.do_symbols_with_same_factors(
        ['usTSLA'], benchmark, buy_factors, sell_factors, capital, show=True)


def sample_813():
    """
    8.1.3 滑点买入卖出价格确定及策略实现
    :return:
    """
    from abupy import AbuSlippageBuyBase

    # 修改g_open_down_rate的值为0.02
    g_open_down_rate = 0.02

    # noinspection PyClassHasNoInit
    class AbuSlippageBuyMean2(AbuSlippageBuyBase):
        def fit_price(self):
            if (self.kl_pd_buy.open / self.kl_pd_buy.pre_close) < (
                        1 - g_open_down_rate):
                # 开盘下跌K_OPEN_DOWN_RATE以上，单子失效
                print(self.factor_name + 'open down threshold')
                return np.inf
            # 买入价格为当天均价
            self.buy_price = np.mean(
                [self.kl_pd_buy['high'], self.kl_pd_buy['low']])
            return self.buy_price

    # 只针对60使用AbuSlippageBuyMean2
    buy_factors2 = [{'slippage': AbuSlippageBuyMean2, 'xd': 60, 'class': AbuFactorBuyBreak},
                    {'xd': 42, 'class': AbuFactorBuyBreak}]

    sell_factor1 = {'xd': 120, 'class': AbuFactorSellBreak}
    sell_factor2 = {'stop_loss_n': 0.5, 'stop_win_n': 3.0, 'class': AbuFactorAtrNStop}
    sell_factor3 = {'class': AbuFactorPreAtrNStop, 'pre_atr_n': 1.0}
    sell_factor4 = {'class': AbuFactorCloseAtrNStop, 'close_atr_n': 1.5}
    sell_factors = [sell_factor1, sell_factor2, sell_factor3, sell_factor4]
    benchmark = AbuBenchmark()
    capital = AbuCapital(1000000, benchmark)
    orders_pd, action_pd, _ = ABuPickTimeExecute.do_symbols_with_same_factors(
        ['usTSLA'], benchmark, buy_factors2, sell_factors, capital, show=True)


def sample_814(show=True):
    """
    8.1.4 对多支股票进行择时
    :return:
    """

    sell_factor1 = {'xd': 120, 'class': AbuFactorSellBreak}
    sell_factor2 = {'stop_loss_n': 0.5, 'stop_win_n': 3.0, 'class': AbuFactorAtrNStop}
    sell_factor3 = {'class': AbuFactorPreAtrNStop, 'pre_atr_n': 1.0}
    sell_factor4 = {'class': AbuFactorCloseAtrNStop, 'close_atr_n': 1.5}
    sell_factors = [sell_factor1, sell_factor2, sell_factor3, sell_factor4]
    benchmark = AbuBenchmark()
    buy_factors = [{'xd': 60, 'class': AbuFactorBuyBreak},
                   {'xd': 42, 'class': AbuFactorBuyBreak}]

    choice_symbols = ['usTSLA', 'usNOAH', 'usSFUN', 'usBIDU', 'usAAPL', 'usGOOG', 'usWUBA', 'usVIPS']
    capital = AbuCapital(1000000, benchmark)
    orders_pd, action_pd, all_fit_symbols_cnt = ABuPickTimeExecute.do_symbols_with_same_factors(choice_symbols,
                                                                                                benchmark, buy_factors,
                                                                                                sell_factors, capital,
                                                                                                show=False)

    metrics = AbuMetricsBase(orders_pd, action_pd, capital, benchmark)
    metrics.fit_metrics()
    if show:
        print('orders_pd[:10]:\n', orders_pd[:10].filter(
            ['symbol', 'buy_price', 'buy_cnt', 'buy_factor', 'buy_pos', 'sell_date', 'sell_type_extra', 'sell_type',
             'profit']))
        print('action_pd[:10]:\n', action_pd[:10])
        metrics.plot_returns_cmp(only_show_returns=True)
    return metrics


def sample_815():
    """
    8.1.5 自定义仓位管理策略的实现
    :return:
    """
    metrics = sample_814(False)
    print('\nmetrics.gains_mean:{}, -metrics.losses_mean:{}'.format(metrics.gains_mean, -metrics.losses_mean))

    from abupy import AbuKellyPosition
    # 42d使用AbuKellyPosition，60d仍然使用默认仓位管理类
    buy_factors2 = [{'xd': 60, 'class': AbuFactorBuyBreak},
                    {'xd': 42, 'position': AbuKellyPosition, 'win_rate': metrics.win_rate,
                     'gains_mean': metrics.gains_mean, 'losses_mean': -metrics.losses_mean,
                     'class': AbuFactorBuyBreak}]

    sell_factor1 = {'xd': 120, 'class': AbuFactorSellBreak}
    sell_factor2 = {'stop_loss_n': 0.5, 'stop_win_n': 3.0, 'class': AbuFactorAtrNStop}
    sell_factor3 = {'class': AbuFactorPreAtrNStop, 'pre_atr_n': 1.0}
    sell_factor4 = {'class': AbuFactorCloseAtrNStop, 'close_atr_n': 1.5}
    sell_factors = [sell_factor1, sell_factor2, sell_factor3, sell_factor4]
    benchmark = AbuBenchmark()
    choice_symbols = ['usTSLA', 'usNOAH', 'usSFUN', 'usBIDU', 'usAAPL', 'usGOOG', 'usWUBA', 'usVIPS']
    capital = AbuCapital(1000000, benchmark)
    orders_pd, action_pd, all_fit_symbols_cnt = ABuPickTimeExecute.do_symbols_with_same_factors(choice_symbols,
                                                                                                benchmark, buy_factors2,
                                                                                                sell_factors, capital,
                                                                                                show=False)
    print(orders_pd[:10].filter(['symbol', 'buy_cnt', 'buy_factor', 'buy_pos']))


def sample_816():
    """
    8.1.6 多支股票使用不同的因子进行择时
    :return:
    """
    # 选定noah和sfun
    target_symbols = ['usSFUN', 'usNOAH']
    # 针对sfun只使用42d向上突破作为买入因子
    buy_factors_sfun = [{'xd': 42, 'class': AbuFactorBuyBreak}]
    # 针对sfun只使用60d向下突破作为卖出因子
    sell_factors_sfun = [{'xd': 60, 'class': AbuFactorSellBreak}]

    # 针对noah只使用21d向上突破作为买入因子
    buy_factors_noah = [{'xd': 21, 'class': AbuFactorBuyBreak}]
    # 针对noah只使用42d向下突破作为卖出因子
    sell_factors_noah = [{'xd': 42, 'class': AbuFactorSellBreak}]

    factor_dict = dict()
    # 构建SFUN独立的buy_factors，sell_factors的dict
    factor_dict['usSFUN'] = {'buy_factors': buy_factors_sfun, 'sell_factors': sell_factors_sfun}
    # 构建NOAH独立的buy_factors，sell_factors的dict
    factor_dict['usNOAH'] = {'buy_factors': buy_factors_noah, 'sell_factors': sell_factors_noah}
    # 初始化资金
    benchmark = AbuBenchmark()
    capital = AbuCapital(1000000, benchmark)
    # 使用do_symbols_with_diff_factors执行
    orders_pd, action_pd, all_fit_symbols = ABuPickTimeExecute.do_symbols_with_diff_factors(
        target_symbols, benchmark, factor_dict, capital)
    print('pd.crosstab(orders_pd.buy_factor, orders_pd.symbol):\n', pd.crosstab(orders_pd.buy_factor, orders_pd.symbol))


def sample_817():
    """
    8.1.7 使用并行来提升择时运行效率
    :return:
    """
    # 要关闭沙盒数据环境，因为沙盒里就那几个股票的历史数据, 下面要随机做50个股票
    from abupy import EMarketSourceType
    abupy.env.g_market_source = EMarketSourceType.E_MARKET_SOURCE_tx

    abupy.env.disable_example_env_ipython()

    # 关闭沙盒后，首先基准要从非沙盒环境换取，否则数据对不齐，无法正常运行
    benchmark = AbuBenchmark()
    # 当传入choice_symbols为None时代表对整个市场的所有股票进行回测
    # noinspection PyUnusedLocal
    choice_symbols = None
    # 顺序获取市场后300支股票
    # noinspection PyUnusedLocal
    choice_symbols = ABuMarket.all_symbol()[-50:]
    # 随机获取300支股票
    choice_symbols = ABuMarket.choice_symbols(50)
    capital = AbuCapital(1000000, benchmark)

    sell_factor1 = {'xd': 120, 'class': AbuFactorSellBreak}
    sell_factor2 = {'stop_loss_n': 0.5, 'stop_win_n': 3.0, 'class': AbuFactorAtrNStop}
    sell_factor3 = {'class': AbuFactorPreAtrNStop, 'pre_atr_n': 1.0}
    sell_factor4 = {'class': AbuFactorCloseAtrNStop, 'close_atr_n': 1.5}
    sell_factors = [sell_factor1, sell_factor2, sell_factor3, sell_factor4]
    buy_factors = [{'xd': 60, 'class': AbuFactorBuyBreak},
                   {'xd': 42, 'class': AbuFactorBuyBreak}]

    orders_pd, action_pd, _ = AbuPickTimeMaster.do_symbols_with_same_factors_process(
        choice_symbols, benchmark, buy_factors, sell_factors,
        capital)

    metrics = AbuMetricsBase(orders_pd, action_pd, capital, benchmark)
    metrics.fit_metrics()
    metrics.plot_returns_cmp(only_show_returns=True)

    abupy.env.enable_example_env_ipython()


"""
    注意所有选股结果等等与书中的结果不一致，因为要控制沙盒数据体积小于50mb， 所以沙盒数据有些symbol只有两年多一点，与原始环境不一致，
    直接达不到选股的min_xd，所以这里其实可以`abupy.env.disable_example_env_ipython()`关闭沙盒环境，直接上真实数据。
"""


def sample_821_1():
    """
    8.2.1_1 选股使用示例
    :return:
    """
    # 选股条件threshold_ang_min=0.0, 即要求股票走势为向上上升趋势
    stock_pickers = [{'class': AbuPickRegressAngMinMax,
                      'threshold_ang_min': 0.0, 'reversed': False}]

    # 从这几个股票里进行选股，只是为了演示方便
    # 一般的选股都会是数量比较多的情况比如全市场股票
    choice_symbols = ['usNOAH', 'usSFUN', 'usBIDU', 'usAAPL', 'usGOOG',
                      'usTSLA', 'usWUBA', 'usVIPS']
    benchmark = AbuBenchmark()
    capital = AbuCapital(1000000, benchmark)
    kl_pd_manager = AbuKLManager(benchmark, capital)
    stock_pick = AbuPickStockWorker(capital, benchmark, kl_pd_manager,
                                    choice_symbols=choice_symbols,
                                    stock_pickers=stock_pickers)
    stock_pick.fit()
    # 打印最后的选股结果
    print('stock_pick.choice_symbols:', stock_pick.choice_symbols)

    # 从kl_pd_manager缓存中获取选股走势数据，注意get_pick_stock_kl_pd为选股数据，get_pick_time_kl_pd为择时
    kl_pd_noah = kl_pd_manager.get_pick_stock_kl_pd('usNOAH')
    # 绘制并计算角度
    deg = ABuRegUtil.calc_regress_deg(kl_pd_noah.close)
    print('noah 选股周期内角度={}'.format(round(deg, 3)))


def sample_821_2():
    """
    8.2.1_2 ABuPickStockExecute
    :return:
    """
    stock_pickers = [{'class': AbuPickRegressAngMinMax,
                      'threshold_ang_min': 0.0, 'threshold_ang_max': 10.0,
                      'reversed': False}]

    choice_symbols = ['usNOAH', 'usSFUN', 'usBIDU', 'usAAPL', 'usGOOG',
                      'usTSLA', 'usWUBA', 'usVIPS']
    benchmark = AbuBenchmark()
    capital = AbuCapital(1000000, benchmark)
    kl_pd_manager = AbuKLManager(benchmark, capital)

    print('ABuPickStockExecute.do_pick_stock_work:\n', ABuPickStockExecute.do_pick_stock_work(choice_symbols, benchmark,
                                                                                              capital, stock_pickers))

    kl_pd_sfun = kl_pd_manager.get_pick_stock_kl_pd('usSFUN')
    print('sfun 选股周期内角度={}'.format(round(ABuRegUtil.calc_regress_deg(kl_pd_sfun.close), 3)))


def sample_821_3():
    """
    8.2.1_3 reversed
    :return:
    """
    # 和上面的代码唯一的区别就是reversed=True
    stock_pickers = [{'class': AbuPickRegressAngMinMax,
                      'threshold_ang_min': 0.0, 'threshold_ang_max': 10.0, 'reversed': True}]

    choice_symbols = ['usNOAH', 'usSFUN', 'usBIDU', 'usAAPL', 'usGOOG',
                      'usTSLA', 'usWUBA', 'usVIPS']
    benchmark = AbuBenchmark()
    capital = AbuCapital(1000000, benchmark)

    print('ABuPickStockExecute.do_pick_stock_work:\n',
          ABuPickStockExecute.do_pick_stock_work(choice_symbols, benchmark, capital, stock_pickers))


def sample_822():
    """
    8.2.2 多个选股因子并行执行
    :return:
    """
    # 选股list使用两个不同的选股因子组合，并行同时生效
    stock_pickers = [{'class': AbuPickRegressAngMinMax,
                      'threshold_ang_min': 0.0, 'reversed': False},
                     {'class': AbuPickStockPriceMinMax, 'threshold_price_min': 50.0,
                      'reversed': False}]

    choice_symbols = ['usNOAH', 'usSFUN', 'usBIDU', 'usAAPL', 'usGOOG',
                      'usTSLA', 'usWUBA', 'usVIPS']
    benchmark = AbuBenchmark()
    capital = AbuCapital(1000000, benchmark)

    print('ABuPickStockExecute.do_pick_stock_work:\n',
          ABuPickStockExecute.do_pick_stock_work(choice_symbols, benchmark, capital, stock_pickers))


def sample_823():
    """
    8.2.3 使用并行来提升回测运行效率
    :return:
    """
    from abupy import EMarketSourceType
    abupy.env.g_market_source = EMarketSourceType.E_MARKET_SOURCE_tx
    abupy.env.disable_example_env_ipython()

    benchmark = AbuBenchmark()
    capital = AbuCapital(1000000, benchmark)

    # 首先随抽取50支股票
    choice_symbols = ABuMarket.choice_symbols(50)
    # 股价在15-50之间
    stock_pickers = [
        {'class': AbuPickStockPriceMinMax, 'threshold_price_min': 15.0,
         'threshold_price_max': 50.0, 'reversed': False}]
    cs = AbuPickStockMaster.do_pick_stock_with_process(capital, benchmark,
                                                       stock_pickers,
                                                       choice_symbols)
    print('len(cs):', len(cs))
    print('cs:\n', cs)


if __name__ == "__main__":
    sample_811()
    # sample_812()
    # sample_813()
    # sample_814()
    # sample_815()
    # sample_816()
    # sample_817()

    # sample_821_1()
    # sample_821_2()
    # sample_821_3()
    # sample_822()
    # sample_823()
