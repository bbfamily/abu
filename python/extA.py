# -*- encoding:utf-8 -*-
from __future__ import print_function
import seaborn as sns
import warnings

# noinspection PyUnresolvedReferences
import abu_local_env
import abupy
from abupy import ABuSymbolPd
from abupy import EMarketSourceType
from abupy import EMarketDataFetchMode
from abupy import AbuFactorBuyBreak
from abupy import AbuFactorAtrNStop
from abupy import AbuFactorPreAtrNStop
from abupy import AbuFactorCloseAtrNStop
from abupy import AbuMetricsBase
from abupy import abu

warnings.filterwarnings('ignore')
sns.set_context(rc={'figure.figsize': (14, 7)})

# 设置选股因子，None为不使用选股因子
stock_pickers = None
# 买入因子依然延用向上突破因子
buy_factors = [{'xd': 60, 'class': AbuFactorBuyBreak},
               {'xd': 42, 'class': AbuFactorBuyBreak}]
# 卖出因子继续使用上一章使用的因子
sell_factors = [
    {'stop_loss_n': 1.0, 'stop_win_n': 3.0,
     'class': AbuFactorAtrNStop},
    {'class': AbuFactorPreAtrNStop, 'pre_atr_n': 1.5},
    {'class': AbuFactorCloseAtrNStop, 'close_atr_n': 1.5}
]

"""
    附录A 量化环境部署

    abu量化系统github地址：https://github.com/bbfamily/abu (您的star是我的动力！)
    abu量化文档教程ipython notebook：https://github.com/bbfamily/abu/tree/master/abupy_lecture

    * 本节建议对照阅读abu量化文档: 第19节 数据源
"""


def sample_a21():
    """
    A.2.1 数据模式的切换
    :return:
    """
    # 表A-1所示
    print(ABuSymbolPd.make_kl_df('601398').tail())

    # 局部使用enable_example_env_ipython，示例
    abupy.env.enable_example_env_ipython()
    # 如果本地有相应股票的缓存，可以使用如下代码强制使用本地缓存数据
    # abupy.env.g_data_fetch_mode = EMarketDataFetchMode.E_DATA_FETCH_FORCE_LOCAL

    # 设置初始资金数
    read_cash = 1000000

    # 择时股票池
    choice_symbols = ['usNOAH', 'usSFUN', 'usBIDU', 'usAAPL', 'usGOOG', 'usTSLA', 'usWUBA', 'usVIPS']
    # 使用run_loop_back运行策略
    abu_result_tuple, _ = abu.run_loop_back(read_cash,
                                            buy_factors, sell_factors, stock_pickers, choice_symbols=choice_symbols,
                                            n_folds=2)
    metrics = AbuMetricsBase(*abu_result_tuple)
    metrics.fit_metrics()
    metrics.plot_returns_cmp()

    # *****************************************************************************************************************
    # 切换数据源
    abupy.env.g_market_source = EMarketSourceType.E_MARKET_SOURCE_tx
    # 强制走网络数据源
    abupy.env.g_data_fetch_mode = EMarketDataFetchMode.E_DATA_FETCH_FORCE_NET
    # 择时股票池
    choice_symbols = ['601398', '600028', '601857', '601318', '600036', '000002', '600050', '600030']
    # 使用run_loop_back运行策略
    abu_result_tuple, _ = abu.run_loop_back(read_cash,
                                            buy_factors, sell_factors, stock_pickers, choice_symbols=choice_symbols,
                                            n_folds=2)

    metrics = AbuMetricsBase(*abu_result_tuple)
    metrics.fit_metrics()
    metrics.plot_returns_cmp()


"""
    A.2.2 目标市场的切换
    A.2.3 A股市场的回测示例

    * 相关内容请阅读abu量化文档：第8节 A股市场的回测， 第20节 A股全市场回测
"""

"""
    A.2.4 港股市场的回测示例

    * 相关内容请阅读abu量化文档：第9节 港股市场的回测
"""

if __name__ == "__main__":
    sample_a21()
    # sample_a23_1()
    # sample_a23_2()
    # sample_a23_2(from_cache=True)
    # sample_a24_1()
    # sample_a24_2()
    # sample_a24_2(from_cache=True)
