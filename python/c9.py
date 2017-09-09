# -*- encoding:utf-8 -*-
from __future__ import print_function
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import warnings

# noinspection PyUnresolvedReferences
import abu_local_env

import abupy

from abupy import AbuMetricsBase

from abupy import AbuFactorBuyBreak
from abupy import AbuFactorAtrNStop
from abupy import AbuFactorPreAtrNStop
from abupy import AbuFactorCloseAtrNStop
# run_loop_back等一些常用且最外层的方法定义在abu中
from abupy import abu

warnings.filterwarnings('ignore')
sns.set_context(rc={'figure.figsize': (14, 7)})
# 使用沙盒数据，目的是和书中一样的数据环境
abupy.env.enable_example_env_ipython()

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
    第九章 量化系统——度量与优化

    abu量化系统github地址：https://github.com/bbfamily/abu (您的star是我的动力！)
    abu量化文档教程ipython notebook：https://github.com/bbfamily/abu/tree/master/abupy_lecture
"""


def sample_91(show=True):
    """
    9.1 度量的基本使用方法
    :return:
    """
    # 设置初始资金数
    read_cash = 1000000
    # 择时股票池
    choice_symbols = ['usNOAH', 'usSFUN', 'usBIDU', 'usAAPL', 'usGOOG',
                      'usTSLA', 'usWUBA', 'usVIPS']
    # 使用run_loop_back运行策略
    abu_result_tuple, kl_pd_manager = abu.run_loop_back(read_cash,
                                                        buy_factors,
                                                        sell_factors,
                                                        stock_pickers,
                                                        choice_symbols=choice_symbols, n_folds=2)
    metrics = AbuMetricsBase(*abu_result_tuple)
    metrics.fit_metrics()
    if show:
        metrics.plot_returns_cmp()
    return metrics


def sample_922():
    """
    9.2.2 度量的可视化
    :return:
    """
    metrics = sample_91(show=False)

    metrics.plot_sharp_volatility_cmp()
    plt.show()

    def sharpe(rets, ann=252):
        return rets.mean() / rets.std() * np.sqrt(ann)

    print('策略sharpe值计算为＝{}'.format(sharpe(metrics.algorithm_returns)))

    metrics.plot_effect_mean_day()
    plt.show()

    metrics.plot_keep_days()
    plt.show()

    metrics.plot_sell_factors()
    plt.show()

    metrics.plot_max_draw_down()
    plt.show()


"""
    9.3 基于grid search寻找因子最优参数
"""

stop_win_range = np.arange(2.0, 4.5, 0.5)
stop_loss_range = np.arange(0.5, 2, 0.5)

sell_atr_nstop_factor_grid = {
    'class': [AbuFactorAtrNStop],
    'stop_loss_n': stop_loss_range,
    'stop_win_n': stop_win_range
}

close_atr_range = np.arange(1.0, 4.0, 0.5)
pre_atr_range = np.arange(1.0, 3.5, 0.5)

sell_atr_pre_factor_grid = {
    'class': [AbuFactorPreAtrNStop],
    'pre_atr_n': pre_atr_range
}

sell_atr_close_factor_grid = {
    'class': [AbuFactorCloseAtrNStop],
    'close_atr_n': close_atr_range
}


def sample_931():
    """
    9.3.1 参数取值范围
    :return:
    """
    print('止盈参数stop_win_n设置范围:{}'.format(stop_win_range))
    print('止损参数stop_loss_n设置范围:{}'.format(stop_loss_range))

    print('暴跌保护止损参数pre_atr_n设置范围:{}'.format(pre_atr_range))
    print('盈利保护止盈参数close_atr_n设置范围:{}'.format(close_atr_range))


def sample_932(show=True):
    """
    9.3.2 参数进行排列组合
    :return:
    """

    from abupy import ABuGridHelper

    sell_factors_product = ABuGridHelper.gen_factor_grid(
        ABuGridHelper.K_GEN_FACTOR_PARAMS_SELL,
        [sell_atr_nstop_factor_grid, sell_atr_pre_factor_grid, sell_atr_close_factor_grid])

    if show:
        print('卖出因子参数共有{}种组合方式'.format(len(sell_factors_product)))
        print('卖出因子组合0形式为{}'.format(sell_factors_product[0]))

    buy_bk_factor_grid1 = {
        'class': [AbuFactorBuyBreak],
        'xd': [42]
    }

    buy_bk_factor_grid2 = {
        'class': [AbuFactorBuyBreak],
        'xd': [60]
    }

    buy_factors_product = ABuGridHelper.gen_factor_grid(
        ABuGridHelper.K_GEN_FACTOR_PARAMS_BUY, [buy_bk_factor_grid1, buy_bk_factor_grid2])

    if show:
        print('买入因子参数共有{}种组合方式'.format(len(buy_factors_product)))
        print('买入因子组合形式为{}'.format(buy_factors_product))

    return sell_factors_product, buy_factors_product


def sample_933():
    """
    9.3.3 GridSearch寻找最优参数
    :return:
    """
    from abupy import GridSearch

    read_cash = 1000000
    choice_symbols = ['usNOAH', 'usSFUN', 'usBIDU', 'usAAPL', 'usGOOG',
                      'usTSLA', 'usWUBA', 'usVIPS']

    sell_factors_product, buy_factors_product = sample_932(show=False)

    grid_search = GridSearch(read_cash, choice_symbols,
                             buy_factors_product=buy_factors_product,
                             sell_factors_product=sell_factors_product)

    from abupy import ABuFileUtil
    """
        注意下面的运行耗时大约1小时多，如果所有cpu都用上的话，也可以设置n_jobs为 < cpu进程数，一边做其它的一边跑
    """
    # 运行GridSearch n_jobs=-1启动cpu个数的进程数
    scores, score_tuple_array = grid_search.fit(n_jobs=-1)

    """
        针对运行完成输出的score_tuple_array可以使用dump_pickle保存在本地，以方便修改其它验证效果。
    """
    ABuFileUtil.dump_pickle(score_tuple_array, '../gen/score_tuple_array')

    print('组合因子参数数量{}'.format(len(buy_factors_product) * len(sell_factors_product)))
    print('最终评分结果数量{}'.format(len(scores)))

    best_score_tuple_grid = grid_search.best_score_tuple_grid
    AbuMetricsBase.show_general(best_score_tuple_grid.orders_pd, best_score_tuple_grid.action_pd,
                                best_score_tuple_grid.capital, best_score_tuple_grid.benchmark)


def sample_934():
    """
    9.3.4 度量结果的评分
    :return:
    """
    from abupy import ABuFileUtil
    score_fn = '../gen/score_tuple_array'
    if not ABuFileUtil.file_exist(score_fn):
        print('../gen/score_tuple_array not exist! please execute sample_933 first!')
        return

    """
        直接读取本地序列化文件
    """
    score_tuple_array = ABuFileUtil.load_pickle(score_fn)
    from abupy import WrsmScorer
    # 实例化一个评分类WrsmScorer，它的参数为之前GridSearch返回的score_tuple_array对象
    scorer = WrsmScorer(score_tuple_array)
    print('scorer.score_pd.tail():\n', scorer.score_pd.tail())

    # score_tuple_array[658]与grid_search.best_score_tuple_grid是一致的
    sfs = scorer.fit_score()
    # 打印前15个高分组合
    print('sfs[::-1][:15]:\n', sfs[::-1][:15])


def sample_935_1():
    """
    9.3.5_1 不同权重的评分: 只考虑投资回报来评分
    :return:
    """
    from abupy import ABuFileUtil
    score_fn = '../gen/score_tuple_array'
    if not ABuFileUtil.file_exist(score_fn):
        print('../gen/score_tuple_array not exist! please execute sample_933 first!')
        return

    """
        直接读取本地序列化文件
    """
    score_tuple_array = ABuFileUtil.load_pickle(score_fn)

    from abupy import WrsmScorer
    # 实例化WrsmScorer，参数weights，只有第二项为1，其他都是0，
    # 代表只考虑投资回报来评分
    scorer = WrsmScorer(score_tuple_array, weights=[0, 1, 0, 0])
    # 返回排序后的队列
    scorer_returns_max = scorer.fit_score()
    # 因为是倒序排序，所以index最后一个为最优参数
    best_score_tuple_grid = score_tuple_array[scorer_returns_max.index[-1]]
    # 由于篇幅，最优结果只打印文字信息
    AbuMetricsBase.show_general(best_score_tuple_grid.orders_pd,
                                best_score_tuple_grid.action_pd,
                                best_score_tuple_grid.capital,
                                best_score_tuple_grid.benchmark,
                                only_info=True)

    # 最后打印出只考虑投资回报下最优结果使用的买入策略和卖出策略
    print('best_score_tuple_grid.buy_factors, best_score_tuple_grid.sell_factors:\n', best_score_tuple_grid.buy_factors,
          best_score_tuple_grid.sell_factors)


def sample_935_2():
    """
    9.3.5_2 不同权重的评分: 只考虑胜率
    :return:
    """
    from abupy import ABuFileUtil
    score_fn = '../gen/score_tuple_array'
    if not ABuFileUtil.file_exist(score_fn):
        print('../gen/score_tuple_array not exist! please execute sample_933 first!')
        return

    """
        直接读取本地序列化文件
    """
    score_tuple_array = ABuFileUtil.load_pickle(score_fn)

    from abupy import WrsmScorer
    # 只有第一项为1，其他都是0代表只考虑胜率来评分
    scorer = WrsmScorer(score_tuple_array, weights=[1, 0, 0, 0])
    # 返回按照评分排序后的队列
    scorer_returns_max = scorer.fit_score()
    # index[-1]为最优参数序号
    best_score_tuple_grid = score_tuple_array[scorer_returns_max.index[-1]]
    AbuMetricsBase.show_general(best_score_tuple_grid.orders_pd,
                                best_score_tuple_grid.action_pd,
                                best_score_tuple_grid.capital,
                                best_score_tuple_grid.benchmark,
                                only_info=False)

    # 最后打印出只考虑胜率下最优结果使用的买入策略和卖出策略
    print('best_score_tuple_grid.buy_factors, best_score_tuple_grid.sell_factors:\n', best_score_tuple_grid.buy_factors,
          best_score_tuple_grid.sell_factors)


"""
    9.4 资金限制对度量的影响

    如下内容不能使用沙盒环境, 建议对照阅读：
        abu量化文档－第十九节 数据源
        第20节 美股交易UMP决策
"""


def sample_94_1():
    """
    9.4_1 下载市场中所有股票的6年数据,
    如果没有运行过abu量化文档－第十九节 数据源：中使用腾讯数据源进行数据更新，需要运行
    如果运行过就不要重复运行了：
    """
    from abupy import EMarketTargetType, EMarketSourceType, EDataCacheType

    # 关闭沙盒数据环境
    abupy.env.disable_example_env_ipython()
    abupy.env.g_market_source = EMarketSourceType.E_MARKET_SOURCE_tx
    abupy.env.g_data_cache_type = EDataCacheType.E_DATA_CACHE_CSV
    # 首选这里预下载市场中所有股票的6年数据(做5年回测，需要预先下载6年数据)
    abu.run_kl_update(start='2011-08-08', end='2017-08-08', market=EMarketTargetType.E_MARKET_TARGET_US)


def sample_94_2(from_cache=False):
    """
    9.4_2 使用切割训练集测试集模式，且生成交易特征，回测训练集交易数据, mac pro顶配大概下面跑了4个小时
    :return:
    """
    # 关闭沙盒数据环境
    abupy.env.disable_example_env_ipython()
    from abupy import EMarketDataFetchMode
    # 因为sample_94_1下载了预先数据，使用缓存，设置E_DATA_FETCH_FORCE_LOCAL
    abupy.env.g_data_fetch_mode = EMarketDataFetchMode.E_DATA_FETCH_FORCE_LOCAL

    # 回测生成买入时刻特征
    abupy.env.g_enable_ml_feature = True
    # 回测将symbols切割分为训练集数据和测试集数据
    abupy.env.g_enable_train_test_split = True
    # 下面设置回测时切割训练集，测试集使用的切割比例参数，默认为10，即切割为10份，9份做为训练，1份做为测试，
    # 由于美股股票数量多，所以切割分为4份，3份做为训练集，1份做为测试集
    abupy.env.g_split_tt_n_folds = 4

    from abupy import EStoreAbu
    if from_cache:
        abu_result_tuple = \
            abu.load_abu_result_tuple(n_folds=5, store_type=EStoreAbu.E_STORE_CUSTOM_NAME,
                                      custom_name='train_us')
    else:
        # 初始化资金200万，资金管理依然使用默认atr
        read_cash = 5000000
        # 每笔交易的买入基数资金设置为万分之15
        abupy.beta.atr.g_atr_pos_base = 0.0015
        # 使用run_loop_back运行策略，因子使用和之前一样，
        # choice_symbols=None为全市场回测，5年历史数据回测
        # 不同电脑运行速度差异大，mac pro顶配大概下面跑了4小时
        # choice_symbols=None为全市场回测，5年历史数据回测
        abu_result_tuple, _ = abu.run_loop_back(read_cash,
                                                buy_factors, sell_factors,
                                                stock_pickers,
                                                choice_symbols=None,
                                                start='2012-08-08', end='2017-08-08')
        # 把运行的结果保存在本地，以便之后分析回测使用，保存回测结果数据代码如下所示
        abu.store_abu_result_tuple(abu_result_tuple, n_folds=5, store_type=EStoreAbu.E_STORE_CUSTOM_NAME,
                                   custom_name='train_us')

    print('abu_result_tuple.action_pd.deal.value_counts():\n', abu_result_tuple.action_pd.deal.value_counts())

    metrics = AbuMetricsBase(*abu_result_tuple)
    metrics.fit_metrics()
    metrics.plot_returns_cmp(only_show_returns=True)


def sample_94_3(from_cache=False, show=True):
    """
    9.4_3 使用切割好的测试数据集快，mac pro顶配大概下面跑了半个小时
    :return:
    """
    # 关闭沙盒数据环境
    abupy.env.disable_example_env_ipython()
    from abupy import EMarketDataFetchMode
    # 因为sample_94_1下载了预先数据，使用缓存，设置E_DATA_FETCH_FORCE_LOCAL
    abupy.env.g_data_fetch_mode = EMarketDataFetchMode.E_DATA_FETCH_FORCE_LOCAL

    abupy.env.g_enable_train_test_split = False
    # 使用切割好的测试数据
    abupy.env.g_enable_last_split_test = True
    # 回测生成买入时刻特征
    abupy.env.g_enable_ml_feature = True

    from abupy import EStoreAbu
    if from_cache:
        abu_result_tuple_test = \
            abu.load_abu_result_tuple(n_folds=5, store_type=EStoreAbu.E_STORE_CUSTOM_NAME,
                                      custom_name='test_us')
    else:
        read_cash = 5000000
        abupy.beta.atr.g_atr_pos_base = 0.007
        choice_symbols = None
        abu_result_tuple_test, kl_pd_manager_test = abu.run_loop_back(read_cash,
                                                                      buy_factors, sell_factors, stock_pickers,
                                                                      choice_symbols=choice_symbols, start='2012-08-08',
                                                                      end='2017-08-08')
        abu.store_abu_result_tuple(abu_result_tuple_test, n_folds=5, store_type=EStoreAbu.E_STORE_CUSTOM_NAME,
                                   custom_name='test_us')

    print('abu_result_tuple_test.action_pd.deal.value_counts():\n', abu_result_tuple_test.action_pd.deal.value_counts())

    metrics = AbuMetricsBase(*abu_result_tuple_test)
    metrics.fit_metrics()
    if show:
        metrics.plot_returns_cmp(only_show_returns=True)
    return metrics


def sample_94_4(from_cache=False):
    """
    满仓乘数
    9.4_4 《量化交易之路》中通过把初始资金扩大到非常大，但是每笔交易的买入基数却不增高，来使交易全部都成交，
    再使用满仓乘数的示例，由于需要再次进行全市场回测，比较耗时。

    下面直接示例通过AbuMetricsBase中的transform_to_full_rate_factor接口将之前的回测结果转换为使用大初始资金回测的结果
    :return:
    """
    metrics_test = sample_94_3(from_cache=True, show=False)

    from abupy import EStoreAbu
    if from_cache:
        test_us_fr = abu.load_abu_result_tuple(n_folds=5, store_type=EStoreAbu.E_STORE_CUSTOM_NAME,
                                               custom_name='test_us_full_rate')
        # 本地读取后使用AbuMetricsBase构造度量对象，参数enable_stocks_full_rate_factor=True, 即使用满仓乘数
        test_frm = AbuMetricsBase(test_us_fr.orders_pd, test_us_fr.action_pd, test_us_fr.capital, test_us_fr.benchmark,
                                  enable_stocks_full_rate_factor=True)
        test_frm.fit_metrics()
    else:
        test_frm = metrics_test.transform_to_full_rate_factor(n_process_kl=4, show=False)
        # 转换后保存起来，下次直接读取，不用再转换了
        from abupy import AbuResultTuple
        test_us_fr = AbuResultTuple(test_frm.orders_pd, test_frm.action_pd, test_frm.capital, test_frm.benchmark)
        abu.store_abu_result_tuple(test_us_fr, n_folds=5, store_type=EStoreAbu.E_STORE_CUSTOM_NAME,
                                   custom_name='test_us_full_rate')

    """
        使用test_frm进行度量结果可以看到所有交易都顺利成交了，策略买入成交比例:100.0000%，但资金利用率显然过低，
        它导致基准收益曲线和策略收益曲线不在一个量级上，无法有效的进行对比
    """
    AbuMetricsBase.show_general(test_frm.orders_pd,
                                test_frm.action_pd, test_frm.capital, test_frm.benchmark, only_show_returns=True)
    """转换出来的test_frm即是一个使用满仓乘数的度量对象，下面使用test_frm直接进行满仓度量即可"""
    print(type(test_frm))
    test_frm.plot_returns_cmp(only_show_returns=True)

    # 如果不需要与基准进行对比，最简单的方式是使用plot_order_returns_cmp
    metrics_test.plot_order_returns_cmp()

"""
    其它市场的回测, A股市场回测全局设置

    请阅读abu量化文档相关章节
"""

if __name__ == "__main__":
    sample_91()
    # sample_922()
    # sample_931()
    # sample_932()
    # 耗时操作
    # sample_933()
    # sample_934()
    # sample_935_1()
    # sample_935_2()
    # sample_94_1()
    # sample_94_2()
    # sample_94_2(from_cache=True)
    # sample_94_3()
    # sample_94_3(from_cache=True)
    # sample_94_4()
    # sample_94_4(from_cache=True)
