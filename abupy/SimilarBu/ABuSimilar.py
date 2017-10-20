# -*- encoding:utf-8 -*-
"""
    相关系数相似应用模块
"""

from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

import functools
import logging
import math
import operator
import os

import pandas as pd

from . import ABuCorrcoef
from . import ABuSimilarDrawing
from .ABuCorrcoef import ECoreCorrType
from ..TradeBu import AbuBenchmark
from ..CoreBu import ABuEnv
from ..CoreBu.ABuParallel import delayed, Parallel
from ..CoreBu.ABuEnv import EMarketDataSplitMode, EMarketTargetType
from ..MarketBu import ABuSymbolPd
from ..MarketBu.ABuMarket import split_k_market, all_symbol
from ..MarketBu.ABuSymbol import IndexSymbol, Symbol
from ..UtilBu.ABuDTUtil import consume_time
from ..UtilBu.ABuProgress import do_clear_output
from ..CoreBu.ABuEnvProcess import add_process_env_sig, AbuEnvProcess
# noinspection PyUnresolvedReferences
from ..CoreBu.ABuFixes import xrange
from ..UtilBu import ABuProgress

"""进行相似度数据收集并行进程数，IO操作偏多，所以分配多个，默认=cpu个数＊2, windows还是..."""
g_process_panel_cnt = ABuEnv.g_cpu_cnt * 2 if ABuEnv.g_is_mac_os else ABuEnv.g_cpu_cnt


def from_local(func):
    """
    现所有相似度应用默认为from_local模式，即需要在有数据的情况下做相似度应用

    为进行相似度数据收集的函数装饰，作用是忽略env中的数据获取模式，改变数据获取模式，
    只使用本地数据模式进行数据收集，完成整个任务后，再恢复之前的数据获取模式
    :param func: 进行相似度应用且有数据收集行为的函数
    """

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        # 临时保存env设置中的g_data_fetch_mode
        fetch_mode = ABuEnv.g_data_fetch_mode
        # 设置数据获取模式为强制本地缓存模式
        ABuEnv.g_data_fetch_mode = ABuEnv.EMarketDataFetchMode.E_DATA_FETCH_FORCE_LOCAL
        if fetch_mode != ABuEnv.EMarketDataFetchMode.E_DATA_FETCH_FORCE_LOCAL:
            # 如果原有设置不是强制本地缓存模式，warning提示
            logging.warning('data from local. run ABu.run_kl_update if you want to get the latest data.')
        result = func(*args, **kwargs)
        # 恢复之前的g_data_fetch_mode
        ABuEnv.g_data_fetch_mode = fetch_mode
        return result

    return wrapper


def from_net(func):
    """
    为进行相似度数据收集的函数装饰，作用是忽略env中的数据获取模式，改变数据获取模式，
    只使用网络数据模式进行数据收集，完成整个任务后，再恢复之前的数据获取模式
    :param func: 进行相似度应用且有数据收集行为的函数
    """

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        # 临时保存env设置中的g_data_fetch_mode
        fetch_mode = ABuEnv.g_data_fetch_mode
        # 设置数据获取模式为强制网络模式
        ABuEnv.g_data_fetch_mode = ABuEnv.EMarketDataFetchMode.E_DATA_FETCH_FORCE_NET
        if fetch_mode != ABuEnv.EMarketDataFetchMode.E_DATA_FETCH_FORCE_NET:
            # 如果原有设置不是强制网络模式，warning提示
            logging.warning('data from net!!!')
        result = func(*args, **kwargs)
        # 恢复之前的g_data_fetch_mode
        ABuEnv.g_data_fetch_mode = fetch_mode
        return result

    return wrapper


@from_local
def _find_similar(symbol, cmp_cnt=None, n_folds=2, start=None, end=None, show_cnt=None, rolling=False,
                  show=True, corr_type=ECoreCorrType.E_CORE_TYPE_PEARS):
    """
    被from_local装饰器装饰 即强制走本地数据，获取全市场symbol涨跌幅度pd.DataFrame对象，
    使用symbol涨跌幅度与全市场symbol涨跌幅度进行相关对比，可视化结果及信息
    :param symbol: 外部指定目标symbol，str对象
    :param cmp_cnt: 相关性对比多少个交易日，int，可选参数
    :param n_folds: 相关性对比n_folds年，int，可选参数
    :param start: 请求的开始日期str对象，可选参数
    :param end: 请求的结束日期str对象，可选参数
    :param show_cnt: 最终结果展示以及可视化相似度个数
    :param rolling: 是否使用时间加权相关计算，与corr_type=ECoreCorrType.E_CORE_TYPE_ROLLING一样，单独拿出来了
    :param show: 是否可视化最终top最相关的股票
    :param corr_type: ECoreCorrType对象，暂时支持皮尔逊，斯皮尔曼，＋－符号相关系数，移动时间加权相关系数
    """
    if isinstance(symbol, Symbol):
        # 如果传递的时Symbol对象，取value
        symbol = symbol.value
    # 获取全市场symbol涨跌幅度pd.DataFrame对象
    market_change_df = _all_market_cg(symbol, cmp_cnt=cmp_cnt, n_folds=n_folds, start=start, end=end)
    if market_change_df is None:
        logging.info('{} data is miss, please update data first!'.format(symbol))
        return
    # 重新赋予标尺实际的交易日数量
    cmp_cnt = market_change_df[symbol].shape[0]
    # symbol涨跌幅度df数据
    benchmark_df = market_change_df[symbol]
    # 清一下输出，太乱
    do_clear_output()
    # 开始使用symbol涨跌幅度与全市场symbol涨跌幅度进行相关对比，可视化结果及信息
    sorted_corr = _handle_market_change_df(market_change_df, cmp_cnt, benchmark_df, show_cnt,
                                           corr_type, rolling, show)
    return sorted_corr


def find_similar_with_se(symbol, start, end, show_cnt=10, rolling=False, show=True,
                         corr_type=ECoreCorrType.E_CORE_TYPE_PEARS):
    """
    固定参数使用start, end参数提供时间范围规则，套接_find_similar，为_find_similar提供时间范围规则
    :param symbol: 外部指定目标symbol，str对象
    :param start: 请求的开始日期str对象
    :param end: 请求的结束日期str对象
    :param show_cnt: 最终结果展示以及可视化相似度个数
    :param rolling: 是否使用时间加权相关计算，与corr_type=ECoreCorrType.E_CORE_TYPE_ROLLING一样，单独拿出来了
    :param show: 是否可视化最终top最相关的股票
    :param corr_type: ECoreCorrType对象，暂时支持皮尔逊，斯皮尔曼，＋－符号相关系数，移动时间加权相关系数
    :return:
    """
    return _find_similar(symbol, start=start, end=end, show_cnt=show_cnt, rolling=rolling, show=show,
                         corr_type=corr_type)


def find_similar_with_folds(symbol, n_folds=2, show_cnt=10, rolling=False, show=True,
                            corr_type=ECoreCorrType.E_CORE_TYPE_PEARS):
    """
    固定参数使用n_folds参数提供时间范围规则，套接_find_similar，为_find_similar提供时间范围规则
    :param symbol: 外部指定目标symbol，str对象
    :param n_folds: 相关性对比n_folds年，int
    :param show_cnt: 最终结果展示以及可视化相似度个数
    :param rolling: 是否使用时间加权相关计算，与corr_type=ECoreCorrType.E_CORE_TYPE_ROLLING一样，单独拿出来了
    :param show: 是否可视化最终top最相关的股票
    :param corr_type: ECoreCorrType对象，暂时支持皮尔逊，斯皮尔曼，＋－符号相关系数，移动时间加权相关系数
    :return:
    """
    return _find_similar(symbol, n_folds=n_folds, show_cnt=show_cnt, rolling=rolling, show=show,
                         corr_type=corr_type)


def find_similar_with_cnt(symbol, cmp_cnt=60, show_cnt=10, rolling=False, show=True,
                          corr_type=ECoreCorrType.E_CORE_TYPE_PEARS):
    """
    固定参数使用cmp_cnt参数提供时间范围规则，套接_find_similar，为_find_similar提供时间范围规则
    :param symbol: 外部指定目标symbol，str对象
    :param cmp_cnt: 相关性对比多少个交易日，int
    :param show_cnt: 最终结果展示以及可视化相似度个数
    :param rolling: 是否使用时间加权相关计算，与corr_type=ECoreCorrType.E_CORE_TYPE_ROLLING一样，单独拿出来了
    :param show: 是否可视化最终top最相关的股票
    :param corr_type: ECoreCorrType对象，暂时支持皮尔逊，斯皮尔曼，＋－符号相关系数，移动时间加权相关系数
    :return:
    """
    return _find_similar(symbol, cmp_cnt=cmp_cnt, show_cnt=show_cnt, rolling=rolling, show=show,
                         corr_type=corr_type)


@add_process_env_sig
def _make_symbols_cg_df(symbols, benchmark):
    """
    相关性金融数据收集，子进程委托函数，子进程通过make_kl_df完成主进程委托的symbols个
    金融数据收集工作，最终返回所有金融时间序列涨跌幅度pd.DataFrame对象
    :param symbols: 可迭代symbols序列，序列中的元素为str对象
    :param benchmark: 进行数据收集使用的标尺对象，数据时间范围确定使用，AbuBenchmark实例对象
    :return: 所有金融时间序列涨跌幅度pd.DataFrame对象
    """

    # 子进程金融数据收集工作, 由于本事是在子进程内工作，所以不再make_kl_df中使用parallel模式，上层进行多任务分配及任务数确定
    panel = ABuSymbolPd.make_kl_df(symbols, data_mode=EMarketDataSplitMode.E_DATA_SPLIT_UNDO, benchmark=benchmark,
                                   show_progress=True)

    if panel is None or panel.empty:
        logging.info('pid {} panel is None'.format(os.getpid()))
        return None
    # 转换panel轴方向，即可方便获取所有金融时间数据的某一个列
    panel = panel.swapaxes('items', 'minor')
    net_cg_df = panel['p_change'].fillna(value=0)
    """
        转轴后直接获取p_change，即所有金融时间序列涨跌幅度pd.DataFrame对象，形如下所示：
                    usF	    usFCAU	usGM	usHMC	usTM	usTSLA	usTTM
        2015-06-25	-0.387	-0.517	-1.308	0.522	-0.391	1.365	-0.029
        2015-06-26	-0.259	1.300	-0.922	0.366	0.437	-0.632	-0.229
        2015-06-29	-2.468	-6.799	-3.345	-2.676	-2.222	-1.898	-2.550
        2015-06-30	-0.067	0.000	0.301	1.250	0.982	2.381	1.353
        2015-07-01	-0.133	0.688	-0.870	-1.605	-0.112	0.332	0.261
        .................................................................
    """
    return net_cg_df


def _make_benchmark_cg_df(symbol, benchmark):
    """
    根据benchmark提取其时间序列对象kl_pd中的p_change列，返回p_change组成的
    新pd.DataFrame对象，行序列名即为symbol
    :param symbol: 标尺对象symbol，str对象
    :param benchmark: 进行数据收集使用的标尺对象，数据时间范围确定使用，AbuBenchmark实例对象
    :return: 返回p_change组成的新pd.DataFrame对象
    """
    kl_pd = benchmark.kl_pd
    net_cg_df = pd.DataFrame({symbol: kl_pd['p_change']}, index=kl_pd.index).fillna(value=0)
    """
        p_change组成的新pd.DataFrame对象，行序列名即为symbol, 形如下所示：
                    us.IXIC
        2014-07-25	-0.50
        2014-07-28	-0.10
        2014-07-29	-0.05
        2014-07-30	0.45
        2014-07-31	-2.09
        2014-08-01	-0.39
        2014-08-04	0.72
    """
    return net_cg_df


def _net_cg_df_create(symbol, benchmark):
    """
    获取env中全市场symbol，切分分配子进程，委托子进程_make_symbols_cg_df函数，
    将子进程返回的金融时间序列涨跌幅度pd.DataFrame对象再次进行连接，组合成为全市场
    symbol涨跌幅度pd.DataFrame对象
    :param symbol: 标尺对象symbol，str对象
    :param benchmark: 进行数据收集使用的标尺对象，数据时间范围确定使用，AbuBenchmark实例对象
    :return: 全市场symbol涨跌幅度pd.DataFrame对象
    """

    # 获取全市场symbol，没有指定市场参数，即根据env中设置的市场来获取所有市场symbol
    choice_symbols = all_symbol()
    # 通过split_k_market将市场symbol切割为子进程需要完成的任务数量
    process_symbols = split_k_market(g_process_panel_cnt, market_symbols=choice_symbols)
    # 因为切割会有余数，所以将原始设置的进程数切换为分割好的个数, 即32 -> 33 16 -> 17
    n_process_pick_stock = len(process_symbols)
    parallel = Parallel(
        n_jobs=n_process_pick_stock, verbose=0, pre_dispatch='2*n_jobs')

    # 暂时关闭多进程进度条，太多, 注意这种全局设置一定要在AbuEnvProcess初始化之前完成
    # ABuProgress.g_show_ui_progress = False
    # _make_symbols_cg_df被装饰器add_process_env_sig装饰，需要进程间内存拷贝对象AbuEnvProcess, 详AbuEnvProcess
    p_nev = AbuEnvProcess()
    change_df_array = parallel(
        delayed(_make_symbols_cg_df)(choice_symbols, benchmark, env=p_nev) for choice_symbols in process_symbols)
    # ABuProgress.g_show_ui_progress = True
    # 还是显示进度条，但是完事时检测一下还有没有ui进度条
    ABuProgress.do_check_process_is_dead()
    """
        如果标尺的涨跌幅已经在choice_symbols中就不单独获取组装了，没有的情况是如：
        eg. env中指定市场参数港股，即g_market_target = EMarketTargetType.E_MARKET_TARGET_HK，但是
        传人的symbol是a股市场中的一支股票，即目的是想从整个港股市场中分析与这支a股股票的相关系数，这时即会
        触发_make_benchmark_cg_df的使用
    """
    change_df_concat = None if symbol in choice_symbols else _make_benchmark_cg_df(symbol, benchmark)
    for change_df in change_df_array:
        if change_df is not None:
            # 将所有子进程返回的金融时间序列涨跌幅度pd.DataFrame对象再次进行连接
            change_df_concat = change_df if change_df_concat is None else pd.concat([change_df, change_df_concat],
                                                                                    axis=1)
    return change_df_concat


@consume_time
def _all_market_cg(symbol, cmp_cnt=None, n_folds=2, start=None, end=None):
    """
    获取全市场symbol涨跌幅度pd.DataFrame对象
    :param symbol: 外部指定目标symbol，str对象
    :param cmp_cnt: 对比多少个交易日，int，可选参数
    :param n_folds: 对比n_folds年，int，可选参数
    :param start: 请求的开始日期 str对象，可选参数
    :param end: 请求的结束日期 str对象，可选参数
    :return: 全市场symbol涨跌幅度pd.DataFrame对象, 形如下所示：
                        e.g.
                            usA	    usAA	usAAC
                2015/7/27	0.76	-1.94	0.59
                2015/7/28	2.12	2.6	    1.3
                2015/7/29	-0.12	2.94	-1.34
                2015/7/30	1.41	-1.77	-4.04
                2015/7/31	-0.05	-1.1	1.39
    """

    if cmp_cnt is not None:
        # 如果有传递对比多少个交易日这个参数，即反向修改n_folds，ceil向上对齐金融序列获取年数
        n_folds = int(math.ceil(cmp_cnt / ABuEnv.g_market_trade_year))
    # 标尺不是使用大盘symbol，而是传人的symbol做为标尺
    benchmark = AbuBenchmark(benchmark=symbol, n_folds=n_folds, start=start, end=end, rs=False)
    if benchmark.kl_pd is None or benchmark.kl_pd.empty:
        logging.info('{} make benchmark get None'.format(symbol))
        return None

    if cmp_cnt is not None and benchmark.kl_pd.shape[0] > cmp_cnt:
        # 再次根据对比多少个交易日这个参数，对齐时间序列
        benchmark.kl_pd = benchmark.kl_pd.iloc[-cmp_cnt:]
    # 有了symbol和benchmark，即可开始获取全市场symbol涨跌幅度pd.DataFrame对象all_market_change_df
    all_market_change_df = _net_cg_df_create(symbol, benchmark)
    return all_market_change_df


def _handle_market_change_df(market_change_df, cmp_cnt, benchmark_df, show_cnt, corr_type, rolling=True, show=True):
    """
    使用benchmark_df与全市场market_change_df进行相关系数计算，可视化结果及信息
    :param market_change_df: 全市场symbol涨跌幅度pd.DataFrame对象
    :param cmp_cnt: 对比多少个交易日，int
    :param benchmark_df: 标尺symbol对应的pd.Series对象
    :param show_cnt: 最终结果展示以及可视化相似度个数
    :param corr_type: ECoreCorrType对象，暂时支持皮尔逊，斯皮尔曼，＋－符号相关系数，移动时间加权相关系数
    :param rolling: 是否使用时间加权相关计算，与corr_type = ECoreCorrType.E_CORE_TYPE_ROLLING一样，单独拿出来了
    :param show: 是否可视化最终top最相关的股票
    :return:
    """
    # 使用[-cmp_cnt:]再次确定时间序列周期
    benchmark_df = benchmark_df.iloc[-cmp_cnt:]
    market_change_df = market_change_df.iloc[-cmp_cnt:]

    if corr_type == ECoreCorrType.E_CORE_TYPE_ROLLING:
        # 把参数时间加权rolling和corr_type设置进行merge
        rolling = True

    if rolling:
        # 时间加权统一使用ABuCorrcoef.rolling_corr单独计算，即使用两个参数方式计算，详见ABuCorrcoef.rolling_corr
        corr_ret = ABuCorrcoef.rolling_corr(market_change_df, benchmark_df)
        corr_ret = pd.Series(corr_ret, index=market_change_df.columns, name=benchmark_df.name)
    else:
        # 其它加权计算统一使用corr_df计算，即统一使用大矩阵计算相关系数后再拿出benchmark_df对应的相关系数列
        corr_ret = ABuCorrcoef.corr_matrix(market_change_df, corr_type)[benchmark_df.name]
    # 对结果进行zip排序，按照相关系统由正相关到负相关排序
    sorted_ret = sorted(zip(corr_ret.index, corr_ret), key=operator.itemgetter(1), reverse=True)
    """
        最终sorted_ret为可迭代序列，形如：
        [('usTSLA', 1.0), ('usSINA', 0.45565379371028253), ('usWB', 0.44811939073120288),
         ('usAEH', 0.37792534372729375), ('usCRESY', 0.37347584342214574),
         ('us.IXIC', 0.36856818073255937), ('usCVG', 0.36841463066151853),
         ('usOCN', 0.36412381487296047), ('usYHOO', 0.36217456000137549), ...............]
    """
    if show:
        # 根据是否是ipython环境决定信息输出函数
        log_func = logging.info if ABuEnv.g_is_ipython else print
        log_func(sorted_ret[:show_cnt])
        # 绘制show_cnt个最相关的股票股价走势
        ABuSimilarDrawing.draw_show_close(sorted_ret, cmp_cnt, show_cnt)

    return sorted_ret


@consume_time
@from_local
def multi_corr_df(corr_jobs, cmp_cnt=252, n_folds=None, start=None, end=None):
    """
    被from_local装饰器装饰 即强制走本地数据，匹配市场对应的benchmark，根据参数
    使用_all_market_cg获取全市场symbol涨跌幅度pd.DataFrame对象change_df使用
    corr_jobs个相关系数计算方法分别计算change_df的相关系数，所有结果组成一个字典返回
    :param corr_jobs: 需要执行相关计算方法ECoreCorrType序列
    :param cmp_cnt: 对比多少个交易日，int
    :param n_folds: 对比n_folds年，int，可选参数
    :param start: 请求的开始日期 str对象，可选参数
    :param end: 请求的结束日期 str对象，可选参数
    :return: 返回相关系数矩阵组成的字典对象，如下所示 eg：

            {'pears':
                            usBIDU    usFB  usGOOG  usNOAH  usSFUN  usTSLA  usVIPS  usWUBA
            usBIDU         1.0000  0.3013  0.3690  0.4015  0.3680  0.3015  0.3706  0.4320
            usFB           0.3013  1.0000  0.6609  0.2746  0.1978  0.4080  0.2856  0.2438
            usGOOG         0.3690  0.6609  1.0000  0.3682  0.1821  0.3477  0.3040  0.2917
            usNOAH         0.4015  0.2746  0.3682  1.0000  0.3628  0.2178  0.4645  0.4488
            usSFUN         0.3680  0.1978  0.1821  0.3628  1.0000  0.2513  0.2843  0.4883
            usTSLA         0.3015  0.4080  0.3477  0.2178  0.2513  1.0000  0.2327  0.3340
            usVIPS         0.3706  0.2856  0.3040  0.4645  0.2843  0.2327  1.0000  0.4189
            usWUBA         0.4320  0.2438  0.2917  0.4488  0.4883  0.3340  0.4189  1.0000

            'sperm':
                            usBIDU    usFB  usGOOG  usNOAH  usSFUN  usTSLA  usVIPS  usWUBA
            usBIDU         1.0000  0.3888  0.4549  0.4184  0.3747  0.3623  0.4333  0.4396
            usFB           0.3888  1.0000  0.7013  0.2927  0.2379  0.4200  0.3123  0.2216
            usGOOG         0.4549  0.7013  1.0000  0.3797  0.2413  0.3871  0.3922  0.3035
            usNOAH         0.4184  0.2927  0.3797  1.0000  0.3581  0.2066  0.4643  0.4382
            usSFUN         0.3747  0.2379  0.2413  0.3581  1.0000  0.2645  0.3890  0.4693
            usTSLA         0.3623  0.4200  0.3871  0.2066  0.2645  1.0000  0.2540  0.2801
            usVIPS         0.4333  0.3123  0.3922  0.4643  0.3890  0.2540  1.0000  0.4080
            usWUBA         0.4396  0.2216  0.3035  0.4382  0.4693  0.2801  0.4080  1.0000 }
    """

    if isinstance(corr_jobs, ECoreCorrType):
        # 如果直接传递进来一个ECoreCorrType，暂时兼容，做成序列
        corr_jobs = [corr_jobs]

    if any([not isinstance(corr_job, ECoreCorrType) for corr_job in corr_jobs]):
        # 序列中的所有元素必须是ECoreCorrType
        raise TypeError('corr_job must ECoreCorrType')

    # 匹配市场对应的benchmark
    if ABuEnv.g_market_target == EMarketTargetType.E_MARKET_TARGET_US:
        # 美股
        benchmark = IndexSymbol.IXIC
    elif ABuEnv.g_market_target == EMarketTargetType.E_MARKET_TARGET_HK:
        # 港股
        benchmark = IndexSymbol.HSI
    elif ABuEnv.g_market_target == EMarketTargetType.E_MARKET_TARGET_CN:
        # a股
        benchmark = IndexSymbol.SH
    elif ABuEnv.g_market_target == EMarketTargetType.E_MARKET_TARGET_FUTURES_CN:
        # 国内期货
        benchmark = IndexSymbol.BM_FUTURES_CN
    elif ABuEnv.g_market_target == EMarketTargetType.E_MARKET_TARGET_TC:
        # 币类市场
        benchmark = IndexSymbol.TC_INX
    elif ABuEnv.g_market_target == EMarketTargetType.E_MARKET_TARGET_OPTIONS_US:
        # 美股期权暂时也以IXIC做为标尺
        benchmark = IndexSymbol.IXIC
    elif ABuEnv.g_market_target == EMarketTargetType.E_MARKET_TARGET_FUTURES_GLOBAL:
        # 国际期货暂时也以BM_FUTURES_GB做为标尺
        benchmark = IndexSymbol.BM_FUTURES_GB
    else:
        # 没匹配上也不抛错误，随便给一个，因为这里要的benchmark主要目的只是做为时间标尺
        benchmark = IndexSymbol.IXIC
    # 根据参数使用_all_market_cg获取全市场symbol涨跌幅度pd.DataFrame对象change_df
    change_df = _all_market_cg(benchmark, cmp_cnt=cmp_cnt, n_folds=n_folds, start=start, end=end)

    # 使用corr_jobs个相关系数计算方法分别计算change_df的相关系数，所有结果组成一个字典返回
    return {corr_job.value: ABuCorrcoef.corr_matrix(change_df, corr_job) for corr_job in corr_jobs}
