# -*- encoding:utf-8 -*-
"""
    技术线内部执行模块
"""
from __future__ import print_function
from __future__ import absolute_import
from __future__ import division

import math
import logging
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from scipy import optimize as sco, stats
from sklearn import metrics
from sklearn.cluster import KMeans
from sklearn.linear_model import LinearRegression

# noinspection PyUnresolvedReferences
from ..CoreBu.ABuFixes import xrange
from ..CoreBu.ABuPdHelper import pd_resample
from ..CoreBu import ABuEnv
from ..CoreBu.ABuEnv import EMarketDataSplitMode
from ..UtilBu.ABuProgress import AbuProgress
from ..UtilBu import ABuRegUtil, ABuScalerUtil
from ..MarketBu import ABuSymbolPd
from ..UtilBu.ABuDTUtil import plt_show

__author__ = '阿布'
__weixin__ = 'abu_quant'

"""模块打印根据环境选择logging.info或者print函数"""
log_func = logging.info if ABuEnv.g_is_ipython else print
"""多项拟合函数寻找阻力，支撑位的poly倍数基数"""
g_upport_resistance_unit = 3


def shift_distance(arr, how, slice_start=0, slice_end=-1, color='r', show=True, show_log=True, ps=True):
    """
    计算序列arr的'位移路程比'，可视化形式为直接三角展示位移，路程直接的关系，返回
    h_distance(三角底边距离), v_distance(三角垂直距离),
    distance(斜边，路程), shift(位移), sd（位移路程比：shift / distance）
    :param arr: numpy array
    :param how: EShiftDistanceHow，计算路程终点点位值使用的计算方法
    :param slice_start: 如果arr是子序列切片，切片start值， int
    :param slice_end: 如果arr是子序列切片，切片end值， int
    :param color: 直角三角边框颜色，str对象 eg：'r', 'red'
    :param show_log: 是否输出位移路程比各个段比值，默认True
    :param show: 是否可视化
    :param ps: 是否立即执行plt.show
    :return: h_distance(三角底边距离), v_distance(三角垂直距离),
            distance(斜边，路程), shift(位移), sd（位移路程比：shift / distance）
    """
    if slice_end == -1:
        # 因为下面还有slice_end - slice_start等操作，所以转换为正切片值
        slice_end = len(arr)
    # 根据slice_start， slice_end做切片
    slice_arr = arr[slice_start:slice_end]

    # 位置0的值即为路程起点点位值
    start_y = slice_arr[0]
    how = shift_distance_how(how)
    # 根据how的方法计算slice_arr出路程终点点位值
    shift_y = how(slice_arr)
    # 直角三角底边distance
    h_distance = slice_end - slice_start
    # 直角三角垂直边distance
    v_distance = abs(shift_y - start_y)
    # 计算斜边值，即路程值
    distance = math.sqrt(h_distance ** 2 + v_distance ** 2)

    # 开始计算位移，首先套上pd.Series做diff结果如下eg形式所示
    diff_slice = pd.Series(slice_arr).diff().fillna(value=0)
    """
        eg: diff_slice
            0     0.0000
            1    -3.7984
            2     7.6824
            3     9.0512
            4     4.3459
            5    -7.7679
                   ...
            44   -3.1482
            45   -1.6426
            46   -2.1216
            47    0.4449
            48   -4.6539
            49    7.1520
    """
    # np.abs(diff_slice).sum计算结果即为位移值
    shift = np.abs(diff_slice).sum()
    # 计算出位移路程比sd
    sd = shift / distance

    if show:
        # 可视化
        start_pos = (slice_start, start_y)
        shift_pos = (slice_end, shift_y)

        # 路程起点点位值 > 路程终点点位值
        if shift_y > start_y:
            end_y = start_y
            top_y = shift_y
            end_pos = (slice_end, end_y)
        else:
            end_y = shift_y
            top_y = start_y
            end_pos = (slice_start, end_y)
        # annotate文字显示两条直接边长度
        plt.annotate('h = {:.2f}, v={:.2f}'.format(h_distance, v_distance), xy=(slice_start, end_y - 8))
        # annotate文字显示路程值
        plt.annotate('distance = {:.2f}'.format(distance), xy=(slice_start, top_y + 15))
        # annotate文字显示位移值
        plt.annotate('shift={:.2f}'.format(shift), xy=(slice_start + 5, top_y + 5))

        # legend显示位移路程比值
        legend = '{}-{} shift/distance: {:.2f}'.format(slice_start, slice_end, sd)

        # 三条边依次相连
        plt.plot([start_pos[0], end_pos[0]],
                 [start_pos[1], end_pos[1]], c=color)

        plt.plot([start_pos[0], shift_pos[0]],
                 [start_pos[1], shift_pos[1]], c=color)

        plt.plot([end_pos[0], shift_pos[0]],
                 [end_pos[1], shift_pos[1]], c=color, label=legend)
        plt.legend(loc=2)
        if ps:
            with plt_show():
                # 是否立即show
                plt.plot(slice_arr)
        if show_log:
            log_func(legend)

    # h_distance(三角底边距离), v_distance(三角垂直距离),distance(斜边路程),shift(位移),sd（位移路程比：shift / distance）
    return h_distance, v_distance, distance, shift, sd


def calc_kl_speed(kl, resample=5):
    """
    计算曲线跟随趋势的速度，由于速度是相对的，所以需要在相同周期内与参数曲线进行比对
    :param kl: pd.Series或者numpy序列
    :param resample: 计算速度数值的重采样周期默认5
    :return: 趋势变化敏感速度
    """
    kl_rp = pd_resample(kl, '{}D'.format(resample), how='mean')
    """
        eg: kl_rp
        2011-07-28    1.0027
        2011-08-02    0.9767
        2011-08-07    0.9189
        2011-08-12    0.9174
        2011-08-17    0.9156
    """
    kl_diff = kl_rp.diff()
    """
        eg:
        2011-08-02   -0.0260
        2011-08-07   -0.0578
        2011-08-12   -0.0015
        2011-08-17   -0.0018
    """
    # 二值化 1 -1, 做为速度参数序列慢线
    # noinspection PyTypeChecker
    kl_trend_slow = pd.Series(np.where(kl_diff > 0, 1, -1))
    """
        eg: kl_trend_fast
            0     -1
            1     -1
            2     -1
            3     -1
            4     -1
            5      1
            6      1
            7     -1
            8     -1
            9     -1
    """
    # 慢线向前错一个周期形成快线
    kl_trend_fast = kl_trend_slow.shift(-1)
    """
        egL kl_trend_slow
            0     -1.0
            1     -1.0
            2     -1.0
            3     -1.0
            4      1.0
            5      1.0
            6     -1.0
            7     -1.0
            8     -1.0
            9     -1.0
    """

    kl_trend_ffs = kl_trend_fast[:-1] * kl_trend_slow[:-1]
    """
        慢线 乘 快线 即符号运算
        eg：kl_trend_ffs
        0      1.0
        1      1.0
        2      1.0
        3      1.0
        4     -1.0
        5      1.0
        6     -1.0
        7      1.0
        8      1.0
        9      1.0
    """

    # 符号相同的占所有和的比例即为趋势变化敏感速度值，注意如果没有在相同周期内与参数曲线进行比对的曲线，本速度值即无意义
    speed = kl_trend_ffs.value_counts()[1] / kl_trend_ffs.value_counts().sum()
    return speed


def calc_pair_speed(symbol, benchmark_symbol, resample=5, speed_key='close',
                    start=None, end=None, n_folds=2, show=False):
    """
    参数传递一组symbol对，获取symbol对的金融时间序列数据，根据speed_key获取曲线序列数据，
    分别通过calc_kl_speed计算symbol对的趋势跟随速度，相关性＊敏感度＝敏感度置信度
    :param symbol: eg: 'AU0'
    :param benchmark_symbol: eg: 'XAU'
    :param resample: 计算速度数值的重采样周期默认5
    :param speed_key: 金融时间序列数据取的曲线序列key，默认'close'
    :param start: 获取金融时间序列的start时间
    :param end: 获取金融时间序列的end时间
    :param n_folds: 获取金融时间序列的n_folds参数
    :param show: 是否可视化symbol对的趋势走势对比
    :return: 参数symbol, benchmark_symbol所对应的趋势变化敏感速度数值，以及相关性＊敏感度＝敏感度置信度
    """
    from ..TradeBu import AbuBenchmark
    from ..SimilarBu import ABuCorrcoef, ECoreCorrType

    benchmark = AbuBenchmark(benchmark_symbol, start=start, end=end, n_folds=n_folds, rs=False)
    if benchmark.kl_pd is None:
        return None, None, None
    benchmark_kl = benchmark.kl_pd
    kl = ABuSymbolPd.make_kl_df(symbol, benchmark=benchmark,
                                data_mode=EMarketDataSplitMode.E_DATA_SPLIT_UNDO)
    if kl is None:
        return None, None, None
    # 通过calc_kl_speed计算趋势跟随速度
    kl_speed = calc_kl_speed(kl[speed_key], resample)
    benchmark_kl_speed = calc_kl_speed(benchmark_kl[speed_key], resample)
    # 两个走势的SPERM相关性
    corr = ABuCorrcoef.corr_xy(kl.close, benchmark_kl.close, ECoreCorrType.E_CORE_TYPE_SPERM)

    if show:
        with plt_show():
            # 可视化symbol对的趋势走势对比
            kl_sl = ABuScalerUtil.scaler_one(kl[speed_key])
            benchmark_kl_sl = ABuScalerUtil.scaler_one(benchmark_kl[speed_key])
            kl_resamp = pd_resample(kl_sl, '{}D'.format(resample), how='mean')
            benchmark_kl_resamp = pd_resample(benchmark_kl_sl, '{}D'.format(resample), how='mean')
            kl_resamp.plot(label='kl', style=['*--'])
            benchmark_kl_resamp.plot(label='benchmark', style=['^--'])
            plt.legend(loc='best')
    # 返回参数symbol, benchmark_symbol所对应的趋势变化敏感速度数值, 以及相关性＊敏感度＝敏感度置信度
    return kl_speed, benchmark_kl_speed, corr


def shift_distance_how(how):
    """
    通过how（EShiftDistanceHow），对应出计算算路程终点点位值使用的计算方法
    注意默认使用shift_distance_close对应标准路程点位值定义方法，其它方法对应的
    路程终点点位值使用的计算方法并非得到最准确的'路程位移比'
    :param how: EShiftDistanceHow对象或者callable即外部可自行设置方法，即计算算路程终点点位值使用的计算方法可自定义
    :return: 计算算路程终点点位值使用的计算方法
    """
    if callable(how):
        # 外部可自行设置方法，即计算算路程终点点位值使用的计算方法可自定义
        return how

    from ..TLineBu.ABuTLine import EShiftDistanceHow

    if how == EShiftDistanceHow.shift_distance_maxmin:
        # 如果p_arr[0] > p_arr[-1]，使用np.min(p_arr)，否则np.max(p_arr)，即上升趋势取max，下跌趋势取min
        how = lambda p_arr: np.min(p_arr) if p_arr[0] > p_arr[-1] else np.max(p_arr)
    elif how == EShiftDistanceHow.shift_distance_close:
        # 对应序列的最后一个点位值，标准路程点位值定义
        how = lambda p_arr: p_arr[-1]
    elif how == EShiftDistanceHow.shift_distance_sum_maxmin:
        # 如果abs(p_arr.max() - p_arr[-1]) > abs(p_arr[-1] - p_arr.min()) 取np.min(p_arr)否则np.max(p_arr)
        # 即最终的点位绝对距离靠近np.min(p_arr)取np.min(p_arr)否则np.max(p_arr)
        how = lambda p_arr: np.min(p_arr) if abs(p_arr.max() - p_arr[-1]) > abs(p_arr[-1] - p_arr.min()) \
            else np.max(p_arr)
    else:
        raise TypeError('how is error how={}'.format(how))
    return how


def regress_trend_channel(arr):
    """
    通过arr计算拟合曲线及上下拟合通道曲线，返回三条拟合曲线，组成拟合通道
    :param arr: numpy array
    :return: y_below, y_fit, y_above
    """
    # 通过ABuRegUtil.regress_y计算拟合曲线和模型reg_mode，不使用缩放参数zoom
    reg_mode, y_fit = ABuRegUtil.regress_y(arr, zoom=False)
    reg_params = reg_mode.params

    x = np.arange(0, len(arr))
    a = reg_params[0]
    b = reg_params[1]
    # 通过argmin寻找出原始序列和拟合序列差值的最小点，差值最小代表点位离拟合曲线远，eg: 100 - 80 < 100 - 90
    min_ind = (arr.T - y_fit).argmin()
    # 根据a, b计算出below值, 注意这里是差，eg: below：100 － 80 ＝ 20
    below = x[min_ind] * b + a - arr[min_ind]
    # 计算x * b + a但- below，即拟合曲线保持相同的斜率整体下移below值
    y_below = x * b + a - below

    # 通过argmax寻找出原始序列和拟合序列差值的最大点，差值最小代表点位离拟合曲线远，eg: 120 - 100 > 120 - 110
    max_ind = (arr.T - y_fit).argmax()
    # 根据a, b计算出above值, 注意这里是差，eg: above 100 - 120 ＝ -20, 即above是负数
    above = x[max_ind] * b + a - arr[max_ind]
    # 计算x * b + a但整天- above，由于above是负数，即相加 即拟合曲线保持相同的斜率整体上移above值
    y_above = x * b + a - above
    return y_below, y_fit, y_above


def bfgs_min_pos(find_min_pos, y_len, linear_interp):
    """
    通过scipy.interpolate.interp1d插值形成的模型，通过sco.fmin_bfgs计算min
    :param find_min_pos: 寻找min的点位值
    :param y_len: 原始序列长度，int
    :param linear_interp: scipy.interpolate.interp1d插值形成的模型
    :return: sco.fmin_bfgs成功找到的值，所有失败的或者异常都返回－1
    """
    try:
        local_min_pos = sco.fmin_bfgs(linear_interp, find_min_pos, disp=False)[0]
    except:
        # 所有失败的或者异常都返回－1
        local_min_pos = -1
    if local_min_pos < 0 or local_min_pos > y_len:
        # 所有失败的或者异常都返回－1
        local_min_pos = -1
    return local_min_pos


def support_resistance_pos(x, support_resistance_y, best_poly=0, label=None):
    """
    分析获取序列阻力位或者支撑位，通过sco.fmin_bfgs寻找阻力位支撑位，阻力位点也是通过sco.fmin_bfgs寻找，
    但是要求传递进来的序列已经是标准化后取反的序列
    eg：
        demean_y = ABuStatsUtil.demean(self.tl)： 首先通过demean将序列去均值
        resistance_y = demean_y * -1 ：阻力位序列要取反
        support_y = demean_y ：支持位序列不需要取反
    sco.fmin_bfgs使用的模型函数为polynomial.Chebyshev多项拟合函数，poly的次数确定
    由ABuRegUtil.search_best_poly得到，即best_poly次多项式拟合回归的趋势曲线可以比较完美的代表原始曲线y的走势，
    为了得到更多的阻力支持位种子点位值，使用：
        np.polynomial.Chebyshev.fit(x, support_resistance_y, best_poly * g_upport_resistance_unit)
        g_upport_resistance_unit默认＝3
    best_poly * 3，即将poly次数又扩大了3倍，可以改变g_upport_resistance_unit获取更多的阻力位支撑位种子点，但
    速度会更慢。
    :param x: 待分析的序列np.array
    :param support_resistance_y:
    :param best_poly: 函数使用者可设置best_poly, 设置后就不使用ABuRegUtil.search_best_poly寻找了
    :param label: 进度条显示的等待文字，str对象
    :return: 阻力位或者支撑位序列
    """
    if best_poly <= 0:
        # 由ABuRegUtil.search_best_poly得到，即best_poly次多项式拟合回归的趋势曲线可以比较完美的代表原始曲线y的走势
        # 注意这里poly_min＝7，即从7次ploy开始，提升效率
        best_poly = ABuRegUtil.search_best_poly(support_resistance_y, poly_min=7, zoom=False, show=False)
        # 根据曲线的长度倍乘best_poly
        best_poly *= int(math.ceil(len(x) / 120))
    best_poly = int(len(x) / 20) if best_poly < int(len(x) / 20) else best_poly

    # 为了得到更多的阻力支持位种子点位值->best_poly * 3，即将poly次数又扩大了3倍
    p = np.polynomial.Chebyshev.fit(x, support_resistance_y, best_poly)

    # 需要使用set，因为需要过滤重复的
    support_resistance = set()
    # 属于耗时操作，构建进度条显示
    with AbuProgress(len(support_resistance_y), 0, label) as progess:
        for index in xrange(0, len(support_resistance_y), 1):
            progess.show(index + 1)
            local_min_pos = int(bfgs_min_pos(index, len(support_resistance_y), p))
            if local_min_pos == -1:
                # 其实主要就是利用这里找不到的情况进行过滤
                continue
            # 将local_min_pos加到集合中
            support_resistance.add(local_min_pos)
    # 为了方便后续api将set转换list
    support_resistance = list(support_resistance)
    return support_resistance


def select_k_support_resistance(support_resistance, thresh=0.06, label='', show=True):
    """
    对阻力位或者支撑位序列从1-序列个数开始聚类，多个聚类器的方差值进行比较，
    通过方差阀值等方法找到最佳聚类个数，最终得到kmean最佳分类器对象
    :param support_resistance: 阻力位或者支撑位序列
    :param thresh: 聚类方差比例阀值，默认0.06
    :param label: 可视化显示的label，主要用来区别阻力位和支撑位的聚类分析结果
    :param show: 是否可视化
    :return: 最佳分类器对象，KMeans类型
    """
    # 阻力位或者支撑位序列从1-序列个数开始聚类
    k_rng = xrange(1, len(support_resistance))
    est_arr = [KMeans(n_clusters=k).fit(support_resistance) for k in k_rng]
    sum_squares = [e.inertia_ for e in est_arr]
    """
        eg: sum_squares 形如
        [568824.24742527946, 63692.462671013389, 23023.512755456246, 11106.460471044047, 4739.3661037023803, 2
        479.206823270833, 1640.5275739375002, 1095.9099614375, 589.28019737500006, 160.28686404166669,
        45.229672916666679, 31.689656250000013, 18.94814375000001, 10.944315625000012, 5.5643156249999999,
        0.6365031249999995]
    """
    # sum_squares[0] = 568824.24742527946 ->
    # 568824.24742527946 / 568824.24742527946, 63692.462671013389 / 568824.24742527946
    diff_squares = [squares / sum_squares[0] for squares in sum_squares]
    diff_squares_pd = pd.Series(diff_squares)

    thresh_pd = diff_squares_pd[diff_squares_pd < thresh]
    if thresh_pd.shape[0] > 0:
        """
            eg：thresh_pd 形如index[0]代表最小的聚类个数值
            2     4.0476e-02
            3     1.9525e-02
            4     8.3319e-03
            5     4.3585e-03
            6     2.8841e-03
            7     1.9266e-03
            8     1.0360e-03
            9     2.8179e-04
        """
        select_k = k_rng[thresh_pd.index[0]]
    else:
        # 没有符合的，就安装最多的个数聚类
        select_k = k_rng[-1]

    # 通过select_k取代分类器
    est = est_arr[select_k - 1]

    if show:
        plt.subplot(211, figsize=ABuEnv.g_plt_figsize)
        plt.title('{}: elbow method to inform k choice'.format(label))
        # 手肘法可视化最佳聚类个数值，通过silhouette_score
        silhouette_score = [metrics.silhouette_score(support_resistance, e.labels_, metric='euclidean')
                            for e in est_arr[1:]]
        plt.plot(k_rng[1:], silhouette_score, 'b*-')
        plt.grid(True)
        plt.ylabel('{}: Silhouette Coefficient'.format(label))

        # 可视化方差最佳聚类
        plt.subplot(212, figsize=ABuEnv.g_plt_figsize)
        plt.plot(k_rng, sum_squares, 'b*-')
        plt.grid(True)
        plt.xlabel('{}: k'.format(label))
        plt.ylabel('{}: Sum of Squares'.format(label))
        # 将前面得到的select_k使用圆圈在图示上进行标注
        plt.plot(select_k, sum_squares[select_k - 1], 'ro', markersize=12, markeredgewidth=1.5,
                 markerfacecolor='None', markeredgecolor='r')
        plt.show()
    # 返回最佳分类器对象，KMeans类型
    return est


def support_resistance_predict(x, y, est, support_resistance, is_support, show=True):
    """
    通过最优聚类器est从阻力位或者支撑位序列support_resistance中的序列进行聚类predict
    对聚类结果进行可视化，最终从每个cluster中找到唯一值代表这个分类，如果是支撑位要找到最小值的index
    如果是阻力位找到最大值的index
    :param x: 可迭代序列x
    :param y: 可迭代序列y
    :param est: 最佳分类器对象，KMeans类型
    :param support_resistance: 阻力位或者支撑位序列
    :param is_support: 是否是进行支撑位support_resistance_predict
    :param show: 是否进行可视化
    :return: 返回从每个cluster中找到唯一值代表这个分类的x值组成的序列。list
    """
    support_resistance_k = est.predict(support_resistance)
    """
        eg: support_resistance_k形如：
        array([0, 0, 1, 1, 0, 1, 1, 0, 0, 2, 1, 0, 0, 2, 0, 0, 0], dtype=int32)
    """
    if show:
        with plt_show():
            # FIXME 这里没有对超出颜色的范围进行处理，如果聚类个数超过颜色，会出错
            colors = np.array(['#FF0054', '#FBD039', '#23C2BC',
                               '#CC99CC', '#CC3399', '#33FF99', '#00CCFF', '#66FF66', '#339999',
                               '#6666CC', '#666666', '#663333', '#660033', '#FF0054', '#FBD039', '#23C2BC',
                               '#CC99CC', '#CC3399', '#33FF99', '#00CCFF', '#66FF66', '#339999',
                               '#6666CC', '#666666', '#663333', '#660033', '#6666CC', '#666666', '#663333',
                               '#660033', '#FF0054', '#FBD039', '#23C2BC', '#CC99CC', '#CC3399', '#33FF99',
                               '#00CCFF', '#66FF66', '#339999'])
            plt.plot(x, y, '-')
            # c=colors[support_resistance_k]即对不同的聚类采用不同的颜色进行标示
            plt.scatter(support_resistance[:, 0], support_resistance[:, 1], c=colors[support_resistance_k], s=60)
            plt.title('{}: k choice'.format('support' if is_support else 'resistance'))
    d_pd = pd.DataFrame(support_resistance, columns=['x', 'y'])
    d_pd['cluster'] = support_resistance_k
    """
        eg：d_pd形如：
                x        y  cluster
        0     0.0  15.5400        0
        1    97.0  21.3475        0
        2   482.0  22.9625        1
        3   387.0  21.6725        1
        4     5.0  16.2350        0
        5   426.0  23.7650        1
        6   459.0  23.2800        1
        7    48.0  13.1475        0
        8    49.0  13.6700        0
        9   283.0  20.9150        2
        10  497.0  25.2240        1
        11   53.0  13.6537        0
        12   23.0  15.3150        0
        13  280.0  19.9900        2
        14  121.0  18.3475        0
        15   27.0  15.2275        0
        16   95.0  23.9475        0
    """
    k_list = list()

    # 最终从每个cluster中找到唯一值代表这个分类
    for k in set(support_resistance_k):
        if is_support:
            # 如果是支撑位要找到最小值的index
            ind = d_pd[d_pd['cluster'] == k]['y'].argmin()
        else:
            # 如果是阻力位找到最大值的index
            ind = d_pd[d_pd['cluster'] == k]['y'].argmax()
        # 最终的结果只要x的值
        choice_x = int(d_pd.iloc[ind]['x'])
        k_list.append(choice_x)
    # 返回从每个cluster中找到唯一值代表这个分类的x值组成的序列
    return k_list


# noinspection PyArgumentList
def plot_support_resistance_trend(x, y, trend_pos, label, only_last=False, plot_org=False, show=True):
    """
    通过trend_pos绘制阻力线或者支撑线，only_last控制只绘制时间序列中最后一个发现的阻力或支撑，
    plot_org控制是否绘制线段还是直线，plot_org＝True时绘制线段，否则通过LinearRegression进行
    直线绘制
    :param x: 可迭代序列x
    :param y: 可迭代序列y
    :param trend_pos: 趋势点x序列，绘制时需要通过原始序列y获取y[x]值
    :param label: 可视化显示的文字，用来区分阻力位和支撑位
    :param only_last: 控制只绘制时间序列中最后一个发现的阻力或支撑
    :param plot_org: 控制是否绘制线段还是直线，控制是否绘制线段还是直线，
                     plot_org＝True时绘制线段，否则通过LinearRegression进行
    :param show: 是否进行可视化
    """
    # trend_pos类型为list容器
    trend_pos.sort()
    if len(trend_pos) < 2:
        log_func('{} len(trend_pos) < 2 !'.format(label))
        return

    if only_last:
        # 只绘制时间序列中最后一个发现的阻力或支撑，因为前面trend_pos.sort()了，即序列中最后两个元素
        trend_pos = trend_pos[-2:]

    y_trend_arr = []
    for ind, trend_start in enumerate(trend_pos):
        if ind == len(trend_pos) - 1:
            continue
        trend_end = trend_pos[ind + 1]

        x_org = [trend_start, trend_end]
        y_org = [y[trend_start], y[trend_end]]

        # 通过LinearRegression学习线段
        reg = LinearRegression()
        reg.fit(np.array(x_org).reshape(-1, 1), np.array(y_org).reshape(-1, 1))

        x_line = [x[0], x[-1]]
        # predict序列的第一个和最后一个点，即将线段变成了直线，延伸了阻力位，支撑位
        y_line = reg.predict(np.array(x_line).reshape(-1, 1)).reshape(-1, )
        # 把端点结果返回给外面
        y_trend_arr.append(y_line)
        if show:
            if plot_org:
                # plot_org＝True时绘制线段
                plt.plot(x_org, y_org, 'o-', label=label)
            else:
                plt.plot(x_line, y_line, 'o-', label=label)
    if show:
        plt.plot(x, y)
    return y_trend_arr


def skeleton_how(how):
    """
    根据how映射计算数据序列骨架点位的方法
    :param how: ESkeletonHow对象或者callable即外部可自行设置方法，即计算数据序列骨架点位的方法可自定义
    :return:
    """
    if callable(how):
        # callable即外部可自行设置方法
        return how

    from ..TLineBu.ABuTLine import ESkeletonHow
    if how == ESkeletonHow.skeleton_min:
        how_func = np.min
    elif how == ESkeletonHow.skeleton_max:
        how_func = np.max
    elif how == ESkeletonHow.skeleton_mean:
        how_func = np.mean
    elif how == ESkeletonHow.skeleton_median:
        how_func = np.median
    elif how == ESkeletonHow.skeleton_close:
        # 取序列最后一个元素做为采样骨架点位
        how_func = lambda arr: arr[-1]
    elif how == ESkeletonHow.skeleton_triangle:
        # 三角模式骨架点位：确定取最大值，最小值，第三个点位how_func提供
        # 如果np.argmax(arr) > np.argmin(arr)即最大值位置在最小值前面，第三点取序列起点，否则取序列终点
        how_func = lambda arr, start: (start, arr[0]) if np.argmax(arr) > np.argmin(arr) else \
            (len(arr) + start, arr[-1])
    else:
        raise TypeError('how is error how={}'.format(how))
    return how_func


def below_above_gen(x, y):
    """
    (y, x) if x > y else (x, y)
    :param x: 支持比较操作的对象
    :param y: 支持比较操作的对象
    """
    return (y, x) if x > y else (x, y)


def find_percent_point(percents, y):
    """
    可视化技术线比例分割的区域, 针对输入的比例迭代操作后
    分别使用stats.scoreatpercentile和 (y.max() - y.min()) * pt + y.min()两种
    方式进行计算的分割值, 返回对象为比例值为key的字典对象
    eg:
        input:
            percents = (0.1, 0.9)
        output:
            {0.1: (15.732749999999999, 15.5075), 0.9: (31.995000000000005, 34.387500000000003)}

    :param percents: 可迭代序列，eg: (0.1, 0.9), [0.3, 0,4, 0.8]
    :param y: 计算分割线的序列
    :return: 比例值为key的字典对象
    """
    percent_point_dict = {pt: (stats.scoreatpercentile(y, np.round(pt * 100, 1)), (y.max() - y.min()) * pt + y.min())
                          for pt in percents}

    return percent_point_dict


# noinspection PyTypeChecker
def find_golden_point_ex(x, y, show=False):
    """统计黄金分割计算方法，以及对应简单可视化操作"""

    sp382 = stats.scoreatpercentile(y, 38.2)
    sp618 = stats.scoreatpercentile(y, 61.8)
    sp50 = stats.scoreatpercentile(y, 50.0)

    if show:
        with plt_show():
            # 可视化操作
            plt.plot(x, y)
            plt.axhline(sp50, color='c')
            plt.axhline(sp618, color='r')
            plt.axhline(sp382, color='g')
            _ = plt.setp(plt.gca().get_xticklabels(), rotation=30)
            plt.legend(['TLine', 'sp50', 'sp618', 'sp382'],
                       bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)

    return sp382, sp50, sp618


def find_golden_point(x, y, show=False):
    """视觉黄金分割计算方法，以及对应简单可视化操作"""

    cs_max = y.max()
    cs_min = y.min()

    sp382 = (cs_max - cs_min) * 0.382 + cs_min
    sp618 = (cs_max - cs_min) * 0.618 + cs_min
    sp50 = (cs_max - cs_min) * 0.5 + cs_min
    if show:
        with plt_show():
            # 可视化操作
            plt.plot(x, y)
            plt.axhline(sp50, color='c')
            plt.axhline(sp618, color='r')
            plt.axhline(sp382, color='g')
            _ = plt.setp(plt.gca().get_xticklabels(), rotation=30)
            plt.legend(['TLine', 'sp50', 'sp618', 'sp382'],
                       bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)

    return sp382, sp50, sp618
