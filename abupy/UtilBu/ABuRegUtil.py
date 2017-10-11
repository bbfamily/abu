# -*- encoding:utf-8 -*-
"""
    拟合工具模块
"""

from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

import math
import logging

import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt
from statsmodels import api as sm, regression
from sklearn import metrics

from ..CoreBu import ABuEnv
from ..CoreBu.ABuPdHelper import pd_rolling_mean
from ..UtilBu.ABuDTUtil import plt_show
from ..UtilBu.ABuStatsUtil import euclidean_distance_xy, manhattan_distances_xy, cosine_distances_xy


log_func = logging.info if ABuEnv.g_is_ipython else print


def regress_xy(x, y, mode=True, zoom=False, show=False):
    """
    使用statsmodels.regression.linear_model进行简单拟合操作，返回model和y_fit
    :param x: 可迭代序列
    :param y: 可迭代序列
    :param mode: 是否需要mode结果，在只需要y_fit且效率需要高时应设置False, 效率差异：
                 mode=False: 1000 loops, best of 3: 778 µs per loop
                 mode=True:  1000 loops, best of 3: 1.23 ms per loop
    :param zoom: 是否缩放x,y
    :param show: 是否可视化结果
    :return: model, y_fit, 如果mode=False，返回的model=None
    """
    if zoom:
        # 将y值 zoom到与x一个级别，不可用ABuScalerUtil.scaler_xy, 因为不管x > y还y > x都拿 x.max() / y.max()
        # TODO ABuScalerUtil中添加使用固定轴进行缩放的功能
        zoom_factor = x.max() / y.max()
        y = zoom_factor * y

    if mode:
        # 加常数1列
        x = sm.add_constant(x)
        model = regression.linear_model.OLS(y, x).fit()

        intercept = model.params[0]
        rad = model.params[1]
        # y = kx + b, x取x[:, 1]，因为add_constant
        y_fit = x[:, 1] * rad + intercept
    else:
        # noinspection PyCallingNonCallable
        y_fit = np.polynomial.Chebyshev.fit(x, y, 1)(x)
        model = None
    if show:
        with plt_show():
            # 取-1因为有OLS add_constant和Chebyshev没有add_constant的两种情况
            x_plot = x[:, -1]
            # 绘制x， y
            plt.plot(x_plot, y)
            # 绘制x， 拟合的y
            plt.plot(x_plot, y_fit)

        with plt_show():
            # 再使用sns绘制，对比拟合结果
            sns.regplot(x=x_plot, y=y)
    return model, y_fit


def regress_y(y, mode=True, zoom=False, show=False):
    """
    使用statsmodels.regression.linear_model进行简单拟合操作, 参数中只提供y序列，
    x使用np.arange(0, len(y))填充
    :param y: 可迭代序列
    :param mode: 是否需要mode结果，在只需要y_fit且效率需要高时应设置False, 效率差异：
             mode=False: 1000 loops, best of 3: 778 µs per loop
             mode=True:  1000 loops, best of 3: 1.23 ms per loop
    :param zoom: 是否缩放x,y
    :param show: 是否可视化结果
    :return: model, y_fit, 如果mode=False，返回的model=None
    """
    x = np.arange(0, len(y))
    return regress_xy(x, y, mode=mode, zoom=zoom, show=show)


def calc_regress_deg(y, show=True):
    """
    将y值 zoom到与x一个级别，之后再fit出弧度转成角度
    1 多个股票的趋势比较提供量化基础，只要同一个时间范围，就可以比较
    2 接近视觉感受到的角度
    :param y:  可迭代序列
    :param show: 是否可视化结果
    :return: deg角度float值
    """
    # 将y值 zoom到与x一个级别
    model, _ = regress_y(y, mode=True, zoom=True, show=show)
    rad = model.params[1]
    # fit出弧度转成角度
    deg = np.rad2deg(rad)
    return deg


def regress_xy_polynomial(x, y, poly=1, zoom=False, show=False):
    """
    多项式拟合, 根据参数poly决定，返回拟合后的y_fit
    :param x: 可迭代序列
    :param y: 可迭代序列
    :param poly: 几次拟合参数，int
    :param zoom: 是否对数据进行缩放
    :param show: 是否可视化显示拟合结果
    :return: y_fit
    """
    if zoom:
        # 将y值 zoom到与x一个级别，不可用ABuScalerUtil.scaler_xy, 因为不管x > y还y > x都拿 x.max() / y.max()
        zoom_factor = x.max() / y.max()
        y = zoom_factor * y

    polynomial = np.polynomial.Chebyshev.fit(x, y, poly)
    # noinspection PyCallingNonCallable
    y_fit = polynomial(x)

    if show:
        with plt_show():
            # 可视化显示拟合结果
            plt.plot(x, y)
            plt.plot(x, y_fit)
            plt.title('{} poly zoom ={}'.format(poly, zoom))

    return y_fit


def regress_y_polynomial(y, poly=1, zoom=False, show=False):
    """
    套接regress_xy_polynomial操作, 参数中只提供y序列，x使用np.arange(0, len(y))填充
    :param y: 可迭代序列
    :param poly: 几次拟合参数，int
    :param zoom: 是否对数据进行缩放
    :param show: 是否可视化显示拟合结果
    :return: y_fit
    """
    x = np.arange(0, len(y))
    return regress_xy_polynomial(x, y, poly=poly, zoom=zoom, show=show)


def metrics_mae(y, y_fit, show=True):
    """
    度量原始序列和拟合后的y_fit的MAE：
         MAE = sum(np.abs(y - y_fit)) / len(y)
    :param y: 原始可迭代序列
    :param y_fit: 拟合可迭代序列
    :param show: 是否输出mae值
    :return: 返回mae值，float
    """
    mae = metrics.mean_absolute_error(y, y_fit)
    if show:
        log_func('MAE={}'.format(mae))
    return mae


def metrics_mse(y, y_fit, show=True):
    """
    度量原始序列和拟合后的y_fit的MSE：
         MSE = sum(np.square(y - y_fit)) / len(y)
    :param y: 原始可迭代序列
    :param y_fit: 拟合可迭代序列
    :param show: 是否输出MSE值
    :return: 返回MSE值，float
    """
    mse = metrics.mean_squared_error(y, y_fit)
    if show:
        log_func('MSE={}'.format(mse))
    return mse


def metrics_rmse(y, y_fit, show=True):
    """
    度量原始序列和拟合后的y_fit的RMSE：
         RMSE = np.sqrt(sum(np.square(y - y_fit)) / len(y))
    :param y: 原始可迭代序列
    :param y_fit: 拟合可迭代序列
    :param show: 是否输出RMSE值
    :return: 返回RMSE值，float
    """
    rmse = np.sqrt(metrics.mean_squared_error(y, y_fit))
    if show:
        log_func('RMSE={}'.format(rmse))
    return rmse


def metrics_euclidean(y, y_fit, show=True):
    """
    度量原始序列和拟合后的y_fit的euclidean欧式距离(L2范数)
    :param y: 原始可迭代序列
    :param y_fit: 拟合可迭代序列
    :param show: 是否输出欧式距离(L2范数)值
    :return: 返回欧式距离(L2范数)值，float
    """
    euclidean = euclidean_distance_xy(y, y_fit, to_similar=False)
    if show:
        log_func('euclidean={}'.format(euclidean))
    return euclidean


def metrics_manhattan(y, y_fit, show=True):
    """
    度量原始序列和拟合后的y_fit的manhattan曼哈顿距离(L1范数)
    :param y: 原始可迭代序列
    :param y_fit: 拟合可迭代序列
    :param show: 是否输出曼哈顿距离(L1范数)值
    :return: 返回曼哈顿距离(L1范数)值，float
    """
    manhattan = manhattan_distances_xy(y, y_fit, to_similar=False)
    if show:
        log_func('manhattan={}'.format(manhattan))
    return manhattan


def metrics_cosine(y, y_fit, show=True):
    """
    度量原始序列和拟合后的y_fit的cosine余弦距离
    :param y: 原始可迭代序列
    :param y_fit: 拟合可迭代序列
    :param show: 是否输出余弦距离值
    :return: 返回曼余弦距离值，float
    """
    cosine = cosine_distances_xy(y, y_fit, to_similar=False)
    if show:
        log_func('cosine={}'.format(cosine))
    return cosine


def valid_poly(y, poly=1, zoom=False, show=True, metrics_func=metrics_rmse):
    """
    验证poly（默认＝1）次多项式拟合回归的趋势曲线是否能代表原始曲线y的走势，
    基础思路：
             1. 对原始曲线y进行窗口均线计算，窗口的大小＝ math.ceil(len(y) / 4)
             eg：
                原始y序列＝504 －> rolling_window = math.ceil(len(y) / 4) = 126
             2. 通过pd_rolling_mean计算出均线的值y_roll_mean
             3  使用metrics_func方法度量原始y值和均线y_roll_mean的距离distance_mean
             3. 通过计算regress_xy_polynomial计算多项式拟合回归的趋势曲线y_fit
             4. 使用metrics_func方法度量原始y值和拟合回归的趋势曲线y_fit的距离distance_fit
             5. 如果distance_fit <= distance_mean即代表拟合曲线可以代表原始曲线y的走势
    :param y: 原始可迭代序列
    :param poly: 几次拟合参数，int
    :param zoom: 是否对y数据进行缩放
    :param show: 是否原始曲线y，均线，以及拟合曲线可视化
    :param metrics_func: 度量始y值和均线y_roll_mean的距离和原始y值和
                         拟合回归的趋势曲线y_fit的距离的方法，默认使用metrics_rmse
    :return: 是否poly次拟合曲线可以代表原始曲线y的走势
    """
    valid = False
    x = np.arange(0, len(y))
    if zoom:
        # 将y值 zoom到与x一个级别，不可用ABuScalerUtil.scaler_xy, 因为不管x > y还y > x都拿 x.max() / y.max()
        zoom_factor = x.max() / y.max()
        y = zoom_factor * y
    # 对原始曲线y进行窗口均线计算，窗口的大小＝ math.ceil(len(y) / 4)
    rolling_window = int(math.ceil(len(y) / 4))
    # 通过pd_rolling_mean计算出均线的值y_roll_mean
    y_roll_mean = pd_rolling_mean(y, window=rolling_window, min_periods=1)
    # 使用metrics_func方法度量原始y值和均线y_roll_mean的距离distance_mean
    distance_mean = metrics_func(y, y_roll_mean, show=False)

    # 通过计算regress_xy_polynomial计算多项式拟合回归的趋势曲线y_fit, 外面做zoom了所以zoom=False
    y_fit = regress_xy_polynomial(x, y, poly=poly, zoom=False, show=False)
    # 使用metrics_func方法度量原始y值和拟合回归的趋势曲线y_fit的距离distance_fit
    distance_fit = metrics_func(y, y_fit, show=False)
    # 如果distance_fit <= distance_mean即代表拟合曲线可以代表原始曲线y的走势
    if distance_fit <= distance_mean:
        valid = True
    if show:
        with plt_show():
            # 原始曲线y，均线，以及拟合曲线可视化
            plt.plot(x, y)
            plt.plot(x, y_roll_mean)
            plt.plot(x, y_fit)
            plt.legend(['close', 'rolling window={}'.format(rolling_window), 'y_fit poly={}'.format(poly)])
            log_func('metrics_func rolling_mean={}, metrics_func y_fit={}'.format(distance_mean, distance_fit))
    return valid


def least_valid_poly(y, zoom=False, show=True, metrics_func=metrics_rmse):
    """
    套接valid_poly，检测至少poly次拟合曲线可以代表原始曲线y的走势
    :param y: 原始可迭代序列
    :param zoom: 是否对y数据进行缩放
    :param show: 是否原始曲线y，均线，以及拟合曲线可视化
    :param metrics_func: 度量始y值和均线y_roll_mean的距离和原始y值和
                         拟合回归的趋势曲线y_fit的距离的方法，默认使用metrics_rmse
    :return: 至少poly次拟合曲线可以代表原始曲线y的走势，int
    """
    poly = 1
    while poly < 100:
        valid = valid_poly(y, poly=poly, zoom=zoom, show=False, metrics_func=metrics_func)
        if valid:
            if show:
                # 这里如果show，就再来了一遍，没在乎效率，在考虑效率情况下不要使用show
                valid_poly(y, poly=poly, zoom=zoom, show=True, metrics_func=metrics_func)
            break
        poly += 1
    return poly


def search_best_poly(y, poly_min=1, poly_max=100, zoom=False, show=True, metrics_func=metrics_rmse):
    """
    寻找poly（1－100）次多项式拟合回归的趋势曲线可以比较完美的代表原始曲线y的走势，
    基础思路：
             1. 对原始曲线y进行窗口均线计算，窗口的大小＝ math.ceil(len(y) / 4)
             eg：
                原始y序列＝504 －> rolling_window = math.ceil(len(y) / 4) = 126
             2. 通过pd_rolling_mean计算出均线的值y_roll_mean
             3  使用metrics_func方法度量原始y值和均线y_roll_mean的距离distance_mean
             3. 迭代计算1-100poly次regress_xy_polynomial的拟合曲线y_fit
             4. 使用metrics_func方法度量原始y值和拟合回归的趋势曲线y_fit的距离distance_fit
             5. 如果distance_fit <= distance_mean* 0.6即代表拟合曲线可以比较完美的代表原始曲线y的走势，停止迭代
             6. 返回停止迭代时的poly次数
    :param y: 原始可迭代序列
    :param poly_min: 寻找最佳拟合次数的最少次数，eg，2
    :param poly_max: 寻找最佳拟合次数的最多次数，eg：20
    :param zoom: 是否对y数据进行缩放
    :param show: 是否原始曲线y，均线，以及拟合曲线可视化
    :param metrics_func: 度量始y值和均线y_roll_mean的距离和原始y值和
                         拟合回归的趋势曲线y_fit的距离的方法，默认使用metrics_rmse
    :return: 返回停止迭代时的poly次数
    """
    x = np.arange(0, len(y))
    if zoom:
        # 将y值 zoom到与x一个级别，不可用ABuScalerUtil.scaler_xy, 因为不管x > y还y > x都拿 x.max() / y.max()
        zoom_factor = x.max() / y.max()
        y = zoom_factor * y
    # 对原始曲线y进行窗口均线计算，窗口的大小＝ math.ceil(len(y) / 4)
    rolling_window = int(math.ceil(len(y) / 4))
    # 通过pd_rolling_mean计算出均线的值y_roll_mean
    y_roll_mean = pd_rolling_mean(y, window=rolling_window, min_periods=1)
    # 使用metrics_func方法度量原始y值和均线y_roll_mean的距离distance_mean
    distance_mean = metrics_func(y, y_roll_mean, show=False)
    poly = poly_min
    while poly < poly_max:
        # 迭代计算1-100poly次regress_xy_polynomial的拟合曲线y_fit, 外面做zoom了所以zoom=False
        y_fit = regress_xy_polynomial(x, y, poly=poly, zoom=False, show=False)
        # 使用metrics_func方法度量原始y值和拟合回归的趋势曲线y_fit的距离distance_fit
        distance_fit = metrics_func(y, y_fit, show=False)
        if distance_fit <= distance_mean * 0.6:
            # 如果distance_fit <= distance_mean* 0.6即代表拟合曲线可以比较完美的代表原始曲线y的走势，停止迭代
            if show:
                with plt_show():
                    # 原始曲线y，均线，以及拟合曲线可视化
                    plt.plot(x, y)
                    plt.plot(x, y_roll_mean)
                    plt.plot(x, y_fit)
                    plt.legend(['close', 'rolling window={}'.format(rolling_window), 'y_fit poly={}'.format(poly)])
                    log_func('metrics_func rolling_mean={}, metrics_func y_fit={}'.format(distance_mean, distance_fit))
            break
        poly += 1
    return poly
