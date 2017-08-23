# -*- encoding:utf-8 -*-
"""
    市场，数据可视化绘制模块
"""
from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

import itertools
import logging
import os
from collections import Iterable
from math import pi

import bokeh.plotting as bp
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from ..CoreBu import ABuEnv
from ..MarketBu import ABuSymbolPd
from ..UtilBu import ABuDateUtil
from ..UtilBu import ABuFileUtil
from ..UtilBu import ABuScalerUtil
from ..CoreBu.ABuFixes import partial

# TODO 可设置的红涨绿跌，还是绿涨红跌
__colorup__ = "red"
__colordown__ = "green"

"""预备颜色序列集，超出序列数量应使用itertools.cycle循环绘制"""
K_PLT_MAP_STYLE = [
    'b', 'c', 'g', 'k', 'm', 'r', 'y', 'w']

"""保存可视化png文件路径"""
K_SAVE_CACHE_PNG_ROOT = os.path.join(ABuEnv.g_project_data_dir, 'save_png')
"""保存可视化html文件路径"""
K_SAVE_CACHE_HTML_ROOT = os.path.join(ABuEnv.g_project_data_dir, 'save_html')

"""暂时只做全据设置，不画量只画价格"""
g_only_draw_price = False


def plot_candle_from_order(order, date_ext=120, day_sum=False, html_bk=False, save=False):
    """
    根据order绘制交易发生在金融时间序列上的位置等信息，对交易进行可视化分析时使用
    :param order: AbuOrder对象转换的pd.DataFrame对象or pd.Series对象
    :param date_ext: int对象 eg. 如交易在2015-06-01执行，如date_ext＝120，择start向前推120天，end向后推120天
    :param day_sum: 端线图 matplotlib的版本有些有bug显示不对
    :param html_bk: 使用bk绘制可交互的k图，在网页上进行交互
    :param save: 是否保存可视化结果在本地
    :return:
    """
    if not isinstance(order, (pd.DataFrame, pd.Series)) and order.shape[0] > 0:
        # order必须是pd.DataFrame对象or pd.Series对象
        raise TypeError('order must DataFrame here!!')

    is_df = isinstance(order, pd.DataFrame)
    if is_df and order.shape[0] == 1:
        # 如果是只有1行pd.DataFrame对象则变成pd.Series
        is_df = False
        # 通过iloc即变成pd.Series对象
        # noinspection PyUnresolvedReferences
        order = order.iloc[0]

    def plot_from_series(p_order, p_want_df):
        """
        根据order对交易进行可视化分析
        :param p_order: AbuOrder对象转换的pd.Series对象
        :param p_want_df: 是否返回交易买入卖出时段金融数据序列
        """

        # 确定交易对象
        target_symbol_inner = p_order['symbol']
        view_index_inner = None
        # 单子都必须有买入时间
        start = ABuDateUtil.fmt_date(p_order['buy_date'])
        # 通过date_ext确定start，即买人单子向前推date_ext天
        start = ABuDateUtil.begin_date(date_ext, date_str=start, fix=False)
        if p_order['sell_type'] != 'keep':
            # 如果有卖出，继续通过sell_date，date_ext确定end时间
            view_index_inner = [pd.to_datetime(str(p_order['buy_date'])), pd.to_datetime(str(p_order['sell_date']))]
            end = ABuDateUtil.fmt_date(p_order['sell_date'])
            # -date_ext 向前
            end = ABuDateUtil.begin_date(-date_ext, date_str=end, fix=False)
        else:
            end = None

        try:
            df = plot_candle_from_symbol(target_symbol_inner, start=start, end=end, day_sum=day_sum,
                                         html_bk=html_bk,
                                         view_index=view_index_inner, save=save)
        except Exception as e:
            logging.exception(e)
            df = None
        if p_want_df:
            return df

    if not is_df:
        return plot_from_series(order, p_want_df=True)
    else:
        # 如果是多个order，排查一下没有交易结果的
        order = order[order['result'] != 0]
        # 如果只有1个order，仍然返回plot_from_series返回的交易范围数据
        want_df = len(order) == 1
        # 多个order迭代执行plot_from_series
        keep_df = order.apply(plot_from_series, axis=1, args=(want_df,))
        if want_df:
            return keep_df


def plot_candle_from_symbol(target_symbol, n_folds=2, start=None, end=None, day_sum=False, html_bk=False,
                            view_index=None, save=False):
    """
    根据target_symbol绘制交易发生在金融时间序列上的位置等信息，对交易进行可视化分析时使用
    :param target_symbol: str对象，代表一个symbol
    :param n_folds: 请求几年的历史回测数据int
    :param start: 请求的开始日期 str对象
    :param end: 请求的结束日期 str对象
    :param day_sum: 端线图 matplotlib的版本有些有bug显示不对
    :param html_bk: 使用bk绘制可交互的k图，在网页上进行交互
    :param view_index: 需要在可视化图中重点标记的交易日信息
        eg. view_index_inner = [pd.to_datetime(str(p_order['buy_date'])), pd.to_datetime(str(p_order['sell_date']))]
    :param save: 是否保存可视化结果在本地
    :return:
    """
    # 通过make_kl_df获取需要的时间序列对象
    kl_pd = ABuSymbolPd.make_kl_df(target_symbol, n_folds=n_folds, start=start, end=end)
    if kl_pd is None or kl_pd.shape[0] == 0:
        logging.debug(target_symbol + ': has net error in data')
        return
    # 绘制获取的kl_pd对象
    plot_candle_form_klpd(kl_pd, day_sum, html_bk, view_index, save=save)

    if isinstance(view_index, Iterable) and len(view_index) == 2:
        """
            即形如[pd.to_datetime(str(p_order['buy_date'])), pd.to_datetime(str(p_order['sell_date']))]
            截取交易买入直到交易卖出这一段金融时间序列
        """
        # noinspection PyUnresolvedReferences
        return kl_pd.loc[view_index[0]:view_index[1]]

    return kl_pd


# 偏函数构成可交换可视化方法
plot_html_symbol = partial(plot_candle_from_symbol, html_bk=True)
# 偏函数构成不可交换可视化方法
plot_symbol = partial(plot_candle_from_symbol, html_bk=False)


def plot_candle_form_klpd(kl_pd, day_sum=False, html_bk=False, view_indexs=None, save=False, name=None):
    """

    :param kl_pd: 金融时间序列，pd.DataFrame对象
    :param day_sum: 端线图 matplotlib的版本有些有bug显示不对
    :param html_bk: 使用bk绘制可交互的k图，在网页上进行交互
    :param view_indexs: 需要在可视化图中重点标记的交易日信息
        eg. view_index_inner = [pd.to_datetime(str(p_order['buy_date'])), pd.to_datetime(str(p_order['sell_date']))]
    :param save: 是否保存可视化结果在本地
    :param name: 外部设置name做为可视化titile，如果不设置取kl_pd.name，即symbol name
    """
    fn = name if name else kl_pd.name if hasattr(kl_pd, 'name') else 'stock'
    plot_candle_stick(kl_pd.index, kl_pd['open'].values, kl_pd['high'].values, kl_pd['low'].values,
                      kl_pd['close'].values, kl_pd['volume'].values, view_indexs,
                      fn, day_sum, html_bk, save)


def plot_candle_stick(date, p_open, high, low, close, volume, view_index, symbol, day_sum, html_bk, save, minute=False):
    """
    展开各个k图绘制数据进行绘制
    :param date: 金融时间序列交易日时间，pd.DataFrame.index对象
    :param p_open: 金融时间序列开盘价格序列，np.array对象
    :param high: 金融时间序列最高价格序列，np.array对象
    :param low: 金融时间序列最低价格序列，np.array对象
    :param close: 金融时间序列收盘价格序列，np.array对象
    :param volume: 金融时间序列成交量序列，np.array对象
    :param view_index: 需要在可视化图中重点标记的交易日信息
        eg. view_index_inner = [pd.to_datetime(str(p_order['buy_date'])), pd.to_datetime(str(p_order['sell_date']))]
    :param symbol: symbol str对象
    :param day_sum: 端线图 matplotlib的版本有些有bug显示不对
    :param html_bk: 使用bk绘制可交互的k图，在网页上进行交互
    :param save: 是否保存可视化结果在本地
    :param minute: 是否是绘制分钟k线图
    """
    if html_bk is False:
        # 绘制不可交互的
        _do_plot_candle(date, p_open, high, low, close, volume, view_index, symbol, day_sum, save, minute)
    else:
        # 通过bk绘制可交互的
        _do_plot_candle_html(date, p_open, high, low, close, symbol, save)


def _do_plot_candle_html(date, p_open, high, low, close, symbol, save):
    """
    bk绘制可交互的k线图
    :param date: 融时间序列交易日时间，pd.DataFrame.index对象
    :param p_open: 金融时间序列开盘价格序列，np.array对象
    :param high: 金融时间序列最高价格序列，np.array对象
    :param low: 金融时间序列最低价格序列，np.array对象
    :param close: 金融时间序列收盘价格序列，np.array对象
    :param symbol: symbol str对象
    :param save: 是否保存可视化结果在本地
    """
    mids = (p_open + close) / 2
    spans = abs(close - p_open)

    inc = close > p_open
    dec = p_open > close

    w = 24 * 60 * 60 * 1000

    t_o_o_l_s = "pan,wheel_zoom,box_zoom,reset,save"

    p = bp.figure(x_axis_type="datetime", tools=t_o_o_l_s, plot_width=1280, title=symbol)
    p.xaxis.major_label_orientation = pi / 4
    p.grid.grid_line_alpha = 0.3

    p.segment(date.to_datetime(), high, date.to_datetime(), low, color="black")
    # noinspection PyUnresolvedReferences
    p.rect(date.to_datetime()[inc], mids[inc], w, spans[inc], fill_color=__colorup__, line_color=__colorup__)
    # noinspection PyUnresolvedReferences
    p.rect(date.to_datetime()[dec], mids[dec], w, spans[dec], fill_color=__colordown__, line_color=__colordown__)

    bp.show(p)
    if save:
        save_dir = os.path.join(K_SAVE_CACHE_HTML_ROOT, ABuDateUtil.current_str_date())
        html_name = os.path.join(save_dir, symbol + ".html")
        ABuFileUtil.ensure_dir(html_name)
        bp.output_file(html_name, title=symbol)


def _do_plot_candle(date, p_open, high, low, close, volume, view_index, symbol, day_sum, save, minute):
    """
    绘制不可交互的k线图
    param date: 融时间序列交易日时间，pd.DataFrame.index对象
    :param p_open: 金融时间序列开盘价格序列，np.array对象
    :param high: 金融时间序列最高价格序列，np.array对象
    :param low: 金融时间序列最低价格序列，np.array对象
    :param close: 金融时间序列收盘价格序列，np.array对象
    :param volume: 金融时间序列成交量序列，np.array对象
    :param symbol: symbol str对象
    :param day_sum: 端线图 matplotlib的版本有些有bug显示不对
    :param save: 是否保存可视化结果在本地
    :param minute: 是否是绘制分钟k线图
    """

    # 需要内部import不然每次import abupy都有warning，特别是子进程很烦人
    try:
        # noinspection PyUnresolvedReferences, PyDeprecation
        import matplotlib.finance as mpf
    except ImportError:
        # 2.2 才会有
        # noinspection PyUnresolvedReferences, PyDeprecation
        import matplotlib.mpl_finance as mpf

    if not g_only_draw_price:
        # 成交量，价格都绘制
        # noinspection PyTypeChecker
        fig, (ax1, ax2) = plt.subplots(2, sharex=True, figsize=(20, 12))
    else:
        # 只绘制价格
        fig, ax1 = plt.subplots(figsize=(6, 6))
    if day_sum:
        # 端线图绘制
        qutotes = []
        for index, (d, o, c, l, h) in enumerate(zip(date, p_open, close, low, high)):
            d = index if minute else mpf.date2num(d)
            val = (d, o, c, l, h)
            qutotes.append(val)
        # plot_day_summary_oclh接口，与mpf.candlestick_ochl不同，即数据顺序为开收低高
        mpf.plot_day_summary_oclh(ax1, qutotes, ticksize=5, colorup=__colorup__, colordown=__colordown__)
    else:
        # k线图绘制
        qutotes = []
        for index, (d, o, c, h, l) in enumerate(zip(date, p_open, close, high, low)):
            d = index if minute else mpf.date2num(d)
            val = (d, o, c, h, l)
            qutotes.append(val)
        # mpf.candlestick_ochl即数据顺序为开收高低
        mpf.candlestick_ochl(ax1, qutotes, width=0.6, colorup=__colorup__, colordown=__colordown__)

    if not g_only_draw_price:
        # 开始绘制成交量
        ax1.set_title(symbol)
        ax1.set_ylabel('ochl')
        ax1.grid(True)
        if not minute:
            ax1.xaxis_date()
        if view_index is not None:
            # 开始绘制买入交易日，卖出交易日，重点突出的点位
            e_list = date.tolist()
            # itertools.cycle循环使用备选的颜色
            for v, csColor in zip(view_index, itertools.cycle(K_PLT_MAP_STYLE)):
                try:
                    v_ind = e_list.index(v)
                except Exception as e:
                    # logging.exception(e)
                    logging.debug(e)
                    # 向前倒一个
                    v_ind = len(close) - 1
                label = symbol + ': ' + str(date[v_ind])
                ax1.plot(v, close[v_ind], 'ro', markersize=12, markeredgewidth=4.5,
                         markerfacecolor='None', markeredgecolor=csColor, label=label)

                # 因为candlestick_ochl 不能label了，所以使用下面的显示文字
                # noinspection PyUnboundLocalVariable
                ax2.plot(v, 0, 'ro', markersize=12, markeredgewidth=0.5,
                         markerfacecolor='None', markeredgecolor=csColor, label=label)
            plt.legend(loc='best')

        # 成交量柱子颜色，收盘价格 > 开盘，即红色
        # noinspection PyTypeChecker
        bar_red = np.where(close >= p_open, volume, 0)
        # 成交量柱子颜色，开盘价格 > 收盘。即绿色
        # noinspection PyTypeChecker
        bar_green = np.where(p_open > close, volume, 0)

        date = date if not minute else np.arange(0, len(date))
        ax2.bar(date, bar_red, facecolor=__colorup__)
        ax2.bar(date, bar_green, facecolor=__colordown__)

        ax2.set_ylabel('volume')
        ax2.grid(True)
        ax2.autoscale_view()
        plt.setp(plt.gca().get_xticklabels(), rotation=30)
    else:
        ax1.grid(False)

    if save:
        # 保存可视化结果在本地
        from pylab import savefig
        save_dir = os.path.join(K_SAVE_CACHE_PNG_ROOT, ABuDateUtil.current_str_date())
        png_dir = os.path.join(save_dir, symbol)
        ABuFileUtil.ensure_dir(png_dir)
        r_cnt = 0
        while True:
            png_name = '{}{}.png'.format(png_dir, '' if r_cnt == 0 else '-{}'.format(r_cnt))
            if not ABuFileUtil.file_exist(png_name):
                break
            r_cnt += 1
        # noinspection PyUnboundLocalVariable
        savefig(png_name)
        fig.clf()
        plt.close(fig)
    else:
        """
            save 了就不show了
        """
        plt.show()


def save_dir_name(html=False):
    """
    外部获取缓存文件夹的绝对路径
    :param html: 是否缓存为html文件
    """
    r_dir = K_SAVE_CACHE_HTML_ROOT if html else K_SAVE_CACHE_PNG_ROOT
    save_dir = os.path.join(r_dir, ABuDateUtil.current_str_date())
    return save_dir


def plot_simple_multi_stock(multi_kl_pd):
    """
    将多个金融时间序列收盘价格缩放到一个价格水平后，可视化价格变动
    :param multi_kl_pd: 可迭代的序列，元素为金融时间序列
    """
    rg_ret = ABuScalerUtil.scaler_matrix([kl_pd.close for kl_pd in multi_kl_pd])
    for i, kl_pd in enumerate(multi_kl_pd):
        plt.plot(kl_pd.index, rg_ret[i])
    plt.show()


def plot_simple_two_stock(two_stcok_dict):
    """
    将两个金融时间序列收盘价格缩放到一个价格水平后，可视化价格变动
    :param two_stcok_dict: 字典形式，key将做为lable进行可视化使用，元素为金融时间序列
    """
    if not isinstance(two_stcok_dict, dict) or len(two_stcok_dict) != 2:
        print('two_stcok_dict type must dict! or len(two_stcok_dict) != 2')
        return

    label_arr = [s_name for s_name in two_stcok_dict.keys()]
    x, y = ABuScalerUtil.scaler_xy(two_stcok_dict[label_arr[0]].close, two_stcok_dict[label_arr[1]].close)
    plt.plot(x, label=label_arr[0])
    plt.plot(y, label=label_arr[1])
    plt.legend(loc=2)
    plt.show()
