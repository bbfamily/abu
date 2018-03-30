# -*- encoding:utf-8 -*-
"""
    交易可视化模块
"""

from __future__ import print_function
from __future__ import absolute_import
from __future__ import division

import copy

import logging
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from ..CoreBu import ABuEnv
from ..UtilBu import ABuDateUtil
from ..UtilBu.ABuProgress import AbuProgress

# noinspection PyUnresolvedReferences
from ..CoreBu.ABuFixes import range
from ..TradeBu.ABuCapital import AbuCapital

g_enable_his_corr = True
g_enable_his_trade = True

__author__ = '阿布'
__weixin__ = 'abu_quant'


def plot_his_trade(orders, kl_pd):
    """
    可视化绘制AbuOrder对象，绘制交易买入时间，卖出时间，价格，生效因子等
    :param orders: AbuOrder对象序列
    :param kl_pd: 金融时间序列，pd.DataFrame对象
    :return:
    """

    if not g_enable_his_trade:
        return

    # 拿出时间序列中最后一个，做为当前价格
    now_price = kl_pd.iloc[-1].close
    all_pd = kl_pd

    # ipython环境绘制在多个子画布上，普通python环境绘制一个show一个
    draw_multi_ax = ABuEnv.g_is_ipython

    # 根据绘制环境设置子画布数量
    ax_cnt = 1 if not draw_multi_ax else len(orders)
    # 根据子画布数量设置画布大小
    plt.figure(figsize=(14, 8 * ax_cnt))
    fig_dims = (ax_cnt, 1)

    with AbuProgress(len(orders), 0) as pg:
        for index, order in enumerate(orders):
            pg.show(index + 1)
            # 迭代所有orders，对每一个AbuOrder对象绘制交易细节
            mask_date = all_pd['date'] == order.buy_date
            st_key = all_pd[mask_date]['key']

            if order.sell_type == 'keep':
                rv_pd = all_pd.iloc[st_key.values[0]:, :]
            else:
                mask_sell_date = all_pd['date'] == order.sell_date
                st_sell_key = all_pd[mask_sell_date]['key']
                rv_pd = all_pd.iloc[st_key.values[0]:st_sell_key.values[0], :]

            if draw_multi_ax:
                # ipython环境绘制在多个子画布上
                plt.subplot2grid(fig_dims, (index, 0))
            # 绘制价格曲线
            plt.plot(all_pd.index, all_pd['close'], label='close')

            try:
                # 填充透明blue, 针对用户一些版本兼容问题进行处理
                plt.fill_between(all_pd.index, 0, all_pd['close'], color='blue', alpha=.18)
                if order.sell_type == 'keep':
                    # 如果单子还没卖出，是否win使用now_price代替sell_price，需＊单子期望的盈利方向
                    order_win = (now_price - order.buy_price) * order.expect_direction > 0
                elif order.sell_type == 'win':
                    order_win = True
                else:
                    order_win = False
                if order_win:
                    # 盈利的使用红色
                    plt.fill_between(rv_pd.index, 0, rv_pd['close'], color='red', alpha=.38)
                else:
                    # 亏损的使用绿色
                    plt.fill_between(rv_pd.index, 0, rv_pd['close'], color='green', alpha=.38)
            except:
                logging.debug('fill_between numpy type not safe!')
            # 格式化买入信息标签
            buy_date_fmt = ABuDateUtil.str_to_datetime(str(order.buy_date), '%Y%m%d')
            buy_tip = 'buy_price:{:.2f}'.format(order.buy_price)

            # 写买入tip信息
            plt.annotate(buy_tip, xy=(buy_date_fmt, all_pd['close'].asof(buy_date_fmt) * 2 / 5),
                         xytext=(buy_date_fmt, all_pd['close'].asof(buy_date_fmt)),
                         arrowprops=dict(facecolor='red'),
                         horizontalalignment='left', verticalalignment='top')

            if order.sell_price is not None:
                # 如果单子卖出，卖出入信息标签使用，收益使用sell_price计算，需＊单子期望的盈利方向
                sell_date_fmt = ABuDateUtil.str_to_datetime(str(order.sell_date), '%Y%m%d')
                pft = (order.sell_price - order.buy_price) * order.buy_cnt * order.expect_direction
                sell_tip = 'sell price:{:.2f}, profit:{:.2f}'.format(order.sell_price, pft)
            else:
                # 如果单子未卖出，卖出入信息标签使用，收益使用now_price计算，需＊单子期望的盈利方向
                sell_date_fmt = ABuDateUtil.str_to_datetime(str(all_pd[-1:]['date'][0]), '%Y%m%d')
                pft = (now_price - order.buy_price) * order.buy_cnt * order.expect_direction
                sell_tip = 'now price:{:.2f}, profit:{:.2f}'.format(now_price, pft)

            # 写卖出tip信息
            plt.annotate(sell_tip, xy=(sell_date_fmt, all_pd['close'].asof(sell_date_fmt) * 2 / 5),
                         xytext=(sell_date_fmt, all_pd['close'].asof(sell_date_fmt)),
                         arrowprops=dict(facecolor='green'),
                         horizontalalignment='left', verticalalignment='top')
            # 写卖出因子信息
            plt.annotate(order.sell_type_extra, xy=(buy_date_fmt, all_pd['close'].asof(sell_date_fmt) / 4),
                         xytext=(buy_date_fmt, all_pd['close'].asof(sell_date_fmt) / 4),
                         arrowprops=dict(facecolor='yellow'),
                         horizontalalignment='left', verticalalignment='top')

            # 写买入因子信息
            if order.buy_factor is not None:
                plt.annotate(order.buy_factor, xy=(buy_date_fmt, all_pd['close'].asof(sell_date_fmt) / 3),
                             xytext=(buy_date_fmt, all_pd['close'].asof(sell_date_fmt) / 3),
                             arrowprops=dict(facecolor='yellow'),
                             horizontalalignment='left', verticalalignment='top')
            # title使用时间序列symbol
            plt.title(order.buy_symbol)
            if not draw_multi_ax:
                # ipython环境绘制在多个子画布上，普通python环境绘制一个show一个
                plt.show()

    plt.show()


def plot_capital_info(capital_pd, init_cash=-1):
    """
    资金信息可视化
    :param capital_pd: AbuCapital对象或者AbuCapital对象的capital_pd
    :param init_cash: 初始化cash，如果capital_pd为AbuCapital对象，即从capital_pd获取
    """

    if isinstance(capital_pd, AbuCapital):
        # 如果是AbuCapital对象进行转换
        init_cash = capital_pd.read_cash
        capital_pd = capital_pd.capital_pd

    plt.figure(figsize=(14, 8))
    if init_cash != -1:
        cb_earn = capital_pd['capital_blance'] - init_cash
        try:
            # 从有资金变化开始的loc开始绘制
            # noinspection PyUnresolvedReferences
            cb_earn = cb_earn.loc[cb_earn[cb_earn != 0].index[0]:]
            cb_earn.plot()
            plt.title('capital_blance earn from none zero point')
            plt.show()
            sns.regplot(x=np.arange(0, cb_earn.shape[0]), y=cb_earn.values, marker='+')
            plt.show()
        except Exception as e:
            logging.exception(e)
            capital_pd['capital_blance'].plot()
            plt.title('capital blance')
            plt.show()

    # 为了画出平滑的曲线，取有值的
    cap_cp = copy.deepcopy(capital_pd)
    cap_cp['stocks_blance'][cap_cp['stocks_blance'] <= 0] = np.nan
    cap_cp['stocks_blance'].fillna(method='pad', inplace=True)
    cap_cp['stocks_blance'].dropna(inplace=True)
    cap_cp['stocks_blance'].plot()
    plt.title('stocks blance')
    plt.show()

    try:
        sns.distplot(capital_pd['capital_blance'], kde_kws={"lw": 3, "label": "capital blance kde"})
        plt.show()
    except Exception as e:
        logging.debug(e)
        capital_pd['capital_blance'].plot(kind='kde')
        plt.title('capital blance kde')
        plt.show()


def plot_bk_xd(bk_summary, kl_pd_xd_mean, title=None):
    """根据有bk_summary属性的bk交易因子进行可视化，暂时未迁移完成"""
    plt.figure()
    plt.plot(list(range(0, len(kl_pd_xd_mean))), kl_pd_xd_mean['close'])
    for bk in bk_summary.bk_xd_obj_list:
        plt.hold(True)
        pc = 'r' if bk.break_sucess is True else 'g'
        plt.plot(bk.break_index, kl_pd_xd_mean['close'][bk.break_index], 'ro', markersize=12, markeredgewidth=1.5,
                 markerfacecolor='None', markeredgecolor=pc)
    if title is not None:
        plt.title(title)
    plt.grid(True)


def plot_kp_xd(kp_summary, kl_pd_xd_mean, title=None):
    """根据有bk_summary属性的kp交易因子进行可视化，暂时未迁移完成"""
    plt.figure()
    plt.plot(list(range(0, len(kl_pd_xd_mean))), kl_pd_xd_mean['close'])

    for kp in kp_summary.kp_xd_obj_list:
        plt.hold(True)
        plt.plot(kp.break_index, kl_pd_xd_mean['close'][kp.break_index], 'ro', markersize=8, markeredgewidth=1.5,
                 markerfacecolor='None', markeredgecolor='r')

    if title is not None:
        plt.title(title)
    plt.grid(True)
