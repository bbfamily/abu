# -*- encoding:utf-8 -*-
"""
    交易执行模块
"""

from __future__ import print_function
from __future__ import absolute_import
from __future__ import division

import logging

import numpy as np
import pandas as pd

from ..UtilBu import ABuDateUtil, AbuProgress
from ..TradeBu.ABuMLFeature import AbuMlFeature
# noinspection PyUnresolvedReferences
from ..CoreBu.ABuFixes import map

__author__ = '阿布'
__weixin__ = 'abu_quant'


def calc_simple_profit(orders, kl_pd):
    """
    计算交易收益，simple的意思是不考虑手续费
    :param orders: AbuOrder对象序列
    :param kl_pd: 金融时间序列，pd.DataFrame对象
    :return:
    """
    all_profit = 0
    now_price = kl_pd[-1:].close
    for order in orders:
        if order.sell_type == 'keep':
            # 单子如果还没有卖出，使用now_price计算收益
            all_profit += (now_price - order.buy_price) * order.buy_cnt * order.expect_direction
        else:
            # 单子如卖出，使用sell_price计算收益
            all_profit += (order.sell_price - order.buy_price) * order.buy_cnt * order.expect_direction
    return all_profit


def make_orders_pd(orders, kl_pd):
    """
    AbuOrder对象序列转换为pd.DataFrame对象，order_pd中每一行代表一个AbuOrder信息
    :param orders: AbuOrder对象序列
    :param kl_pd: 金融时间序列，pd.DataFrame对象
    """
    ret_orders_pd = None
    for index, order in enumerate(orders):
        # 迭代order，将每一个AbuOrder对象转换为一个pd.DataFrame对象
        order_pd = pd.DataFrame(np.array([order.buy_date, order.buy_price, order.buy_cnt, order.buy_factor,
                                          order.buy_symbol, order.buy_pos,
                                          order.buy_type_str, order.expect_direction,
                                          order.sell_type_extra, order.sell_date, order.sell_price, order.sell_type,
                                          order.ml_features]).reshape(1, -1),
                                index=[index],
                                columns=['buy_date', 'buy_price', 'buy_cnt', 'buy_factor', 'symbol', 'buy_pos',
                                         'buy_type_str', 'expect_direction',
                                         'sell_type_extra',
                                         'sell_date',
                                         'sell_price', 'sell_type', 'ml_features'])

        # 从原始金融时间序列中找到key，赋予order_pd['key']
        mask = kl_pd[kl_pd['date'] == order.buy_date]
        order_pd['key'] = mask['key'].values[0]
        # 将所有order_pd concat生成一个pd.DataFrame对象
        ret_orders_pd = order_pd if ret_orders_pd is None else pd.concat([ret_orders_pd, order_pd])

    # 转换连接好的pd.DataFrame对象的index赋予对应的时间，形成交易时间序列
    dates_fmt = list(map(lambda date: ABuDateUtil.fmt_date(date), ret_orders_pd['buy_date'].tolist()))
    dates_pd = pd.to_datetime(dates_fmt)
    ret_orders_pd.index = dates_pd

    # 把除字符串类型外的所有进行列类型进行显示转换，因为支持py3
    ret_orders_pd['sell_price'] = ret_orders_pd['sell_price'].astype(float)
    ret_orders_pd['sell_date'] = ret_orders_pd['sell_date'].fillna(0).astype(int)

    ret_orders_pd['buy_price'] = ret_orders_pd['buy_price'].astype(float)
    ret_orders_pd['buy_date'] = ret_orders_pd['buy_date'].astype(int)
    ret_orders_pd['buy_cnt'] = ret_orders_pd['buy_cnt'].astype(float)
    ret_orders_pd['expect_direction'] = ret_orders_pd['expect_direction'].astype(float)

    # 计算收益
    c_ss = (ret_orders_pd['sell_price'] - ret_orders_pd['buy_price']) * ret_orders_pd[
        'buy_cnt'] * ret_orders_pd['expect_direction']
    ret_orders_pd['profit'] = np.round(c_ss.values, decimals=2)

    # 判定单子最终是否盈利 win：1，loss：－1. keep：0
    # noinspection PyTypeChecker
    ret_orders_pd['result'] = np.where(ret_orders_pd['sell_type'] == 'win', 1, -1)
    # 针对还是keep的单子置0
    # noinspection PyTypeChecker
    ret_orders_pd['result'] = np.where(ret_orders_pd['sell_type'] == 'keep', 0, ret_orders_pd['result'])
    # 如果单子开启了特征收集，将收集的特征添加到对应的交易中，详阅读AbuMlFeature
    AbuMlFeature().unzip_ml_feature(ret_orders_pd)
    return ret_orders_pd


def transform_action(orders_pd):
    """
    将在make_orders_pd中交易订单构成的pd.DataFrame对象进行拆解，分成买入交易行为及数据，卖出交易行为和数据，
    按照买卖时间顺序，转换构造交易行为顺序序列
    :param orders_pd: 交易订单构成的pd.DataFrame对象
    :return: 交易行为顺序序列 pd.DataFrame对象
    """

    # 从order中摘出买入交易行为
    buy_actions = orders_pd.loc[:, ['buy_date', 'buy_price', 'buy_cnt', 'symbol', 'expect_direction', 'sell_price']]
    # action = buy
    buy_actions['action'] = 'buy'
    # ACTION和order都有的action使用首字母大写，内容小写区分开
    buy_actions = buy_actions.rename(columns={'buy_date': 'Date', 'buy_price': 'Price', 'buy_cnt': 'Cnt',
                                              'sell_price': 'Price2', 'expect_direction': 'Direction'})
    buy_actions.index = np.arange(0, buy_actions.shape[0])

    # 从order中摘出卖出交易行为
    sell_actions = orders_pd.loc[:, ['sell_date', 'sell_price', 'buy_cnt', 'symbol', 'expect_direction', 'buy_price']]
    # action = sell
    sell_actions['action'] = 'sell'
    # action和order都有的action使用首字母大写，内容小写区分开
    sell_actions = sell_actions.rename(columns={'sell_date': 'Date', 'sell_price': 'Price', 'buy_cnt': 'Cnt',
                                                'buy_price': 'Price2', 'expect_direction': 'Direction'})
    sell_actions.index = np.arange(buy_actions.shape[0], buy_actions.shape[0] + sell_actions.shape[0])

    # 把买入交易行为和卖出交易行为连起来
    action_pd = pd.concat([buy_actions, sell_actions])

    # 根据时间和买卖行为排序，即构成时间行为顺序
    # noinspection PyUnresolvedReferences
    action_pd = action_pd.sort_values(['Date', 'action'])
    action_pd.index = np.arange(0, action_pd.shape[0])
    # action中干掉所有keep的单子, 只考虑Price列，即drop卖出行为Price是nan的
    action_pd = action_pd.dropna(subset=['Price'])
    # 一定要先把date转换成int sort_values
    action_pd['Date'] = action_pd['Date'].astype(int)
    action_pd = action_pd.sort_values(['Date', 'action'])
    return action_pd


def apply_action_to_capital(capital, action_pd, kl_pd_manager, show_progress=True):
    """
    多个金融时间序列对应的多个交易行为action_pd，在考虑资金类AbuCapital对象的情况下，对AbuCapital对象进行
    资金时间序列更新，以及判定在有限资金的情况下，交易行为是否可以执行
    :param capital: 资金类AbuCapital实例化对象
    :param action_pd: 交易行为构成的pd.DataFrame对象
    :param kl_pd_manager: 金融时间序列管理对象，AbuKLManager实例
    :param show_progress: 是否显示进度条，默认True
    :return:
    """
    if action_pd.empty:
        logging.info('apply_action_to_capital action_pd.empty!!!')
        return

    # 如果交易symbol数量 > 10000个显示初始化进度条
    init_show_progress = (show_progress and len(set(action_pd.symbol)) > 10000)
    # 资金时间序列初始化各个symbol对应的持仓列，持仓价值列
    capital.apply_init_kl(action_pd, show_progress=init_show_progress)

    # 如果交易symbol数量 > 1个显示apply进度条
    show_apply_act_progress = (show_progress and len(set(action_pd.symbol)) > 1)
    # 外部new一个进度条，为使用apply的操作使用
    with AbuProgress(len(action_pd), 0, label='capital.apply_action') as progress:
        # 针对每一笔交易进行buy，sell细节处理，涉及有限资金是否成交判定
        action_pd['deal'] = action_pd.apply(capital.apply_action, axis=1,
                                            args=(progress if show_apply_act_progress else None,))

    # 如果交易symbol数量 > 1000个显示apply进度条
    show_apply_kl = (show_progress and len(set(action_pd.symbol)) > 1000)
    # 根据交易行为产生的持仓列，持仓价值列更新资金时间序列
    capital.apply_kl(action_pd, kl_pd_manager, show_progress=show_apply_kl)

    # filter出所有持仓价值列
    stock_worths = capital.capital_pd.filter(regex='.*_worth')
    # 所有持仓价值列的sum形成stocks_blance列
    capital.capital_pd['stocks_blance'] = stock_worths.sum(axis=1)
    # stocks_blance ＋ cash_blance（现金余额）＝ capital_blance（总资产价值）列
    capital.capital_pd['capital_blance'] = capital.capital_pd['stocks_blance'] + capital.capital_pd['cash_blance']
