# -*- encoding:utf-8 -*-
"""
    手续费模块
"""

from __future__ import print_function
from __future__ import absolute_import
from __future__ import division

import logging
from contextlib import contextmanager

import numpy as np
import pandas as pd

from ..MarketBu.ABuSymbolFutures import AbuFuturesCn
from ..CoreBu.ABuFixes import partial
from ..CoreBu import ABuEnv
from ..CoreBu.ABuEnv import EMarketTargetType
from .ABuOrder import OrderMarket
from ..MarketBu import ABuMarket

__author__ = '阿布'
__weixin__ = 'abu_quant'


# noinspection PyUnusedLocal
def calc_commission_us(trade_cnt, price):
    """
    美股计算交易费用：每股0.01，最低消费2.99
    :param trade_cnt: 交易的股数（int）
    :param price: 每股的价格（美元）（暂不使用，只是保持接口统一）
    :return: 计算结果手续费
    """
    # 每股手续费0.01
    commission = trade_cnt * 0.01
    if commission < 2.99:
        # 最低消费2.99
        commission = 2.99
    return commission


def calc_commission_cn(trade_cnt, price):
    """
    a股计算交易费用：印花税＋佣金： 印花税万3，佣金万2.5
    :param trade_cnt: 交易的股数（int）
    :param price: 每股的价格（人民币）
    :return: 计算结果手续费
    """
    cost = trade_cnt * price
    # 印花税万3，
    tax = cost * 0.0003
    # 佣金万2.5
    commission = cost * 0.00025
    # 佣金最低5
    commission = commission if commission > 5 else 5
    commission += tax
    return commission


def calc_commission_hk(trade_cnt, price):
    """
    h股计算交易费用：印花税＋佣金： 佣金千分之二，印花税千分之一
    :param trade_cnt: 交易的股数（int）
    :param price: 每股的价格（人民币）
    :return: 计算结果手续费
    """
    cost = trade_cnt * price
    # 印花税千分之一
    tax = cost * 0.001
    # 佣金千分之二，
    commission = cost * 0.002
    commission += tax
    return commission


def calc_commission_tc(trade_cnt, price):
    """
    币类计算交易费用：只简单计算手续费，双向都使用流通币计算手续费，不涉及卖出使用币类的手续计算，
    如需要更精确计算，请使用自定义计算费率，即在AbuCommission初始化中自定义计算手续费的方法
    :param trade_cnt: 交易的币个数（int）
    :param price: 每币的价格（人民币）
    :return: 计算结果手续费
    """
    cost = trade_cnt * price
    # 双向都使用流通币计算手续费，千分之2
    commission = cost * 0.002
    return commission


# noinspection PyUnusedLocal
def calc_commission_futures_cn(trade_cnt, price, symbol_name):
    """
    期货计算交易费用：首先查询对应商品单位交易量（每手单位数量），以及每手手续费，再计算对应手续费
    :param trade_cnt: 交易的单位数量（int）
    :param price: 买入的价格（暂不使用，只是保持接口统一）
    :param symbol_name: 商品查询symbol
    :return: 计算结果手续费
    """
    min_unit = 10
    commission_unit = 10
    # 查询商品期货的对应df
    q_df = AbuFuturesCn().query_symbol(symbol_name)
    if q_df is not None:
        # 每手单位数量
        min_unit = q_df.min_unit.values[0]
        # 每手交易手续费
        commission_unit = q_df.commission.values[0]
    commission = trade_cnt / min_unit * commission_unit
    return commission


def calc_options_us(trade_cnt, price):
    """
    美股期权：差别很大，最好外部自定义自己的计算方法，这里只简单按照0.0035计算
    :param trade_cnt: 交易的股数（int）
    :param price: 每股的价格（美元
    :return: 计算结果手续费
    """
    cost = trade_cnt * price
    # 美股期权各个券商以及个人方式差别很大，最好外部自定义计算方法，这里只简单按照0.0035计算
    commission = cost * 0.0035
    return commission


def calc_commission_futures_global(trade_cnt, price):
    """
    国际期货：差别很大，最好外部自定义自己的计算方法，这里只简单按照0.002计算
    :param trade_cnt: 交易的股数（int）
    :param price: 每股的价格（美元）
    :return: 计算结果手续费
    """
    cost = trade_cnt * price
    # 国际期货各个券商以及代理方式差别很大，最好外部自定义计算方法，这里只简单按照0.002计算
    commission = cost * 0.002
    return commission


class AbuCommission(object):
    """交易手续费计算，记录，分析类，在AbuCapital中实例化"""

    def __init__(self, commission_dict):
        """
        :param commission_dict: 代表用户自定义手续费计算dict对象，
                                key：buy_commission_func， 代表用户自定义买入计算费用方法
                                key：sell_commission_func，代表用户自定义卖出计算费用方法
        """
        self.commission_dict = commission_dict
        # 对象内部记录交易的pd.DataFrame对象，列设定
        self.df_columns = ['type', 'date', 'symbol', 'commission']
        # 构建手续费记录pd.DataFrame对象commission_df
        self.commission_df = pd.DataFrame(columns=self.df_columns)

    def __str__(self):
        """打印对象显示：如果有手续费记录，打印记录df，否则打印commission_df.info"""
        if self.commission_df.shape[0] == 0:
            return str(self.commission_df.info())
        return str(self.commission_df)

    __repr__ = __str__

    # noinspection PyMethodMayBeStatic
    def _commission_enter(self, a_order):
        """
        通过a_order对象进行交易对象市场类型转换，分配对应手续费计算方法
        :param a_order: 交易单对象AbuOrder实例
        :return:
        """

        # 如果使用env中统一设置，即不需要通过OrderMarket对单子查询市场，提高运行效率，详ABuMarket
        market = ABuEnv.g_market_target if ABuMarket.g_use_env_market_set \
            else OrderMarket(a_order).symbol_market
        # 不同的市场不同的计算手续费方法
        if market == EMarketTargetType.E_MARKET_TARGET_US:
            # 美股
            calc_commission_func = calc_commission_us
        elif market == EMarketTargetType.E_MARKET_TARGET_CN:
            # a股
            calc_commission_func = calc_commission_cn
        elif market == EMarketTargetType.E_MARKET_TARGET_HK:
            # h股
            calc_commission_func = calc_commission_hk
        elif market == EMarketTargetType.E_MARKET_TARGET_TC:
            # 币类
            calc_commission_func = calc_commission_tc
        elif market == EMarketTargetType.E_MARKET_TARGET_FUTURES_CN:
            # 期货
            calc_commission_func = partial(calc_commission_futures_cn, symbol_name=a_order.buy_symbol)
        elif market == EMarketTargetType.E_MARKET_TARGET_OPTIONS_US:
            # 美股期权
            calc_commission_func = calc_options_us
        elif market == EMarketTargetType.E_MARKET_TARGET_FUTURES_GLOBAL:
            # 国际期货
            calc_commission_func = calc_commission_futures_global
        else:
            raise TypeError('buy_stock market error!!!')
        return calc_commission_func

    @contextmanager
    def buy_commission_func(self, a_order):
        """
        外部用with as 返回的list中需要加入计算的最终结果，否则不进行内部交易费用记录
        :param a_order: 买单对象AbuOrder实例
        """
        if self.commission_dict is not None and 'buy_commission_func' in self.commission_dict:
            # 如果有自定义计算交易费的方法使用自定义的
            buy_func = self.commission_dict['buy_commission_func']
        else:
            buy_func = self._commission_enter(a_order)

        # 使用list因为是可变类型，需要将外面的结果带回来
        commission_list = list()
        yield buy_func, commission_list

        # 如果有外部有append，说明需要记录手续费，且执行计算成功
        if len(commission_list) == 1:
            commission = commission_list[0]
            # 将买单对象AbuOrder实例中的数据转换成交易记录需要的np.array对象
            record = np.array(['buy', a_order.buy_date, a_order.buy_symbol, commission]).reshape(1, 4)
            record_df = pd.DataFrame(record, columns=self.df_columns)
            self.commission_df = self.commission_df.append(record_df)
        else:
            logging.info('buy_commission_func calc error')

    @contextmanager
    def sell_commission_func(self, a_order):
        """
        外部用with as 返回的list中需要加入计算的最终结果，否则不进行内部交易费用记录
        :param a_order: 卖单对象AbuOrder实例
        """
        if self.commission_dict is not None and 'sell_commission_func' in self.commission_dict:
            # 如果有自定义计算交易费的方法使用自定义的
            sell_func = self.commission_dict['sell_commission_func']
        else:
            sell_func = self._commission_enter(a_order)
        # 使用list因为是可变类型，需要将外面的结果带回来
        commission_list = list()

        yield sell_func, commission_list

        if len(commission_list) == 1:
            commission = commission_list[0]
            # 将卖单对象AbuOrder实例中的数据转换成交易记录需要的np.array对象
            record = np.array(['sell', a_order.sell_date, a_order.buy_symbol, commission]).reshape(1, 4)
            record_df = pd.DataFrame(record, columns=self.df_columns)
            self.commission_df = self.commission_df.append(record_df)
        else:
            logging.info('sell_commission_func calc error!!!')
