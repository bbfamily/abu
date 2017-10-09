# -*- encoding:utf-8 -*-
"""
    交易订单模块
"""

from __future__ import print_function
from __future__ import absolute_import
from __future__ import division

import math

import numpy as np

from ..CoreBu import ABuEnv
from ..CoreBu.ABuEnv import EMarketTargetType
from ..MarketBu.ABuSymbolFutures import AbuFuturesCn, AbuFuturesGB
from ..MarketBu.ABuHkUnit import AbuHkUnit
from ..MarketBu import ABuMarket
from ..MarketBu.ABuMarket import MarketMixin

__author__ = '阿布'
__weixin__ = 'abu_quant'


class OrderMarket(MarketMixin):
    """根据AbuOrder对象，设置混入类MarketMixin中symbol_name"""

    def __init__(self, order):
        """
        设置混入类MarketMixin中symbol_name，以获取symbol_market等信息
        :param order: AbuOrder对象
        """
        if isinstance(order, AbuOrder):
            self.symbol_name = order.buy_symbol
        else:
            raise TypeError('order is AbuOrder object!!!')


# noinspection PyAttributeOutsideInit
class AbuOrder(object):
    """交易订单类"""

    # 多个因子买入条件可能生成几百万个order对象使用__slots__降低内存消耗
    __slots__ = ('order_deal', 'buy_symbol', 'buy_date', 'buy_factor', 'buy_factor_class',
                 'buy_price', 'buy_cnt', 'buy_pos', 'sell_date',
                 'buy_type_str', 'expect_direction',
                 'sell_type', 'keep_days', 'sell_price', 'sell_type_extra', 'ml_features')

    def __init__(self):
        """初始设置只需要将order_deal设置未成交状态"""
        self.order_deal = False

    def fit_buy_order(self, day_ind, factor_object):
        """
        根据买入交易日当当天数据以及买入因子，拟合计算买入订单
        :param day_ind: 买入交易发生的时间索引，即对应self.kl_pd.key
        :param factor_object: ABuFactorBuyBases子类实例对象
        """
        kl_pd = factor_object.kl_pd
        # 要执行买入当天的数据
        kl_pd_buy = kl_pd.iloc[day_ind + 1]
        # 买入因子名称
        factor_name = factor_object.factor_name if hasattr(factor_object, 'factor_name') else 'unknown'
        # 日内滑点决策类
        slippage_class = factor_object.slippage_class
        # 仓位管理类设置
        position_class = factor_object.position_class
        # 初始资金，也可修改策略使用剩余资金
        read_cash = factor_object.capital.read_cash
        # 实例化滑点类
        fact = slippage_class(kl_pd_buy, factor_name)
        # 执行fit_price, 计算决策买入价格
        bp = fact.fit()
        # 如果滑点类中决定不买入，撤单子，bp就返回正无穷
        if bp < np.inf:
            """
                实例化仓位管理类
                仓位管理默认保证金比例是1，即没有杠杆，修改ABuPositionBase.g_deposit_rate可提高融资能力，
                如果margin＝2－>ABuPositionBase.g_deposit_rate = 0.5, 即只需要一半的保证金，也可同过构建
                时使用关键字参数完成保证金比例传递
            """
            position = position_class(kl_pd_buy, factor_name, factor_object.kl_pd.name, bp, read_cash,
                                      **factor_object.position_kwargs)

            market = ABuEnv.g_market_target if ABuMarket.g_use_env_market_set else position.symbol_market
            """
                由于模块牵扯复杂，暂时不迁移保证金融资相关模块，期货不使用杠杆，即回测不牵扯资金总量的评估
                if market == EMarketTargetType.E_MARKET_TARGET_FUTURES_CN:
                    deposit_rate = 0.10
                    q_df = AbuFuturesCn().query_symbol(factor_object.kl_pd.name)
                    if q_df is not None:
                        deposit_rate = q_df.min_deposit.values[0]
                    # 重新设置保证金比例
                    position.deposit_rate = deposit_rate
            """
            # 执行fit_position，通过仓位管理计算买入的数量
            bc = position.fit_position(factor_object)
            if np.isnan(bc):
                return

            if market != EMarketTargetType.E_MARKET_TARGET_TC:
                # 除了比特币市场外，都向下取整数到最小交易单位个数
                buy_cnt = int(math.floor(bc))
            else:
                # 币类市场可以买非整数个, 保留三位小数
                buy_cnt = round(bc, 3)

            if market == EMarketTargetType.E_MARKET_TARGET_US:
                # 美股1
                min_cnt = 1
            elif market == EMarketTargetType.E_MARKET_TARGET_TC:
                # 国内一般只支持到0.01个
                min_cnt = 0.01
            elif market == EMarketTargetType.E_MARKET_TARGET_CN:
                # A股最小100一手
                min_cnt = 100
                # 向最小的手量看齐
                buy_cnt -= buy_cnt % min_cnt
            elif market == EMarketTargetType.E_MARKET_TARGET_HK:
                # 港股从AbuHkUnit读取数据，查询对应symbol每一手的交易数量
                min_cnt = AbuHkUnit().query_unit(factor_object.kl_pd.name)
                # 向最小的手量看齐
                buy_cnt -= buy_cnt % min_cnt
            elif market == EMarketTargetType.E_MARKET_TARGET_FUTURES_CN:
                # 国内期货，查询最少一手单位
                min_cnt = AbuFuturesCn().query_min_unit(factor_object.kl_pd.name)
                # 向最小的手量看齐
                buy_cnt -= buy_cnt % min_cnt
            elif market == EMarketTargetType.E_MARKET_TARGET_OPTIONS_US:
                # 美股期权最小合约单位1contract，代表100股股票权利
                min_cnt = 100
                buy_cnt -= buy_cnt % min_cnt
            elif market == EMarketTargetType.E_MARKET_TARGET_FUTURES_GLOBAL:
                # 国际期货, 查询最少一手单位
                min_cnt = AbuFuturesGB().query_min_unit(factor_object.kl_pd.name)
                buy_cnt -= buy_cnt % min_cnt
            else:
                raise TypeError('ABuEnv.g_market_target ERROR, market={}, g_market_target={}'.format(
                    market, ABuEnv.g_market_target))

            if buy_cnt < min_cnt:
                # 不够买最少单位量
                return

            # 如下生成order内部数据
            self.buy_symbol = kl_pd.name
            # 订单写入买入日期
            self.buy_date = int(kl_pd_buy.date)
            # 订单写入买入因子名字
            self.buy_factor = factor_name
            # 订单对象中添加买入因子类名，和buy_factor不同没有具体参数等唯一key
            self.buy_factor_class = factor_object.__class__.__name__
            # 订单写入买入价格
            self.buy_price = bp
            # 订单写入买入数量
            self.buy_cnt = buy_cnt
            # 订单写入仓位管理类名称
            self.buy_pos = position.__class__.__name__
            # 订单写入买入类型，call or put
            self.buy_type_str = factor_object.buy_type_str
            # 订单写入买入因子期望方向
            self.expect_direction = factor_object.expect_direction

            # 如下卖出信息具体写入在fit_sell_order中
            # 订单卖出时间
            self.sell_date = None
            # 订单卖出类型，keep：持有
            self.sell_type = 'keep'
            # 交易日持有天数
            self.keep_days = 0

            # 订单卖出价格
            self.sell_price = None
            # 订单卖出额外信息
            self.sell_type_extra = ''

            # 订单买入，卖出特征
            self.ml_features = None
            # 订单形成
            self.order_deal = True

    def fit_sell_order(self, day_ind, factor_object):
        """
        根据卖出交易日当当天数据以及卖出因子，拟合计算卖出信息，完成订单
        :param day_ind: 卖出交易发生的时间索引，即对应self.kl_pd.key
        :param factor_object: AbuFactorSellBase子类实例对象
        """

        if self.sell_type != 'keep':
            # 保证外部不需要过滤单子，内部自己过滤已经卖出成交的订单
            return

        # 卖出策略中进行特征合成, 以及ump拦截卖出行为
        if factor_object.make_sell_order(self, day_ind):
            kl_pd_sell = factor_object.kl_pd.iloc[day_ind + 1]

            # 日内滑点决策类
            slippage_class = factor_object.slippage_class
            # 卖出因子名称
            factor_name = factor_object.factor_name if hasattr(factor_object, 'factor_name') else 'unknown'
            # 实例化日内滑点决策类，进行具体卖出价格决策
            sell_price = slippage_class(kl_pd_sell, factor_name).fit()

            if sell_price == -np.inf:
                # 如果卖出执行返回负无穷说明无法卖出，例如跌停
                return

            self.sell_price = sell_price
            # 卖出原因其它描述
            sell_type_extra = factor_object.sell_type_extra if hasattr(factor_object, 'sell_type_extra') else 'unknown'
            self.sell_type_extra = sell_type_extra
            if self.buy_type_str == 'call':
                # call卖出类型:  win = self.sell_price > self.buy_price
                self.sell_type = 'win' if self.sell_price > self.buy_price else 'loss'
            else:
                # put卖出类型:  loss = self.sell_price > self.buy_price
                self.sell_type = 'loss' if self.sell_price > self.buy_price else 'win'
            # 卖出日期写入单子
            self.sell_date = int(kl_pd_sell.date)

    def __str__(self):
        """打印对象显示：buy_symbol， buy_price， buy_cnt， buy_date，buy_factor，sell_date，sell_type， sell_price"""
        return 'buy Symbol = ' + str(self.buy_symbol) + '\n' \
               + 'buy Prices = ' + str(self.buy_price) + '\n' \
               + 'buy cnt = ' + str(self.buy_cnt) + '\n' \
               + 'buy date = ' + str(self.buy_date) + '\n' \
               + 'buy factor = ' + str(self.buy_factor) + '\n' \
               + 'sell date = ' + str(self.sell_date) + '\n' \
               + 'sell type = ' + str(self.sell_type) + '\n' \
               + 'sell Price = ' + str(self.sell_price) + '\n'

    __repr__ = __str__
