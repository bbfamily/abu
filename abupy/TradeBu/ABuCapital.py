# -*- encoding:utf-8 -*-
"""
    资金模块，不区分美元，人民币等类型，做美股交易默认当作美元，a股默认当作人民币
"""

from __future__ import print_function
from __future__ import absolute_import
from __future__ import division

import logging

import numpy as np
import pandas as pd

from ..UtilBu.ABuProgress import AbuProgress
from ..UtilBu import ABuDateUtil
from ..TradeBu.ABuOrder import AbuOrder
from ..TradeBu.ABuCommission import AbuCommission
from ..CoreBu.ABuBase import PickleStateMixin

__author__ = '阿布'
__weixin__ = 'abu_quant'


class AbuCapital(PickleStateMixin):
    """资金类"""

    def __init__(self, init_cash, benchmark, user_commission_dict=None):
        """
        :param init_cash: 初始资金值，注意这里不区分美元，人民币等类型，做美股交易默认当作美元，a股默认当作人民币，int
        :param benchmark: 资金回测时间标尺，做为资金类表格时间范围确定使用，AbuBenchmark实例对象
        :param user_commission_dict: dict对象，可自定义交易手续费计算方法，详情查看AbuCommission
        """
        self.read_cash = init_cash

        kl_pd = benchmark.kl_pd
        if kl_pd is None:
            # 要求基准必须有数据
            raise ValueError('CapitalClass init klPd is None')

        # 根据基准时间序列，制作相同的时序资金对象capital_pd(pd.DcataFrame对象)
        self.capital_pd = pd.DataFrame(
            {
                'cash_blance': np.NAN * kl_pd.shape[0],
                'stocks_blance': np.zeros(kl_pd.shape[0]),
                'atr21': kl_pd['atr21'],
                'date': kl_pd['date']
            },
            index=kl_pd.index)

        self.capital_pd['date'] = self.capital_pd['date'].astype(int)
        # cash_blance除了第一个其它都是nan
        self.capital_pd.loc[
            self.capital_pd.index[0], 'cash_blance'] = self.read_cash
        # 构造交易手续费对象AbuCommission，如果user自定义手续费计算方法，通过user_commission_dict传入
        self.commission = AbuCommission(user_commission_dict)

    def __str__(self):
        """打印对象显示：capital_pd.info commission_df.info"""
        return 'capital_pd:\n{}\ncommission_pd:\n{}'.format(self.capital_pd.info(),
                                                            self.commission.commission_df.info())

    __repr__ = __str__

    def __len__(self):
        """对象长度：时序资金对象capital_pd的行数，即self.capital_pd.shape[0]"""
        return self.capital_pd.shape[0]

    def init_k_line(self, a_symbol):
        """
        每一个交易对象在时序资金对象capital_pd上都添加对应的call keep（买涨持仓量），call worth（买涨总价值），
        put keep（买跌持仓量），put worth（买跌总价值）
        :param a_symbol: symbol str对象
        """

        # 买涨持仓量
        call_keep = '_call_keep'
        if self.capital_pd.columns.tolist().count(a_symbol + call_keep) == 0:
            self.capital_pd[a_symbol + call_keep] = np.NAN * \
                                                    self.capital_pd.shape[0]
        # 买跌持仓量
        put_keep = '_put_keep'
        if self.capital_pd.columns.tolist().count(a_symbol + put_keep) == 0:
            self.capital_pd[a_symbol + put_keep] = np.NAN * \
                                                   self.capital_pd.shape[0]
        # 买涨总价值
        call_worth = '_call_worth'
        if self.capital_pd.columns.tolist().count(a_symbol + call_worth) == 0:
            self.capital_pd[a_symbol + call_worth] = np.NAN * \
                                                     self.capital_pd.shape[0]

        # 买跌总价值
        put_worth = '_put_worth'
        if self.capital_pd.columns.tolist().count(a_symbol + put_worth) == 0:
            self.capital_pd[a_symbol + put_worth] = np.NAN * \
                                                    self.capital_pd.shape[0]

    def apply_init_kl(self, action_pd, show_progress):
        """
        根据回测交易在时序资金对象capital_pd上新建对应的call，put列
        :param action_pd: 回测交易行为对象，pd.DataFrame对象
        :param show_progress: 外部设置是否需要显示进度条
        """
        # 使用set筛选唯一的symbol交易序列
        symbols = set(action_pd.symbol)
        # 单进程进度条
        with AbuProgress(len(symbols), 0, label='apply_init_kl...') as progress:
            for pos, symbol in enumerate(symbols):
                if show_progress:
                    progress.show(a_progress=pos + 1)
                # 迭代symbols，新建对应的call，put列
                self.init_k_line(symbol)

    def apply_k_line(self, a_k_day, kl_pd, buy_type_head):
        """
        在apply_kl中的do_apply_kl方法中时序资金对象capital进行apply的对应方法，
        即迭代金融时间序列的每一个交易日，根据持仓量计算每一个交易日的市场价值
        :param a_k_day: 每一个被迭代中的时间，即每一个交易日数据
        :param kl_pd: 正在被apply迭代的金融时间序列本体，pd.DataFrame对象
        :param buy_type_head: 代表交易类型，范围（_call，_put）
        :return:
        """
        if a_k_day[kl_pd.name + buy_type_head + '_keep'] > 0:
            kl = kl_pd[kl_pd['date'] == a_k_day['date']]
            if kl is None or kl.shape[0] == 0:
                # 前提是当前交易日有对应的持仓
                return

            # 今天的收盘价格
            td_close = kl['close'].values[0]
            if buy_type_head == '_put':
                # 针对buy put对价格进行转换，主要是针对put特殊处理
                today_key = kl.key.values[0]
                if today_key > 0:
                    yd_close = kl_pd.iloc[today_key - 1].close
                    # 如果是买跌，实时市场收益以昨天为基础进行计算，即＊－1进行方向计算，以收益来映射重新定义今天的收盘价格
                    td_close = (td_close - yd_close) * -1 + yd_close
            # 根据持仓量即处理后的今日收盘价格，进行今日价值计算
            self.capital_pd.loc[kl.index, [kl_pd.name + buy_type_head + '_worth']] \
                = np.round(td_close * a_k_day[kl_pd.name + buy_type_head + '_keep'], 3)

    def apply_kl(self, action_pd, kl_pd_manager, show_progress):
        """
        apply_action之后对实际成交的交易分别迭代更新时序资金对象capital_pd上每一个交易日的实时价值
        :param action_pd: 回测结果生成的交易行为构成的pd.DataFrame对象
        :param kl_pd_manager: 金融时间序列管理对象，AbuKLManager实例
        :param show_progress: 是否显示进度条
        """

        # 在apply_action之后形成deal列后，set出考虑资金下成交了的交易序列
        deal_symbols_set = set(action_pd[action_pd['deal'] == 1].symbol)

        def do_apply_kl(kl_pd, buy_type_head):
            """
            根据金融时间序列在时序资金对象capital_pd上进行call（买涨），put（买跌）的交易日实时价值更新
            :param kl_pd: 金融时间序列，pd.DataFrame对象
            :param buy_type_head: 代表交易类型，范围（_call，_put）
            """
            # cash_blance对na进行pad处理
            self.capital_pd['cash_blance'].fillna(method='pad', inplace=True)
            # symbol对应列持仓量对na进行处理
            self.capital_pd[kl_pd.name + buy_type_head + '_keep'].fillna(method='pad', inplace=True)
            self.capital_pd[kl_pd.name + buy_type_head + '_keep'].fillna(0, inplace=True)

            # 使用apply在axis＝1上，即每一个交易日上对持仓量及市场价值进行更新
            self.capital_pd.apply(self.apply_k_line, axis=1, args=(kl_pd, buy_type_head))

            # symbol对应列市场价值对na进行处理
            self.capital_pd[kl_pd.name + buy_type_head + '_worth'].fillna(method='pad', inplace=True)
            self.capital_pd[kl_pd.name + buy_type_head + '_worth'].fillna(0, inplace=True)

            # 纠错处理把keep=0但是worth被pad的进行二次修正
            fe_mask = (self.capital_pd[kl_pd.name + buy_type_head + '_keep'] == 0) & (
                self.capital_pd[kl_pd.name + buy_type_head + '_worth'] > 0)
            # 筛出需要纠错的index
            fe_index = self.capital_pd[fe_mask].index
            # 将需要纠错的对应index上市场价值进行归零处理
            cp_w = self.capital_pd[kl_pd.name + buy_type_head + '_worth']
            cp_w.loc[fe_index] = 0

        # 单进程进度条
        with AbuProgress(len(deal_symbols_set), 0, label='apply_kl...') as progress:
            for pos, deal_symbol in enumerate(deal_symbols_set):
                if show_progress:
                    progress.show(a_progress=pos + 1)
                # 从kl_pd_manager中获取对应的金融时间序列kl，每一个kl分别进行call（买涨），put（买跌）的交易日实时价值更新
                kl = kl_pd_manager.get_pick_time_kl_pd(deal_symbol)
                # 进行call（买涨）的交易日实时价值更新
                do_apply_kl(kl, '_call')
                # 进行put（买跌）的交易日实时价值更新
                do_apply_kl(kl, '_put')

    def apply_action(self, a_action, progress):
        """
        在回测结果生成的交易行为构成的pd.DataFrame对象上进行apply对应本方法，即
        将交易行为根据资金情况进行处理，处理手续费以及时序资金对象capital_pd上的
        数据更新
        :param a_action: 每一个被迭代中的action，即每一个交易行为
        :param progress: 进度条对象
        :return: 是否成交deal bool
        """
        # 区别买入行为和卖出行为
        is_buy = True if a_action['action'] == 'buy' else False
        # 从action数据构造AbuOrder对象
        order = AbuOrder()
        order.buy_symbol = a_action['symbol']
        order.buy_cnt = a_action['Cnt']
        if is_buy:
            # 如果是买单，sell_price = price2 ,详情阅读ABuTradeExecute中transform_action
            order.buy_price = a_action['Price']
            order.sell_price = a_action['Price2']
        else:
            # 如果是卖单，buy_price = price2 ,详情阅读ABuTradeExecute中transform_action
            order.sell_price = a_action['Price']
            order.buy_price = a_action['Price2']
        # 交易发生的时间
        order.buy_date = a_action['Date']
        order.sell_date = a_action['Date']
        # 交易的方向
        order.expect_direction = a_action['Direction']

        # 对买单和卖单分别进行处理，确定是否成交deal
        deal = self.buy_stock(order) if is_buy else self.sell_stock(order)

        if progress is not None:
            progress.show()

        return deal

    def buy_stock(self, a_order):
        """
        在apply_action中每笔交易进行处理，根据买单计算cost，在时序资金对象capital_pd上修改对应cash_blance，
        以及更新对应symbol上的持仓量
        :param a_order: 在apply_action中由action转换的AbuOrder对象
        :return: 是否成交deal bool
        """

        # 首先使用commission对象计算手续费
        with self.commission.buy_commission_func(a_order) as (buy_func, commission_list):
            commission = buy_func(a_order.buy_cnt, a_order.buy_price)
            # 将上下文管理器中返回的commission_list中添加计算结果commission，内部根据list长度决定写入手续费记录pd.DataFrame
            commission_list.append(commission)
        # cost = 买单数量 ＊ 单位价格 ＋ 手续费
        order_cost = a_order.buy_cnt * a_order.buy_price + commission
        # 买单时间转换成pd时间日期对象
        time_ind = pd.to_datetime(ABuDateUtil.fmt_date(a_order.buy_date))
        # pd时间日期对象置换出对应的index number
        num_index = self.capital_pd.index.tolist().index(time_ind)

        # cash_blance初始化init中除了第一个其它都是nan
        cash_blance = self.capital_pd['cash_blance'].dropna()

        # 先截取cash_blance中 < time_ind的时间序列
        bl_ss = cash_blance[cash_blance.index <= time_ind]
        if bl_ss is None or bl_ss.shape[0] == 0:
            logging.info('bl_ss.shape[0] == 0 ' + str(a_order.buy_date))
            return False
        # 截取的bl_ss中选中最后一个元素做为判定买入时刻的cash值
        cash = bl_ss.iloc[-1]
        # 判定买入时刻的cash值是否能够钱买入
        if cash >= order_cost and a_order.buy_cnt > 0:
            # 够的话，买入，先将cash － cost
            cash -= order_cost
            # 根据a_order.expect_direction确定是要更新call的持仓量还是put的持仓量
            buy_type_keep = '_call_keep' if a_order.expect_direction == 1.0 else '_put_keep'
            # 前提1: 资金时间序列中有这个a_order.buy_symbol + buy_type_keep列
            has_cond1 = self.capital_pd.columns.tolist().count(a_order.buy_symbol + buy_type_keep) > 0
            # 前提2: 对应这个列从单子买入日期index开始，在dropna后还有shape说明本就有持仓
            has_cond2 = self.capital_pd[a_order.buy_symbol +
                                        buy_type_keep].iloc[:num_index + 1].dropna().shape[0] > 0

            keep_cnt = 0
            if has_cond1 and has_cond2:
                # 前提1 + 前提2->本就有持仓, 拿到之前的持仓量
                keep_cnt = self.capital_pd[a_order.buy_symbol
                                           + buy_type_keep].iloc[:num_index + 1].dropna()[-1]

            keep_cnt += a_order.buy_cnt

            # TODO 这里迁移之前逻辑，删除了一些未迁移模块的逻辑，所以看起来多写了一个流程，需要重构逻辑
            # 将计算好的cash更新到资金时间序列中对应的位置
            self.capital_pd.loc[time_ind, ['cash_blance']] = np.round(cash, 3)
            # 资金时间序列中之后的cash_blance也对应开始减去order_cost
            self.capital_pd.loc[cash_blance[cash_blance.index > time_ind].index, ['cash_blance']] \
                -= order_cost
            # 在更新持仓量前，取出之前的数值
            org_cnt = self.capital_pd.loc[time_ind][a_order.buy_symbol + buy_type_keep]
            # 更新持仓量
            self.capital_pd.loc[time_ind, [a_order.buy_symbol + buy_type_keep]] = keep_cnt
            if not np.isnan(org_cnt):
                # 对多个因子作用在同一个symbol上，且重叠了持股时间提前更新之后的持仓量
                keep_pos = self.capital_pd[a_order.buy_symbol + buy_type_keep].dropna()
                self.capital_pd.loc[keep_pos[keep_pos.index > time_ind].index, [a_order.buy_symbol + buy_type_keep]] \
                    += a_order.buy_cnt
            return True
        else:
            return False

    def sell_stock(self, a_order):
        """
        在apply_action中每笔交易进行处理，根据卖单计算cost，在时序资金对象capital_pd上修改对应cash_blance，
        以及更新对应symbol上的持仓量
        :param a_order: 在apply_action中由action转换的AbuOrder对象
        :return: 是否成交deal bool
        """

        # 卖单时间转换成pd时间日期对象
        time_ind = pd.to_datetime(ABuDateUtil.fmt_date(a_order.sell_date))
        # # pd时间日期对象置换出对应的index number
        num_index = self.capital_pd.index.tolist().index(time_ind)
        # 根据a_order.expect_direction确定是要更新call的持仓量还是put的持仓量
        buy_type_keep = '_call_keep' if a_order.expect_direction == 1.0 else '_put_keep'
        # 前提1: 资金时间序列中有这个a_order.buy_symbol + buy_type_keep列
        has_cond1 = self.capital_pd.columns.tolist().count(a_order.buy_symbol + buy_type_keep) > 0
        # 前提2: 对应这个列从单子买入日期index开始，在dropna后还有shape说明本就有持仓
        has_cond2 = self.capital_pd[a_order.buy_symbol + buy_type_keep].iloc[:num_index + 1].dropna().shape[0] > 0

        if has_cond1 and has_cond2:
            # 有持仓, 拿到之前的持仓量
            keep_cnt = self.capital_pd[a_order.buy_symbol + buy_type_keep].iloc[:num_index + 1].dropna()[-1]
            sell_cnt = a_order.buy_cnt

            if keep_cnt < sell_cnt:
                # 忽略一个问题，就算是当时买时的这个单子没有成交，这里也试图卖出当时设想买入的股数
                sell_cnt = keep_cnt
            if sell_cnt == 0:
                # 有可能由于没买入的单子，造成没有成交
                return False
            # 更新对应持仓量
            keep_cnt -= sell_cnt
            # 将卖出价格转换成call，put都可计算收益的价格，不要进行计算公式合并，保留冗余，便于理解
            sell_earn_price = (a_order.sell_price - a_order.buy_price) * a_order.expect_direction + a_order.buy_price
            order_earn = sell_earn_price * sell_cnt

            # 使用commission对象计算手续费
            with self.commission.sell_commission_func(a_order) as (sell_func, commission_list):
                commission = sell_func(sell_cnt, a_order.sell_price)
                # 将上下文管理器中返回的commission_list中添加计算结果commission，内部根据list长度决定写入手续费记录pd.DataFrame
                commission_list.append(commission)

            # cash_blance初始化init中除了第一个其它都是nan
            cash_blance = self.capital_pd['cash_blance'].dropna()
            # 截取 < time_indd的cash_blance中最后一个元素做为cash值
            cash = cash_blance[cash_blance.index <= time_ind].iloc[-1]
            cash += (order_earn - commission)
            # 将计算好的cash更新到资金时间序列中对应的位置
            self.capital_pd.loc[time_ind, ['cash_blance']] = np.round(cash, 3)

            # TODO 这里迁移之前逻辑，删除了一些未迁移模块的逻辑，所以看起来多写了一个流程，需要重构逻辑
            self.capital_pd.loc[cash_blance[cash_blance.index > time_ind].index, ['cash_blance']] \
                += (order_earn - commission)

            org_cnt = self.capital_pd.loc[time_ind][a_order.buy_symbol + buy_type_keep]
            # 更新持仓量
            self.capital_pd.loc[time_ind, [a_order.buy_symbol + buy_type_keep]] = keep_cnt

            if not np.isnan(org_cnt):
                # 针对diff factor same stock的情况
                keep_pos = self.capital_pd[a_order.buy_symbol + buy_type_keep].dropna()
                self.capital_pd.loc[keep_pos[keep_pos.index > time_ind].index, [a_order.buy_symbol + buy_type_keep]] \
                    -= sell_cnt
            return True
        else:
            return False
