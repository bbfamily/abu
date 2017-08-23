# -*- encoding:utf-8 -*-
"""
    梦想中的机器学习股票数据环境
"""
import numpy as np
from abupy import ABuSymbolPd
import sklearn.preprocessing as preprocessing

"""
    是否开启date_week噪音
"""
g_with_date_week_noise = True


def _gen_another_word_price(kl_another_word):
    """
    生成股票在另一个世界中的价格
    :param kl_another_word:
    :return:
    """
    for ind in np.arange(2, kl_another_word.shape[0]):
        # 前天数据
        bf_yesterday = kl_another_word.iloc[ind - 2]
        # 昨天
        yesterday = kl_another_word.iloc[ind - 1]
        # 今天
        today = kl_another_word.iloc[ind]
        # 生成今天的收盘价格
        kl_another_word.close[ind] = _gen_another_word_price_rule(yesterday.close, yesterday.volume,
                                                                  bf_yesterday.close, bf_yesterday.volume,
                                                                  today.volume, today.date_week)


def _gen_another_word_price_rule(yesterday_close, yesterday_volume, bf_yesterday_close, bf_yesterday_volume,
                                 today_volume, date_week):
    """
        通过前天收盘量价，昨天收盘量价，今天的量，构建另一个世界中的价格模型
    """
    price_change = yesterday_close - bf_yesterday_close
    volume_change = yesterday_volume - bf_yesterday_volume

    # 如果量和价变动一致，今天价格涨，否则跌
    sign = 1.0 if price_change * volume_change > 0 else -1.0

    # 通过date_week生成噪音，否则之后分类100%分对
    if g_with_date_week_noise:
        # 噪音的先决条件是今天的量是这三天最大的
        gen_noise = today_volume > np.max([yesterday_volume, bf_yesterday_volume])
        # 如果是周五，下跌
        if gen_noise and date_week == 4:
            sign = -1.0
        # 如果是周一，上涨
        elif gen_noise and date_week == 0:
            sign = 1.0

    # 今天的涨跌幅度基础是price_change（昨天前天的价格变动）
    price_base = abs(price_change)
    # 今天的涨跌幅度变动因素
    price_factor = np.mean([today_volume / yesterday_volume, today_volume / bf_yesterday_volume])

    # 如果涨跌幅度超过10%，限制上限，下限为10%
    if abs(price_base * price_factor) < yesterday_close * 0.10:
        today_price = yesterday_close + sign * price_base * price_factor
    else:
        today_price = yesterday_close + sign * yesterday_close * 0.10
    return today_price


def change_real_to_another_word(symbol):
    """
    将原始真正的股票数据只保留价格的头两个，量，周几，将其它价格使用_gen_another_word_price变成另一个世界价格
    :param symbol:
    :return:
    """
    kl_pd = ABuSymbolPd.make_kl_df(symbol)
    if kl_pd is not None:
        kl_dream = kl_pd.filter(['close', 'date_week', 'volume'])
        # 只保留原始头两天的交易收盘价格
        kl_dream['close'][2:] = np.nan
        # 将其它价格变成另一个世界中价格
        _gen_another_word_price(kl_dream)
        return kl_dream


def gen_pig_three_feature(kl_another_word):
    """
        猪老三构建特征模型函数
    """
    # 回顾预测的y值
    kl_another_word['regress_y'] = kl_another_word.close.pct_change()
    # 前天收盘价格
    kl_another_word['bf_yesterday_close'] = 0
    # 昨天收盘价格
    kl_another_word['yesterday_close'] = 0
    # 昨天收盘成交量
    kl_another_word['yesterday_volume'] = 0
    # 前天收盘成交量
    kl_another_word['bf_yesterday_volume'] = 0

    # 今天收盘成交量, 不用了用了之后更接近完美，但也算是使用了未来数据，虽然可以狡辩说为快收盘时候买入
    # kl_deram['feature_today_volume'] = kl_deram['volume']

    # 对其特征
    kl_another_word['bf_yesterday_close'][2:] = kl_another_word['close'][:-2]
    kl_another_word['bf_yesterday_volume'][2:] = kl_another_word['volume'][:-2]
    kl_another_word['yesterday_close'][1:] = kl_another_word['close'][:-1]
    kl_another_word['yesterday_volume'][1:] = kl_another_word['volume'][:-1]

    # 特征1: 价格差
    kl_another_word['feature_price_change'] = kl_another_word['yesterday_close'] - kl_another_word['bf_yesterday_close']
    # 特征2: 成交量差
    kl_another_word['feature_volume_Change'] = kl_another_word['yesterday_volume'] - kl_another_word[
        'bf_yesterday_volume']

    # 特征3: 涨跌sign
    kl_another_word['feature_sign'] = np.sign(
        kl_another_word['feature_price_change'] * kl_another_word['feature_volume_Change'])

    # 为之后kmena实例准备数据
    kmean_date_week = kl_another_word['date_week']

    # 构建噪音特征, 因为猪老三也不可能全部分析正确真实的特征因素，这里引入一些噪音特征
    # 成交量乘积
    kl_another_word['feature_volume_noise'] = kl_another_word['yesterday_volume'] * kl_another_word[
        'bf_yesterday_volume']
    # 价格乘积
    kl_another_word['feature_price_noise'] = kl_another_word['yesterday_close'] * kl_another_word['bf_yesterday_close']

    # 将数据标准化
    scaler = preprocessing.StandardScaler()
    kl_another_word['feature_price_change'] = scaler.fit_transform(
        kl_another_word['feature_price_change'].values.reshape(-1, 1))
    kl_another_word['feature_volume_Change'] = scaler.fit_transform(
        kl_another_word['feature_volume_Change'].values.reshape(-1, 1))
    kl_another_word['feature_volume_noise'] = scaler.fit_transform(
        kl_another_word['feature_volume_noise'].values.reshape(-1, 1))
    kl_another_word['feature_price_noise'] = scaler.fit_transform(
        kl_another_word['feature_price_noise'].values.reshape(-1, 1))

    # 只筛选feature_开头的特征和regress_y
    kl_pig_three_feature = kl_another_word.filter(regex='regress_y|feature_*')[2:]
    return kl_pig_three_feature, kmean_date_week[2:]
