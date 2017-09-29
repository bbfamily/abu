# -*- encoding:utf-8 -*-
"""
    abupy中使用的金融时间序列分析模块, 模块真的方法真的参数都为abupy中格式化好的kl如下

    eg:
                close	high	low	p_change	open	pre_close	volume	date	date_week	key	atr21	atr14
    2016-07-20	228.36	229.800	225.00	1.38	226.47	225.26	2568498	20160720	2	499	9.1923	8.7234
    2016-07-21	220.50	227.847	219.10	-3.44	226.00	228.36	4428651	20160721	3	500	9.1711	8.7251
    2016-07-22	222.27	224.500	218.88	0.80	221.99	220.50	2579692	20160722	4	501	9.1858	8.7790
    2016-07-25	230.01	231.390	221.37	3.48	222.27	222.27	4490683	20160725	0	502	9.2669	8.9298
    2016-07-26	225.93	228.740	225.63	-1.77	227.34	230.01	41833	20160726	1	503	9.1337	8.7541
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from collections import Iterable

import logging

import numpy as np
import pandas as pd

from ..CoreBu import ABuEnv
from ..CoreBu.ABuPdHelper import pd_resample

__author__ = '阿布'
__weixin__ = 'abu_quant'

log_func = logging.info if ABuEnv.g_is_ipython else print


def _df_dispatch(df, dispatch_func):
    """
    根据df的类型分发callable的执行方法，

    :param df: abupy中格式化好的kl，或者字典，或者可迭代序列
    :param dispatch_func: 分发的可执行的方法
    """
    if isinstance(df, pd.DataFrame):
        # 参数只是pd.DataFrame
        return dispatch_func(df)
    elif isinstance(df, dict) and all([isinstance(_df, pd.DataFrame) for _df in df.values()]):
        # 参数只是字典形式
        return [dispatch_func(df[df_key], df_key) for df_key in df]
    elif isinstance(df, Iterable) and all([isinstance(_df, pd.DataFrame) for _df in df]):
        # 参数只是可迭代序列
        return [dispatch_func(_df) for _df in df]
    else:
        log_func('df type is error! {}'.format(type(df)))


def _df_dispatch_concat(df, dispatch_func):
    """
    根据df的类型分发callable的执行方法，如果是字典或者可迭代类型的返回值使用
    pd.concat连接起来

    :param df: abupy中格式化好的kl，或者字典，或者可迭代序列
    :param dispatch_func: 分发的可执行的方法
    """

    if isinstance(df, pd.DataFrame):
        # 参数只是pd.DataFrame
        return dispatch_func(df)
    elif isinstance(df, dict) and all([isinstance(_df, pd.DataFrame) for _df in df.values()]):
        # 参数只是字典形式
        return pd.concat([dispatch_func(df[df_key], df_key) for df_key in df], axis=1)
    elif isinstance(df, Iterable) and all([isinstance(_df, pd.DataFrame) for _df in df]):
        # 参数只是可迭代序列
        return pd.concat([dispatch_func(_df) for _df in df], axis=1)
    else:
        log_func('df type is error! {}'.format(type(df)))


def resample_close_mean(df, bins=None):
    """
    对金融时间序列进行变换周期重新采样，对重新采样的结果进行pct_change处理
    ，对pct_change序列取abs绝对值，对pct_change绝对值序列取平均，即算出
    重新采样的周期内的平均变化幅度

    eg:
    tsla = ABuSymbolPd.make_kl_df('usTSLA')
    ABuKLUtil.resample_close_mean(tsla)

    out:
                resample
        5D	    0.0340
        10D	    0.0468
        21D	    0.0683
        42D	    0.0805
        60D	    0.1002
        90D	    0.0931
        120D    0.0939

    :param df: abupy中格式化好的kl，或者字典，或者可迭代序列
    :param bins: 默认eg:  ['5D', '10D', '21D', '42D', '60D', '90D', '120D']
    :return: pd.DataFrame
    """

    def _resample_close_mean(p_df, df_name=''):
        resample_dict = {}
        for _bin in bins:
            change = abs(pd_resample(p_df.close, _bin, how='mean').pct_change()).mean()
            """
                eg: pd_resample(p_df.close, bin, how='mean')

                    2014-07-23    249.0728
                    2014-09-03    258.3640
                    2014-10-15    240.8663
                    2014-11-26    220.1552
                    2015-01-07    206.0070
                    2015-02-18    198.0932
                    2015-04-01    217.9791
                    2015-05-13    251.3640
                    2015-06-24    266.4511
                    2015-08-05    244.3334
                    2015-09-16    236.2250
                    2015-10-28    222.0441
                    2015-12-09    222.0574
                    2016-01-20    177.2303
                    2016-03-02    226.8766
                    2016-04-13    230.6000
                    2016-05-25    216.7596
                    2016-07-06    222.6420

                    abs(pd_resample(p_df.close, bin, how='mean').pct_change())

                    2014-09-03    0.037
                    2014-10-15    0.068
                    2014-11-26    0.086
                    2015-01-07    0.064
                    2015-02-18    0.038
                    2015-04-01    0.100
                    2015-05-13    0.153
                    2015-06-24    0.060
                    2015-08-05    0.083
                    2015-09-16    0.033
                    2015-10-28    0.060
                    2015-12-09    0.000
                    2016-01-20    0.202
                    2016-03-02    0.280
                    2016-04-13    0.016
                    2016-05-25    0.060
                    2016-07-06    0.027

                    abs(pd_resample(p_df.close, bin, how='mean').pct_change()).mean():

                    0.080
            """
            resample_dict[_bin] = change
        resample_df = pd.DataFrame.from_dict(resample_dict, orient='index')
        resample_df.columns = ['{}resample'.format(df_name)]
        return resample_df

    if bins is None:
        bins = ['5D', '10D', '21D', '42D', '60D', '90D', '120D']
    return _df_dispatch_concat(df, _resample_close_mean)


def bcut_change_vc(df, bins=None):
    """
    eg:
        tsla = ABuSymbolPd.make_kl_df('usTSLA')
        ABuKLUtil.bcut_change_vc(tsla)

        out:
                p_change	rate
        (0, 3]	209	0.4147
        (-3, 0]	193	0.3829
        (3, 7]	47	0.0933
        (-7, -3]	44	0.0873
        (-10, -7]	6	0.0119
        (7, 10]	3	0.0060
        (10, inf]	1	0.0020
        (-inf, -10]	1	0.0020

    :param df: abupy中格式化好的kl，或者字典，或者可迭代序列
    :param bins: 默认eg：[-np.inf, -10, -7, -3, 0, 3, 7, 10, np.inf]
    :return: pd.DataFrame
    """

    def _bcut_change_vc(p_df, df_name=''):
        dww = pd.DataFrame(pd.cut(p_df.p_change, bins=bins).value_counts())
        # 计算各个bin所占的百分比
        dww['{}rate'.format(df_name)] = dww.p_change.values / dww.p_change.values.sum()
        if len(df_name) > 0:
            dww.rename(columns={'p_change': '{}'.format(df_name)}, inplace=True)
        return dww

    if bins is None:
        bins = [-np.inf, -10, -7, -3, 0, 3, 7, 10, np.inf]
    return _df_dispatch_concat(df, _bcut_change_vc)


def qcut_change_vc(df, q=10):
    """
    eg:
        tsla = ABuSymbolPd.make_kl_df('usTSLA')
        ABuKLUtil.qcut_change_vc(tsla)

        out:
            change
        0	[-10.45, -3.002]
        1	(-3.002, -1.666]
        2	(-1.666, -0.93]
        3	(-0.93, -0.396]
        4	(-0.396, 0.065]
        5	(0.065, 0.48]
        6	(0.48, 1.102]
        7	(1.102, 1.922]
        8	(1.922, 3.007]
        9	(3.007, 11.17]

    :param df: abupy中格式化好的kl，或者字典，或者可迭代序列
    :param q: 透传qcut使用的q参数，默认10，10等分
    :return: pd.DataFrame
    """

    def _qcut_change_vc(p_df, df_name=''):
        dww = pd.qcut(p_df.p_change, q).value_counts().index.values
        # 构造Categories使用DataFrame套Series
        dww = pd.Series(dww)
        # 涨跌从负向正开始排序
        dww.sort_values(inplace=True)
        dww = pd.DataFrame(dww)
        # 排序后index重新从0开始排列
        dww.index = np.arange(0, q)
        dww.columns = ['{}change'.format(df_name)]
        return dww

    return _df_dispatch_concat(df, _qcut_change_vc)


def date_week_mean(df):
    """
        eg:

        tsla = ABuSymbolPd.make_kl_df('usTSLA')
        ABuKLUtil.date_week_mean(tsla)

        out:
        周一    0.0626
        周二    0.0475
        周三    0.0881
        周四    0.2691
        周五   -0.2838
    :param df: abupy中格式化好的kl，或者字典，或者可迭代序列
    :return: pd.Series或者pd.DataFrame
    """

    def _date_week_win(p_df, df_name=''):
        dww = p_df.groupby('date_week')['p_change'].mean()
        # 将周几这个信息变成中文
        dww.rename(index={6: '周日', 0: '周一', 1: '周二', 2: '周三', 3: '周四', 4: '周五', 5: '周六'},
                   inplace=True)
        # p_change变成对应的pchange
        dww = pd.DataFrame(dww)
        dww.rename(columns={'p_change': '{}_p_change'.format(df_name)}, inplace=True)
        return dww

    return _df_dispatch_concat(df, _date_week_win)


def date_week_win(df):
    """
    eg:
        tsla = ABuSymbolPd.make_kl_df('usTSLA')
        ABuKLUtil.date_week_win(tsla)

        out：
                0	1	win
            date_week
            0	44	51	0.5368
            1	55	48	0.4660
            2	48	57	0.5429
            3	44	57	0.5644
            4	53	47	0.470

    :param df: bupy中格式化好的kl，或者字典，或者可迭代序列
    :return: pd.Series或者pd.DataFrame
    """

    def _date_week_win(p_df, df_name=''):
        _df = p_df.copy()
        win_key = '{}win'.format(df_name)
        _df[win_key] = _df['p_change'] > 0
        _df[win_key] = _df[win_key].astype(int)

        dww = pd.concat([pd.crosstab(_df.date_week, _df[win_key]), _df.pivot_table([win_key], index='date_week')],
                        axis=1)
        # 将周几这个信息变成中文
        # noinspection PyUnresolvedReferences
        dww.rename(index={6: '周日', 0: '周一', 1: '周二', 2: '周三', 3: '周四', 4: '周五', 5: '周六'}, inplace=True)
        return dww

    return _df_dispatch_concat(df, _date_week_win)


def wave_change_rate(df):
    """
    eg:
        tsla = ABuSymbolPd.make_kl_df('usTSLA')
        ABuKLUtil.wave_change_rate(tsla)

        out:
        日振幅涨跌幅比：1.794156

    :param df: abupy中格式化好的kl，或者字典，或者可迭代序列
    """

    def _wave_change_rate(p_df, df_name=''):
        wave = ((p_df.high - p_df.low) / p_df.pre_close) * 100
        # noinspection PyUnresolvedReferences
        wave_rate = wave.mean() / np.abs(p_df['p_change']).mean()

        print('{}日振幅涨跌幅比：{:2f}, {}日统计套利条件'.format(
            df_name, wave_rate, '具备' if wave_rate > 1.80 else '不具备'))
    _df_dispatch(df, _wave_change_rate)


def p_change_stats(df):
    """
    eg :
        tsla = ABuSymbolPd.make_kl_df('usTSLA')
        ABuKLUtil.p_change_stats(tsla)

        out:

        日涨幅平均值1.861, 共260个交易日上涨走势
        日跌幅平均值-1.906, 共244个交易日下跌走势
        日平均涨跌比0.977, 上涨下跌数量比:1.066

    :param df: abupy中格式化好的kl，或者字典，或者可迭代序列
    """

    def _p_change_stats(p_df, df_name=''):
        p_change_up = p_df[p_df['p_change'] > 0].p_change
        p_change_down = p_df[p_df['p_change'] < 0].p_change
        print('{}日涨幅平均值{:.3f}, 共{}个交易日上涨走势'.format(df_name, p_change_up.mean(), p_change_up.count()))
        print('{}日跌幅平均值{:.3f}, 共{}个交易日下跌走势'.format(df_name, p_change_down.mean(), p_change_down.count()))
        print('{}日平均涨跌比{:.3f}, 上涨下跌数量比:{:.3f}\n'.format(
            df_name, abs(p_change_up.mean() / p_change_down.mean()), p_change_up.count() / p_change_down.count()))

    _df_dispatch(df, _p_change_stats)


def date_week_wave(df):
    """
    根据周几分析金融时间序列中的日波动:

    eg:
        tsla = ABuSymbolPd.make_kl_df('usTSLA')
        ABuKLUtil.date_week_wave(tsla)

        out:
            usTSLAwave
            date_week
            周一  3.8144
            周二  3.3326
            周三  3.3932
            周四  3.3801
            周五  2.9923

    :param df: abupy中格式化好的kl，或者字典，或者可迭代序列
    :return: pd.Series或者pd.DataFrame
    """

    def _date_week_wave(p_df, df_name=''):
        # 要改df所以copy
        df_copy = p_df.copy()
        wave_key = '{}wave'.format(df_name)
        # 计算波动: * 100目的是和金融序列中的p_change单位一致
        df_copy[wave_key] = ((df_copy.high - df_copy.low) / df_copy.pre_close) * 100
        dww = df_copy.groupby('date_week')[wave_key].mean()
        # 将周几这个信息变成中文
        dww.rename(index={6: '周日', 0: '周一', 1: '周二', 2: '周三', 3: '周四', 4: '周五', 5: '周六'}, inplace=True)
        return dww

    return _df_dispatch_concat(df, _date_week_wave)
