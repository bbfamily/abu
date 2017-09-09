# -*- encoding:utf-8 -*-
from __future__ import print_function
from __future__ import division

import warnings
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats
# noinspection PyUnresolvedReferences
import abu_local_env
import abupy
from abupy import ABuSymbolPd
from abupy import pd_rolling_std, pd_ewm_std, pd_rolling_mean

warnings.filterwarnings('ignore')
sns.set_context(rc={'figure.figsize': (14, 7)})
# 使用沙盒数据，目的是和书中一样的数据环境
abupy.env.enable_example_env_ipython()

tsla_df = ABuSymbolPd.make_kl_df('usTSLA', n_folds=2)

"""
    第五章 量化工具——可视化

    abu量化系统github地址：https://github.com/bbfamily/abu (您的star是我的动力！)
    abu量化文档教程ipython notebook：https://github.com/bbfamily/abu/tree/master/abupy_lecture
"""


# noinspection PyUnresolvedReferences
def plot_demo(axs=None, just_series=False):
    """
    绘制tsla的收盘价格曲线
    :param axs: axs为子画布，稍后会详细讲解
    :param just_series: 是否只绘制一条收盘曲线使用series，后面会用到
    :return:
    """
    # 如果参数传入子画布则使用子画布绘制，下一节会使用
    drawer = plt if axs is None else axs
    # Series对象tsla_df.close，红色
    drawer.plot(tsla_df.close, c='r')
    if not just_series:
        # 为曲线不重叠，y变量加了10个单位tsla_df.close.values + 10
        # numpy对象tsla_df.close.index ＋ tsla_df.close.values，绿色
        drawer.plot(tsla_df.close.index, tsla_df.close.values + 10,
                    c='g')
        # 为曲线不重叠，y变量加了20个单位
        # list对象，numpy.tolist()将numpy对象转换为list对象，蓝色
        drawer.plot(tsla_df.close.index.tolist(),
                    (tsla_df.close.values + 20).tolist(), c='b')

    plt.xlabel('time')
    plt.ylabel('close')
    plt.title('TSLA CLOSE')
    plt.grid(True)


def sample_511():
    """
    5.1.1 matplotlib可视化基础
    :return:
    """
    print('tsla_df.tail():\n', tsla_df.tail())

    plot_demo()
    plt.show()


def sample_512():
    """
    5.1.2 matplotlib子画布及loc的使用
    :return:
    """
    _, axs = plt.subplots(nrows=2, ncols=2, figsize=(14, 10))
    # 画布0，loc：0 plot_demo中传入画布，则使用传入的画布绘制
    drawer = axs[0][0]
    plot_demo(drawer)
    drawer.legend(['Series', 'Numpy', 'List'], loc=0)
    # 画布1，loc：1
    drawer = axs[0][1]
    plot_demo(drawer)
    drawer.legend(['Series', 'Numpy', 'List'], loc=1)
    # 画布2，loc：2
    drawer = axs[1][0]
    plot_demo(drawer)
    drawer.legend(['Series', 'Numpy', 'List'], loc=2)
    # 画布3，loc：2， 设置bbox_to_anchor，在画布外的相对位置绘制
    drawer = axs[1][1]
    plot_demo(drawer)
    drawer.legend(['Series', 'Numpy', 'List'], bbox_to_anchor=(1.05, 1),
                  loc=2,
                  borderaxespad=0.)
    plt.show()


def sample_513():
    """
    5.1.3 k线图的绘制
    :return:
    """
    import matplotlib.finance as mpf

    __colorup__ = "red"
    __colordown__ = "green"
    # 为了示例清晰，只拿出前30天的交易数据绘制蜡烛图，
    tsla_part_df = tsla_df[:30]
    fig, ax = plt.subplots(figsize=(14, 7))
    qutotes = []

    for index, (d, o, c, h, l) in enumerate(
            zip(tsla_part_df.index, tsla_part_df.open, tsla_part_df.close,
                tsla_part_df.high, tsla_part_df.low)):
        # 蜡烛图的日期要使用matplotlib.finance.date2num进行转换为特有的数字值
        d = mpf.date2num(d)
        # 日期，开盘，收盘，最高，最低组成tuple对象val
        val = (d, o, c, h, l)
        # 加val加入qutotes
        qutotes.append(val)
    # 使用mpf.candlestick_ochl进行蜡烛绘制，ochl代表：open，close，high，low
    mpf.candlestick_ochl(ax, qutotes, width=0.6, colorup=__colorup__,
                         colordown=__colordown__)
    ax.autoscale_view()
    ax.xaxis_date()
    plt.show()


def sample_52():
    """
    5.2 使用bokeh交互可视化
    :return:
    """
    from abupy import ABuMarketDrawing
    ABuMarketDrawing.plot_candle_form_klpd(tsla_df, html_bk=True)


"""
    5.3 使用pandas可视化数据
"""


def sample_531_1():
    """
    5.3.1_1 绘制股票的收益，及收益波动情况 demo list
    :return:
    """
    # 示例序列
    demo_list = np.array([2, 4, 16, 20])
    # 以三天为周期计算波动
    demo_window = 3
    # pd.rolling_std * np.sqrt
    print('pd.rolling_std(demo_list, window=demo_window, center=False) * np.sqrt(demo_window):\n',
          pd_rolling_std(demo_list, window=demo_window, center=False) * np.sqrt(demo_window))

    print('pd.Series([2, 4, 16]).std() * np.sqrt(demo_window):', pd.Series([2, 4, 16]).std() * np.sqrt(demo_window))
    print('pd.Series([4, 16, 20]).std() * np.sqrt(demo_window):', pd.Series([4, 16, 20]).std() * np.sqrt(demo_window))
    print('np.sqrt(pd.Series([2, 4, 16]).var() * demo_window):', np.sqrt(pd.Series([2, 4, 16]).var() * demo_window))


def sample_531_2():
    """
    5.3.1_2 绘制股票的收益，及收益波动情况
    :return:
    """
    tsla_df_copy = tsla_df.copy()
    # 投资回报
    tsla_df_copy['return'] = np.log(tsla_df['close'] / tsla_df['close'].shift(1))

    # 移动收益标准差
    tsla_df_copy['mov_std'] = pd_rolling_std(tsla_df_copy['return'],
                                             window=20,
                                             center=False) * np.sqrt(20)
    # 加权移动收益标准差，与移动收益标准差基本相同，只不过根据时间权重计算std
    tsla_df_copy['std_ewm'] = pd_ewm_std(tsla_df_copy['return'], span=20,
                                         min_periods=20,
                                         adjust=True) * np.sqrt(20)

    tsla_df_copy[['close', 'mov_std', 'std_ewm', 'return']].plot(subplots=True, grid=True)
    plt.show()


def sample_532():
    """
    5.3.2 绘制股票的价格与均线
    :return:
    """
    tsla_df.close.plot()
    # ma 30
    # pd_rolling_mean(tsla_df.close, window=30).plot()
    pd_rolling_mean(tsla_df.close, window=30).plot()
    # ma 60
    # pd.rolling_mean(tsla_df.close, window=60).plot()
    pd_rolling_mean(tsla_df.close, window=60).plot()
    # ma 90
    # pd.rolling_mean(tsla_df.close, window=90).plot()
    pd_rolling_mean(tsla_df.close, window=90).plot()
    # loc='best'即自动寻找适合的位置
    plt.legend(['close', '30 mv', '60 mv', '90 mv'], loc='best')
    plt.show()


def sample_533():
    """
    5.3.3 其它pandas统计图形种类
    :return:
    """
    # iloc获取所有低开高走的下一个交易日组成low_to_high_df，由于是下一个交易日
    # 所以要对满足条件的交易日再次通过iloc获取，下一个交易日index用key.values + 1
    # key序列的值即为0-len(tsla_df), 即为交易日index，详情查阅本章初tail
    low_to_high_df = tsla_df.iloc[tsla_df[
                                      (tsla_df.close > tsla_df.open) & (
                                          tsla_df.key != tsla_df.shape[
                                              0] - 1)].key.values + 1]

    # 通过where将下一个交易日的涨跌幅通过ceil，floor向上，向下取整
    change_ceil_floor = np.where(low_to_high_df['p_change'] > 0,
                                 np.ceil(
                                     low_to_high_df['p_change']),
                                 np.floor(
                                     low_to_high_df['p_change']))

    # 使用pd.Series包裹，方便之后绘制
    change_ceil_floor = pd.Series(change_ceil_floor)
    print('低开高收的下一个交易日所有下跌的跌幅取整和sum: ' + str(
        change_ceil_floor[change_ceil_floor < 0].sum()))

    print('低开高收的下一个交易日所有上涨的涨幅取整和sum: ' + str(
        change_ceil_floor[change_ceil_floor > 0].sum()))

    # 2 * 2: 四张子图
    _, axs = plt.subplots(nrows=2, ncols=2, figsize=(12, 10))
    # 竖直柱状图，可以看到-1的柱子最高, 图5-7左上
    change_ceil_floor.value_counts().plot(kind='bar', ax=axs[0][0])
    # 水平柱状图，可以看到-1的柱子最长, 图5-7右上
    change_ceil_floor.value_counts().plot(kind='barh', ax=axs[0][1])
    # 概率密度图，可以看到向左偏移, 图5-7左下
    change_ceil_floor.value_counts().plot(kind='kde', ax=axs[1][0])
    # 圆饼图，可以看到－1所占的比例最高, -2的比例也大于＋2，图5-7右下
    change_ceil_floor.value_counts().plot(kind='pie', ax=axs[1][1])
    plt.show()


def sample_54_1():
    """
    5.4 使用seaborn可视化数据
    :return:
    """
    sns.distplot(tsla_df['p_change'], bins=80)
    plt.show()

    sns.boxplot(x='date_week', y='p_change', data=tsla_df)
    plt.show()

    sns.jointplot(tsla_df['high'], tsla_df['low'])
    plt.show()


def sample_54_2():
    """
    5.4 使用seaborn可视化数据
    :return:
    """
    change_df = pd.DataFrame({'tsla': tsla_df.p_change})
    # join usGOOG
    change_df = change_df.join(pd.DataFrame({'goog': ABuSymbolPd.make_kl_df('usGOOG', n_folds=2).p_change}),
                               how='outer')
    # join usAAPL
    change_df = change_df.join(pd.DataFrame({'aapl': ABuSymbolPd.make_kl_df('usAAPL', n_folds=2).p_change}),
                               how='outer')
    # join usFB
    change_df = change_df.join(pd.DataFrame({'fb': ABuSymbolPd.make_kl_df('usFB', n_folds=2).p_change}),
                               how='outer')
    # join usBIDU
    change_df = change_df.join(pd.DataFrame({'bidu': ABuSymbolPd.make_kl_df('usBIDU', n_folds=2).p_change}),
                               how='outer')

    change_df = change_df.dropna()
    # 表5-2所示
    print('change_df.head():\n', change_df.head())

    # 使用corr计算数据的相关性
    corr = change_df.corr()
    _, ax = plt.subplots(figsize=(8, 5))
    # sns.heatmap热力图展示每组股票涨跌幅的相关性
    sns.heatmap(corr, ax=ax)
    plt.show()


"""
    5.5 实例1:可视化量化策略的交易区间，卖出原因
"""


def sample_55_1():
    """
    5.5 可视化量化策略的交易区间，卖出原因
    :return:
    """

    def plot_trade(buy_date, sell_date):
        # 找出2014-07-28对应时间序列中的index作为start
        start = tsla_df[tsla_df.index == buy_date].key.values[0]
        # 找出2014-09-05对应时间序列中的index作为end
        end = tsla_df[tsla_df.index == sell_date].key.values[0]

        # 使用5.1.1封装的绘制tsla收盘价格时间序列函数plot_demo
        # just_series＝True, 即只绘制一条曲线使用series数据
        plot_demo(just_series=True)

        # 将整个时间序列都填充一个底色blue，注意透明度alpha=0.08是为了
        # 之后标注其他区间透明度高于0.08就可以清楚显示
        plt.fill_between(tsla_df.index, 0, tsla_df['close'], color='blue',
                         alpha=.08)

        # 标注股票持有周期绿色，使用start和end切片周期
        # 透明度alpha=0.38 > 0.08
        plt.fill_between(tsla_df.index[start:end], 0,
                         tsla_df['close'][start:end], color='green',
                         alpha=.38)

        # 设置y轴的显示范围，如果不设置ylim，将从0开始作为起点显示，效果不好
        plt.ylim(np.min(tsla_df['close']) - 5,
                 np.max(tsla_df['close']) + 5)
        # 使用loc='best'
        plt.legend(['close'], loc='best')

    # 标注交易区间2014-07-28到2014-09-05, 图5-12所示
    plot_trade('2014-07-28', '2014-09-05')
    plt.show()

    def plot_trade_with_annotate(buy_date, sell_date, annotate):
        """
        :param buy_date: 交易买入日期
        :param sell_date: 交易卖出日期
        :param annotate: 卖出原因
        :return:
        """
        # 标注交易区间buy_date到sell_date
        plot_trade(buy_date, sell_date)
        # annotate文字，asof：从tsla_df['close']中找到index:sell_date对应值
        plt.annotate(annotate,
                     xy=(sell_date, tsla_df['close'].asof(sell_date)),
                     arrowprops=dict(facecolor='yellow'),
                     horizontalalignment='left', verticalalignment='top')

    plot_trade_with_annotate('2014-07-28', '2014-09-05',
                             'sell for stop loss')
    plt.show()


def sample_55_2():
    """
    5.5 可视化量化策略的交易区间，卖出原因
    :return:
    """

    def plot_trade(buy_date, sell_date):
        # 找出2014-07-28对应时间序列中的index作为start
        start = tsla_df[tsla_df.index == buy_date].key.values[0]
        # 找出2014-09-05对应时间序列中的index作为end
        end = tsla_df[tsla_df.index == sell_date].key.values[0]
        # 使用5.1.1封装的绘制tsla收盘价格时间序列函数plot_demo
        # just_series＝True, 即只绘制一条曲线使用series数据
        plot_demo(just_series=True)
        # 将整个时间序列都填充一个底色blue，注意透明度alpha=0.08是为了
        # 之后标注其他区间透明度高于0.08就可以清楚显示
        plt.fill_between(tsla_df.index, 0, tsla_df['close'], color='blue',
                         alpha=.08)
        # 标注股票持有周期绿色，使用start和end切片周期，透明度alpha=0.38 > 0.08
        if tsla_df['close'][end] < tsla_df['close'][start]:
            # 如果赔钱了显示绿色
            plt.fill_between(tsla_df.index[start:end], 0,
                             tsla_df['close'][start:end], color='green',
                             alpha=.38)
            is_win = False
        else:
            # 如果挣钱了显示红色
            plt.fill_between(tsla_df.index[start:end], 0,
                             tsla_df['close'][start:end], color='red',
                             alpha=.38)
            is_win = True

        # 设置y轴的显示范围，如果不设置ylim，将从0开始作为起点显示
        plt.ylim(np.min(tsla_df['close']) - 5,
                 np.max(tsla_df['close']) + 5)
        # 使用loc='best'
        plt.legend(['close'], loc='best')
        # 将是否盈利结果返回
        return is_win

    def plot_trade_with_annotate(buy_date, sell_date):
        """
        :param buy_date: 交易买入日期
        :param sell_date: 交易卖出日期
        :return:
        """
        # 标注交易区间buy_date到sell_date
        is_win = plot_trade(buy_date, sell_date)
        # 根据is_win来判断是否显示止盈还是止损卖出
        plt.annotate(
            'sell for stop win' if is_win else 'sell for stop loss',
            xy=(sell_date, tsla_df['close'].asof(sell_date)),
            arrowprops=dict(facecolor='yellow'),
            horizontalalignment='left', verticalalignment='top')

    # 区间2014-07-28到2014-09-05
    plot_trade_with_annotate('2014-07-28', '2014-09-05')
    # 区间2015-01-28到2015-03-11
    plot_trade_with_annotate('2015-01-28', '2015-03-11')
    # 区间2015-04-10到2015-07-10
    plot_trade_with_annotate('2015-04-10', '2015-07-10')
    # 区间2015-10-2到2015-10-14
    plot_trade_with_annotate('2015-10-2', '2015-10-14')
    # 区间2016-02-10到2016-04-11
    plot_trade_with_annotate('2016-02-10', '2016-04-11')
    plt.show()


"""
    5.6 实例2:标准化两个股票的观察周期
"""

goog_df = ABuSymbolPd.make_kl_df('usGOOG', n_folds=2)


def plot_two_stock(tsla, goog, axs=None):
    # 如果有传递子画布，使用子画布，否则plt
    drawer = plt if axs is None else axs
    # tsla red
    drawer.plot(tsla, c='r')
    # google greeen
    drawer.plot(goog, c='g')
    # 显示网格
    drawer.grid(True)
    # 图例标注
    drawer.legend(['tsla', 'google'], loc='best')


def sample_56_1():
    """
    5.6 标准化两个股票的观察周期
    :return:
    """
    # mean:打印均值，median：打印中位数
    print(round(goog_df.close.mean(), 2), round(goog_df.close.median(), 2))
    # 表5-3所示
    print('goog_df.tail():\n', goog_df.tail())

    plot_two_stock(tsla_df.close, goog_df.close)
    plt.title('TSLA and Google CLOSE')
    # x轴时间
    plt.xlabel('time')
    # y轴收盘价格
    plt.ylabel('close')
    plt.show()


def sample_56_2():
    """
    5.6 标准化两个股票的观察周期
    :return:
    """

    # noinspection PyShadowingNames
    def two_mean_list(one, two, type_look='look_max'):
        """
        只针对俩个输入的均值归一化
        :param one:
        :param two:
        :param type_look:
        :return:
        """
        one_mean = one.mean()
        two_mean = two.mean()
        if type_look == 'look_max':
            """
                向较大的均值序列看齐
            """
            one, two = (one, one_mean / two_mean * two) \
                if one_mean > two_mean else (
                one * two_mean / one_mean, two)
        elif type_look == 'look_min':
            """
                向较小的均值序列看齐
            """
            one, two = (one * two_mean / one_mean, two) \
                if one_mean > two_mean else (
                one, two * one_mean / two_mean)
        return one, two

    def regular_std(group):
        # z-score规范化也称零-均值规范化
        return (group - group.mean()) / group.std()

    def regular_mm(group):
        # 最小-最大规范化
        return (group - group.min()) / (group.max() - group.min())

    # 2行2列，4个画布
    _, axs = plt.subplots(nrows=2, ncols=2, figsize=(14, 10))

    # 第一个regular_std, 如图5-16左上所示
    drawer = axs[0][0]
    plot_two_stock(regular_std(tsla_df.close), regular_std(goog_df.close),
                   drawer)
    drawer.set_title('(group - group.mean()) / group.std()')

    # 第二个regular_mm，如图5-16右上所示
    drawer = axs[0][1]
    plot_two_stock(regular_mm(tsla_df.close), regular_mm(goog_df.close),
                   drawer)
    drawer.set_title(
        '(group - group.min()) / (group.max() - group.min())')

    # 第三个向较大的序列看齐，如图5-16左上所示
    drawer = axs[1][0]
    one, two = two_mean_list(tsla_df.close, goog_df.close,
                             type_look='look_max')
    plot_two_stock(one, two, drawer)
    drawer.set_title('two_mean_list type_look=look_max')

    # 第四个向较小的序列看齐，如图5-16右下所示
    drawer = axs[1][1]
    one, two = two_mean_list(tsla_df.close, goog_df.close,
                             type_look='look_min')
    plot_two_stock(one, two, drawer)
    drawer.set_title('two_mean_list type_look=look_min')
    plt.show()


def sample_56_3():
    """
    5.6 标准化两个股票的观察周期
    :return:
    """
    _, ax1 = plt.subplots()
    ax1.plot(tsla_df.close, c='r', label='tsla')
    # 第一个ax的标注
    ax1.legend(loc=2)
    ax1.grid(False)
    # 反向y轴 twinx
    ax2 = ax1.twinx()
    ax2.plot(goog_df.close, c='g', label='google')
    # 第二个ax的标志
    ax2.legend(loc=1)
    plt.show()


# noinspection PyTypeChecker
def sample_571_1():
    """
    5.7.1 黄金分割线的定义方式
    :return:
    """
    # 收盘价格序列中的最大值
    cs_max = tsla_df.close.max()
    # 收盘价格序列中的最小值
    cs_min = tsla_df.close.min()

    sp382 = (cs_max - cs_min) * 0.382 + cs_min
    sp618 = (cs_max - cs_min) * 0.618 + cs_min
    print('视觉上的382: ' + str(round(sp382, 2)))
    print('视觉上的618: ' + str(round(sp618, 2)))

    sp382_stats = stats.scoreatpercentile(tsla_df.close, 38.2)
    sp618_stats = stats.scoreatpercentile(tsla_df.close, 61.8)

    print('统计上的382: ' + str(round(sp382_stats, 2)))
    print('统计上的618: ' + str(round(sp618_stats, 2)))


# noinspection PyTypeChecker
def sample_571_2():
    """
    5.7.1 黄金分割线的定义方式
    :return:
    """
    from collections import namedtuple

    # 收盘价格序列中的最大值
    cs_max = tsla_df.close.max()
    # 收盘价格序列中的最小值
    cs_min = tsla_df.close.min()

    sp382 = (cs_max - cs_min) * 0.382 + cs_min
    sp618 = (cs_max - cs_min) * 0.618 + cs_min
    sp382_stats = stats.scoreatpercentile(tsla_df.close, 38.2)
    sp618_stats = stats.scoreatpercentile(tsla_df.close, 61.8)

    def plot_golden():
        # 从视觉618和统计618中筛选更大的值
        above618 = np.maximum(sp618, sp618_stats)
        # 从视觉618和统计618中筛选更小的值
        below618 = np.minimum(sp618, sp618_stats)
        # 从视觉382和统计382中筛选更大的值
        above382 = np.maximum(sp382, sp382_stats)
        # 从视觉382和统计382中筛选更小的值
        below382 = np.minimum(sp382, sp382_stats)

        # 绘制收盘价
        plt.plot(tsla_df.close)
        # 水平线视觉382
        plt.axhline(sp382, c='r')
        # 水平线统计382
        plt.axhline(sp382_stats, c='m')
        # 水平线视觉618
        plt.axhline(sp618, c='g')
        # 水平线统计618
        plt.axhline(sp618_stats, c='k')

        # 填充618 red
        plt.fill_between(tsla_df.index, above618, below618,
                         alpha=0.5, color="r")
        # 填充382 green
        plt.fill_between(tsla_df.index, above382, below382,
                         alpha=0.5, color="g")

        # 最后使用namedtuple包装上，方便获取
        return namedtuple('golden', ['above618', 'below618', 'above382',
                                     'below382'])(
            above618, below618, above382, below382)

    golden = plot_golden()

    # 根据绘制顺序标注名称
    plt.legend(['close', 'sp382', 'sp382_stats', 'sp618', 'sp618_stats'],
               loc='best')
    plt.show()

    print('理论上的最高盈利: {}'.format(golden.above618 - golden.below382))

    return golden


def sample_572():
    """
    5.7.2 多维数据绘制示例
    :return:
    """
    from itertools import product

    buy_rate = [0.20, 0.25, 0.30]
    sell_rate = [0.70, 0.80, 0.90]

    def find_percent_point(percent, y_org, want_max):
        """
        :param percent: 比例
        :param y_org: close价格序列
        :param want_max: 是否返回大的值
        :return:
        """
        cs_max = y_org.max()
        cs_min = y_org.min()

        # 如果want_max 就使用maximum否则minimum
        maxmin_mum = np.maximum if want_max else np.minimum
        # 每次都计算统计上和视觉上，根据want_max返回大的值above，或小的值below
        return maxmin_mum(
            # 统计上的计算
            stats.scoreatpercentile(y_org, np.round(percent * 100, 1)),
            # 视觉上的计算
            (cs_max - cs_min) * percent + cs_min)

    # 存储结果list
    result = list()
    # 先将0.382, 0.618这一组放入结果队列中

    golden = sample_571_2()
    result.append(
        (0.382, 0.618, round(golden.above618 - golden.below382, 2)))

    # 将buy_rate和sell_rate做笛卡尔积排列各种组合
    for (buy, sell) in product(buy_rate, sell_rate):
        # 如果是买入比例want_max为False，因为只计算理论最高盈利，只需要最below
        profit_below = find_percent_point(buy, tsla_df.close, False)
        # 如果是卖出比例want_max为True，因为只计算理论最高盈利，只需要最above
        profit_above = find_percent_point(sell, tsla_df.close, True)
        # 最终将买入比例，卖出比例，理论最高盈利append
        result.append((buy, sell,
                       round(profit_above - profit_below, 2)))
    # 最后使用np.array套上result
    result = np.array(result)
    print('result:\n', result)

    # 1. 通过scatter点图
    cmap = plt.get_cmap('jet', 20)
    cmap.set_under('gray')
    fig, ax = plt.subplots(figsize=(8, 5))
    # scatter点图，result[:, 0]:x，result[:, 1]:y, result[:, 2]:c
    cax = ax.scatter(result[:, 0], result[:, 1], c=result[:, 2],
                     cmap=cmap, vmin=np.min(result[:, 2]),
                     vmax=np.max(result[:, 2]))
    fig.colorbar(cax, label='max profit', extend='min')
    plt.grid(True)
    plt.xlabel('buy rate')
    plt.ylabel('sell rate')
    plt.show()

    # 2. 通过mpl_toolkits.mplot3d
    # noinspection PyUnresolvedReferences
    from mpl_toolkits.mplot3d import Axes3D

    fig = plt.figure(figsize=(9, 6))
    ax = fig.gca(projection='3d')
    ax.view_init(30, 60)
    ax.scatter3D(result[:, 0], result[:, 1], result[:, 2], c='r', s=50,
                 cmap='spring')
    ax.set_xlabel('buy rate')
    ax.set_ylabel('sell rate')
    ax.set_zlabel('max profit')
    plt.show()


# noinspection PyTypeChecker
def sample_581():
    """
    5.8.1 MACD指标的可视化
    :return:
    """
    from abupy import nd
    nd.macd.plot_macd_from_klpd(tsla_df)


def sample_582_1():
    """
    5.8.2_1 ATR指标的可视化, 使用talib
    :return:
    """
    from abupy import nd
    nd.atr.plot_atr_from_klpd(tsla_df)

if __name__ == "__main__":
    sample_511()
    # sample_512()
    # sample_513()
    # sample_52()
    # sample_531_1()
    # sample_531_2()
    # sample_532()
    # sample_533()
    # sample_54_1()
    # sample_54_2()
    # sample_55_1()
    # sample_55_2()
    # sample_56_1()
    # sample_56_2()
    # sample_56_3()
    # sample_571_1()
    # sample_571_2()
    # sample_572()
    # sample_581()
    # sample_582_1()
