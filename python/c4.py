# -*- encoding:utf-8 -*-
from __future__ import print_function
from __future__ import division

import warnings
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

# noinspection PyUnresolvedReferences
import abu_local_env
import abupy
from abupy import ABuSymbolPd
from abupy import xrange, pd_resample

warnings.filterwarnings('ignore')
sns.set_context(rc={'figure.figsize': (14, 7)})
# 使用沙盒数据，目的是和书中一样的数据环境
abupy.env.enable_example_env_ipython()

stock_day_change = np.load('../gen/stock_day_change.npy')


"""
    第四章 量化工具——pandas

    abu量化系统github地址：https://github.com/bbfamily/abu (您的star是我的动力！)
    abu量化文档教程ipython notebook：https://github.com/bbfamily/abu/tree/master/abupy_lecture
"""


def sample_411():
    """
    4.1.1 DataFrame构建及方法
    :return:
    """
    print('stock_day_change.shape:', stock_day_change.shape)

    # 下面三种写法输出完全相同，输出如表4-1所示
    print('head():\n', pd.DataFrame(stock_day_change).head())
    print('head(5):\n', pd.DataFrame(stock_day_change).head(5))
    print('[:5]:\n', pd.DataFrame(stock_day_change)[:5])


def sample_412():
    """
    4.1.2 索引行列序列
    :return:
    """
    # 股票0 -> 股票stock_day_change.shape[0]
    stock_symbols = ['股票 ' + str(x) for x in
                     xrange(stock_day_change.shape[0])]
    # 通过构造直接设置index参数，head(2)就显示两行，表4-2所示
    print('pd.DataFrame(stock_day_change, index=stock_symbols).head(2):\n',
          pd.DataFrame(stock_day_change, index=stock_symbols).head(2))
    # 从2017-1-1向上时间递进，单位freq='1d'即1天
    days = pd.date_range('2017-1-1',
                         periods=stock_day_change.shape[1], freq='1d')
    # 股票0 -> 股票stock_day_change.shape[0]
    stock_symbols = ['股票 ' + str(x) for x in
                     xrange(stock_day_change.shape[0])]
    # 分别设置index和columns
    df = pd.DataFrame(stock_day_change, index=stock_symbols, columns=days)
    # 表4-3所示
    print('df.head(2):\n', df.head(2))


def sample_413():
    """
    4.1.3 金融时间序列
    :return:
    """
    days = pd.date_range('2017-1-1',
                         periods=stock_day_change.shape[1], freq='1d')
    stock_symbols = ['股票 ' + str(x) for x in
                     xrange(stock_day_change.shape[0])]
    df = pd.DataFrame(stock_day_change, index=stock_symbols, columns=days)

    # df做个转置
    df = df.T
    # 表4-4所示
    print('df.head():\n', df.head())

    df_20 = pd_resample(df, '21D', how='mean')
    # 表4-5所示
    print('df_20.head():\n', df_20.head())


def sample_414():
    """
    4.1.4 Series构建及方法
    :return
    """
    days = pd.date_range('2017-1-1',
                         periods=stock_day_change.shape[1], freq='1d')
    stock_symbols = ['股票 ' + str(x) for x in
                     xrange(stock_day_change.shape[0])]
    df = pd.DataFrame(stock_day_change, index=stock_symbols, columns=days)
    df = df.T

    print('df.head():\n', df.head())
    df_stock0 = df['股票 0']
    # 打印df_stock0类型
    print('type(df_stock0):', type(df_stock0))
    # 打印出Series的前5行数据, 与DataFrame一致
    print('df_stock0.head():\n', df_stock0.head())

    df_stock0.cumsum().plot()
    plt.show()


def sample_415():
    """
    4.1.5 重采样数据
    :return
    """
    days = pd.date_range('2017-1-1',
                         periods=stock_day_change.shape[1], freq='1d')
    stock_symbols = ['股票 ' + str(x) for x in
                     xrange(stock_day_change.shape[0])]
    df = pd.DataFrame(stock_day_change, index=stock_symbols, columns=days)
    df = df.T
    df_stock0 = df['股票 0']

    # 以5天为周期重采样（周k）
    df_stock0_5 = pd_resample(df_stock0.cumsum(), '5D', how='ohlc')
    # 以21天为周期重采样（月k），
    # noinspection PyUnusedLocal
    df_stock0_20 = pd_resample(df_stock0.cumsum(), '21D', how='ohlc')
    # 打印5天重采样，如下输出2017-01-01, 2017-01-06, 2017-01-11, 表4-6所示
    print('df_stock0_5.head():\n', df_stock0_5.head())

    from abupy import ABuMarketDrawing
    # 图4-2所示
    ABuMarketDrawing.plot_candle_stick(df_stock0_5.index,
                                       df_stock0_5['open'].values,
                                       df_stock0_5['high'].values,
                                       df_stock0_5['low'].values,
                                       df_stock0_5['close'].values,
                                       np.random.random(len(df_stock0_5)),
                                       None, 'stock', day_sum=False,
                                       html_bk=False, save=False)

    print('type(df_stock0_5.open.values):', type(df_stock0_5['open'].values))
    print('df_stock0_5.open.index:\n', df_stock0_5['open'].index)
    print('df_stock0_5.columns:\n', df_stock0_5.columns)


"""
    4.2 基本数据分析示例
"""
# n_folds=2两年
tsla_df = ABuSymbolPd.make_kl_df('usTSLA', n_folds=2)


def sample_420():
    # 表4-7所示
    print('tsla_df.tail():\n', tsla_df.tail())


def sample_421():
    """
    4.2.1 数据整体分析
    :return:
    """
    print('tsla_df.info():\n', tsla_df.info())
    print('tsla_df.describe():\n', tsla_df.describe())

    tsla_df[['close', 'volume']].plot(subplots=True, style=['r', 'g'], grid=True)
    plt.show()


def sample_422():
    """
    4.2.2 索引选取和切片选择
    :return:
    """

    # 2014-07-23至2014-07-31 开盘价格序列
    print('tsla_df.loc[x:x, x]\n', tsla_df.loc['2014-07-23':'2014-07-31', 'open'])

    # 2014-07-23至2014-07-31 所有序列，表4-9所示
    print('tsla_df.loc[x:x]\n', tsla_df.loc['2014-07-23':'2014-07-31'])

    # [1:5]：(1，2，3，4)，[2:6]: (2, 3, 4, 5)
    # 表4-10所示
    print('tsla_df.iloc[1:5, 2:6]:\n', tsla_df.iloc[1:5, 2:6])

    # 切取所有行[2:6]: (2, 3, 4, 5)列
    print('tsla_df.iloc[:, 2:6]:\n', tsla_df.iloc[:, 2:6])
    # 选取所有的列[35:37]:(35, 36)行，表4-11所示
    print('tsla_df.iloc[35:37]:\n', tsla_df.iloc[35:37])

    # 指定一个列
    print('tsla_df.close[0:3]:\n', tsla_df.close[0:3])
    # 通过组成一个列表选择多个列，表4-12所示
    print('tsla_df[][0:3]:\n', tsla_df[['close', 'high', 'low']][0:3])


def sample_423():
    """
    4.2.3 逻辑条件进行数据筛选
    :return:
    """
    # abs为取绝对值的意思，不是防抱死，表4-13所示
    print('tsla_df[np.abs(tsla_df.p_change) > 8]:\n', tsla_df[np.abs(tsla_df.p_change) > 8])
    print('tsla_df[(np.abs(tsla_df.p_change) > 8) & (tsla_df.volume > 2.5 * tsla_df.volume.mean())]:\n',
          tsla_df[(np.abs(tsla_df.p_change) > 8) & (tsla_df.volume > 2.5 * tsla_df.volume.mean())])


def sample_424_1():
    """
    4.2.4_1 数据转换与规整
    :return:
    """
    # 数据序列值排序
    print('tsla_df.sort_index(by=p_change)[:5]:\n', tsla_df.sort_index(by='p_change')[:5])
    print('tsla_df.sort_index(by=p_change, ascending=False)[:5]:\n',
          tsla_df.sort_index(by='p_change', ascending=False)[:5])

    # 如果一行的数据中存在na就删除这行
    tsla_df.dropna()
    # 通过how控制 如果一行的数据中全部都是na就删除这行
    tsla_df.dropna(how='all')
    # 使用指定值填充na， inplace代表就地操作，即不返回新的序列在原始序列上修改
    tsla_df.fillna(tsla_df.mean(), inplace=True)


def sample_424_2():
    """
    4.2.4_1 数据转换处理 pct_change
    :return:
    """
    print('tsla_df.close[:3]:\n', tsla_df.close[:3])
    print('tsla_df.close.pct_change()[:3]:\n', tsla_df.close.pct_change()[:3])
    print('(223.54 - 222.49) / 222.49, (223.57 - 223.54) / 223.54:', (223.54 - 222.49) / 222.49,
          (223.57 - 223.54) / 223.54)

    # pct_change对序列从第二项开始向前做减法在除以前一项，这样的针对close做pct_change后的结果就是涨跌幅
    change_ratio = tsla_df.close.pct_change()
    print('change_ratio.tail():\n', change_ratio.tail())

    # 将change_ratio转变成与tsla_df.p_change字段一样的百分百，同样保留两位小数
    print('np.round(change_ratio[-5:] * 100, 2):\n', np.round(change_ratio[-5:] * 100, 2))

    fmt = lambda x: '%.2f' % x
    print('tsla_df.atr21.map(fmt).tail():\n', tsla_df.atr21.map(fmt).tail())


def sample_425():
    """
    4.2.5 数据本地序列化操作
    :return:
    """
    tsla_df.to_csv('../gen/tsla_df.csv', columns=tsla_df.columns, index=True)
    tsla_df_load = pd.read_csv('../gen/tsla_df.csv', parse_dates=True, index_col=0)
    print('tsla_df_load.head():\n', tsla_df_load.head())


"""
    4.3 实例1：寻找股票异动涨跌幅阀值
"""


def sample_431():
    """
    4.3.1 数据的离散化
    :return:
    """
    tsla_df.p_change.hist(bins=80)
    plt.show()

    cats = pd.qcut(np.abs(tsla_df.p_change), 10)
    print('cats.value_counts():\n', cats.value_counts())

    # 将涨跌幅数据手工分类，从负无穷到－7，－5，－3，0， 3， 5， 7，正无穷
    bins = [-np.inf, -7.0, -5, -3, 0, 3, 5, 7, np.inf]
    cats = pd.cut(tsla_df.p_change, bins)
    print('bins cats.value_counts():\n', cats.value_counts())

    # cr_dummies为列名称前缀
    change_ration_dummies = pd.get_dummies(cats, prefix='cr_dummies')
    print('change_ration_dummies.head():\n', change_ration_dummies.head())


def sample_432():
    """
    4.3.2 concat, append, merge的使用
    :return:
    """
    # 将涨跌幅数据手工分类，从负无穷到－7，－5，－3，0， 3， 5， 7，正无穷
    bins = [-np.inf, -7.0, -5, -3, 0, 3, 5, 7, np.inf]
    cats = pd.cut(tsla_df.p_change, bins)
    change_ration_dummies = pd.get_dummies(cats, prefix='cr_dummies')

    # noinspection PyUnresolvedReferences
    print('pd.concat([tsla_df, change_ration_dummies], axis=1).tail():\n ',
          pd.concat([tsla_df, change_ration_dummies], axis=1).tail())

    # pd.concat的连接axis＝0：纵向连接atr>14的df和p_change > 10的df
    pd.concat([tsla_df[tsla_df.p_change > 10],
               tsla_df[tsla_df.atr14 > 16]], axis=0)

    # 直接使用DataFrame对象append，结果与上面pd.concat的结果一致, 表4-20所示
    print('tsla_df[tsla_df.p_change > 10].append(tsla_df[tsla_df.atr14 > 16]):\n',
          tsla_df[tsla_df.p_change > 10].append(tsla_df[tsla_df.atr14 > 16]))


"""
    4.4 实例2 ：星期几是这个股票的‘好日子’
"""


def sample_441():
    """
    4.4.1 构建交叉表
    :return:
    """
    # noinspection PyTypeChecker
    tsla_df['positive'] = np.where(tsla_df.p_change > 0, 1, 0)
    print('tsla_df.tail():\n', tsla_df.tail())
    xt = pd.crosstab(tsla_df.date_week, tsla_df.positive)
    print('xt:\n', xt)

    xt_pct = xt.div(xt.sum(1).astype(float), axis=0)
    print('xt_pct:\n', xt_pct)

    xt_pct.plot(
        figsize=(8, 5),
        kind='bar',
        stacked=True,
        title='date_week -> positive')
    plt.xlabel('date_week')
    plt.ylabel('positive')
    plt.show()


def sample_442():
    """
    4.4.2 构建透视表
    :return:
    """
    # noinspection PyTypeChecker
    tsla_df['positive'] = np.where(tsla_df.p_change > 0, 1, 0)
    print('tsla_df.pivot_table([positive], index=[date_week]):\n',
          tsla_df.pivot_table(['positive'], index=['date_week']))
    print('tsla_df.groupby([date_week, positive])[positive].count():\n',
          tsla_df.groupby(['date_week', 'positive'])['positive'].count())


"""
    4.5 实例3 ：跳空缺口
"""

jump_pd = pd.DataFrame()
jump_threshold = tsla_df.close.median() * 0.03


def judge_jump(p_today):
    global jump_pd
    if p_today.p_change > 0 and (p_today.low - p_today.pre_close) > jump_threshold:
        """
            符合向上跳空
        """
        # jump记录方向 1向上
        p_today['jump'] = 1
        # 向上跳能量＝（今天最低 － 昨收）／ 跳空阀值
        p_today['jump_power'] = (p_today.low - p_today.pre_close) / jump_threshold
        jump_pd = jump_pd.append(p_today)
    elif p_today.p_change < 0 and (p_today.pre_close - p_today.high) > jump_threshold:
        """
            符合向下跳空
        """
        # jump记录方向 －1向下
        p_today['jump'] = -1
        # 向下跳能量＝（昨收 － 今天最高）／ 跳空阀值
        p_today['jump_power'] = (p_today.pre_close - p_today.high) / jump_threshold
        jump_pd = jump_pd.append(p_today)


def sample_45_1():
    """
    4.5 实例3 ：跳空缺口
    :return:
    """
    for kl_index in np.arange(0, tsla_df.shape[0]):
        # 通过ix一个一个拿
        today = tsla_df.ix[kl_index]
        judge_jump(today)

    # filter按照顺序只显示这些列, 表4-26所示
    print('jump_pd.filter([jump, jump_power, close, date, p_change, pre_close]):\n',
          jump_pd.filter(['jump', 'jump_power', 'close', 'date', 'p_change', 'pre_close']))


def sample_45_2():
    """
    4.5 实例3 ：跳空缺口
    :return:
    """
    # axis=1即行数据，tsla_df的每一条行数据即为每一个交易日数据
    tsla_df.apply(judge_jump, axis=1)
    print('jump_pd:\n', jump_pd)

    from abupy import ABuMarketDrawing
    # view_indexs传入jump_pd.index，即在k图上使用圆来标示跳空点
    ABuMarketDrawing.plot_candle_form_klpd(tsla_df, view_indexs=jump_pd.index)
    plt.show()


"""
    4.6 pandas三维面板的使用
"""


def sample_46():
    """
    4.6 pandas三维面板的使用
    :return:
    """
    # disable_example_env_ipython不再使用沙盒数据，因为沙盒里面没有相关tsla行业的数据啊
    abupy.env.disable_example_env_ipython()

    from abupy import ABuIndustries
    r_symbol = 'usTSLA'
    # 这里获取了和TSLA电动车处于同一行业的股票组成pandas三维面板Panel数据
    p_date, _ = ABuIndustries.get_industries_panel_from_target(r_symbol, show=False)
    print('type(p_date):', type(p_date))
    print('p_date:\n', p_date)

    print('p_date[usTTM].head():\n', p_date['usTTM'].head())

    p_data_it = p_date.swapaxes('items', 'minor')
    print('p_data_it:\n', p_data_it)

    p_data_it_close = p_data_it['close'].dropna(axis=0)
    print('p_data_it_close.tail():\n', p_data_it_close.tail())

    from abupy import ABuScalerUtil
    # ABuScalerUtil.scaler_std将所有close的切面数据做(group - group.mean()) / group.std()标示化，为了可视化在同一范围
    p_data_it_close = ABuScalerUtil.scaler_std(p_data_it_close)
    p_data_it_close.plot()
    plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
    plt.ylabel('Price')
    plt.xlabel('Time')
    plt.show()


if __name__ == "__main__":
    sample_411()
    # sample_412()
    # sample_413()
    # sample_414()
    # sample_415()
    # sample_420()
    # sample_421()
    # sample_422()
    # sample_423()
    # sample_424_1()
    # sample_424_2()
    # sample_425()
    # sample_431()
    # sample_432()
    # sample_441()
    # sample_442()
    # sample_45_1()
    # sample_45_2()
    # sample_46()
