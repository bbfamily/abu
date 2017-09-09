# -*- encoding:utf-8 -*-
from __future__ import print_function
import seaborn as sns
import numpy as np
from sklearn import metrics
import warnings
import ast

# noinspection PyUnresolvedReferences
import abu_local_env
import abupy
from abupy import ml
from abupy import AbuMetricsBase, EStoreAbu, abu
from abupy import ABuMarketDrawing

from abupy import AbuFactorBuyBreak
from abupy import AbuFactorAtrNStop
from abupy import AbuFactorPreAtrNStop
from abupy import AbuFactorCloseAtrNStop
from abupy import EMarketTargetType, EMarketDataFetchMode
from abupy import AbuUmpMainDeg
from abupy import AbuUmpMainJump
from abupy import AbuUmpMainPrice
from abupy import AbuUmpMainWave

# 设置选股因子，None为不使用选股因子
stock_pickers = None
# 买入因子依然延用向上突破因子
buy_factors = [{'xd': 60, 'class': AbuFactorBuyBreak},
               {'xd': 42, 'class': AbuFactorBuyBreak}]

# 卖出因子继续使用上一章使用的因子
sell_factors = [
    {'stop_loss_n': 1.0, 'stop_win_n': 3.0, 'class': AbuFactorAtrNStop},
    {'class': AbuFactorPreAtrNStop, 'pre_atr_n': 1.5},
    {'class': AbuFactorCloseAtrNStop, 'close_atr_n': 1.5}
]

warnings.filterwarnings('ignore')
sns.set_context(rc={'figure.figsize': (14, 7)})

"""
    第11章 量化系统-机器学习•ABU

    abu量化系统github地址：https://github.com/bbfamily/abu (您的star是我的动力！)
    abu量化文档教程ipython notebook：https://github.com/bbfamily/abu/tree/master/abupy_lecture

    * 因为需要全市场回测所以本章无法使用沙盒数据，《量化交易之路》中的原始示例使用的是美股市场，这里的示例改为使用A股市场。
    * 本节可以对照阅读abu量化文档第20-23节内容
    * 本节的基础是在abu量化文档中第20节内容完成运行后有A股训练集交易和A股测试集交易数据之后
"""


def load_abu_result_tuple():
    abupy.env.g_market_target = EMarketTargetType.E_MARKET_TARGET_CN
    abupy.env.g_data_fetch_mode = EMarketDataFetchMode.E_DATA_FETCH_FORCE_LOCAL
    abu_result_tuple_train = abu.load_abu_result_tuple(n_folds=5, store_type=EStoreAbu.E_STORE_CUSTOM_NAME,
                                                       custom_name='train_cn')
    abu_result_tuple_test = abu.load_abu_result_tuple(n_folds=5, store_type=EStoreAbu.E_STORE_CUSTOM_NAME,
                                                      custom_name='test_cn')
    metrics_train = AbuMetricsBase(*abu_result_tuple_train)
    metrics_train.fit_metrics()
    metrics_test = AbuMetricsBase(*abu_result_tuple_test)
    metrics_test.fit_metrics()

    return abu_result_tuple_train, abu_result_tuple_test, metrics_train, metrics_test


def sample_110():
    abu_result_tuple_train, abu_result_tuple_test, metrics_train, metrics_test = load_abu_result_tuple()
    metrics_train.plot_returns_cmp(only_show_returns=True)
    metrics_test.plot_returns_cmp(only_show_returns=True)


def sample_111():
    """
    11.1 搜索引擎与量化交易

    请对照阅读ABU量化系统使用文档 ：第16节 UMP主裁交易决策 中相关内容

    :return:
    """
    abu_result_tuple_train, abu_result_tuple_test, metrics_train, metrics_test = load_abu_result_tuple()
    orders_pd_train = abu_result_tuple_train.orders_pd

    # 选择失败的前20笔交易绘制交易快照
    # 这里只是示例，实战中根据需要挑选，rank或者其他方式
    plot_simple = orders_pd_train[orders_pd_train.profit_cg < 0][:20]
    # save=True保存在本地，文件保存在~/abu/data/save_png/中
    ABuMarketDrawing.plot_candle_from_order(plot_simple, save=True)


"""
    11.2 主裁

    请对照阅读ABU量化系统使用文档 ：第15节 中相关内容
"""


def sample_112():
    """
    11.2.1 角度主裁, 11.2.2 使用全局最优对分类簇集合进行筛选
    :return:
    """

    abu_result_tuple_train, abu_result_tuple_test, metrics_train, metrics_test = load_abu_result_tuple()
    orders_pd_train = abu_result_tuple_train.orders_pd
    # 参数为orders_pd
    ump_deg = AbuUmpMainDeg(orders_pd_train)
    # df即由之前ump_main_make_xy生成的类df，表11-1所示
    print('ump_deg.fiter.df.head():\n', ump_deg.fiter.df.head())

    # 耗时操作，大概需要10几分钟，具体根据电脑性能，cpu情况
    _ = ump_deg.fit(brust_min=False)
    print('ump_deg.cprs:\n', ump_deg.cprs)
    max_failed_cluster = ump_deg.cprs.loc[ump_deg.cprs.lrs.argmax()]
    print('失败概率最大的分类簇{0}, 失败率为{1:.2f}%, 簇交易总数{2}, 簇平均交易获利{3:.2f}%'.format(
        ump_deg.cprs.lrs.argmax(), max_failed_cluster.lrs * 100, max_failed_cluster.lcs, max_failed_cluster.lms * 100))

    cpt = int(ump_deg.cprs.lrs.argmax().split('_')[0])
    print('cpt:\n', cpt)
    ump_deg.show_parse_rt(ump_deg.rts[cpt])

    max_failed_cluster_orders = ump_deg.nts[ump_deg.cprs.lrs.argmax()]

    print('max_failed_cluster_orders:\n', max_failed_cluster_orders)

    ml.show_orders_hist(max_failed_cluster_orders,
                        ['buy_deg_ang21', 'buy_deg_ang42', 'buy_deg_ang60', 'buy_deg_ang252'])
    print('分类簇中deg_ang60平均值为{0:.2f}'.format(
        max_failed_cluster_orders.buy_deg_ang60.mean()))

    print('分类簇中deg_ang21平均值为{0:.2f}'.format(
        max_failed_cluster_orders.buy_deg_ang21.mean()))

    print('分类簇中deg_ang42平均值为{0:.2f}'.format(
        max_failed_cluster_orders.buy_deg_ang42.mean()))

    print('分类簇中deg_ang252平均值为{0:.2f}'.format(
        max_failed_cluster_orders.buy_deg_ang252.mean()))

    ml.show_orders_hist(orders_pd_train, ['buy_deg_ang21', 'buy_deg_ang42', 'buy_deg_ang60', 'buy_deg_ang252'])
    print('训练数据集中deg_ang60平均值为{0:.2f}'.format(
        orders_pd_train.buy_deg_ang60.mean()))

    print('训练数据集中deg_ang21平均值为{0:.2f}'.format(
        orders_pd_train.buy_deg_ang21.mean()))

    print('训练数据集中deg_ang42平均值为{0:.2f}'.format(
        orders_pd_train.buy_deg_ang42.mean()))

    print('训练数据集中deg_ang252平均值为{0:.2f}'.format(
        orders_pd_train.buy_deg_ang252.mean()))

    """
        11.2.2 使用全局最优对分类簇集合进行筛选
    """
    brust_min = ump_deg.brust_min()
    print('brust_min:', brust_min)

    llps = ump_deg.cprs[(ump_deg.cprs['lps'] <= brust_min[0]) & (ump_deg.cprs['lms'] <= brust_min[1]) & (
        ump_deg.cprs['lrs'] >= brust_min[2])]
    print('llps:\n', llps)

    print(ump_deg.choose_cprs_component(llps))
    ump_deg.dump_clf(llps)


"""
    11.2.3 跳空主裁
"""


def sample_1123():
    """
    11.2.3 跳空主裁
    :return:
    """
    abu_result_tuple_train, abu_result_tuple_test, metrics_train, metrics_test = load_abu_result_tuple()
    orders_pd_train = abu_result_tuple_train.orders_pd
    ump_jump = AbuUmpMainJump.ump_main_clf_dump(orders_pd_train, save_order=False)
    print(ump_jump.fiter.df.head())

    print('失败概率最大的分类簇{0}'.format(ump_jump.cprs.lrs.argmax()))
    # 拿出跳空失败概率最大的分类簇
    max_failed_cluster_orders = ump_jump.nts[ump_jump.cprs.lrs.argmax()]
    # 显示失败概率最大的分类簇，表11-6所示
    print('max_failed_cluster_orders:\n', max_failed_cluster_orders)

    ml.show_orders_hist(max_failed_cluster_orders, feature_columns=['buy_diff_up_days', 'buy_jump_up_power',
                                                                    'buy_diff_down_days', 'buy_jump_down_power'])

    print('分类簇中jump_up_power平均值为{0:.2f}， 向上跳空平均天数{1:.2f}'.format(
        max_failed_cluster_orders.buy_jump_up_power.mean(), max_failed_cluster_orders.buy_diff_up_days.mean()))

    print('分类簇中jump_down_power平均值为{0:.2f}, 向下跳空平均天数{1:.2f}'.format(
        max_failed_cluster_orders.buy_jump_down_power.mean(), max_failed_cluster_orders.buy_diff_down_days.mean()))

    print('训练数据集中jump_up_power平均值为{0:.2f}，向上跳空平均天数{1:.2f}'.format(
        orders_pd_train.buy_jump_up_power.mean(), orders_pd_train.buy_diff_up_days.mean()))

    print('训练数据集中jump_down_power平均值为{0:.2f}, 向下跳空平均天数{1:.2f}'.format(
        orders_pd_train.buy_jump_down_power.mean(), orders_pd_train.buy_diff_down_days.mean()))


"""
    11.2.4 价格主裁
"""


def sample_1124():
    """
    11.2.4 价格主裁
    :return:
    """
    abu_result_tuple_train, abu_result_tuple_test, metrics_train, metrics_test = load_abu_result_tuple()
    orders_pd_train = abu_result_tuple_train.orders_pd
    ump_price = AbuUmpMainPrice.ump_main_clf_dump(orders_pd_train, save_order=False)
    print('ump_price.fiter.df.head():\n', ump_price.fiter.df.head())

    print('失败概率最大的分类簇{0}'.format(ump_price.cprs.lrs.argmax()))

    # 拿出价格失败概率最大的分类簇
    max_failed_cluster_orders = ump_price.nts[ump_price.cprs.lrs.argmax()]
    # 表11-8所示
    print('max_failed_cluster_orders:\n', max_failed_cluster_orders)


"""
    11.2.5 波动主裁
"""


def sample_1125():
    """
    11.2.5 波动主裁
    :return:
    """
    abu_result_tuple_train, abu_result_tuple_test, metrics_train, metrics_test = load_abu_result_tuple()
    orders_pd_train = abu_result_tuple_train.orders_pd
    # 文件保存在~/abu/data/save_png/中
    ump_wave = AbuUmpMainWave.ump_main_clf_dump(orders_pd_train, save_order=True)
    print('ump_wave.fiter.df.head():\n', ump_wave.fiter.df.head())

    print('失败概率最大的分类簇{0}'.format(ump_wave.cprs.lrs.argmax()))
    # 拿出波动特征失败概率最大的分类簇
    max_failed_cluster_orders = ump_wave.nts[ump_wave.cprs.lrs.argmax()]
    # 表11-10所示
    print('max_failed_cluster_orders:\n', max_failed_cluster_orders)

    ml.show_orders_hist(max_failed_cluster_orders, feature_columns=['buy_wave_score1', 'buy_wave_score3'])

    print('分类簇中wave_score1平均值为{0:.2f}'.format(
        max_failed_cluster_orders.buy_wave_score1.mean()))

    print('分类簇中wave_score3平均值为{0:.2f}'.format(
        max_failed_cluster_orders.buy_wave_score3.mean()))

    ml.show_orders_hist(orders_pd_train, feature_columns=['buy_wave_score1', 'buy_wave_score1'])

    print('训练数据集中wave_score1平均值为{0:.2f}'.format(
        orders_pd_train.buy_wave_score1.mean()))

    print('训练数据集中wave_score3平均值为{0:.2f}'.format(
        orders_pd_train.buy_wave_score1.mean()))


"""
    11.2.6 验证主裁是否称职

    请对照阅读ABU量化系统使用文档 ：第21节 A股UMP决策 中相关内容
"""


def sample_1126():
    """
    11.2.6 验证主裁是否称职
    :return:
    """
    """
        需要有运行之前的代码即有本地化后的裁判，然后通过如下代码直接加载
    """
    ump_deg = AbuUmpMainDeg(predict=True)
    ump_jump = AbuUmpMainJump(predict=True)
    ump_price = AbuUmpMainPrice(predict=True)
    ump_wave = AbuUmpMainWave(predict=True)

    def apply_ml_features_ump(order, predicter, need_hit_cnt):
        if not isinstance(order.ml_features, dict):
            # 低版本pandas dict对象取出来会成为str
            ml_features = ast.literal_eval(order.ml_features)
        else:
            ml_features = order.ml_features

        return predicter.predict_kwargs(need_hit_cnt=need_hit_cnt, **ml_features)

    abu_result_tuple_train, abu_result_tuple_test, metrics_train, metrics_test = load_abu_result_tuple()
    # 选取有交易结果的数据order_has_result
    order_has_result = abu_result_tuple_test.orders_pd[abu_result_tuple_test.orders_pd.result != 0]
    # 角度主裁开始裁决
    order_has_result['ump_deg'] = order_has_result.apply(apply_ml_features_ump, axis=1, args=(ump_deg, 2,))
    # 跳空主裁开始裁决
    order_has_result['ump_jump'] = order_has_result.apply(apply_ml_features_ump, axis=1, args=(ump_jump, 2,))
    # 波动主裁开始裁决
    order_has_result['ump_wave'] = order_has_result.apply(apply_ml_features_ump, axis=1, args=(ump_wave, 2,))
    # 价格主裁开始裁决
    order_has_result['ump_price'] = order_has_result.apply(apply_ml_features_ump, axis=1, args=(ump_price, 2,))

    block_pd = order_has_result.filter(regex='^ump_*')
    block_pd['sum_bk'] = block_pd.sum(axis=1)
    block_pd['result'] = order_has_result['result']

    block_pd = block_pd[block_pd.sum_bk > 0]
    print('四个裁判整体拦截正确率{:.2f}%'.format(
        block_pd[block_pd.result == -1].result.count() / block_pd.result.count() * 100))
    print('block_pd.tail():\n', block_pd.tail())

    def sub_ump_show(block_name):
        sub_block_pd = block_pd[(block_pd[block_name] == 1)]
        # 如果失败就正确 －1->1 1->0
        # noinspection PyTypeChecker
        sub_block_pd.result = np.where(sub_block_pd.result == -1, 1, 0)
        return metrics.accuracy_score(sub_block_pd[block_name], sub_block_pd.result)

    print('角度裁判拦截正确率{:.2f}%'.format(sub_ump_show('ump_deg') * 100))
    print('跳空裁判拦截正确率{:.2f}%'.format(sub_ump_show('ump_jump') * 100))
    print('波动裁判拦截正确率{:.2f}%'.format(sub_ump_show('ump_wave') * 100))
    print('价格裁判拦截正确率{:.2f}%'.format(sub_ump_show('ump_price') * 100))


"""
    11.2.7 在abu系统中开启主裁拦截模式

    请对照阅读ABU量化系统使用文档 ：第21节 A股UMP决策 中相关内容
"""

"""
11.3.1 角度边裁
请对照阅读ABU量化系统使用文档 ：第17节 UMP边裁交易决策，第21节 A股UMP决策 中相关内容

11.3.2 价格边裁
请对照阅读ABU量化系统使用文档 ：第17节 UMP边裁交易决策，第21节 A股UMP决策 中相关内容

11.3.3 波动边裁
请对照阅读ABU量化系统使用文档 ：第17节 UMP边裁交易决策，第21节 A股UMP决策 中相关内容

11.3.4 综合边裁
请对照阅读ABU量化系统使用文档 ：第17节 UMP边裁交易决策，第21节 A股UMP决策 中相关内容

11.3.5 验证边裁是否称职

请对照阅读ABU量化系统使用文档 ：第21节 A股UMP决策 中相关内容

11.3.6 在abu系统中开启边裁拦截模式

请对照阅读ABU量化系统使用文档 ：第21节 A股UMP决策 中相关内容

"""

if __name__ == "__main__":
    sample_111()
    # sample_112()
    # sample_1123()
    # sample_1124()
    # sample_1125()
    # sample_1126()
