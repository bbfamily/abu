# -*- encoding:utf-8 -*-
from __future__ import print_function
import seaborn as sns
import warnings
import numpy as np
# noinspection PyUnresolvedReferences
import abu_local_env
from abupy import tl
from abupy import abu
import abupy

warnings.filterwarnings('ignore')
sns.set_context(rc={'figure.figsize': (14, 7)})

"""
    量化相关性分析

    abu量化系统github地址：https://github.com/bbfamily/abu (您的star是我的动力！)
    abu量化文档教程ipython notebook：https://github.com/bbfamily/abu/tree/master/abupy_lecture

    本节建议对照阅读abu量化文档：第14节 量化相关性分析应用
"""


def sample_b0():
    """
    相关分析默认强制使用local数据，所以本地无缓存，请先进行数据更新

    如果没有运行过abu量化文档－第十九节 数据源：中使用腾讯数据源进行数据更新，需要运行
    如果运行过就不要重复运行了：
    """
    from abupy import EMarketTargetType, EMarketSourceType, EDataCacheType

    # 关闭沙盒数据环境
    abupy.env.disable_example_env_ipython()
    abupy.env.g_market_source = EMarketSourceType.E_MARKET_SOURCE_tx
    abupy.env.g_data_cache_type = EDataCacheType.E_DATA_CACHE_CSV
    # 首选这里预下载市场中所有股票的6年数据(做5年回测，需要预先下载6年数据)
    abu.run_kl_update(start='2011-08-08', end='2017-08-08', market=EMarketTargetType.E_MARKET_TARGET_US)


def sample_b1():
    """
    B1 皮尔逊相关系数
    :return:
    """
    arr1 = np.random.rand(10000)
    arr2 = np.random.rand(10000)

    corr = np.cov(arr1, arr2) / np.std(arr1) * np.std(arr2)
    print('corr:\n', corr)
    print('corr[0, 1]:', corr[0, 1])

    print('np.corrcoef(arr1, arr2)[0, 1]:', np.corrcoef(arr1, arr2)[0, 1])


# noinspection PyTypeChecker
def sample_b2():
    """
    B2 斯皮尔曼秩相关系数
    :return:
    """
    arr1 = np.random.rand(10000)
    arr2 = arr1 + np.random.normal(0, .2, 10000)

    print('np.corrcoef(arr1, arr2)[0, 1]:', np.corrcoef(arr1, arr2)[0, 1])

    import scipy.stats as stats
    demo_list = [1, 2, 10, 100, 2, 1000]
    print('原始序列: ', demo_list)
    print('序列的秩: ', list(stats.rankdata(demo_list)))

    # 实现斯皮尔曼秩相关系数
    def spearmanr(a, b=None, axis=0):
        a, outaxis = _chk_asarray(a, axis)
        ar = np.apply_along_axis(stats.rankdata, outaxis, a)
        br = None
        if b is not None:
            b, axisout = _chk_asarray(b, axis)
            br = np.apply_along_axis(stats.rankdata, axisout, b)
        return np.corrcoef(ar, br, rowvar=outaxis)

    def _chk_asarray(a, axis):
        if axis is None:
            a = np.ravel(a)
            outaxis = 0
        else:
            a = np.asarray(a)
            outaxis = axis
        if a.ndim == 0:
            a = np.atleast_1d(a)
        return a, outaxis

    print('spearmanr(arr1, arr2)[0, 1]:', spearmanr(arr1, arr2)[0, 1])

    """
        scipy.stats中直接封装斯皮尔曼秩相关系数函数stats.spearmanr()函数
        注意下面的方法速度没有上述自己实现计算spearmanr相关系数的方法快，因为附加计算了pvalue
    """
    print('stats.spearmanr(arr1, arr2):', stats.spearmanr(arr1, arr2))


"""
    B3 相关性使用示例
"""

"""
    【示例1】使用abu量化系统中的ABuSimilar.find_similar_with_xxx()函数找到与目标股票相关程度最高的股票可视化
"""


def sample_b3_1():
    """
    【示例1】使用abu量化系统中的ABuSimilar.find_similar_with_xxx()函数找到与目标股票相关程度最高的股票可视化
    :return:
    """
    # find_similar_with_cnt可视化与tsla相关top10，以及tsla相关性dict：cmp_cnt=252(252天)，加权相关，E_CORE_TYPE_PEARS(皮尔逊)
    from abupy import find_similar_with_cnt, ECoreCorrType
    _ = find_similar_with_cnt('usTSLA', cmp_cnt=252, show_cnt=10, rolling=True, show=True,
                              corr_type=ECoreCorrType.E_CORE_TYPE_PEARS)

    # find_similar_with_se可视化与tsla相关top10，以及tsla相关性dict：从'2012-01-01'直到'2017-01-01'5年数据，非加权相关，皮尔逊
    from abupy import find_similar_with_se
    _ = find_similar_with_se('usTSLA', start='2012-01-01', end='2017-01-01', show_cnt=10, rolling=False,
                             show=True, corr_type=ECoreCorrType.E_CORE_TYPE_PEARS)

    # find_similar_with_folds可视化与tsla相关top10，以及tsla相关性dict：n_folds=3(3年数据)，
    # 非加权相关，E_CORE_TYPE_SPERM斯皮尔曼
    from abupy import find_similar_with_folds
    _ = find_similar_with_folds('usTSLA', n_folds=3, show_cnt=10, rolling=False, show=True,
                                corr_type=ECoreCorrType.E_CORE_TYPE_SPERM)


"""
    【示例2】使用abu量化系统中的ABuTLSimilar.calc_similar()函数计算两支股票相对整个市场的相关性评级rank
"""


def sample_b3_2():
    """
    【示例2】使用abu量化系统中的ABuTLSimilar.calc_similar()函数计算两支股票相对整个市场的相关性评级rank
    :return:
    """
    # 以整个市场作为观察者，usTSLA与usNOAH的相关性
    rank_score, sum_rank = tl.similar.calc_similar('usNOAH', 'usTSLA')
    print('rank_score', rank_score)
    from abupy import find_similar_with_cnt
    net_cg_ret = find_similar_with_cnt('usTSLA', cmp_cnt=252, show=False)

    # 以usTSLA作为观察者，它与usNOAH的相关性数值
    for ncr in net_cg_ret:
        if ncr[0] == 'usNOAH':
            print(ncr[1])
            break
    """
        以整个市场作为观察者，与usTSLA相关性TOP 10可视化
        直接将calc_similar返回的sum_rank传入calc_similar_top直接用，不用再计算了
    """
    tl.similar.calc_similar_top('usTSLA', sum_rank)


"""
    【示例3】相关与协整组成的一个简单量化选股策略, 使用封装好的函数coint_similar()
"""


def sample_b3_3():
    """
    【示例3】相关与协整组成的一个简单量化选股策略, 使用封装好的函数coint_similar()
    :return:
    """
    tl.similar.coint_similar('usTSLA')


"""
    【示例4】abu量化系统选股结合相关性，编写相关性选股策略
"""


def sample_b3_4():
    """
    【示例4】abu量化系统选股结合相关性，编写相关性选股策略
    AbuPickSimilarNTop源代码请自行阅读，只简单示例使用。
    :return:
    """
    from abupy import AbuPickSimilarNTop
    from abupy import AbuPickStockWorker
    from abupy import AbuBenchmark, AbuCapital, AbuKLManager

    benchmark = AbuBenchmark()

    # 选股因子AbuPickSimilarNTop， 寻找与usTSLA相关性不低于0.95的股票
    # 这里内部使用以整个市场作为观察者方式计算，即取值范围0-1
    stock_pickers = [{'class': AbuPickSimilarNTop,
                      'similar_stock': 'usTSLA', 'threshold_similar_min': 0.95}]

    # 从这几个股票里进行选股，只是为了演示方便，一般的选股都会是数量比较多的情况比如全市场股票
    choice_symbols = ['usNOAH', 'usSFUN', 'usBIDU', 'usAAPL', 'usGOOG', 'usTSLA', 'usWUBA', 'usVIPS']

    capital = AbuCapital(1000000, benchmark)
    kl_pd_manager = AbuKLManager(benchmark, capital)
    stock_pick = AbuPickStockWorker(capital, benchmark, kl_pd_manager, choice_symbols=choice_symbols,
                                    stock_pickers=stock_pickers)
    stock_pick.fit()
    print('stock_pick.choice_symbols:\n', stock_pick.choice_symbols)

    """
        通过选股因子first_choice属性执行批量优先选股操作，具体阅读源代码
    """
    # 选股因子AbuPickSimilarNTop， 寻找与usTSLA相关性不低于0.95的股票
    # 通过设置'first_choice':True，进行优先批量操作，默认从对应市场选股
    stock_pickers = [{'class': AbuPickSimilarNTop, 'first_choice': True,
                      'similar_stock': 'usTSLA', 'threshold_similar_min': 0.95}]
    benchmark = AbuBenchmark()
    capital = AbuCapital(1000000, benchmark)
    kl_pd_manager = AbuKLManager(benchmark, capital)
    stock_pick = AbuPickStockWorker(capital, benchmark, kl_pd_manager, choice_symbols=None,
                                    stock_pickers=stock_pickers)
    stock_pick.fit()
    print('stock_pick.choice_symbols:\n', stock_pick.choice_symbols)

if __name__ == "__main__":
    # sample_b0()
    sample_b1()
    # sample_b2()
    # sample_b3_1()
    # sample_b3_2()
    # sample_b3_3()
    # sample_b3_4()
