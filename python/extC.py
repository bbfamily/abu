# -*- encoding:utf-8 -*-
from __future__ import print_function
import seaborn as sns
import warnings

# noinspection PyUnresolvedReferences
import abu_local_env
import abupy
from abupy import EStoreAbu, abu
from abupy import ABuSymbolPd
from abupy import tl
from abupy import nd
from abupy import ABuMarketDrawing

warnings.filterwarnings('ignore')
sns.set_context(rc={'figure.figsize': (14, 7)})

"""
    附录C-量化统计分析及指标应用

    abu量化系统github地址：https://github.com/bbfamily/abu (您的star是我的动力！)
    abu量化文档教程ipython notebook：https://github.com/bbfamily/abu/tree/master/abupy_lecture

    本节内容建议对照阅读abu量化文档：第13节 量化技术分析应用
"""


def sample_c1():
    """
    C.1 量化统计分析应用
    :return:
    """
    tsla_df = ABuSymbolPd.make_kl_df('usTSLA', n_folds=2)

    jumps = tl.jump.calc_jump(tsla_df)
    print('jumps:\n', jumps)

    # sw[0]代表非时间因素的jump_power，sw[1]代表时间加权因素的jump_power，当sw[0]=1时与非加权方式相同，具体实现请参考源代码
    filter_jumps = tl.jump.calc_jump_line_weight(tsla_df, sw=(0.5, 0.5))
    print('filter_jumps:\n', filter_jumps)

    # tl.wave.calc_wave_abs()函数可视化价格波动情况
    tl.wave.calc_wave_abs(tsla_df, xd=21, show=True)


"""
    C.2 量化技术指标应用: 对量化策略失败结果的人工分析
"""


def sample_c2():
    """
    C.2 量化技术指标应用: 对量化策略失败结果的人工分析
    :return:
    """
    abupy.env.disable_example_env_ipython()

    # 从之前章节的缓存中读取交易数据
    abu_result_tuple_train = abu.load_abu_result_tuple(n_folds=5, store_type=EStoreAbu.E_STORE_CUSTOM_NAME,
                                                       custom_name='train_cn')
    # 只筛选orders中有交易结果的单子
    has_result = abu_result_tuple_train.orders_pd[
        abu_result_tuple_train.orders_pd.result == -1]

    # 随便拿一个交易数据作为示例
    sample_order = has_result.ix[100]
    _ = ABuMarketDrawing.plot_candle_from_order(sample_order)

    nd.macd.plot_macd_from_order(sample_order, date_ext=252)
    nd.boll.plot_boll_from_order(has_result.ix[100], date_ext=252)
    nd.ma.plot_ma_from_order(has_result.ix[100], date_ext=252, time_period=[10, 20, 30, 60, 90, 120])


if __name__ == "__main__":
    sample_c1()
    # sample_c2()
