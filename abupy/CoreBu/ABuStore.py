# -*- encoding:utf-8 -*-
"""针对交易回测结果存储，读取模块"""

import os
from collections import namedtuple
from enum import Enum
import datetime

import numpy as np
import pandas as pd

from ..CoreBu import ABuEnv
from ..UtilBu import ABuFileUtil


# noinspection PyClassHasNoInit
class AbuResultTuple(namedtuple('AbuResultTuple',
                                ('orders_pd',
                                 'action_pd',
                                 'capital',
                                 'benchmark'))):
    """
        使用abu.run_loop_back返回的nametuple对象：

        orders_pd：回测结果生成的交易订单构成的pd.DataFrame对象
        action_pd: 回测结果生成的交易行为构成的pd.DataFrame对象
        capital:   资金类AbuCapital实例化对象
        benchmark: 交易基准对象，AbuBenchmark实例对象
    """
    __slots__ = ()

    def __repr__(self):
        """打印对象显示：orders_pd.info, action_pd.info, capital, benchmark"""
        return "orders_pd:{}\naction_pd:{}\ncapital:{}\nbenchmark:{}".format(
            self.orders_pd.info(),
            self.action_pd.info(),
            self.capital, self.benchmark)


class EStoreAbu(Enum):
    """保存回测结果的enum类型"""

    """保存普通类型，存储文件后缀为空"""
    E_STORE_NORMAL = 0

    """保存训练集回测，存储文件后缀为train"""
    E_STORE_TRAIN = 1
    """保存测试集交易回测，存储文件后缀为test"""
    E_STORE_TEST = 2

    """保存测试集交易使用主裁ump进行回测，存储文件后缀为test_ump"""
    E_STORE_TEST_UMP = 3
    """保存测试集交易使用主裁＋边裁ump进行回测，存储文件后缀为test_ump_with_edge"""
    E_STORE_TEST_UMP_WITH_EDGE = 4

    """保存测回测，存储文件后缀为自定义字符串"""
    E_STORE_CUSTOM_NAME = 5


def dump_custom_abu_index_csv(custom_name, custom_desc):
    """
    将回测模块的回测结果文件做index描述记录的保存，特定保存已custom_name为索引index，
    custom_desc为内容的一行一列的DataFrame数据，数据保存在cache目录中保存为csv类型文件custom_abu_index.csv
    :param custom_name: custom_name为索引index
    :param custom_desc: 构成DataFrame数据custom_desc为内容
    """
    # 无描述显示No description
    custom_desc = 'No description' if custom_desc is None else custom_desc
    # 特定的一行一列的DataFrame数据
    index_path = np.array([custom_desc]).reshape(1, 1)
    # custom_name为索引index，custom_desc为内容
    index_df = pd.DataFrame(index_path, index=[custom_name],
                            columns=['description'])

    # 数据保存在cache目录中
    index_csv_path = os.path.join(ABuEnv.g_project_data_dir, 'cache', 'custom_abu_index.csv')
    ABuFileUtil.ensure_dir(index_csv_path)

    index_csv_df = ABuFileUtil.load_df_csv(index_csv_path)
    if index_csv_df is not None:
        # 如果已存在相同索引直接返回，不再写入
        if custom_name in index_csv_df.index:
            return
        index_csv_df = index_csv_df.append(index_df)
    else:
        index_csv_df = index_df
    # 最终dump为csv文件
    ABuFileUtil.dump_df_csv(index_csv_path, index_csv_df)


def dump_custom_ump_index_csv(custom_name, ump_unique, is_main_ump, custom_desc):
    """
    将ump训练好的数据文件做index描述记录的保存，特定保存已custom_name + ump_unique为索引index，
    custom_desc, is_main_ump, ump_unique为内容的一行3列的DataFrame数据，数据保存在ump缓存目录
    中保存为csv类型文件custom_ump_index.csv
    :param custom_name: custom_name + ump_unique为索引index
    :param ump_unique: ump类的标识str类型，ump.class_unique_id()
    :param is_main_ump: 是主裁还是边裁标识str类型，eg：main or edge
    :param custom_desc: ump训练数据的描述str
    """
    # 无描述显示No description
    custom_desc = 'No description' if custom_desc is None else custom_desc
    # 特定的一行3列的DataFrame数据
    index_path = np.array([custom_desc, ump_unique, is_main_ump]).reshape(1, 3)
    # custom_desc, is_main_ump, ump_unique为内容的一行3列的DataFrame数据
    index_df = pd.DataFrame(index_path, index=['{}:{}'.format(ump_unique, custom_name)],
                            columns=['description', 'ump_unique', 'is_main_ump'])

    # 数据保存在ump缓存目录
    index_csv_path = os.path.join(ABuEnv.g_project_data_dir, 'ump', 'custom_ump_index.csv')
    ABuFileUtil.ensure_dir(index_csv_path)

    index_csv_df = ABuFileUtil.load_df_csv(index_csv_path)
    if index_csv_df is not None:
        if custom_name in index_csv_df.index:
            return
        index_csv_df = index_csv_df.append(index_df)
    else:
        index_csv_df = index_df
    # 最终dump为csv文件
    ABuFileUtil.dump_df_csv(index_csv_path, index_csv_df)


def _load_custom_index(*paths):
    """执行读取csv，通过abu数据目录 + *paths参数构成读取路径"""
    index_csv_path = os.path.join(ABuEnv.g_project_data_dir, *paths)
    index_csv_df = ABuFileUtil.load_df_csv(index_csv_path)
    return index_csv_df


def load_custom_abu_index():
    """读取回测结果索引描述csv，通过abu数据目录 + cache + custom_abu_index.csv构成读取路径"""
    return _load_custom_index('cache', 'custom_abu_index.csv')


def load_custom_ump_index():
    """读取裁判ump训练索引描述csv，通过abu数据目录 + ump + custom_ump_index.csv构成读取路径"""
    return _load_custom_index('ump', 'custom_ump_index.csv')


def _del_custom_index(custom_name, *paths):
    """执行删除索引描述csv中某一特定行，custom_name为行名称，即执行drop"""
    index_csv_path = os.path.join(ABuEnv.g_project_data_dir, *paths)
    index_csv_df = ABuFileUtil.load_df_csv(index_csv_path)

    if custom_name in index_csv_df.index:
        index_csv_df.drop(custom_name, inplace=True)
        ABuFileUtil.dump_df_csv(index_csv_path, index_csv_df)


def del_custom_abu_index(custom_name):
    """删除回测结果索引描述csv中某一特定行，custom_name为行名称，即执行drop"""
    return _del_custom_index(custom_name, 'cache', 'custom_abu_index.csv')


def del_custom_ump_index(custom_name):
    """删除裁判ump训练索引描述csv中某一特定行，custom_name为行名称，即执行drop"""
    return _del_custom_index(custom_name, 'ump', 'custom_ump_index.csv')


def _cache_abu_result_path(n_folds, store_type, custom_name):
    """由外部参数返回所有单子存贮路径"""
    fn_root = ABuEnv.g_project_cache_dir
    fn_head = '' if n_folds is None else 'n{}_'.format(n_folds)

    # 根据EStoreAbu来决定fn_head
    if store_type == EStoreAbu.E_STORE_TEST:
        fn_head += 'test'
    elif store_type == EStoreAbu.E_STORE_TEST_UMP:
        fn_head += 'test_ump'
    elif store_type == EStoreAbu.E_STORE_TEST_UMP_WITH_EDGE:
        fn_head += 'test_ump_with_edge'
    elif store_type == EStoreAbu.E_STORE_TRAIN:
        fn_head += 'train'
    elif store_type == EStoreAbu.E_STORE_CUSTOM_NAME:
        fn_head += custom_name
    elif store_type != EStoreAbu.E_STORE_NORMAL:
        raise ValueError('store_type error!!!')

    # eg: n2_test_orders_pd
    orders_key = fn_head + '_orders_pd'
    orders_path = os.path.join(fn_root, orders_key)
    # 只需要ensure_dir第一个就可以了
    ABuFileUtil.ensure_dir(orders_path)

    # eg: n2_test_action_pd
    action_key = fn_head + '_action_pd'
    action_path = os.path.join(fn_root, action_key)

    # eg: n2_test_capital
    capital_path = os.path.join(fn_root, fn_head + '_capital')

    # eg: n2_test_benchmark
    benchmark_path = os.path.join(fn_root, fn_head + '_benchmark')

    return orders_path, orders_key, action_path, action_key, capital_path, benchmark_path


def store_abu_result_tuple(abu_result_tuple, n_folds=None, store_type=EStoreAbu.E_STORE_NORMAL,
                           custom_name=None):
    """
    保存abu.run_loop_back的回测结果AbuResultTuple对象，根据n_folds，store_type参数
    来定义存储的文件名称

    :param abu_result_tuple: AbuResultTuple对象类型
    :param n_folds: 回测执行了几年，只影响存贮文件名
    :param store_type: 回测保存类型EStoreAbu类型，只影响存贮文件名
    :param custom_name: 如果store_type=EStoreAbu.E_STORE_CUSTOM_NAME时需要的自定义文件名称
    """
    orders_path, orders_key, action_path, action_key, capital_path, benchmark_path = _cache_abu_result_path(
        n_folds, store_type, custom_name)
    # abu_result_tuple.orders_pd使用dump_hdf5存储
    ABuFileUtil.dump_hdf5(orders_path, abu_result_tuple.orders_pd, orders_key)
    # abu_result_tuple.action_pd使用dump_hdf5存储
    ABuFileUtil.dump_hdf5(action_path, abu_result_tuple.action_pd, action_key)
    # abu_result_tuple.capital使用dump_pickle存储AbuCapital对象
    ABuFileUtil.dump_pickle(abu_result_tuple.capital, capital_path)
    # abu_result_tuple.benchmark使用dump_pickle存储AbuBenchmark对象
    ABuFileUtil.dump_pickle(abu_result_tuple.benchmark, benchmark_path)


def load_abu_result_tuple(n_folds=None, store_type=EStoreAbu.E_STORE_NORMAL, custom_name=None):
    """
    读取使用store_abu_result_tuple保存的回测结果，根据n_folds，store_type参数
    来定义读取的文件名称，依次读取orders_pd，action_pd，capital，benchmark后构造
    AbuResultTuple对象返回

    :param n_folds: 回测执行了几年，只影响读取的文件名
    :param store_type: 回测保存类型EStoreAbu类型，只影响读取的文件名
    :param custom_name: 如果store_type=EStoreAbu.E_STORE_CUSTOM_NAME时需要的自定义文件名称
    :return: AbuResultTuple对象
    """

    orders_path, orders_key, action_path, action_key, capital_path, benchmark_path = _cache_abu_result_path(
        n_folds, store_type, custom_name)
    # load_hdf5读取pd.DataFrame对象orders_pd
    orders_pd = ABuFileUtil.load_hdf5(orders_path, orders_key)
    # load_hdf5读取pd.DataFrame对象action_pd
    action_pd = ABuFileUtil.load_hdf5(action_path, action_key)
    # load_pickle读取AbuCapital对象capital，AbuCapital混入了类PickleStateMixin
    capital = ABuFileUtil.load_pickle(capital_path)
    # load_pickle读取AbuBenchmark对象benchmark，AbuBenchmark混入了类PickleStateMixin
    benchmark = ABuFileUtil.load_pickle(benchmark_path)
    # 构建返回AbuResultTuple对象
    return AbuResultTuple(orders_pd, action_pd, capital, benchmark)


def delete_abu_result_tuple(n_folds=None, store_type=EStoreAbu.E_STORE_NORMAL, custom_name=None, del_index=False):
    """
    删除本地store_abu_result_tuple保存的回测结果，根据n_folds，store_type参数
    来定义读取的文件名称，依次读取orders_pd，action_pd，capital，benchmark后构造
    AbuResultTuple对象返回

    :param n_folds: 回测执行了几年，只影响读取的文件名
    :param store_type: 回测保存类型EStoreAbu类型，只影响读取的文件名
    :param custom_name: 如果store_type=EStoreAbu.E_STORE_CUSTOM_NAME时需要的自定义文件名称
    :param del_index: 是否删除index csv
    """

    orders_path, _, action_path, _, capital_path, benchmark_path = _cache_abu_result_path(
        n_folds, store_type, custom_name)

    # 获取各个单子路径后依次删除
    ABuFileUtil.del_file(orders_path)
    ABuFileUtil.del_file(action_path)
    ABuFileUtil.del_file(capital_path)
    ABuFileUtil.del_file(benchmark_path)

    if del_index:
        # 删除回测所对应的描述文件索引行
        del_custom_abu_index(custom_name)


def store_abu_result_out_put(abu_result_tuple, show_log=True):
    """
    保存abu.run_loop_back的回测结果AbuResultTuple对象，根据当前时间戳保存来定义存储的文件夹名称，
    不同于保存在cache中，将保存在out_put文件夹中，且所有单子都使用csv进行保存，不使用hdf5进行保存
    保证外部的可读性
    1. 交易单: orders.csv
    2. 行动单: actions.csv
    3. 资金单: capital.csv
    4. 手续费: commission.csv
    """
    base_dir = 'out_put'
    # 时间字符串
    date_dir = datetime.datetime.now().strftime("%Y_%m_%d_%H_%M_%S_%f")
    fn = os.path.join(ABuEnv.g_project_data_dir, base_dir, date_dir, 'orders.csv')
    ABuFileUtil.ensure_dir(fn)

    fn = os.path.join(ABuEnv.g_project_data_dir, base_dir, date_dir, 'actions.csv')
    ABuFileUtil.dump_df_csv(fn, abu_result_tuple.action_pd)
    if show_log:
        print('save {} suc!'.format(fn))

    fn = os.path.join(ABuEnv.g_project_data_dir, base_dir, date_dir, 'capital.csv')
    ABuFileUtil.dump_df_csv(fn, abu_result_tuple.capital.capital_pd)
    if show_log:
        print('save {} suc!'.format(fn))

    fn = os.path.join(ABuEnv.g_project_data_dir, base_dir, date_dir, 'commission.csv')
    ABuFileUtil.dump_df_csv(fn, abu_result_tuple.capital.commission.commission_df)
    if show_log:
        print('save {} suc!'.format(fn))
