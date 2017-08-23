# -*- encoding:utf-8 -*-
"""针对交易回测结果存储，读取模块"""

import os
from collections import namedtuple
from enum import Enum

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


def store_abu_result_tuple(abu_result_tuple, n_folds, store_type=None, custom_name=None):
    """
    保存abu.run_loop_back的回测结果AbuResultTuple对象，根据n_folds，store_type参数
    来定义存储的文件名称

    :param abu_result_tuple: AbuResultTuple对象类型
    :param n_folds: 回测执行了几年，只影响存贮文件名
    :param store_type: 回测保存类型EStoreAbu类型，只影响存贮文件名
    :param custom_name: 如果store_type=EStoreAbu.E_STORE_CUSTOM_NAME时需要的自定义文件名称
    """
    fn_root = ABuEnv.g_project_cache_dir
    fn_head = 'n{}_'.format(n_folds)

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

    # eg: n2_test_orders_pd
    key = fn_head + '_orders_pd'
    fn = os.path.join(fn_root, key)
    ABuFileUtil.ensure_dir(fn)
    # abu_result_tuple.orders_pd使用dump_hdf5存储
    ABuFileUtil.dump_hdf5(fn, abu_result_tuple.orders_pd, key)

    # eg: n2_test_action_pd
    key = fn_head + '_action_pd'
    fn = os.path.join(fn_root, key)
    # abu_result_tuple.action_pd使用dump_hdf5存储
    ABuFileUtil.dump_hdf5(fn, abu_result_tuple.action_pd, key)

    # eg: n2_test_capital
    fn = os.path.join(fn_root, fn_head + '_capital')
    # abu_result_tuple.capital使用dump_pickle存储AbuCapital对象
    ABuFileUtil.dump_pickle(abu_result_tuple.capital, fn)

    # eg: n2_test_benchmark
    fn = os.path.join(fn_root, fn_head + '_benchmark')
    # abu_result_tuple.benchmark使用dump_pickle存储AbuBenchmark对象
    ABuFileUtil.dump_pickle(abu_result_tuple.benchmark, fn)


def load_abu_result_tuple(n_folds, store_type, custom_name=None):
    """
    读取使用store_abu_result_tuple保存的回测结果，根据n_folds，store_type参数
    来定义读取的文件名称，依次读取orders_pd，action_pd，capital，benchmark后构造
    AbuResultTuple对象返回

    :param n_folds: 回测执行了几年，只影响读取的文件名
    :param store_type: 回测保存类型EStoreAbu类型，只影响读取的文件名
    :param custom_name: 如果store_type=EStoreAbu.E_STORE_CUSTOM_NAME时需要的自定义文件名称
    :return: AbuResultTuple对象
    """
    fn_root = ABuEnv.g_project_cache_dir
    fn_head = 'n{}_'.format(n_folds)
    # 根据EStoreAbu来决定fn_head
    if store_type == EStoreAbu.E_STORE_TRAIN:
        fn_head += 'train'
    elif store_type == EStoreAbu.E_STORE_TEST:
        fn_head += 'test'
    elif store_type == EStoreAbu.E_STORE_TEST_UMP:
        fn_head += 'test_ump'
    elif store_type == EStoreAbu.E_STORE_TEST_UMP_WITH_EDGE:
        fn_head += 'test_ump_with_edge'
    elif store_type == EStoreAbu.E_STORE_CUSTOM_NAME:
        fn_head += custom_name
    elif store_type != EStoreAbu.E_STORE_NORMAL:
        raise ValueError('store_type error!!!')

    # eg: n2_test_orders_pd
    key = fn_head + '_orders_pd'
    fn = os.path.join(fn_root, key)
    # load_hdf5读取pd.DataFrame对象orders_pd
    orders_pd = ABuFileUtil.load_hdf5(fn, key)

    # eg: n2_test_action_pd
    key = fn_head + '_action_pd'
    fn = os.path.join(fn_root, key)
    # load_hdf5读取pd.DataFrame对象action_pd
    action_pd = ABuFileUtil.load_hdf5(fn, key)

    fn = os.path.join(fn_root, fn_head + '_capital')
    # load_pickle读取AbuCapital对象capital，AbuCapital混入了类PickleStateMixin
    capital = ABuFileUtil.load_pickle(fn)

    fn = os.path.join(fn_root, fn_head + '_benchmark')
    # load_pickle读取AbuBenchmark对象benchmark，AbuBenchmark混入了类PickleStateMixin
    benchmark = ABuFileUtil.load_pickle(fn)
    # 构建返回AbuResultTuple对象
    return AbuResultTuple(orders_pd, action_pd, capital, benchmark)
