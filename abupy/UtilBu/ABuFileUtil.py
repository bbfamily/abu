# coding=utf-8
"""
    文件处理读取写入
"""
from __future__ import print_function
from __future__ import division
from __future__ import absolute_import

import logging
import os
import shutil
from contextlib import contextmanager

import functools
import pandas as pd

from .ABuDTUtil import warnings_filter
# noinspection PyUnresolvedReferences
from ..CoreBu.ABuFixes import pickle, Pickler, Unpickler, as_bytes

"""HDF5_COMP_LEVEL：压缩级别：0-9，如果修改了压缩级别，需要删除之前的物理文件"""
HDF5_COMP_LEVEL = 4
""" HDF5_COMP_LIB: 使用的压缩库:'blosc', 'bzip2', 'lzo', 'zlib', 如果修改了压缩库，需要删除之前的物理文件"""
HDF5_COMP_LIB = 'blosc'

"""HDF5内部存贮依然会使用pickle，即python版本切换，本地文件会有协议冲突：使用所支持的最高协议进行dump"""
K_SET_PICKLE_HIGHEST_PROTOCOL = False
"""HDF5内部存贮依然会使用pickle，即python版本切换，本地文件会有协议冲突：python2, python3协议兼容模式，使用protocol=0"""
K_SET_PICKLE_ZERO_PROTOCOL = False


def ensure_dir(a_path):
    """
    确保a_path所在路径文件夹存在，如果a_path是文件将确保它上一级
    文件夹的存在
    :param a_path: str对象, 相对路径或者绝对路径
    """
    if os.path.isdir(a_path):
        a_dir = a_path
    else:
        a_dir = os.path.dirname(a_path)
    if not os.path.exists(a_dir):
        os.makedirs(a_dir)


def ensure_file(a_path):
    """
    确保a_path所在路径文件存在，首先会使用ensure_dir确保文件夹存在，
    确定a_path是file会确保文件存在
    :param a_path: str对象, 相对路径或者绝对路径
    :return:
    """
    ensure_dir(a_path)
    open(a_path, 'a+').close()


def file_exist(a_path):
    """
    a_path是否存在
    :param a_path: str对象, 相对路径或者绝对路径
    """
    return os.path.exists(a_path)


def copy_file(source, target_dir):
    """
    拷贝文件操作，支持文件夹拷贝操作
    :param source: 文件名或者文件夹，str对象, 相对路径或者绝对路径
    :param target_dir: 文件名或者文件夹，str对象, 相对路径或者绝对路径
    """

    if os.path.exists(source):
        logging.error('copy_file source={} not exists!'.format(source))
        return

    ensure_dir(target_dir)

    if os.path.isdir(source):
        shutil.copytree(source, target_dir)
    else:
        shutil.copy(source, target_dir)


def del_file(a_path):
    """
    删除文件操作，支持文件夹删除操作
    :param a_path: 文件名或者文件夹，str对象, 相对路径或者绝对路径
    """
    if not file_exist(a_path):
        return

    if os.path.isdir(a_path):
        shutil.rmtree(a_path)
    else:
        os.remove(a_path)


def load_pickle(file_name):
    """
    读取python序列化的本地文件
    :param file_name: 文件名，str对象, 相对路径或者绝对路径
    :return:
    """
    if not file_exist(file_name):
        logging.error('load_pickle file_name={} not exists!'.format(file_name))
        return None

    # TODO 根据文件大小，决定这里是否需要tip wait
    print('please wait! load_pickle....:', file_name)

    try:
        with open(file_name, "rb") as unpickler_file:
            unpickler = Unpickler(unpickler_file)
            ret = unpickler.load()
    except EOFError:
        print('unpickler file with EOFError, please check {} is 0kb!!!'.format(file_name))
        ret = {}

    return ret


def dump_pickle(input_obj, file_name, how='normal'):
    """
    存贮python序列化的本地文件
    :param input_obj: 需要进行序列化的对象
    :param file_name: 文件名，str对象, 相对路径或者绝对路径
    :param how: 序列化协议选择，默认normal不特殊处理，
                zero使用python2, python3协议兼容模式，使用protocol=0，
                high使用支持的最高协议
    """
    ensure_dir(file_name)

    print('please wait! dump_pickle....:', file_name)

    try:
        with open(file_name, "wb") as pick_file:
            if K_SET_PICKLE_HIGHEST_PROTOCOL or how == 'high':
                """使用所支持的最高协议进行dump"""
                pickle.dump(input_obj, pick_file, pickle.HIGHEST_PROTOCOL)
            elif K_SET_PICKLE_ZERO_PROTOCOL or how == 'zero':
                """python2, python3协议兼容模式，使用protocol=0"""
                pickle.dump(input_obj, pick_file, 0)
            else:
                pickler = Pickler(pick_file)
                pickler.dump(input_obj)
    except Exception as e:
        logging.exception(e)


"""hdf5批量处理时保存HDFStore对象，为避免反复open，close"""
__g_batch_h5s = None


def __start_batch_h5s(file_name, mode):
    """
    使用pd.HDFStore打开file_name对象保存在全局__g_batch_h5s中
    :param file_name: hdf5文件路径名
    :param mode: 打开hdf5文件模式。eg：w, r, a
    """
    global __g_batch_h5s
    __g_batch_h5s = pd.HDFStore(file_name, mode, complevel=HDF5_COMP_LEVEL, complib=HDF5_COMP_LIB)


def __end_batch_h5s():
    """
    如果__g_batch_h5s中hdf5对象仍然是打开的，进行flush，后close
    """
    global __g_batch_h5s
    if __g_batch_h5s is not None and __g_batch_h5s.is_open:
        __g_batch_h5s.flush()
        __g_batch_h5s.close()
        __g_batch_h5s = None


def batch_h5s(h5_fn, mode='a'):
    """
    使用装饰器方式对hdf5操作进行批量处理，外部使用：
        eg： 详见ABuSymbolPd.py
            @batch_h5s(h5s_fn)
            def _batch_save():
                for df_dict in df_dicts:
                    # 每一个df_dict是一个并行的序列返回的数据
                    for ind, (key_tuple, df) in enumerate(df_dict.values()):
                        # (key_tuple, df)是保存kl需要的数据, 迭代后直接使用save_kline_df
                        save_kline_df(df, *key_tuple)
                        if df is not None:
                            print("save kl {}_{}_{} {}/{}".format(key_tuple[0].value, key_tuple[1], key_tuple[2], ind,
                                                                  df.shape[0]))
                    # 完成一层循环一次，即批量保存完一个并行的序列返回的数据后，进行清屏
                    do_clear_output()
    :param h5_fn: hdf5文件路径名, 如果为None。即忽略整个批处理流程
    :param mode: 打开hdf5文件模式。eg：w, r, a
    :return:
    """

    def _batch_h5s(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            if h5_fn is not None:
                __start_batch_h5s(h5_fn, mode)
            ret = func(*args, **kwargs)
            if h5_fn is not None:
                __end_batch_h5s()
            return ret

        return wrapper

    return _batch_h5s


@contextmanager
def batch_ctx_h5s(h5_fn, mode='a'):
    """
    使用上下文管理器方式对hdf5操作进行批量处理，与装饰器模式batch_h5s
    功能相同，为外部不方便封装为具体操作函数时使用
        eg：
            with batch_ctx_h5s(h5s_fn):
                for df_dict in df_dicts:
                    # 每一个df_dict是一个并行的序列返回的数据
                    for ind, (key_tuple, df) in enumerate(df_dict.values()):
                        # (key_tuple, df)是保存kl需要的数据, 迭代后直接使用save_kline_df
                        save_kline_df(df, *key_tuple)
                        if df is not None:
                            print("save kl {}_{}_{} {}/{}".format(key_tuple[0].value, key_tuple[1], key_tuple[2], ind,
                                                                  df.shape[0]))
                    # 完成一层循环一次，即批量保存完一个并行的序列返回的数据后，进行清屏
                    do_clear_output()

    :param h5_fn: hdf5文件路径名, 如果为None。即忽略整个批处理流程
    :param mode: 打开hdf5文件模式。eg：w, r, a
    :return:
    """
    if h5_fn is not None:
        __start_batch_h5s(h5_fn, mode)

    yield

    if h5_fn is not None:
        __end_batch_h5s()


@warnings_filter
def dump_del_hdf5(file_name, dump_dict, del_array=None):
    """
    对hdf5进行删除和保存操作，del_array中的key先删除，后保存dump_dict
    字典中的数据
    :param file_name: hdf5文件路径名
    :param dump_dict: 需要保存到hdf5中的数据字典
    :param del_array: 需要从hdf5中删除的key序列
    """
    global __g_batch_h5s

    def do_dump_del_hdf5(h5s):
        """
        对hdf5进行删除和保存操作执行函数
        :param h5s: hdf5对象句柄
        :return:
        """
        if del_array is not None:
            # 先执行删除操作
            for del_key in del_array:
                if h5s.__contains__(del_key):
                    del h5s[del_key]

        for input_key in dump_dict:
            input_obj = dump_dict[input_key]
            h5s[input_key] = input_obj

    if __g_batch_h5s is None:
        # 如果批处理句柄没有打着，使用with pd.HDFStore先打开
        with pd.HDFStore(file_name, 'a', complevel=HDF5_COMP_LEVEL, complib=HDF5_COMP_LIB) as h5s_obj:
            do_dump_del_hdf5(h5s_obj)
    else:
        # 使用批处理句柄__g_batch_h5s操作
        do_dump_del_hdf5(__g_batch_h5s)


@warnings_filter
def dump_hdf5(file_name, input_obj, input_key):
    """
    对hdf5进行保存操作
    :param file_name: hdf5文件路径名
    :param input_obj: 保存的数据对象
    :param input_key: 保存的数据key
    """
    global __g_batch_h5s

    if __g_batch_h5s is None:
        # 如果批处理句柄没有打着，使用with pd.HDFStore先打开
        with pd.HDFStore(file_name, 'a', complevel=HDF5_COMP_LEVEL, complib=HDF5_COMP_LIB) as h5s:
            h5s[input_key] = input_obj
    else:
        # 使用批处理句柄__g_batch_h5s操作
        __g_batch_h5s[input_key] = input_obj


def del_hdf5(file_name, key):
    """
    对hdf5进行删除操作
    :param file_name: hdf5文件路径名
    :param key: 保存的数据key
    """
    if not file_exist(file_name):
        return

    if __g_batch_h5s is None:
        # 如果批处理句柄没有打着，使用with pd.HDFStore先打开
        with pd.HDFStore(file_name, 'a', complevel=HDF5_COMP_LEVEL, complib=HDF5_COMP_LIB) as h5s:
            if h5s.__contains__(key):
                del h5s[key]
    else:
        # 使用批处理句柄__g_batch_h5s操作
        if __g_batch_h5s.__contains__(key):
            del __g_batch_h5s[key]


def load_hdf5(file_name, key):
    """
    读取hdf5中的数据
    :param file_name: df5文件路径名
    :param key: 保存的数据key
    """
    global __g_batch_h5s

    if not file_exist(file_name):
        return None

    def _load_hdf5(h5s):
        load_obj = None
        if h5s.__contains__(key):
            try:
                load_obj = h5s[key]
            except (AttributeError, TypeError):
                # 'NoneType' attribute 'T' is just None
                # TypeError: 'len() of unsized object'
                # 低版本hdf5 bug导致的存贮异常情况的读取，忽略，计为正常损耗
                pass
        return load_obj

    if __g_batch_h5s is None:
        # 如果批处理句柄没有打着，使用with pd.HDFStore先打开
        with pd.HDFStore(file_name, 'a', complevel=HDF5_COMP_LEVEL, complib=HDF5_COMP_LIB) as h5s_obj:
            return _load_hdf5(h5s_obj)
    else:
        # 使用批处理句柄__g_batch_h5s操作
        return _load_hdf5(__g_batch_h5s)


def dump_df_csv(file_name, df):
    """
    将pd.DataFrame对象保存在csv中
    :param file_name: 保存csv的文件名称
    :param df: 需要保存的pd.DataFrame对象
    """
    if df is not None:
        # TODO 为效率，不应该在函数内部ensure_dir，确保使用dump_df_csv需要在外部ensure_dir
        ensure_dir(file_name)
        df.to_csv(file_name, columns=df.columns, index=True, encoding='utf-8')


def load_df_csv(file_name):
    """
    从csv文件中实例化pd.DataFrame对象
    :param file_name: 保存csv的文件名称
    :return: pd.DataFrame对象
    """
    if file_exist(file_name):
        return pd.read_csv(file_name, index_col=0)
    return None


def save_file(ct, file_name):
    """
    将内容ct保存文件
    :param ct: 内容str对象
    :param file_name: 保存的文件名称
    :return:
    """
    ensure_dir(file_name)
    with open(file_name, 'wb') as f:
        f.write(ct)
