# -*- encoding:utf-8 -*-
"""
    通用装饰器, 上下文管理器工具模块
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import functools
import logging
import pdb
import time
import warnings
from collections import Iterable
from contextlib import contextmanager

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

from ..CoreBu import ABuEnv
from ..CoreBu.ABuFixes import six


def warnings_filter(func):
    """
        作用范围：函数装饰器 (模块函数或者类函数)
        功能：被装饰的函数上的警告不会打印，忽略
    """

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        warnings.simplefilter('ignore')
        ret = func(*args, **kwargs)
        if not ABuEnv.g_ignore_all_warnings:
            # 如果env中的设置不是忽略所有才恢复
            warnings.simplefilter('default')
        return ret

    return wrapper


def singleton(cls):
    """
        作用范围：类装饰器
        功能：被装饰后类变成单例类
    """

    instances = {}

    @functools.wraps(cls)
    def get_instance(*args, **kw):
        if cls not in instances:
            # 不存在实例instances才进行构造
            instances[cls] = cls(*args, **kw)
        return instances[cls]

    return get_instance


# TODO 放在这里不合适，还要和ABuScalerUtil中的装饰器arr_to_pandas重复代码进行重构
def arr_to_pandas(arr):
    """
        函数装饰器：将可以迭代的序列转换为pd.DataFrame或者pd.Series，支持
        np.ndarray，list，dict, list，set，嵌套可迭代序列, 混嵌套可迭代序列
    """
    # TODO Iterable和six.string_types的判断抽出来放在一个模块，做为Iterable的判断来使用
    if not isinstance(arr, Iterable) or isinstance(arr, six.string_types):
        return arr

    if not isinstance(arr, pd.DataFrame) or isinstance(arr, pd.Series):
        if isinstance(arr, np.ndarray) and len(arr.shape) > 1 and arr.shape[1] > 1:
            # np.ndarray > 1列的转换为pd.DataFrame
            arr = pd.DataFrame(arr)
        elif isinstance(arr, dict):
            # 针对dict转换pd.DataFrame，注意顺序不能在all(isinstance(arr_item, Iterable)下面
            arr = pd.DataFrame(arr)
        elif all(isinstance(arr_item, Iterable) for arr_item in arr):
            # 如果子序列的元素也都是可以迭代的，那么先转np.array，然后再DataFrame
            arr = pd.DataFrame(np.array(arr))
        else:
            # 否则序列对象转换为pd.Series
            arr = pd.Series(arr)
    return arr


def params_to_pandas(func):
    """
        函数装饰器：不定参数装饰器，定参数转换使用ABuScalerUtil中的装饰器arr_to_pandas(func)
        将被装饰函数中的参数中所有可以迭代的序列转换为pd.DataFrame或者pd.Series
    """

    @functools.wraps(func)
    def wrapper(*arg, **kwargs):
        # 把arg中的可迭代序列转换为pd.DataFrame或者pd.Series
        arg_list = [arr_to_pandas(param) for param in arg]
        # 把kwargs中的可迭代序列转换为pd.DataFrame或者pd.Series
        arg_dict = {param_key: arr_to_pandas(kwargs[param_key]) for param_key in kwargs}
        return func(*arg_list, **arg_dict)

    return wrapper


# TODO 放在这里不合适，还要和ABuScalerUtil中的装饰器arr_to_numpy重复代码进行重构
def arr_to_numpy(arr):
    """
        函数装饰器：将可以迭代的序列转换为np.array，支持pd.DataFrame或者pd.Series
        ，list，dict, list，set，嵌套可迭代序列, 混嵌套可迭代序列
    """
    # TODO Iterable和six.string_types的判断抽出来放在一个模块，做为Iterable的判断来使用
    if not isinstance(arr, Iterable) or isinstance(arr, six.string_types):
        return arr

    if not isinstance(arr, np.ndarray):
        if isinstance(arr, pd.DataFrame) or isinstance(arr, pd.Series):
            # 如果是pandas直接拿values
            arr = arr.values
        elif isinstance(arr, dict):
            # 针对dict转换np.array
            arr = np.array(list(arr.values())).T
        else:
            arr = np.array(arr)
    return arr


def params_to_numpy(func):
    """
        函数装饰器：不定参数装饰器，定参数转换使用ABuScalerUtil中的装饰器arr_to_numpy(func)
        将被装饰函数中的参数中所有可以迭代的序列转换为np.array
    """

    @functools.wraps(func)
    def wrapper(*arg, **kwargs):
        # 把arg中的可迭代序列转换为np.array
        arg_list = [arr_to_numpy(param) for param in arg]
        # 把kwargs中的可迭代序列转换为np.array
        arg_dict = {param_key: arr_to_numpy(kwargs[param_key]) for param_key in kwargs}
        return func(*arg_list, **arg_dict)

    return wrapper


def catch_error(return_val=None, log=True):
    """
    作用范围：函数装饰器 (模块函数或者类函数)
    功能：捕获被装饰的函数中所有异常，即忽略函数中所有的问题，用在函数的执行级别低，且不需要后续处理
    :param return_val: 异常后返回的值，
                eg:
                    class A:
                        @ABuDTUtil.catch_error(return_val=100)
                        def a_func(self):
                            raise ValueError('catch_error')
                            return 100
                    in: A().a_func()
                    out: 100
    :param log: 是否打印错误日志
    """

    def decorate(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                logging.exception(e) if log else logging.debug(e)
                return return_val

        return wrapper

    return decorate


def consume_time(func):
    """
    作用范围：函数装饰器 (模块函数或者类函数)
    功能：简单统计被装饰函数运行时间
    """

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        print('{} cost {}s'.format(func.__name__, round(end_time - start_time, 3)))
        return result

    return wrapper


def empty_wrapper(func):
    """
    作用范围：函数装饰器 (模块函数或者类函数)
    功能：空装饰器，为fix版本问题使用，或者分逻辑功能实现使用
    """

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        return func(*args, **kwargs)

    return wrapper


# noinspection PyUnusedLocal
def empty_wrapper_with_params(*p_args, **p_kwargs):
    """
    作用范围：函数装饰器 (模块函数或者类函数)
    功能：带参数空装饰器，为fix版本问题使用，或者分逻辑功能实现使用
    """

    def decorate(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            return func(*args, **kwargs)

        return wrapper

    return decorate


def except_debug(func):
    """
    作用范围：函数装饰器 (模块函数或者类函数)
    功能：debug，调试使用，装饰在有问题函数上，发生问题打出问题后，再运行一次函数，可以用s跟踪问题了
    """

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            pdb.set_trace()
            print(e)
            # 再来一遍用s跟踪进去
            return func(*args, **kwargs)

    return wrapper


@contextmanager
def plt_show():
    """
        在conda5.00封装的matplotlib中全局rc的figsize在使用notebook并且开启直接show的模式下
        代码中显示使用plt.show会将rc中的figsize重置，所以需要显示使用plt.show的地方，通过plt_show
        上下文管理器进行规范控制：
        1. 上文figsize设置ABuEnv中的全局g_plt_figsize
        2. 下文显示调用plt.show()
    """
    plt.figure(figsize=ABuEnv.g_plt_figsize)
    yield
    plt.show()
