# -*- encoding:utf-8 -*-
"""
    预处理函数参数或返回值
"""

from __future__ import print_function
from __future__ import absolute_import
from __future__ import division

from functools import wraps

from ..CoreBu.ABuFixes import zip
from ..CheckBu.ABuFuncUtil import *

try:
    from ..ExtBu.six.moves import zip_longest
except ImportError:
    from six.moves import zip_longest

__author__ = '夜猫'
__weixin__ = 'abu_quant'


def arg_process(*arg_funcs, **kwarg_funcs):
    """
    【装饰器】
    将funcs函数作用在原函数的参数上；func函数只包括一个参数: return_val.
    :param arg_funcs: func函数tuple。
    :param kwarg_funcs: func函数dict
    :return: 
    """

    def _decorator(func):
        # 检查待bind参数
        check_bind(func, *arg_funcs, **kwarg_funcs)
        # 提取funcs字典
        funcs = bind_partial(func, *arg_funcs, **kwarg_funcs)

        @wraps(func)
        def wrapper(*args, **kwargs):
            # 拼装经过函数处理后的新参数
            args_new = [f(arg) if f else arg for arg, f in zip(args, six.itervalues(funcs))]
            kwargs_new = {k: funcs[k](arg) if k in funcs and funcs[k] else arg
                          for k, arg in six.iteritems(kwargs)}
            # 调用函数
            return func(*args_new, **kwargs_new)

        return wrapper

    return _decorator


def return_process(*funcs, **_unused):
    """
    【装饰器】
    将funcs函数作用在被装饰的函数的返回值上
    :param funcs: 一个或者多个处理对应参数的func函数，多个func函数时需要和返回值数量对齐；func函数只包括一个参数: return_val
    :param _unused: 用来屏蔽dict参数
    :return: 返回被funcs函数处理过的函数返回值
    """
    # 屏蔽dict参数
    if _unused:
        raise TypeError("return_process() doesn't accept dict processors")

    def _decorator(f):
        def wrapper(*args, **kw):
            # 拿到返回值
            return_vals = f(*args, **kw)
            # 用函数处理返回值
            return _apply_funcs(return_vals, funcs)

        return wrapper

    return _decorator


def _apply_funcs(return_vals, funcs):
    """
    将func函数作用在被装饰的函数的返回值上
    """
    # 检查iterable
    if not isinstance(return_vals, tuple):
        return_vals = (return_vals,)
    try:
        iter(funcs)
    except TypeError:
        funcs = (funcs,) if funcs else ()

    # 检查函数
    if not funcs:
        return return_vals

    # 函数和返回值不对齐
    if 1 < len(return_vals) < len(funcs):
        raise TypeError(
            "In _apply_funcs(), len(funcs) == {} more than len(return_vals) == {}".format(
                len(funcs), len(return_vals)
            )
        )
    # 函数多于返回值
    if 1 == len(return_vals) < len(funcs):
        raise TypeError(
            "In _apply_funcs(), only 1 return value with len(processors) == {}".format(len(funcs),
                                                                                       len(return_vals))
        )

    # 将函数作用在返回值上
    return tuple([f(v) if f else v for v, f in zip_longest(return_vals, funcs)])
