# -*- encoding:utf-8 -*-
"""
    函数对象的工具类
"""

from __future__ import print_function
from __future__ import absolute_import
from __future__ import division

from collections import OrderedDict
from abc import ABCMeta

from ..CoreBu.ABuFixes import six

try:
    from inspect import getfullargspec as getargspec
except ImportError:

    from inspect import getargspec

__author__ = '夜猫'
__weixin__ = 'abu_quant'


class ArgNoDefault(six.with_metaclass(ABCMeta, TypeError)):
    """没有默认值的参数对象"""
    pass


def get_func_name(func):
    """
    获取函数名称
    :param func: 传入函数
    :return: 
    """
    try:
        func_name = str(func.__name__) + '()'
    except AttributeError:
        func_name = str(func)
    return func_name


def get_arg_defaults(func):
    """
    获取函数默认值字典；没有默认值时对应NoDefaultArg对象
    :param func: 传入函数
    :return: 函数参数名：默认值
    """
    # 解包函数参数及默认值
    argspec = getargspec(func)
    spec_args = argspec.args if argspec.args else []
    defaults = argspec.defaults if argspec.defaults else ()
    # 拼装默认值dict
    no_defaults = (ArgNoDefault(),) * (len(spec_args) - len(defaults))
    args_defaults = dict(zip(spec_args, no_defaults + defaults))
    return args_defaults


def check_bind(func, *args, **kwargs):
    """
    检查要bind的对象和原函数func的参数是否对齐；对齐失败，raise TypeError
    :param func:  原函数
    :param args: 要bind的tuple对象
    :param kwargs: 要bind的dict对象
    :return: 
    """
    argspec = getargspec(func)
    spec_args = argspec.args if argspec.args else []
    # 检查 kwargs 中是否有不存在的参数名
    bad_names = set(kwargs.keys()) - set(spec_args)
    if bad_names:
        raise TypeError(
            "Got unknown arguments: {}".format(str(bad_names))
        )
    # 检查args的参数是否过长
    l_arg_len = len(args)
    if len(spec_args) < l_arg_len + len(kwargs):
        raise TypeError(
            "Function with {} arguments, but got {} arguments to bind".format(len(spec_args),
                                                                              l_arg_len + len(kwargs))
        )
    # 检查 kwargs 中是否和 args的参数冲突
    arg_inds = OrderedDict(zip(spec_args, range(len(spec_args))))
    for k, v in six.iteritems(kwargs):
        if l_arg_len > arg_inds[k]:
            raise TypeError(
                "dict argument crash on tuple argument:  {}".format(str(k))
            )


def bind_partial(func, *args, **kwargs):
    """
    绑定func的参数和对应的参数对象
    :param func: 原函数
    :param args: 要bind的tuple对象
    :param kwargs: 要bind的dict对象
    :return: 绑定后的字典
    """
    # 解包函数参数
    argspec = getargspec(func)
    spec_args = argspec.args if argspec.args else []

    # 拼装函数参数和对应的参数对象
    bind_dict = OrderedDict(zip(spec_args, [None] * len(spec_args)))
    for k, v in six.iteritems(kwargs):
        bind_dict[k] = v
    for k, v in zip(six.iterkeys(bind_dict), args):
        bind_dict[k] = v

    return bind_dict
