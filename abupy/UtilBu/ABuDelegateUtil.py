# -*- encoding:utf-8 -*-
"""
    委托工具模块
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import functools
from functools import update_wrapper
from operator import attrgetter

from ..CoreBu.ABuFixes import signature

__author__ = '阿布'
__weixin__ = 'abu_quant'


def first_delegate_has_method(delegate, check_params=True):
    """
    装饰在类函数上，如果delegate有定义对应名称方法，优先使用delegate中的方法，否则使用被装饰的方法

        eg:

        class A:
        def a_func(self):
            print('a.a_func')

        class B:
            def __init__(self):
                self.a = A()

            @ABuDelegateUtil.first_delegate_has_method('a')
            def a_func(self):
                print('b.a_func')

        in: B().a_func()
        out: a.a_func

    :param delegate: str对象，被委托的类属性对象名称，从被装饰的方法的类成员变量中寻找对应名字的对象
    :param check_params: 是否检测方法签名是否相同，默认检测
    """

    def decorate(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            if len(args) > 0:
                wrap_self = args[0]
                # 首先从被装饰的类对象实例获取delegate
                delegate_obj = getattr(wrap_self, delegate, None)
                # 被装饰的类对象实例存在且存在和func.__name__一样的方法
                if delegate_obj is not None and hasattr(delegate_obj, func.__name__):
                    # 被委托的对象的函数的方法签名
                    delegate_params = list(signature(getattr(delegate_obj, func.__name__)).parameters.keys())
                    # 被装饰的函数的方法签名
                    func_params = list(signature(func).parameters.keys())[1:]
                    # print(func_params)
                    # print(delegate_params)
                    # TODO 增加检测规范，如只检测参数order及类型，不全匹配名字
                    if not check_params or delegate_params == func_params:
                        # 一致就优先使用被委托的对象的同名函数
                        return getattr(delegate_obj, func.__name__)(*args[1:], **kwargs)
            return func(*args, **kwargs)

        return wrapper

    return decorate


def replace_word_delegate_has_method(delegate, key_word, replace_word, check_params=True):
    """
    不在delegate中寻找完全一样的方法名字，在被装饰的方法名字中的key_word替换为replace_word后再在delegate中寻找，找到优先使用
    否则继续使用被装饰的方法
        eg:
            class A:
                def a_func(self):
                    print('a.a_func')

            class B:
                def __init__(self):
                    self.a = A()

                @ABuDelegateUtil.replace_word_delegate_has_method('a', key_word='b', replace_word='a')
                def b_func(self):
                    print('b.b_func')

            in: B().b_func()
            out: a.a_func

    :param delegate: str对象，被委托的类属性对象名称，从被装饰的方法的类成员变量中寻找对应名字的对象
    :param key_word: 被装饰的函数名称中将被replace_word替换的key_word，str对象
    :param replace_word: 替换key_word形成要寻找的在被委托函数中的名字，str对象
    :param check_params: 是否检测方法签名是否相同，默认检测
    """

    def decorate(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            if len(args) > 0:
                wrap_self = args[0]
                # 首先从被装饰的类对象实例获取delegate
                delegate_obj = getattr(wrap_self, delegate, None)
                if delegate_obj is not None:
                    #  被装饰的类对象实例存在
                    org_func_name = func.__name__
                    if len(replace_word) > 0 and key_word in org_func_name:
                        # 使用replace_word替换原始函数名称org_func_name中的key_word
                        delegate_func_name = org_func_name.replace(key_word, replace_word)
                    else:
                        delegate_func_name = org_func_name

                    if hasattr(delegate_obj, delegate_func_name):
                        # 被装饰的类对象中确实存在delegate_func_name
                        delegate_params = list(signature(getattr(delegate_obj, delegate_func_name)).parameters.keys())
                        func_params = list(signature(func).parameters.keys())[1:]
                        # TODO 增加检测规范，如只检测参数order及类型，不全匹配名字
                        if not check_params or delegate_params == func_params:
                            # 参数命名一致就优先使用被委托的对象的函数
                            return getattr(delegate_obj, delegate_func_name)(*args[1:], **kwargs)
            return func(*args, **kwargs)

        return wrapper

    return decorate


class _IffHasAttrDescriptor(object):
    """
        摘自sklearn中metaestimators.py _IffHasAttrDescriptor
    """

    def __init__(self, fn, delegate_names, attribute_name):
        self.fn = fn
        self.delegate_names = delegate_names
        self.attribute_name = attribute_name

        # update the docstring of the descriptor
        update_wrapper(self, fn)

    def __get__(self, obj, p_type=None):
        # raise an AttributeError if the attribute is not present on the object
        if obj is not None:
            # delegate only on instances, not the classes.
            # this is to allow access to the docstrings.
            for delegate_name in self.delegate_names:
                try:
                    delegate = attrgetter(delegate_name)(obj)
                except AttributeError:
                    continue
                else:
                    getattr(delegate, self.attribute_name)
                    break
            else:
                attrgetter(self.delegate_names[-1])(obj)

        # lambda, but not partial, allows help() to work with update_wrapper
        out = lambda *args, **kwargs: self.fn(obj, *args, **kwargs)
        # update the docstring of the returned function
        update_wrapper(out, self.fn)
        return out


def if_delegate_has_method(delegate):
    """
        摘自sklearn中metaestimators.py if_delegate_has_method
        如果delegate有定义对应方法，才实际定义方法，否则被装饰的方法撤销
    """
    if isinstance(delegate, list):
        delegate = tuple(delegate)
    if not isinstance(delegate, tuple):
        delegate = (delegate,)

    return lambda fn: _IffHasAttrDescriptor(fn, delegate,
                                            attribute_name=fn.__name__)
