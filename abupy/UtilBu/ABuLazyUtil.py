# -*- encoding:utf-8 -*-
"""
    lazy工具模块
"""

from __future__ import print_function
from __future__ import absolute_import
from __future__ import division

import sys
import weakref


class LazyFunc(object):
    """描述器类：作用在类中需要lazy的对象方法上"""

    def __init__(self, func):
        """
        外部使用eg：
            class BuyCallMixin(object):
                @LazyFunc
                def buy_type_str(self):
                    return "call"

                @LazyFunc
                def expect_direction(self):
                    return 1.0
        """
        self.func = func
        self.cache = weakref.WeakKeyDictionary()

    def __get__(self, instance, owner):
        """描述器__get__，使用weakref.WeakKeyDictionary将以实例化的instance加入缓存"""
        if instance is None:
            return self
        try:
            return self.cache[instance]
        except KeyError:
            ret = self.func(instance)
            self.cache[instance] = ret
            return ret

    def __set__(self, instance, value):
        """描述器__set__，raise AttributeError，即禁止外部set值"""
        raise AttributeError("LazyFunc set value!!!")

    def __delete__(self, instance):
        """描述器___delete__从weakref.WeakKeyDictionary cache中删除instance"""
        del self.cache[instance]


class LazyClsFunc(LazyFunc):
    """
        描述器类：
        作用在类中需要lazy的类方法上，实际上只是使用__get__(owner, owner)
        替换原始__get__(self, instance, owner)
    """

    def __get__(self, instance, owner):
        """描述器__get__，使用__get__(owner, owner)替换原始__get__(self, instance, owner)"""
        return super(LazyClsFunc, self).__get__(owner, owner)


def add_doc(func, doc):
    """Lazy add doc"""
    func.__doc__ = doc


def import_module(name):
    """Lazy impor _module"""
    __import__(name)
    return sys.modules[name]
