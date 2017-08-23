# -*- encoding:utf-8 -*-
"""
    检查类，检查函数对象、函数参数、函数返回值
"""

from __future__ import print_function
from __future__ import absolute_import
from __future__ import division

from toolz import valmap
from functools import wraps

from ..CheckBu.ABuFuncUtil import *
from ..CheckBu.ABuChecks import *
from ..CheckBu.ABuProcessor import arg_process, return_process

__author__ = '夜猫'
__weixin__ = 'abu_quant'


class _NoInstances(six.with_metaclass(ABCMeta, type)):
    """阻止实例化"""

    def __call__(cls, *args, **kwargs):
        raise TypeError("Can't instantiate directly")


class FuncChecker(six.with_metaclass(ABCMeta, _NoInstances)):
    """函数相关的检查类"""

    @staticmethod
    def check_iscallable(func):
        """
        检查传入参数对象是否是函数；不是函数raise CheckError
        :param func: 传入参数对象
        """
        if not callable(func):
            raise CheckError('%s is not callable' % get_func_name(func))


class ArgChecker(six.with_metaclass(ABCMeta, _NoInstances)):
    """函数参数相关的检查类"""

    @staticmethod
    def check_type(*ty_args, **ty_kwargs):
        """
        【装饰器】
        检查输入参数类型；检查失败raise CheckError
        :param ty_args: 类型tuple
        :param ty_kwargs: 类型dict
        :return: 
        """
        # 检查是否有不合规的tuple参数
        for ty in ty_args:
            if not isinstance(ty, (type, tuple)):
                raise TypeError(
                    "check_type() expected a type or tuple of types"
                    ", but got {type_} instead.".format(
                        type_=ty,
                    )
                )
        # 检查是否有不合规的dict参数
        for name, ty in six.iteritems(ty_kwargs):
            if not isinstance(ty, (type, tuple)):
                raise TypeError(
                    "check_type() expected a type or tuple of types for "
                    "argument '{name}', but got {type_} instead.".format(
                        name=name, type_=ty,
                    )
                )
        # 将type_check作用在函数参数上
        return arg_process(*map(type_check, list(ty_args)), **valmap(type_check, ty_kwargs))

    @staticmethod
    def check_bound(*bd_args, **bd_kwargs):
        """
        【装饰器】
        检查输入参数是否在某一范围内；检查失败raise CheckError
        传入参数形式应为`` (min_value, max_value)``.
        ``None`` 可以作为 ``min_value`` 或 ``max_value``，相当于正负无穷
        :param bd_args: tuple范围参数
        :param bd_kwargs: dict范围参数
        :return: 
        """
        # 将bound_valid_and_check作用在函数参数上
        return arg_process(*map(bound_valid_and_check, list(bd_args)),
                           **valmap(bound_valid_and_check, bd_kwargs))

    @staticmethod
    def _check_default(check_no, *names, **_unused):
        """
        【装饰器】
        检查函数参数是否有或者没有默认值；检查失败raise CheckError
        :param staticmethod: 检查有 or 没有
        :param names: 待检查参数的名称
        :param _unused: 屏蔽dict参数
        :return: 
        """
        # 屏蔽dict参数
        if _unused:
            raise TypeError("_check_default() doesn't accept dict processors")
        check_err_msg = '' if check_no else 'no '

        def decorate(func):
            # 获取默认参数字典
            arg_defaults = get_arg_defaults(func)
            for name in names:
                if name not in arg_defaults:
                    # 传入了并不存在的参数名
                    raise TypeError(get_func_name(func) + ' has no argument named ' + name)
                if bool(check_no) != (isinstance(arg_defaults[name], ArgNoDefault)):
                    # 检查失败
                    raise CheckError(
                        'In ' + get_func_name(func) + ' argument {} has {}default'.format(name, check_err_msg))

            @wraps(func)
            def wrapper(*args, **kwargs):
                # 直接返回被装饰函数结果
                return func(*args, **kwargs)

            return wrapper

        return decorate

    @staticmethod
    def check_hasdefault(*names, **_unused):
        """
        【装饰器】
        检查函数参数是否有默认值；检查失败raise CheckError
        :param names: 待检查参数的名称
        :param _unused: 屏蔽dict参数
        :return: 
        """

        return ArgChecker._check_default(False, *names, **_unused)

    @staticmethod
    def check_nodefault(*names, **_unused):
        """
        【装饰器】
        检查函数参数是否没有默认值；检查失败raise CheckError
        :param names: 待检查参数的名称
        :param _unused: 屏蔽dict参数
        :return: 
        """

        return ArgChecker._check_default(True, *names, **_unused)

    @staticmethod
    def check_hasargs(func):
        """
        【装饰器】
        检查函数是否有*args参数；检查失败raise CheckError
        :param func: 传入函数对象
        :return: 
        """
        # 解包函数参数及默认值
        argspec = getargspec(func)
        spec_args = argspec.args if argspec.args else []
        defaults = argspec.defaults if argspec.defaults else ()
        if len(spec_args) - len(defaults) == 0:
            # 函数没有tuple参数
            raise CheckError(get_func_name(func) + ' has no args')

        @wraps(func)
        def wrapper(*args_inner, **kwargs):
            # 直接返回被装饰函数结果
            return func(*args_inner, **kwargs)

        return wrapper

    @staticmethod
    def check_haskwargs(func):
        """
        【装饰器
        检查函数是否有**kwargs参数；检查失败raise CheckError
        :param func: 传入函数对象
        :return: 
        """
        # 解包函数参数及默认值
        argspec = getargspec(func)
        defaults = argspec.defaults if argspec.defaults else ()
        if not defaults:
            # 函数没有dict参数
            raise CheckError(get_func_name(func) + ' has no kwargs')

        @wraps(func)
        def wrapper(*args, **kwargs):
            # 直接返回被装饰函数结果
            return func(*args, **kwargs)

        return wrapper

    @staticmethod
    def check_subset(*ss_args, **ss_kwargs):
        """
        【装饰器】
        检查输入参数是否是某一集合的子集；检查失败raise CheckError
        :param ss_args: 参数集合tuple
        :param ss_kwargs: 参数集合dict
        :return: 
        """
        # 检查是否有不合规的tuple参数
        for ss in ss_args:
            if not isinstance(ss, (list, set, type(None))):
                raise TypeError(
                    "check_subset() expected a list or set or None of values"
                    ", but got {subset_} or tuple instead.".format(
                        subset_=str(type(ss)),
                    )
                )
        # 检查是否有不合规的dict参数
        for name, ss in six.iteritems(ss_kwargs):
            if not isinstance(ss, (list, set, type(None))):
                raise TypeError(
                    "check_subset() expected a list or set of values for "
                    "argument '{name_}', but got {subset_} or tuple instead.".format(
                        name_=name, subset_=str(type(ss)),
                    )
                )
        # 将subset_check函数作用在函数参数上
        return arg_process(*map(subset_check, list(ss_args)), **valmap(subset_check, ss_kwargs))


class ReturnChecker(six.with_metaclass(ABCMeta, _NoInstances)):
    """函数返回值相关的检查类"""

    @staticmethod
    def check_type(*types, **_unused):
        """
        【装饰器】
        检查返回值类型；检查失败raise CheckError
        :param types: 类型tuple
        :param _unused: 屏蔽dict参数
        :return: 
        """
        # 屏蔽dict参数
        if _unused:
            raise TypeError("check_type() doesn't accept dict processors")

        # 检查是否有不合格的tuple参数
        for type_ in types:
            if not isinstance(type_, (type, tuple, type(None))):  # tuple or 内置类型
                raise TypeError(
                    "check_return_type() expected a type or tuple of types, but got {type_msg} instead.".format(
                        type_msg=type_,
                    )
                )
        # 将type_check函数作用在函数返回值上
        return return_process(*map(type_check, list(types)))

    @staticmethod
    def check_bound(*bounds, **_unused):
        """
        【装饰器】
        检查返回参数是否在某一范围内；检查失败raise CheckError
        传入参数形式应为`` (min_value, max_value)``.
        ``None`` 可以作为 ``min_value`` 或 ``max_value``，相当于正负无穷
        :param bounds: tuple范围参数
        :param _unused: 屏蔽dict参数
        :return: 
        """
        # 屏蔽dict参数
        if _unused:
            raise TypeError("check_bound() doesn't accept dict processors")
        # 将bound_valid_and_check函数作用在函数返回值上
        return return_process(*map(bound_valid_and_check, list(bounds)))

    @staticmethod
    def check_subset(*ss_args, **_unused):
        """
        【装饰器】
        检查输入参数是否是某一集合的子集；检查失败raise CheckError
        :param ss_args: 参数集合tuple
        :param _unused: 屏蔽dict参数
        :return: 
        """
        # 屏蔽dict参数
        if _unused:
            raise TypeError("check_subset() doesn't accept dict processors")
        # 检查传入的tuple参数
        for ss in ss_args:
            if not isinstance(ss, (list, set, type(None))):
                raise TypeError(
                    "check_subset() expected a list or set or None of values"
                    ", but got {subset_} or tuple instead.".format(
                        subset_=str(type(ss)),
                    )
                )
        # 将subset_check函数作用在函数返回值上
        return return_process(*map(subset_check, list(ss_args)))
