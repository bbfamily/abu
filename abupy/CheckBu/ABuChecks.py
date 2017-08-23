# -*- encoding:utf-8 -*-
"""
    检查范围的函数
"""

from __future__ import print_function
from __future__ import absolute_import
from __future__ import division

import numpy as np
from abc import ABCMeta

from ..CoreBu.ABuFixes import six

__author__ = '夜猫'
__weixin__ = 'abu_quant'


class CheckError(six.with_metaclass(ABCMeta, TypeError)):
    """Check失败的Error类型"""
    pass


def bound_check(bound):
    """
    制作检查数值范围的check_fail函数
    """
    (min_val, max_val) = bound
    # 转换None到inf
    min_val = -np.inf if min_val is None else min_val
    max_val = np.inf if max_val is None else max_val
    # 准备错误message
    error_msg = "function expected a return value inclusively between %s and %s" % (min_val, max_val)

    def _check(value):
        """范围检查逻辑"""
        return not (min_val <= value <= max_val)

    def _bound_check(val):
        """
        检查数值范围的函数；检查失败raise CheckError
        """
        if _check(val):
            raise CheckError(error_msg)
        else:
            return val

    return _bound_check


def bound_valid_and_check(bound):
    """
    检查bound的输入参数格式；失败raise TypeError
    传入参数形式应为`` (min_value, max_value)``.
    """

    def valid_bound(t):
        # 检查bound传入参数格式
        return (
            isinstance(t, tuple)
            and len(t) == 2
            and t != (None, None)
        )

    if not valid_bound(bound):
        # 参数格式错误
        raise TypeError(
            "function expected a tuple of bounds,"
            "but got {} instead.".format(bound)
        )
    return bound_check(bound)


def subset_check(subset):
    """
    制作检查是否子集的check函数
    """

    def _check(arg_val):
        """
        检查数值是否子集的check函数；检查失败raise CheckError
        """
        if subset is not None and arg_val not in set(subset):
            raise CheckError(
                'Value {} is not the subset of {}'.format(arg_val, str(subset))
            )
        return arg_val

    return _check


def type_check(arg_ty):
    """
    制作检查数值类型的check函数
    """

    def _check(arg_val):
        """
        检查数值类型的check函数；检查失败raise CheckError
        """
        if arg_ty and not isinstance(arg_val, arg_ty):
            raise CheckError(
                'Value {} is not {}'.format(arg_val, arg_ty)
            )
        return arg_val

    return _check
