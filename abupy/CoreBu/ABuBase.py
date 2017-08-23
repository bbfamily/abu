# -*- encoding:utf-8 -*-
"""
    类基础通用模块
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys
import logging

import pandas as pd
# noinspection PyUnresolvedReferences
from ..CoreBu.ABuFixes import signature, Parameter, pickle
from ..CoreBu import ABuEnv

__author__ = '阿布'
__weixin__ = 'abu_quant'


class FreezeAttrMixin(object):
    """冻结对外设置属性混入类，设置抛异常"""

    def _freeze(self):
        """冻结属性设置接口"""
        object.__setattr__(self, "__frozen", True)

    def __setattr__(self, key, value):
        if getattr(self, "__frozen", False) and not (key in type(self).__dict__ or key == "_cache"):
            raise AttributeError("You cannot add any new attribute '{key}'".format(key=key))
        object.__setattr__(self, key, value)


class PickleStateMixin(object):
    """混入有本地序列化需求的类"""

    # pickle的最高支持版本
    _pickle_highest_protocol = pickle.HIGHEST_PROTOCOL
    # python版本，也可简单使用是否py3
    _python_version = str(sys.version_info)
    # windows or mac os
    _is_mac_os = ABuEnv.g_is_mac_os
    # 是否考虑本身的版本version, 默认不考虑，忽略abupy的版本号
    skip_abupy_version = True

    def __getstate__(self):
        from .. import __version__
        _abupy_version = __version__
        self.pick_extend_work()
        return dict(self.__dict__.items(), _abupy_version=_abupy_version,
                    _pickle_highest_protocol=self._pickle_highest_protocol,
                    _python_version=self._python_version,
                    _is_mac_os=self._is_mac_os)

    def __setstate__(self, state):
        """开始从本地序列化文件转换为python对象，即unpick"""

        # 从本地序列化文件中读取的pickle的最高支持版本, 默认0
        pickle_highest_protocol = state.pop("_pickle_highest_protocol", 0)
        # 从本地序列化文件中读取的abupy的版本号, 默认0.0.1
        old_abupy_version = state.pop("_abupy_version", '0.0.1')
        # 从本地序列化文件中读取的python版本号, 默认2.7.0
        python_version = state.pop("_python_version", '2.7.0')
        # 从本地序列化文件中读取的平台信息, 默认False，即windows
        platform_version = state.pop("_is_mac_os", False)

        if self.skip_abupy_version:
            # 忽略abupy的版本号
            _abupy_version = old_abupy_version
        else:
            from .. import __version__
            _abupy_version = __version__

        if self._pickle_highest_protocol != pickle_highest_protocol \
                or _abupy_version != old_abupy_version or self._python_version != python_version \
                or self._is_mac_os != platform_version:
            """只要有一个信息不一致，打印info，即有序列化读取失败的可能"""
            logging.info(
                "unpickle {} : "
                "old pickle_highest_protocol={},"
                "now pickle_highest_protocol={}, "
                "old abupy_version={}, "
                "now abupy_version={}, "
                "old python_version={}, "
                "now python_version={}, "
                "old platform_version={}, "
                "now platform_version={}, ".format(
                    self.__class__.__name__,
                    pickle_highest_protocol, self._pickle_highest_protocol,
                    old_abupy_version, _abupy_version,
                    python_version, self._python_version,
                    platform_version, self._is_mac_os))

        self.__dict__.update(state)
        # 混入对象可覆盖unpick_extend_work方法，完成对象特有的unpick工作
        self.unpick_extend_work(state)

    # noinspection PyMethodMayBeStatic
    def pick_extend_work(self):
        """混入对象可覆盖pick_extend_work方法，完成对象特有的__getstate__工作"""
        pass

    def unpick_extend_work(self, state):
        """混入对象可覆盖unpick_extend_work方法，完成对象特有的__setstate__工作"""
        pass


class AbuParamBase(object):
    """对象基础类，实现对象基本信息打印，调试查看接口"""

    @classmethod
    def get_params(cls):
        # init中特意找了被类装饰器替换了的deprecated_original方法，即原始init方法
        init = getattr(cls.__init__, 'deprecated_original', cls.__init__)
        if init is object.__init__:
            # 非自定义init返回空
            return list()
        # 获取init的参数签名
        init_signature = signature(init)
        # 过滤self和func(*args), 和func(**kwargs)
        parameters = [p for p in init_signature.parameters.values()
                      if p.name != 'self' and p.kind != Parameter.VAR_KEYWORD and p.kind != Parameter.VAR_POSITIONAL]
        return sorted([p.name for p in parameters])

    def _filter_attr(self, user):
        """根据user设置，返回所有类属性key或者用户定义类属性key"""
        if not user:
            return self.__dict__.keys()

        # 只筛选用户定义类属性key
        user_attr = list(filter(
            lambda attr: not attr.startswith('_'), self.__dict__.keys()))
        return user_attr

    def to_dict(self, user=True):
        """for debug show dict"""
        return {attr: self.__dict__[attr] for attr in self._filter_attr(user)}

    def to_series(self, user=True):
        """for notebook debug show series"""
        return pd.Series(self.to_dict(user))

    def __str__(self):
        """打印对象显示：class name, params"""
        class_name = self.__class__.__name__
        return '%s(%s)' % (class_name, self.get_params())

    __repr__ = __str__
