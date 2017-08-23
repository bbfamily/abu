# -*- encoding:utf-8 -*-
"""
    Deprecated警告模块
"""

import warnings

from ..CoreBu.ABuFixes import six


class AbuDeprecated(object):
    """支持装饰类或者方法，在使用类或者方法时警告Deprecated信息"""

    def __init__(self, tip_info=''):
        # 用户自定义警告信息tip_info
        self.tip_info = tip_info

    def __call__(self, obj):
        if isinstance(obj, six.class_types):
            # 针对类装饰
            return self._decorate_class(obj)
        else:
            # 针对方法装饰
            return self._decorate_fun(obj)

    def _decorate_class(self, cls):
        """实现类装饰警告Deprecated信息"""

        msg = "class {} is deprecated".format(cls.__name__)
        if self.tip_info:
            msg += "; {}".format(self.tip_info)
        # 取出原始init
        init = cls.__init__

        def wrapped(*args, **kwargs):
            warnings.warn(msg, category=DeprecationWarning)
            return init(*args, **kwargs)

        cls.__init__ = wrapped

        wrapped.__name__ = '__init__'
        wrapped.__doc__ = self._update_doc(init.__doc__)
        # init成为deprecated_original，必须要使用这个属性名字，在其它地方，如AbuParamBase会寻找原始方法找它
        wrapped.deprecated_original = init

        return cls

    def _decorate_fun(self, fun):
        """实现方法装饰警告Deprecated信息"""

        msg = "func {} is deprecated".format(fun.__name__)
        if self.tip_info:
            msg += "; {}".format(self.tip_info)

        def wrapped(*args, **kwargs):
            warnings.warn(msg, category=DeprecationWarning)
            return fun(*args, **kwargs)

        # 更新func及文档信息
        wrapped.__name__ = fun.__name__
        wrapped.__dict__ = fun.__dict__
        wrapped.__doc__ = self._update_doc(fun.__doc__)

        return wrapped

    def _update_doc(self, func_doc):
        """更新文档信息，把原来的文档信息进行合并格式化, 即第一行为deprecated_doc(Deprecated: tip_info)，下一行为原始func_doc"""
        deprecated_doc = "Deprecated"
        if self.tip_info:
            """如果有tip format tip"""
            deprecated_doc = "{}: {}".format(deprecated_doc, self.tip_info)
        if func_doc:
            # 把原来的文档信息进行合并格式化, 即第一行为deprecated_doc，下一行为原始func_doc
            func_doc = "{}\n{}".format(deprecated_doc, func_doc)
        return func_doc
