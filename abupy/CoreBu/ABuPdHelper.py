# -*- encoding:utf-8 -*-
"""
    封装pandas中版本兼容问题，保持接口规范情况下，避免警告
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import functools
from collections import Iterable

import pandas as pd
from ..CoreBu.ABuFixes import partial
from ..CoreBu.ABuFixes import six

__author__ = '阿布'
__weixin__ = 'abu_quant'

try:
    # noinspection PyUnresolvedReferences
    from pandas.tseries.resample import DatetimeIndexResampler
    g_pandas_has_resampler = True
except ImportError:
    try:
        # noinspection PyUnresolvedReferences
        from pandas.core.resample import DatetimeIndexResampler
        g_pandas_has_resampler = True
    except ImportError:
        g_pandas_has_resampler = False

try:
    # noinspection PyUnresolvedReferences
    from pandas.core.window import EWM
    g_pandas_has_ewm = True
except ImportError:
    g_pandas_has_ewm = False

try:
    # noinspection PyUnresolvedReferences
    from pandas.core.window import Rolling
    g_pandas_has_rolling = True
except ImportError:
    g_pandas_has_rolling = False

try:
    # noinspection PyUnresolvedReferences
    from pandas.core.window import Expanding
    g_pandas_has_expanding = True
except ImportError:
    g_pandas_has_expanding = False


def __pd_object_covert_start(iter_obj):
    """
    _pd_object_covert中进行参数检测及转换
    :param iter_obj: 将要进行操作的可迭代序列
    :return: 操作之后的返回值是否需要转换为np.array
    """
    if isinstance(iter_obj, (pd.Series, pd.DataFrame)):
        # 如果本身就是(pd.Series, pd.DataFrame)，返回对返回值不需要转换，即False
        return iter_obj, False
    # TODO Iterable和six.string_types的判断抽出来放在一个模块，做为Iterable的判断来使用
    if isinstance(iter_obj, Iterable) and not isinstance(iter_obj, six.string_types):
        # 可迭代对象使用pd.Series进行包装，且返回对返回值需要转换为np.array，即True
        return pd.Series(iter_obj), True
    raise TypeError('pd_object must support Iterable!!!')


def _pd_object_covert(func):
    """针对参数序列进行pandas处理的事前，事后装饰器"""

    @functools.wraps(func)
    def wrapper(pd_object, pd_object_cm, how, *args, **kwargs):
        """事前装饰工作__pd_object_covert_start，事后根据是否需要转换为np.array工作"""
        # 事前装饰工作__pd_object_covert_start
        pd_object, ret_covert = __pd_object_covert_start(pd_object)
        ret = func(pd_object, pd_object_cm, how, *args, **kwargs)
        # 事后根据是否需要转换为np.array工作
        if ret is not None and ret_covert:
            return ret.values
        return ret

    return wrapper


@_pd_object_covert
def _pd_rolling(pd_object, pd_object_cm, how, *args, **kwargs):
    """
    被_pd_object_covert装饰，对pandas中的rolling操作，根据pandas version版本自动选择调用方式
    :param pd_object: 可迭代的序列，pd.Series, pd.DataFrame或者只是Iterable
    :param pd_object_cm: 与pd_object相同，针对需要两个pandas对象或者序列执行的操作，如corr，cov等
    :param how: 代表方法操作名称，eg. mean, std, var
    :return:
    """
    if g_pandas_has_rolling:
        """pandas版本高，使用如pd_object.rolling直接调用"""
        rolling_obj = pd_object.rolling(*args, **kwargs)
        if hasattr(rolling_obj, how):
            if pd_object_cm is None:
                return getattr(rolling_obj, how)()
            # 需要两个pd_object进行的操作, getattr(rolling_obj, how)(pd_object_cm)
            return getattr(rolling_obj, how)(pd_object_cm)
    else:
        """pandas版本低，使用如pd.rolling_mean方法调用"""
        how_func = 'rolling_{}'.format(how)
        if hasattr(pd, how_func):
            if pd_object_cm is None:
                return getattr(pd, how_func)(pd_object, *args, **kwargs)
            # 需要两个pd_object进行的操作，getattr(pd, how_func)(pd_object, pd_object_cm, *args, **kwargs)
            return getattr(pd, how_func)(pd_object, pd_object_cm, *args, **kwargs)
    raise RuntimeError('_pd_rolling {} getattr error'.format(how))


"""没有全部导出，只导出常用的"""
pd_rolling_mean = partial(_pd_rolling, how='mean', pd_object_cm=None)
pd_rolling_median = partial(_pd_rolling, how='median', pd_object_cm=None)
pd_rolling_std = partial(_pd_rolling, how='std', pd_object_cm=None)
pd_rolling_var = partial(_pd_rolling, how='var', pd_object_cm=None)
pd_rolling_max = partial(_pd_rolling, how='max', pd_object_cm=None)
pd_rolling_min = partial(_pd_rolling, how='min', pd_object_cm=None)
pd_rolling_sum = partial(_pd_rolling, how='sum', pd_object_cm=None)
pd_rolling_kurt = partial(_pd_rolling, how='kurt', pd_object_cm=None)
pd_rolling_skew = partial(_pd_rolling, how='skew', pd_object_cm=None)
pd_rolling_corr = partial(_pd_rolling, how='corr')
pd_rolling_cov = partial(_pd_rolling, how='cov')


@_pd_object_covert
def _pd_ewm(pd_object, pd_object_cm, how, *args, **kwargs):
    """
    被_pd_object_covert装饰，对pandas中的ewm操作，根据pandas version版本自动选择调用方式
    :param pd_object: 可迭代的序列，pd.Series, pd.DataFrame或者只是Iterable
    :param pd_object_cm: 与pd_object相同，针对需要两个pandas对象或者序列执行的操作，如corr，cov等
    :param how: 代表方法操作名称，eg. mean, std, var
    :return:
    """
    if g_pandas_has_ewm:
        """pandas版本高，使用如pd_object.ewm直接调用"""
        ewm_obj = pd_object.ewm(*args, **kwargs)
        if hasattr(ewm_obj, how):
            if pd_object_cm is None:
                return getattr(ewm_obj, how)()
            # 需要两个pd_object进行的操作
            return getattr(ewm_obj, how)(pd_object_cm)
    else:
        """pandas版本低，使用如pd.ewmstd方法调用"""
        if how == 'mean':
            # pd.ewma特殊代表加权移动平均，所以使用a替换mean
            how = 'a'
        how_func = 'ewm{}'.format(how)
        if hasattr(pd, how_func):
            if pd_object_cm is None:
                return getattr(pd, how_func)(pd_object, *args, **kwargs)
            # 需要两个pd_object进行的操作
            return getattr(pd, how_func)(pd_object, pd_object_cm, *args, **kwargs)
    raise RuntimeError('_pd_ewm {} getattr error'.format(how))


"""没有全部导出，只导出常用的"""
pd_ewm_mean = partial(_pd_ewm, how='mean', pd_object_cm=None)
pd_ewm_std = partial(_pd_ewm, how='std', pd_object_cm=None)
pd_ewm_var = partial(_pd_ewm, how='var', pd_object_cm=None)
pd_ewm_corr = partial(_pd_ewm, how='corr')
pd_ewm_cov = partial(_pd_ewm, how='cov')


@_pd_object_covert
def _pd_expanding(pd_object, pd_object_cm, how, *args, **kwargs):
    """
    对pandas中的expanding操作，根据pandas version版本自动选择调用方式
    :param pd_object: 可迭代的序列，pd.Series, pd.DataFrame或者只是Iterable
    :param pd_object_cm: 与pd_object相同，针对需要两个pandas对象或者序列执行的操作，如corr，cov等
    :param how: 代表方法操作名称，eg. mean, std, var
    :return:
    """
    if g_pandas_has_expanding:
        """pandas版本高，使用如pd_object.expanding直接调用"""
        rolling_obj = pd_object.expanding(*args, **kwargs)
        if hasattr(rolling_obj, how):
            if pd_object_cm is None:
                return getattr(rolling_obj, how)()
            else:
                # 需要两个pd_object进行的操作
                return getattr(rolling_obj, how)(pd_object_cm)
    else:
        """pandas版本低，使用如pd.expanding_mean方法调用"""
        how_func = 'expanding_{}'.format(how)
        if hasattr(pd, how_func):
            if pd_object_cm is None:
                return getattr(pd, how_func)(pd_object, *args, **kwargs)
            else:
                # 需要两个pd_object进行的操作
                return getattr(pd, how_func)(pd_object, pd_object_cm, *args, **kwargs)
    raise RuntimeError('_pd_expanding {} getattr error'.format(how))


"""没有全部导出，只导出常用的"""
pd_expanding_mean = partial(_pd_expanding, how='mean', pd_object_cm=None)
pd_expanding_median = partial(_pd_expanding, how='median', pd_object_cm=None)
pd_expanding_std = partial(_pd_expanding, how='std', pd_object_cm=None)
pd_expanding_var = partial(_pd_expanding, how='var', pd_object_cm=None)
pd_expanding_max = partial(_pd_expanding, how='max', pd_object_cm=None)
pd_expanding_min = partial(_pd_expanding, how='min', pd_object_cm=None)
pd_expanding_sum = partial(_pd_expanding, how='sum', pd_object_cm=None)
pd_expanding_kurt = partial(_pd_expanding, how='kurt', pd_object_cm=None)
pd_expanding_skew = partial(_pd_expanding, how='skew', pd_object_cm=None)
pd_expanding_corr = partial(_pd_expanding, how='corr')
pd_expanding_cov = partial(_pd_expanding, how='cov')


def pd_resample(pd_object, rule, *args, **kwargs):
    """
    对pandas中的resample操作，根据pandas version版本自动选择调用方式
    :param pd_object: 可迭代的序列，pd.Series, pd.DataFrame或者只是Iterable
    :param rule: 具体的resample函数中需要的参数 eg. 21D, 即重采样周期值
    :return:
    """
    if g_pandas_has_resampler:
        """pandas版本高，使用如pd_object.resample('21D').mean()直接调用"""
        how = kwargs.pop('how', '')
        rep_obj = pd_object.resample(rule)
        if hasattr(rep_obj, how):
            return getattr(rep_obj, how)()
        print('rep_obj how is error set!!!')
        return rep_obj
    else:
        """pandas版本低，使用如pd_object.resample('21D').how方法调用, 也就不用kwargs.pop('how', '')了"""
        return pd_object.resample(rule, *args, **kwargs)
