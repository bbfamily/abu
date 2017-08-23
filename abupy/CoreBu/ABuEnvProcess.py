# -*- encoding:utf-8 -*-
"""
    多任务子进程拷贝跟随主进程设置模块
"""

from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import functools

# noinspection PyUnresolvedReferences
from ..CoreBu.ABuFixes import filter
from ..CoreBu.ABuFixes import signature, Parameter

__author__ = '阿布'
__weixin__ = 'abu_quant'


def add_process_env_sig(func):
    """
    初始化装饰器时给被装饰函数添加env关键字参数，在wrapper中将env对象进行子进程copy
    由于要改方法签名，多个装饰器的情况要放在最下面
    :param func:
    :return:
    """

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        # env = kwargs.pop('env', None)
        if 'env' in kwargs:
            """
                实际上linux, mac os上并不需要进行进程间模块内存拷贝，
                子进程fork后携带了父进程的内存信息，win上是需要的，
                暂时不做区分，都进行进程间的内存拷贝，如特别在乎效率的
                情况下基于linux系统，mac os可以不需要拷贝，如下：
                if kwargs['env'] is not None and not ABuEnv.g_is_mac_os:
                    # 只有windows进行内存设置拷贝
                    env.copy_process_env()
            """
            # if kwargs['env'] is not None and not ABuEnv.g_is_mac_os:
            env = kwargs.pop('env', None)
            if env is not None:
                # 将主进程中的env拷贝到子进程中
                env.copy_process_env()
        return func(*args, **kwargs)

    # 获取原始函数参数签名，给并行方法添加env参数
    sig = signature(func)

    if 'env' not in list(sig.parameters.keys()):
        parameters = list(sig.parameters.values())
        # 通过强制关键字参数，给方法加上env
        parameters.append(Parameter('env', Parameter.KEYWORD_ONLY, default=None))
        # wrapper的__signature__进行替换
        wrapper.__signature__ = sig.replace(parameters=parameters)

    return wrapper


class AbuEnvProcess(object):
    """多任务主进程内存设置拷贝执行者类"""

    def __init__(self):
        """迭代注册了的需要拷贝内存设置的模块，通过筛选模块中以g_或者_g_开头的属性，将这些属性拷贝为类属性变量"""
        for module in self.register_module():
            # 迭代注册了的需要拷贝内存设置的模块, 筛选模块中以g_或者_g_开头的, 且不能callable，即不是方法
            sig_env = list(filter(
                lambda _sig: not callable(_sig) and (_sig.startswith('g_') or _sig.startswith('_g_')), dir(module)))

            module_name = module.__name__
            # map(lambda sig: setattr(self, '{}_{}'.format(module_name, sig), module.__dict__[sig]), sig_env)
            for sig in sig_env:
                # 模块中的属性拷贝为类属性变量，key＝module_name_sig
                setattr(self, '{}_{}'.format(module_name, sig), module.__dict__[sig])

    # noinspection PyMethodMayBeStatic
    def register_module(self):
        """
        注册需要拷贝内存的模块，不要全局模块注册，否则很多交叉引用，也不要做为类变量存储否则多进程传递pickle时会出错
        :return:
        """
        from ..BetaBu import ABuAtrPosition, ABuPositionBase
        from ..CoreBu import ABuEnv
        from ..SimilarBu import ABuCorrcoef
        from ..SlippageBu import ABuSlippageBuyMean, ABuSlippageBuyBase, ABuSlippageSellBase
        from ..TradeBu import ABuMLFeature
        from ..UmpBu import ABuUmpManager, ABuUmpMainBase, ABuUmpEdgeBase
        from ..TLineBu import ABuTLSimilar
        from ..MarketBu import ABuMarket
        from ..AlphaBu import ABuPickTimeWorker
        from ..FactorSellBu import ABuFactorCloseAtrNStop, ABuFactorPreAtrNStop
        from ..PickStockBu import ABuPickSimilarNTop
        from ..UtilBu import ABuProgress

        # TODO 将每个模块中全局设置放在一个模块配置代码文件中，这里只将所有模块配置代码文件加载
        return [ABuAtrPosition, ABuPositionBase, ABuEnv, ABuCorrcoef, ABuProgress,
                ABuSlippageBuyMean, ABuSlippageSellBase, ABuSlippageBuyBase, ABuUmpMainBase, ABuUmpEdgeBase,
                ABuMLFeature, ABuUmpManager, ABuTLSimilar, ABuPickTimeWorker,
                ABuFactorCloseAtrNStop, ABuMarket, ABuFactorPreAtrNStop, ABuPickSimilarNTop]

    def copy_process_env(self):
        """为子进程拷贝主进程中的设置执行，在add_process_env_sig装饰器中调用，外部不应主动使用"""
        for module in self.register_module():
            # 迭代注册了的需要拷贝内存设置的模块, 筛选模块中以g_或者_g_开头的, 且不能callable，即不是方法
            sig_env = list(filter(
                lambda sig: not callable(sig) and (sig.startswith('g_') or sig.startswith('_g_')), dir(module)))
            module_name = module.__name__
            for _sig in sig_env:
                # 格式化类变量中对应模块属性的key
                name = '{}_{}'.format(module_name, _sig)
                # 根据应模块属性的key（name）getattr获取属性值
                val = getattr(self, name)
                # 为子模块内存变量进行值拷贝
                module.__dict__[_sig] = val
                # print(name, val)

    def __str__(self):
        """打印对象显示：注册需要拷贝内存的模块中在AbuEnvProcess对象属性的映射key值，以及value值"""

        str_dict = dict()
        for module in self.register_module():
            sig_env = list(filter(
                lambda sig: not callable(sig) and (sig.startswith('g_') or sig.startswith('_g_')), dir(module)))
            module_name = module.__name__
            for _sig in sig_env:
                # format对象属性的映射key值
                name = '{}_{}'.format(module_name, _sig)
                # 根据映射key值getattr出value值
                attr_str = getattr(self, name)
                str_dict[name] = attr_str
        return str(str_dict)

    __repr__ = __str__
