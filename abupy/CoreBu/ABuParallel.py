# -*- encoding:utf-8 -*-
"""
    并行封装模块，主要针对不同平台统一接口规范：

    windows 上使用joblib进行长时间的多任务，如超过10小时以上时，在任何最后有系统pop任务
    的错误，所以windows上使用ProcessPoolExecutor进行多任务，套上Parallel和delayed保持接口的
    通用性及规范统一
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import functools

from ..CoreBu import ABuEnv

__author__ = '阿布'
__weixin__ = 'abu_quant'

# if ABuEnv.g_is_mac_os:
if False:
    """
        对外版本不再使用joblib避免python2，python3，mac, windows等joblib最后的pop一直出不来的bug，
        有能力可自行打开，在mac下python3环境且cpu足够快建议打开
    """
    # MAC 直接import Parallel, delayed
    # noinspection PyUnresolvedReferences
    from ..ExtBu.joblib import Parallel, delayed
else:
    # windows需要使用ProcessPoolExecutor
    try:
        # noinspection PyCompatibility
        from concurrent.futures import ProcessPoolExecutor
    except ImportError:
        from ..ExtBu.futures import ProcessPoolExecutor


    def delayed(function):
        """
        将function通过functools.wraps及delayed_function进行保留，但不执行
        :param function:
        :return:
        """
        def delayed_function(*args, **kwargs):
            """将function以及参数返回为tuple，tuple[0]为原始function"""
            return function, args, kwargs

        try:
            delayed_function = functools.wraps(function)(delayed_function)
        except AttributeError:
            raise TypeError('wraps fails on some callable objects')
        return delayed_function


    # noinspection PyUnusedLocal
    class Parallel(object):
        """封装ProcessPoolExecutor进行并行任务执行操作"""

        def __init__(self, n_jobs=1, backend='multiprocessing', verbose=0,
                     pre_dispatch='2 * n_jobs', batch_size='auto',
                     temp_folder=None, max_nbytes='1M', mmap_mode='r'):
            """
            :param n_jobs: 并行启动的进程数，任务数量
            :param backend: 无意义，只是为了统一接口规范，与joblib.Parallel保持一样的参数
            :param verbose: 无意义，只是为了统一接口规范，与joblib.Parallel保持一样的参数
            :param pre_dispatch: 无意义，只是为了统一接口规范，与joblib.Parallel保持一样的参数
            :param batch_size: 无意义，只是为了统一接口规范，与joblib.Parallel保持一样的参数
            :param temp_folder: 无意义，只是为了统一接口规范，与joblib.Parallel保持一样的参数
            :param max_nbytes: 无意义，只是为了统一接口规范，与joblib.Parallel保持一样的参数
            :param mmap_mode: 无意义，只是为了统一接口规范，与joblib.Parallel保持一样的参数
            """
            self.n_jobs = n_jobs

        def __call__(self, iterable):
            """为与joblib并行保持一致，内部使用ProcessPoolExecutor开始工作"""

            result = []

            def when_done(r):
                """ProcessPoolExecutor每一个进程结束后结果append到result中"""
                result.append(r.result())

            if self.n_jobs <= 0:
                # 主要为了适配 n_jobs = -1，joblib中启动cpu个数个进程并行执行
                self.n_jobs = ABuEnv.g_cpu_cnt

            if self.n_jobs == 1:
                # 如果只开一个进程，那么只在主进程(或当前运行的子进程)里运行，方便pdb debug且与joblib运行方式保持一致
                for jb in iterable:
                    result.append(jb[0](*jb[1], **jb[2]))
            else:
                with ProcessPoolExecutor(max_workers=self.n_jobs) as pool:
                    for jb in iterable:
                        # 这里iterable里每一个元素是delayed.delayed_function保留的tuple
                        future_result = pool.submit(jb[0], *jb[1], **jb[2])
                        future_result.add_done_callback(when_done)
            return result


def run_in_thread(func, *args, **kwargs):
    """
    多线程工具函数，不涉及返回值等细节处理时使用
    :param func: 被线程委托的函数
    :return: 返回Thread线程对象
    """
    from threading import Thread
    thread = Thread(target=func, args=args, kwargs=kwargs)
    thread.daemon = True
    thread.start()
    return thread


def run_in_subprocess(func, *args, **kwargs):
    """
    多进程工具函数，不涉及返回值等细节处理时使用
    :param func: 被进程委托的函数
    :return: 返回multiprocessing进程对象
    """
    from multiprocessing import Process
    process = Process(target=func, args=args, kwargs=kwargs)
    process.daemon = True
    process.start()
    return process
