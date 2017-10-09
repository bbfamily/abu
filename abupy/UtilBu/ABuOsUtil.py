# -*- encoding:utf-8 -*-
"""操作系统工具函数整合模块"""

from __future__ import print_function
from __future__ import absolute_import
from __future__ import division

import logging

from ..CoreBu import ABuEnv
from ..UtilBu.ABuDTUtil import catch_error

log_func = logging.info if ABuEnv.g_is_ipython else print


@catch_error()
def show_msg(title, msg, log=False):
    """
    统一平台弹窗信息提示，被catch_error装饰，即不应该被提示中断程序，
    特别长任务的情况下
    :param title: 弹窗信息标题
    :param msg: 弹窗信息内容
    :param log: 是否通过logging.info打印信息
    :return:
    """
    # 由于catch_error忽略错误，所有主要信息还是先打印
    if log:
        log_func(u'{}\n{}'.format(title, msg))
    if ABuEnv.g_is_mac_os:
        from ..UtilBu.ABuMacUtil import show_msg as do_show_msg
    else:
        from ..UtilBu.ABuWinUtil import show_msg as do_show_msg
    do_show_msg(title, msg)


def socket_bind_recv(socket_fn, cmd_handler):
    """
    进程间socket或者共享内存通信，接受消息，处理消息，外层应处理catch_error
    :param socket_fn: socket文件名称或者共享内存名称
    :param cmd_handler: cmd处理函数，callable类型
    """

    # TODO 使用ZeroMQ进行重新对接
    if ABuEnv.g_is_mac_os:
        from ..UtilBu.ABuMacUtil import socket_bind_recv as do_socket_bind_recv
    else:
        from ..UtilBu.ABuWinUtil import socket_bind_recv as do_socket_bind_recv
    do_socket_bind_recv(socket_fn, cmd_handler)


def socket_send_msg(socket_fn, msg):
    """
    进程间socket或全局共享内通信，发送消息，外层应处理catch_error
    :param socket_fn: socket文件名称或者共享内存名称
    :param msg: 字符串类型需要传递的数据，不需要encode，内部进行encode
    """
    # TODO 使用ZeroMQ进行重新对接
    if ABuEnv.g_is_mac_os:
        from ..UtilBu.ABuMacUtil import socket_send_msg as do_socket_send_msg
    else:
        from ..UtilBu.ABuWinUtil import socket_send_msg as do_socket_send_msg
    do_socket_send_msg(socket_fn, msg)


@catch_error(return_val=0)
def fold_free_size_mb(folder):
    """
    统一平路径可用空间获取，被catch_error装饰，即不应该被提示中断程序，return_val＝0，出错也返回0
    :param folder: 路径或盘符信息
    :return: folder下的可用空间大小
    """
    if ABuEnv.g_is_mac_os:
        from ..UtilBu.ABuMacUtil import fold_free_size_mb as do_fold_free_size_mb
    else:
        from ..UtilBu.ABuWinUtil import fold_free_size_mb as do_fold_free_size_mb
    do_fold_free_size_mb(folder)
