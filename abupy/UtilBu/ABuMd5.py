# -*- encoding:utf-8 -*-
"""
    md5, crc32等加密，变换匹配模块
"""

from __future__ import print_function
from __future__ import absolute_import
from __future__ import division

import hashlib
import os
from binascii import crc32

# noinspection PyUnresolvedReferences
from ..CoreBu.ABuFixes import xrange
from ..CoreBu import ABuEnv

K_BYTE_SIZE = 1024


def _md5_obj():
    """根据python版本返回md5实例"""
    md5_obj = hashlib.md5() if ABuEnv.g_is_py3 else hashlib.new("md5")
    return md5_obj


def md5_from_binary(binary):
    """对字符串进行md5, 返回md5后32位字符串对象"""
    m = _md5_obj()
    m.update(binary.encode('utf-8'))
    return m.hexdigest()


def md5_from_file(fn, block_m=1):
    """
    对文件进行md5, 分块读取文件
    :param fn: 目标文件路径
    :param block_m: 分块读取大小，默认1mb
    :return: md5后32位字符串对象, md5失败等问题返回0
    """
    opened = False
    f_obj = None
    if hasattr(fn, "read"):
        f_obj = fn
    else:
        if os.path.exists(fn) and os.path.isfile(fn):
            f_obj = open(fn, "rb")
            opened = True
    if f_obj:
        block_b = block_m * K_BYTE_SIZE * K_BYTE_SIZE
        try:
            m = _md5_obj()
            while True:
                fb = f_obj.read(block_b)
                if not fb:
                    break
                m.update(fb)
        finally:
            if opened:
                f_obj.close()
        return m.hexdigest()
    else:
        return 0


def crc32_from_file(fn, block_m=1):
    """
    对文件进行crc32, 分块读取文件
    :param fn: 目标文件路径
    :param block_m: 分块读取大小，默认1mb
    :return: crc32后返回的16进制字符串 eg. '0x00000000'
    """
    if os.path.exists(fn) and os.path.isfile(fn):
        block_b = block_m * K_BYTE_SIZE * K_BYTE_SIZE
        crc = 0
        f = open(fn, "rb")
        while True:
            fb = f.read(block_b)
            if not fb:
                break
            crc = crc32(fb, crc)
        f.close()
        res = ''
        for _ in xrange(4):
            t = crc & 0xFF
            crc >>= 8
            res = '%02x%s' % (t, res)
        return "0x" + res
    else:
        return 0
