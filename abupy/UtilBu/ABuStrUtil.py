# -*- encoding:utf-8 -*-
"""
    字符工具模块
"""

from __future__ import print_function
from __future__ import absolute_import
from __future__ import division

import random
import re

# noinspection PyUnresolvedReferences
from ..CoreBu.ABuFixes import xrange
from ..CoreBu.ABuFixes import six
from ..CoreBu import ABuEnv

K_CN_RE = re.compile(u'[\u4e00-\u9fa5]+')


def _create_random_tmp(salt_count, seed):
    """
    从seed种子字符池中随机抽取salt_count个字符，返回生成字符串,
    注意抽取属于有放回抽取方法
    :param salt_count: 生成的字符序列的长度
    :param seed: 字符串对象，做为生成序列的种子字符池
    :return: 返回生成字符串
    """
    # TODO random.choice有放回抽取方法, 添加参数支持无放回抽取模式
    sa = [random.choice(seed) for _ in xrange(salt_count)]
    salt = ''.join(sa)
    return salt


def create_random_with_num(salt_count):
    """
    种子字符池 = "0123456789", 从种子字符池中随机抽取salt_count个字符, 返回生成字符串,
    :param salt_count: 生成的字符序列的长度
    :return: 返回生成字符串
    """
    seed = "0123456789"
    return _create_random_tmp(salt_count, seed)


def create_random_with_alpha(salt_count):
    """
    种子字符池 = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ",
    从种子字符池中随机抽取salt_count个字符, 返回生成字符串,
    :param salt_count: 生成的字符序列的长度
    :return: 返回生成字符串
    """
    seed = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ"
    return _create_random_tmp(salt_count, seed)


def create_random_with_num_low(salt_count):
    """
    种子字符池 = "abcdefghijklmnopqrstuvwxyz0123456789",
    从种子字符池中随机抽取salt_count个字符, 返回生成字符串,
    :param salt_count: 生成的字符序列的长度
    :return: 返回生成字符串
    """
    seed = "abcdefghijklmnopqrstuvwxyz0123456789"
    return _create_random_tmp(salt_count, seed)


def to_unicode(text, encoding=None, errors='strict'):
    """
    to_native_str对py2生效，对six.text_type直接返回，其它的encode，默认utf-8
    """
    if isinstance(text, six.text_type):
        return text
    if not isinstance(text, (bytes, six.text_type)):
        raise TypeError('to_unicode must receive a bytes, str or unicode '
                        'object, got %s' % type(text).__name__)
    if encoding is None:
        encoding = 'utf-8'

    try:
        decode_text = text.decode(encoding, errors)
    except:
        # 切换试一下，不行就需要上层处理
        decode_text = text.decode('gbk' if encoding == 'utf-8' else 'utf-8', errors)
    return decode_text


def to_bytes(text, encoding=None, errors='strict'):
    """
    to_native_str对py3生效，对bytes直接返回，其它的encode，默认utf-8
    """
    if isinstance(text, bytes):
        return text
    if not isinstance(text, six.string_types):
        raise TypeError('to_bytes must receive a unicode, str or bytes '
                        'object, got %s' % type(text).__name__)
    if encoding is None:
        encoding = 'utf-8'
    try:
        encode_text = text.encode(encoding, errors)
    except:
        # 切换试一下，不行就需要上层处理
        encode_text = text.encode('gbk' if encoding == 'utf-8' else 'utf-8', errors)
    return encode_text


def to_native_str(text, encoding=None, errors='strict'):
    """
    套接to_unicode和to_bytes针对python版本不同处理

        python2 to_bytes
        python3 to_unicode
    """
    if not ABuEnv.g_is_py3:
        return to_bytes(text, encoding, errors)
    else:
        return to_unicode(text, encoding, errors)


def str_is_num10(a_str):
    """通过int(a_str, 10)测试字符串数字是否是10进制"""
    # noinspection PyBroadException
    try:
        int(a_str, 10)
        return True
    except:
        return False


def str_is_num16(a_str):
    """通过int(a_str, 16)测试字符串数字是否是16进制"""
    # noinspection PyBroadException
    try:
        int(a_str, 16)
        return True
    except:
        return False


def str_is_cn(a_str):
    """
        通过正则表达式判断字符串中是否含有中文
        返回结果只判断是否search结果为None, 不返回具体匹配结果
        eg:
            K_CN_RE.search(a_str)('abc') is None
            return False
            K_CN_RE.search(a_str)('abc哈哈') -> <_sre.SRE_Match object; span=(3, 5), match='哈哈'>
            return True
    """
    # a_str = to_unicode(a_str)
    # return any(u'\u4e00' <= c <= u'\u9fa5' for c in a_str)
    return K_CN_RE.search(to_unicode(a_str)) is not None


def digit_str(item):
    """
        从第一个字符开始删除，直到所有字符都是数字为止，或者item长度 < 2
        eg:
            input:  ABuStrUtil.digit_str('sh000001')
            output: 000001

            input:  ABuStrUtil.digit_str('shszsh000001')
            output: 000001
    :param item: 字符串对象
    :return: 过滤head字母的字符串对象
    """
    while True:
        if item.isdigit():
            break
        if len(item) < 2:
            break
        item = item[1:]
    return item


def var_name(var, glb):
    """
    eg：
        in:  a = 5
        in:  var_name(a, globals())
        out: 'a'

    :param var: 要查的变量对象
    :param glb: globals()
    :return: var对象对应的名称
    """
    for vn in glb:
        if glb[vn] is var:
            return vn
    return 'unkonw'
