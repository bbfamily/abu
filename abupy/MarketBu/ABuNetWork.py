# -*- encoding:utf-8 -*-
"""
    网络统一接口模块
"""

from __future__ import print_function
from __future__ import absolute_import
from __future__ import division

import logging
import requests
import time
import ast

from requests.packages.urllib3.exceptions import ReadTimeoutError

"""g_enable_lru_cache针对lru_cache是否开启，考虑到目前爬取的数据fetch url都会有时间戳等可变字段，所以默认关闭"""
g_enable_lru_cache = False
g_lru_cache_max = 300
if g_enable_lru_cache:
    # 开启import lru_cache
    from ..CoreBu.ABuFixes import lru_cache
else:
    # 导入一个空的装饰器as lru_cache
    from ..UtilBu.ABuDTUtil import empty_wrapper_with_params as lru_cache
# 设置requests库的日志级别
logging.getLogger("requests").setLevel(logging.WARNING)


@lru_cache(maxsize=g_lru_cache_max)
def get(url, params=None, headers=None, retry=3, **kwargs):
    """
    :param url: 请求base url
    :param params: url params参数
    :param headers: http head头信息
    :param retry: 重试次数，默认retry=3
    :param kwargs: 透传给requests.get，可设置ua等，超时等参数
    """
    req_count = 0
    while req_count < retry:
        # 重试retry次
        try:
            resp = requests.get(url=url, params=params, headers=headers, **kwargs)
            if resp.status_code == 200 or resp.status_code == 206:
                # 如果200，206返回，否则继续走重试
                return resp
        except ReadTimeoutError:
            # 超时直接重试就行，不打日志
            pass
        except Exception as e:
            logging.exception(e)
        req_count += 1
        time.sleep(0.5)
        continue
    return None


def post(url, params=None, headers=None, retry=3, **kwargs):
    """
    :param url: 请求base url
    :param params: url params参数
    :param headers: http head头信息
    :param retry: 重试次数，默认retry=3
    :param kwargs: 透传给requests.get，可设置ua等，超时等参数
    """
    req_count = 0
    while req_count < retry:
        try:
            resp = requests.post(url=url, params=params, headers=headers, **kwargs)
            return resp
        except Exception as e:
            logging.exception(e)
            req_count += 1
            time.sleep(0.5)
            continue
    return None


def ast_parse_js(js_var):
    """
    通过ast模块解析Javascript字符串
    :param js_var: Javascript字符串
    :return: map, dict or value
    """
    js_mode = ast.parse(js_var)
    body_head = js_mode.body[0]

    def _parse(node):
        if isinstance(node, ast.Expr):
            return _parse(node.value)
        elif isinstance(node, ast.Num):
            return node.n
        elif isinstance(node, ast.Str):
            return node.s
        elif isinstance(node, ast.Name):
            return node.id
        elif isinstance(node, ast.Dict):
            return dict(zip(map(_parse, node.keys), map(_parse, node.values)))
        elif isinstance(node, ast.List):
            return map(_parse, node.elts)
        else:
            raise NotImplementedError(node.__class__)

    return _parse(body_head)


def parse_js(js_var):
    """
    通过eval解析Javascript字符串
    :param js_var: Javascript字符串
    :return: dict
    """
    obj = eval(js_var, type('Dummy', (dict,), dict(__getitem__=lambda s, n: n))())
    return obj
