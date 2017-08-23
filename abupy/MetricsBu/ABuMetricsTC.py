# -*- encoding:utf-8 -*-
"""比特币度量模块"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


from ..MetricsBu.ABuMetricsFutures import AbuMetricsFutures

__author__ = '阿布'
__weixin__ = 'abu_quant'


class AbuMetricsTC(AbuMetricsFutures):
    """比特币，莱特币等币类型度量，自扩张使用，暂时继承AbuMetricsFutures，即不涉及benchmark，user可继承扩展需求"""
