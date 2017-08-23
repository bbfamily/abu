# -*- encoding:utf-8 -*-
"""直观可视化制作qcut的bins点"""

from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

import math
import logging

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import itertools

# noinspection PyUnresolvedReferences
from ..CoreBu.ABuFixes import filter

__all__ = ['show_orders_hist']


def show_orders_hist(df, feature_columns, show=True, only_hist=True, show_pie=False):
    """
    可视化统计feature_columns序列所指定的特征在df中的直方图或者饼状图，
    根据参数only_hist是否进行pd.qcut统计输出

    eg:
        from abupy import AbuML, ml
        ttn_raw = AbuML.load_ttn_raw_df()
        ml.show_orders_hist(ttn_raw, ['Age', 'Fare', 'Pclass'])

    :param df: pd.DataFrame对象
    :param feature_columns: 特征名称序列，eg：['Age', 'Fare', 'Pclass']
    :param show: 是否可视化直方图或者饼状图
    :param show_pie: 是否优先考虑绘制饼状图，默认false
    :param only_hist: 是否进行pd.qcut统计输出
    """
    if not isinstance(df, pd.DataFrame):
        logging.info('df must pd.DataFrame, not type {}'.format(type(df)))
        return

    # 第一步过滤不在在特征列中的feature_columns元素
    feature_columns = list(filter(lambda x: df.columns.tolist().count(x) > 0, feature_columns))
    # 第二步过滤feature_columns元素中类型不是int或者float的
    feature_columns = list(
        filter(
            lambda x: df[x].dtype == int or df[x].dtype == float or df[x].dtype == np.uint or df[x].dtype == np.uint8,
            feature_columns))
    # 第三步过滤feature_columns元素中所指特征列中unique==1的，eg：1列全是1，全是0，没办法做bin
    feature_columns = list(filter(lambda x: len(np.unique(df[x])) > 1, feature_columns))

    axs_list = None
    if len(feature_columns) == 0:
        # 晒没了的情况，直接返回
        logging.info('{}\n{}\nnot exist! or unique==1!, or dtype != int or float'.format(
            df.columns, df.dtypes))
        return

    if show:
        # 如果可视化直方图，先确定子画布列数，一行放两个，取math.ceil，eg：3 ／2 ＝ 2
        n_rows = int(math.ceil(len(feature_columns) / 2))
        # 行高取5，总高度：n_rows * 5
        fig_h = n_rows * 5
        # plt.subplots生成子画布
        _, axs = plt.subplots(nrows=n_rows, ncols=2, figsize=(14, fig_h))
        # 如果是多于1个的即展开字画本序列为1d序列
        axs_list = axs if n_rows == 1 else list(itertools.chain.from_iterable(axs))

    for ind, feature in enumerate(feature_columns):
        feature_unique = len(np.unique(df[feature]))
        ax = None
        if axs_list is not None:
            ax = axs_list[ind]
            ax.set_title(feature)
        if show_pie and feature_unique < 10:
            # 如果特征的值unique < 10个，通过value_counts直接画饼图
            df[feature].value_counts().plot(ax=ax, kind='pie')
        else:
            # 画直方图
            bins = int(feature_unique / 50) if feature_unique / 50 > 10 else 10
            df[feature].hist(ax=ax, bins=bins)

        if only_hist:
            # 只做可视化就continue
            continue

        try:
            # qcut切分10等份
            cats = pd.qcut(df[feature], 10)
        except Exception:
            # 某一个数据超出q的数量导致无法分
            import pandas.core.algorithms as algos
            bins = algos.quantile(np.unique(df[feature]), np.linspace(0, 1, 10 + 1))
            # noinspection PyProtectedMember,PyUnresolvedReferences
            cats = pd.tools.tile._bins_to_cuts(df[feature], bins, include_lowest=True)

        logging.info('{0} show hist and qcuts'.format(feature))
        """
            Age show hist and qcuts
            (31.8, 36]    91
            (14, 19]      87
            (41, 50]      78
            [0.42, 14]    77
            (22, 25]      70
            (19, 22]      67
            (28, 31.8]    66
            (50, 80]      64
            (25, 28]      61
            (36, 41]      53
            Name: Age, dtype: int64
        """
        logging.info(cats.value_counts())
