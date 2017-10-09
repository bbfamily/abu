# -*- encoding:utf-8 -*-
"""
    组合参数辅助模块
"""

from __future__ import print_function
from __future__ import absolute_import
from __future__ import division
import copy
from itertools import product

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from .ABuGridSearch import ParameterGrid
from scipy import stats
# noinspection PyUnresolvedReferences
from ..CoreBu.ABuFixes import range, xrange

__author__ = '阿布'
__weixin__ = 'abu_quant'

# TODO 使用enum代替K常量
# 代表买因子参数组合
K_GEN_FACTOR_PARAMS_BUY = 0
# 代表卖因子参数组合
K_GEN_FACTOR_PARAMS_SELL = 1


def gen_factor_grid(type_param, factors, need_empty_sell=False):
    """
    :param type_param: grid目标，为K_GEN_FACTOR_PARAMS_BUY或K_GEN_FACTOR_PARAMS_SELL需要重构
    :param factors: 可迭代序列，元素为因子dict 如：
                    {'class': [AbuFactorBuyBreak], 'xd': [42]}, {'class': [AbuFactorBuyBreak],'xd': [60]}
    :param need_empty_sell: 只针对卖出因子组合添加一个完全不使用任何卖出因子的组合
    :return: 返回因子dict的组合参数序列
    """
    
    # 通过ParameterGrid将factor包装，即通过ParameterGrid将dict对象product(*values)，详阅读ParameterGrid
    grid_params = [ParameterGrid(factor) for factor in factors]
    # 进行product调用ParameterGrid__iter__进行product(*values)
    factor_params = product(*grid_params)
    factor_params = [list(pd_cls) for pd_cls in factor_params]

    if len(factors) > 1:
        # 把单独一个factor的加进去
        for grid_single in grid_params:
            for single in grid_single:
                factor_params.append([single])
    if need_empty_sell and type_param == K_GEN_FACTOR_PARAMS_SELL:
        # 只有sell的factor要加个空的，买的因子要是全空就没办法玩了
        factor_params.append([])  # 最后加一个完全不使用因子的

    return factor_params


def score_pd_plot(grid_score_pd, y_key, x_key=None):
    """对最优结果score可视化，暂时未迁移完整，需迁移其余最优模块后可用"""
    if x_key is not None:
        xt = pd.crosstab(grid_score_pd[x_key], grid_score_pd[y_key])
        xt_pct = xt.div(xt.sum(1).astype(float), axis=0)
        xt_pct.plot(kind='bar',
                    stacked=True,
                    title=str(x_key) + ' -> ' + str(y_key))
        plt.xlabel(str(x_key))
        plt.ylabel(str(y_key))
    else:
        for col in grid_score_pd.columns:
            if col.startswith('Y_'):
                continue
            xt = pd.crosstab(grid_score_pd[col], grid_score_pd[y_key])
            xt_pct = xt.div(xt.sum(1).astype(float), axis=0)
            xt_pct.plot(kind='bar',
                        stacked=True,
                        title=str(col) + ' -> ' + str(y_key))
            plt.xlabel(str(col))
            plt.ylabel(str(y_key))
            plt.show()


# noinspection PyTypeChecker
def make_grid_score_pd(grid_scores, score_index=0):
    """对最优结果score分析处理，暂时未迁移完整，需迁移其余最优模块后可用"""
    unique_sell_factor = {slFac['class'] for grid in grid_scores for slFac in grid[2]}
    unique_buy_factor = {byFac['class'] for grid in grid_scores for byFac in grid[3]}

    grid_pd = pd.DataFrame([alpha_params[1] for alpha_params in grid_scores])
    factor_pd = pd.DataFrame([factor_params[2] for factor_params in grid_scores])
    factor_buy_pd = pd.DataFrame([factor_params[3] for factor_params in grid_scores])

    org_columns = factor_pd.shape[1]
    for columns_ind in xrange(0, org_columns):
        factor_pd[str(columns_ind) + 'class'] = factor_pd[columns_ind].apply(lambda x:
                                                                             None if x is None else x['class'])

    org_buy_columns = factor_buy_pd.shape[1]
    for columns_ind in xrange(0, org_buy_columns):
        factor_buy_pd[str(columns_ind) + 'class'] = factor_buy_pd[columns_ind].apply(lambda x:
                                                                                     None if x is None else x['class'])

    def rm_noise_key(fun_dict):
        copy_dict = copy.deepcopy(fun_dict)
        if 'draw' in copy_dict:
            del copy_dict['draw']
        if 'show' in copy_dict:
            del copy_dict['show']
        if 'class' in copy_dict:
            del copy_dict['class']
        return copy_dict

    def make_factor_pd(x_pd, sel_fac):
        def make_factor_pd_inner(p_x_pd):
            if p_x_pd is None:
                return 0
            if p_x_pd == sel_fac:
                return 1
            return 0

        y_ret = x_pd.apply(make_factor_pd_inner)
        if np.count_nonzero(y_ret) > 0:
            return 1
        return 0

    def make_factor_param_pd(x_pd, unique_facts):
        def make_factor_param_pd_inner(p_x_pd):
            if p_x_pd is None:
                return
            class_key = p_x_pd['class'].__name__
            x_pd_copy = rm_noise_key(p_x_pd)
            for item in x_pd_copy.items():
                unique_item = class_key + ':' + str(item[0]) + ':' + str(item[1])
                unique_facts.append(unique_item)

        x_pd.apply(make_factor_param_pd_inner)

    for selFac in unique_sell_factor:
        grid_pd[selFac.__name__] = factor_pd.iloc[:, org_columns:].apply(
            make_factor_pd, args=(selFac,), axis=1)
    for buyFac in unique_buy_factor:
        grid_pd[buyFac.__name__] = factor_buy_pd.iloc[:, org_buy_columns:].apply(
            make_factor_pd, args=(buyFac,), axis=1)

    unique_factor_params = []
    unique_factor_buy_params = []
    factor_pd.iloc[:, 0:org_columns].apply(make_factor_param_pd, args=(unique_factor_params,), axis=1)
    factor_buy_pd.iloc[:, 0:org_buy_columns].apply(make_factor_param_pd, args=(unique_factor_buy_params,), axis=1)

    unique_factor_params = set(unique_factor_params)
    unique_factor_buy_params = set(unique_factor_buy_params)

    def dummies_params(x_pd, p_sel_fac_param):
        def dummies_params_inner(p_x_pd):
            if p_x_pd is None:
                return 0
            class_key = p_x_pd['class'].__name__
            x_pd_copy = rm_noise_key(p_x_pd)
            for item in x_pd_copy.items():
                unique_item = class_key + ':' + str(item[0]) + ':' + str(item[1])
                if p_sel_fac_param == unique_item:
                    return 1
            return 0

        y_ret = x_pd.apply(dummies_params_inner)
        if np.count_nonzero(y_ret) > 0:
            return 1
        return 0

    for sel_fac_param in unique_factor_params:
        grid_pd[sel_fac_param] = factor_pd.iloc[:, 0:org_columns].apply(
            dummies_params, args=(sel_fac_param,), axis=1)
    for buy_fac_param in unique_factor_buy_params:
        grid_pd[buy_fac_param] = factor_pd.iloc[:, 0:org_columns].apply(
            dummies_params, args=(buy_fac_param,), axis=1)

    grid_pd['Y_REG'] = [score[0][score_index] if isinstance(score[0], list) else score[0]
                        for score in grid_scores]

    grid_pd['Y_LOG_MEDIAN'] = np.where(
        grid_pd['Y_REG'] > grid_pd['Y_REG'].median(), 1, 0)
    grid_pd['Y_LOG_618'] = np.where(
        grid_pd['Y_REG'] > stats.scoreatpercentile(grid_pd['Y_REG'], 61.8), 1, 0)

    return grid_pd
