# -*- encoding:utf-8 -*-
"""

计算需要动态分配权重，规避分析

"""
import AtrIndicator

__author__ = 'BBFamily'


def calc_dynamic_stop_loss(kl_pd_pre):
    """
        SymbolPd 中周期计算动态stoploass,
        按照时间分配权重， 使用分位数的mean与
        最新close，计算f1, f2, f3, f4再计算
        weights
    """
    stop_loss_atr14mean, stop_loss_atr14mean2, stop_loss_atr14mean3, last14 = AtrIndicator.get_atr14_mean(
        kl_pd_pre['high'].values,
        kl_pd_pre['low'].values, kl_pd_pre['preClose'].values)
    stop_loss_atr21mean, stop_loss_atr21mean2, stop_loss_atr21mean3, last21 = AtrIndicator.get_atr21_mean(
        kl_pd_pre['high'].values,
        kl_pd_pre['low'].values, kl_pd_pre['preClose'].values)

    f1 = (stop_loss_atr14mean / kl_pd_pre.iloc[-1]['close'])
    f2 = (stop_loss_atr14mean3 / kl_pd_pre.iloc[-1]['close'])
    f3 = (stop_loss_atr14mean2 / kl_pd_pre.iloc[-1]['close'])
    f4 = last14 / kl_pd_pre.iloc[-1]['close']

    weight4 = 1 if f1 + f2 + f3 + f4 > 1 else f1 + f2 + f3 + f4
    weight3 = 1 - weight4 if f1 + f2 + f3 > 1 - weight4 else f1 + f2 + f3
    weight2 = 1 - weight4 - weight3 if f1 + f2 > 1 - weight4 - weight3 else f1 + f2
    weight1 = 1 - weight4 - weight3 - weight2

    stop_loss_atr14mean = stop_loss_atr14mean * weight1 + stop_loss_atr14mean2 * weight2 + \
                          stop_loss_atr14mean2 * weight3 + last14 * weight4

    f_ex1 = (stop_loss_atr21mean / kl_pd_pre.iloc[-1]['close'])
    f_ex2 = (stop_loss_atr21mean3 / kl_pd_pre.iloc[-1]['close'])
    f_ex3 = (stop_loss_atr21mean2 / kl_pd_pre.iloc[-1]['close'])
    f_ex4 = last21 / kl_pd_pre.iloc[-1]['close']

    weight_ex4 = 1 if f_ex1 + f_ex2 + f_ex3 + f_ex4 > 1 else f_ex1 + f_ex2 + f_ex3 + f_ex4
    weight_ex3 = 1 - weight_ex4 if f_ex1 + f_ex2 + f_ex3 > 1 - weight_ex4 else f_ex1 + f_ex2 + f_ex3
    weight_ex2 = 1 - weight_ex4 - weight_ex3 if f_ex1 + f_ex2 > 1 - weight_ex4 - weight_ex3 else f_ex1 + f_ex2
    weight_ex1 = 1 - weight_ex4 - weight_ex3 - weight_ex2

    stop_loss_atr21mean = stop_loss_atr21mean * weight_ex1 + \
                          stop_loss_atr21mean2 * weight_ex2 + \
                          stop_loss_atr21mean3 * weight_ex3 + last21 * weight_ex4

    # print(weight4, weight3, weight2, weight1)
    stop_loss = stop_loss_atr14mean + stop_loss_atr21mean
    return stop_loss
