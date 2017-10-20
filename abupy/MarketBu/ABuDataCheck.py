# -*- encoding:utf-8 -*-
"""本地缓存监测模块"""

from __future__ import print_function
from __future__ import absolute_import
from __future__ import division

import os
import logging
import math

from ..UtilBu import ABuFileUtil
from ..CoreBu import ABuEnv
from ..CoreBu.ABuEnv import EDataCacheType, EMarketTargetType, EMarketDataFetchMode
# noinspection PyUnresolvedReferences
from ..CoreBu.ABuFixes import filter, partial
from ..MarketBu.ABuMarket import is_in_sand_box
from ..UtilBu.ABuOsUtil import show_msg
from ..MarketBu.ABuSymbolPd import check_symbol_in_local_csv

__author__ = '阿布'
__weixin__ = 'abu_quant'

"""基于不同系统的提示框使用partial包装title以及显示log"""
show_msg_toast_func = partial(show_msg, u'提示', log=True)


def browser_down_csv_zip(open_browser=False):
    """浏览器打开教程使用的csv数据百度云地址"""
    try:
        if open_browser:
            import webbrowser
            webbrowser.open('https://pan.baidu.com/s/1geNZgqf', new=0, autoraise=True)
            show_msg_toast_func(u'提取密码: gvtr')
    except:
        pass
    finally:
        logging.info(u'建议直接从百度云下载教程中使用的csv格式美股，A股，港股，币类，期货6年日k数据: ')
        logging.info(u'下载地址: https://pan.baidu.com/s/1geNZgqf')
        logging.info(u'提取密码: gvtr')
        logging.info(u'下载完成后解压zip得到\'csv\'文件夹到\'{}\'目录下'.format(ABuEnv.g_project_data_dir))


# noinspection PyProtectedMember
def check_symbol_data_mode(choice_symbols):
    """在考虑choice_symbols为None, 可以全市场工作时，检测是否需要提示下载csv数据或者使用数据下载界面进行操作"""
    if ABuEnv._g_enable_example_env_ipython and choice_symbols is not None:
        # 沙盒模式下 and choice_symbols不是none
        not_in_sb_list = list(filter(lambda symbol: not is_in_sand_box(symbol), choice_symbols))
        if len(not_in_sb_list) > 0:
            logging.info(
                u'当前数据模式为\'沙盒模式\'无{}数据，'
                u'请在\'分析设置\'中切换数据模式并确认数据可获取！'
                u'非沙盒模式建议先用\'数据下载界面操作\'进行数据下载'
                u'之后设置数据模式为\'开放数据模式\'，联网模式使用\'本地数据模式\''.format(not_in_sb_list))
            browser_down_csv_zip()
            return False

    is_stock_market = \
        ABuEnv.g_market_target == EMarketTargetType.E_MARKET_TARGET_FUTURES_CN or \
        ABuEnv.g_market_target == EMarketTargetType.E_MARKET_TARGET_US or \
        ABuEnv.g_market_target == EMarketTargetType.E_MARKET_TARGET_HK

    if is_stock_market and not ABuEnv._g_enable_example_env_ipython and choice_symbols is None:
        # 非沙盒模式下要做全股票市场全市场回测
        if ABuEnv.g_data_fetch_mode != EMarketDataFetchMode.E_DATA_FETCH_FORCE_LOCAL:
            logging.info(
                u'未选择任何回测目标且在非沙盒数据模式下，判定为进行全市场回测'
                u'为了提高运行效率，请将联网模式修改为\'本地数据模式\'，如需要进行数据更新，'
                u'请先使用\'数据下载界面操作\'进行数据更新！')
            browser_down_csv_zip()
            return False
        else:
            if ABuEnv.g_data_cache_type == EDataCacheType.E_DATA_CACHE_CSV:
                # csv模式下
                if not ABuFileUtil.file_exist(ABuEnv.g_project_kl_df_data_csv):
                    # 股票类型全市场回测，但没有数据
                    logging.info(
                        u'未选择任何回测目标且在非沙盒数据模式下，判定为进行全市场回测'
                        u'为了提高运行效率, 只使用\'本地数据模式\'进行回测，但未发现本地缓存数据，'
                        u'如需要进行数据更新'
                        u'请先使用\'数据下载界面操作\'进行数据更新！')
                    browser_down_csv_zip()
                    return False
                elif len(os.listdir(ABuEnv.g_project_kl_df_data_csv)) < 100:
                    # 股票类型全市场回测，但数据不足
                    logging.info(
                        u'未选择任何回测目标且在非沙盒数据模式下，判定为进行全市场回测'
                        u'为了提高运行效率, 只使用\'本地数据模式\'进行回测，发现本地缓存数据不足，'
                        u'只有{}支股票历史数据信息'
                        u'如需要进行数据更新'
                        u'请先使用\'数据下载界面操作\'进行数据更新！'.format(
                            len(os.listdir(ABuEnv.g_project_kl_df_data_csv))))
                    browser_down_csv_zip()
                    return False
            elif ABuEnv.g_data_cache_type == EDataCacheType.E_DATA_CACHE_HDF5 \
                    and not ABuFileUtil.file_exist(ABuEnv.g_project_kl_df_data):
                # hdf5模式下文件不存在
                logging.info(
                    u'未选择任何回测目标且在非沙盒数据模式下，判定为进行全市场回测'
                    u'为了提高运行效率, 只使用\'本地数据模式\'进行回测'
                    u'hdf5模式下未发现hdf5本地缓存数据，'
                    u'如需要进行数据更新'
                    u'请先使用\'数据下载界面操作\'进行数据更新！')
                browser_down_csv_zip()
                return False
    return True


def check_symbol_data(choice_symbols):
    """在choice_symbols不可以为None, 不可以全市场工作时，检测是否需要提示下载csv数据或者使用数据下载界面进行操作"""

    # noinspection PyProtectedMember
    if ABuEnv._g_enable_example_env_ipython and choice_symbols is not None:
        # 沙盒模式下 and choice_symbols不是none
        not_in_sb_list = list(filter(lambda symbol: not is_in_sand_box(symbol), choice_symbols))
        if len(not_in_sb_list) > 0:
            logging.info(
                u'当前数据模式为\'沙盒模式\'无{}数据，'
                u'请在\'设置\'中切换数据模式并确认数据在本地存在！'
                u'最优参数grid search暂不支持实时网络数据模式！'
                u'所以非沙盒模式需要先用\'数据下载界面操作\'进行数据下载'.format(not_in_sb_list))
            browser_down_csv_zip()
            return False
    else:
        # 非沙盒数据模式下
        if ABuEnv.g_data_cache_type == EDataCacheType.E_DATA_CACHE_CSV:
            # csv模式下，一个csv数据都没有
            if not ABuFileUtil.file_exist(ABuEnv.g_project_kl_df_data_csv):
                # 股票类型全市场回测，但没有数据
                logging.info(
                    u'未发现本地缓存数据，最优参数grid search暂不支持实时网络数据模式！'
                    u'所以非沙盒模式需要先用\'数据下载界面操作\'进行数据下载')
                browser_down_csv_zip()
                return False
            elif len(os.listdir(ABuEnv.g_project_kl_df_data_csv)) < 100:
                # 未下载云盘上的csv为前提条件
                not_in_local_csv = list(filter(lambda symbol:
                                               not check_symbol_in_local_csv(symbol), choice_symbols))
                # 需要grid search的symbol中有30%以上不在本地缓存中提示下载数据
                if not_in_local_csv > math.ceil(len(choice_symbols) * 0.3):
                    logging.info(
                        u'{}未发现本地缓存数据，最优参数grid search暂不支持实时网络数据模式！'
                        u'需要先用\'数据下载界面操作\'进行数据下载'.format(not_in_local_csv))
                    browser_down_csv_zip()
                    return False

        elif ABuEnv.g_data_cache_type == EDataCacheType.E_DATA_CACHE_HDF5 \
                and not ABuFileUtil.file_exist(ABuEnv.g_project_kl_df_data):
            # hdf5模式下文件不存在
            logging.info(
                u'未发现本地缓存数据，最优参数grid search暂不支持实时网络数据模式！'
                u'所以非沙盒模式需要先用\'数据下载界面操作\'进行数据下载')
            browser_down_csv_zip()
            return False
    return True


def all_market_env_check():
    """确定要做全市场相关类型的操作，检测本地数据文件"""
    # noinspection PyProtectedMember
    if ABuEnv._g_enable_example_env_ipython:
        # 沙盒环境下不需要检测
        return True

    if ABuEnv.g_data_cache_type == EDataCacheType.E_DATA_CACHE_CSV:
        # csv模式下
        if not ABuFileUtil.file_exist(ABuEnv.g_project_kl_df_data_csv):
            # 全市场回测，但没有数据
            logging.info(
                u'全市场相关操作为了提高运行效率, 只使用\'本地数据模式\'进行回测，但未发现本地缓存数据，'
                u'如需要进行数据更新'
                u'请先使用\'数据下载界面操作\'进行数据更新！')
            browser_down_csv_zip()
            return False
        elif len(os.listdir(ABuEnv.g_project_kl_df_data_csv)) < 30:
            # 全市场回测，但数据不足, 这里取30
            logging.info(
                u'全市场相关操作为了提高运行效率, 只使用\'本地数据模式\'进行回测，发现本地缓存数据不足，'
                u'只有{}支股票历史数据信息'
                u'如需要进行数据更新'
                u'请先使用\'数据下载界面操作\'进行数据更新！'.format(
                    len(os.listdir(ABuEnv.g_project_kl_df_data_csv))))
            browser_down_csv_zip()
            return False
    elif ABuEnv.g_data_cache_type == EDataCacheType.E_DATA_CACHE_HDF5 \
            and not ABuFileUtil.file_exist(ABuEnv.g_project_kl_df_data):
        # hdf5模式下文件不存在
        logging.info(
            u'全市场相关为了提高运行效率, 只使用\'本地数据模式\'进行回测'
            u'hdf5模式下未发现hdf5本地缓存数据，'
            u'如需要进行数据更新'
            u'请先使用\'数据下载界面操作\'进行数据更新！')
        browser_down_csv_zip()
        return False
    return True
