# -*- encoding:utf-8 -*-
"""量化技术分析工具图形可视化基础模块"""

from __future__ import print_function
from __future__ import absolute_import
from __future__ import division

import logging
import functools
from contextlib import contextmanager

import ipywidgets as widgets
import pandas as pd

from ..CoreBu import ABuEnv
from ..WidgetBu.ABuWGBase import WidgetBase, show_msg_func
from ..WidgetBu.ABuWGBSymbol import WidgetSymbolChoice
from ..WidgetBu.ABuWGBRunBase import WidgetEnvSetMixin, WidgetTimeModeMixin
from ..MarketBu import ABuSymbolPd
from ..MarketBu.ABuDataCheck import browser_down_csv_zip
from ..UtilBu import ABuProgress
from ..TLineBu.ABuTL import AbuTLine

__author__ = '阿布'
__weixin__ = 'abu_quant'


class WidgetToolSet(WidgetBase, WidgetEnvSetMixin, WidgetTimeModeMixin):
    """基础设置界面"""

    def __init__(self):
        """初始化基础设置界面"""

        mdm_box = self.init_env_set_ui()
        tm_box = self.init_time_mode_ui()
        self.sc = WidgetSymbolChoice()
        self.sc.cs_tip.value = u'如果股池为空，将使用示例的symbol进行分析'
        # 默认1年的数据分析
        self.run_years.value = 1
        # self.widget = widgets.Box([self.sc.widget, tm_box, mdm_box], layout=self.widget_layout)
        self.widget = widgets.VBox([self.sc.widget, tm_box, mdm_box])

    def time_mode_str(self):
        """实现混入WidgetTimeModeMixin，声明时间模块代表分析"""
        return u'分析'


def single_fetch_symbol_analyse(func):
    """定参数装饰器函数：获取设置中的1个symbol进行数据获取后调用被装饰函数"""

    @functools.wraps(func)
    def wrapper(self, bt):
        symbol = self._choice_symbol_single()
        kl = self._fetch_single_kl(symbol)
        ABuProgress.clear_output()
        # ABuProgress.clear_std_output()
        if kl is not None:
            # 清理之前的输出结果
            kl_tl = AbuTLine(kl.close, 'kl')
            return func(self, kl, kl_tl, bt)
        else:
            self.info_change_set_mode(symbol)

    return wrapper


def multi_fetch_symbol_analyse(func):
    """定参数装饰器函数：获取设置中的多个symbol数据获取组成字典后调用被装饰函数"""

    @functools.wraps(func)
    def wrapper(self, bt):
        choice_symbol = self._choice_symbol_multi()
        kl_dict = self._fetch_multi_kl(choice_symbol)
        ABuProgress.clear_output()
        # ABuProgress.clear_std_output()
        if kl_dict is not None and len(kl_dict) > 0:
            return func(self, kl_dict, bt)
        else:
            self.info_change_set_mode(choice_symbol)

    return wrapper


def multi_fetch_symbol_df_analyse(col_key):
    """
    定参数装饰器函数：获取设置中的多个symbol数据获取组成字典后
    通过参数col_key获取所有金融序列中的某一列形成新的dataframe，
    将dataframe传递给被装饰函数
    :param col_key: 金融序列中的某一列名称，eg：p_change, close
    """

    def decorate(func):
        @functools.wraps(func)
        def wrapper(self, bt):
            choice_symbol = self._choice_symbol_multi()
            cg_df = self._fetch_multi_kl_col(choice_symbol, col_key=col_key)
            ABuProgress.clear_output()
            # ABuProgress.clear_std_output()
            if cg_df is not None and len(cg_df) > 0:
                return func(self, cg_df, bt)
            else:
                self.info_change_set_mode(choice_symbol)

        return wrapper

    return decorate


# noinspection PyUnusedLocal,PyProtectedMember
class WidgetToolBase(WidgetBase):
    """技术分析基类"""

    def __init__(self, tool_set):
        """
            构造通用的layout css:
            label_layout, tool_layout, scroll_widget_layout,
            description_layout
        """
        if not isinstance(tool_set, WidgetToolSet):
            raise TypeError('tool_set must isinstance WidgetToolSet, not {}!'.format(type(tool_set)))
        self.tool_set = tool_set
        self.label_layout = widgets.Layout(width='300px', align_items='stretch')
        self.tool_layout = widgets.Layout(align_items='stretch', justify_content='space-between')
        self.scroll_widget_layout = widgets.Layout(overflow_x='scroll',
                                                   # flex_direction='row',
                                                   display='flex')
        self.description_layout = widgets.Layout(height='150px')
        # 默认不启动可滚动因子界面，因为对外的widget版本以及os操作系统不统一
        self.scroll_factor_box = False
        self._sub_children_group_cnt = 3

    def _sub_children(self, children, n_split):
        """将children每n_split个为一组，组装子children_group序列"""
        sub_children_cnt = int(len(children) / n_split)
        if sub_children_cnt == 0:
            sub_children_cnt = 1
        group_adjacent = lambda a, k: zip(*([iter(a)] * k))
        children_group = list(group_adjacent(children, sub_children_cnt))
        residue_ind = -(len(children) % sub_children_cnt) if sub_children_cnt > 0 else 0
        if residue_ind < 0:
            children_group.append(children[residue_ind:])
        return children_group

    @contextmanager
    def data_mode_recover(self, is_example_mode):
        """
        上下文管理器函数：
        1. 上文通过参数is_example_mode与全局的沙盒模式是否相同改变设置
        2. 下文承接是否上文修改了沙盒模式设置，恢复上文的修改
        :param is_example_mode: 临时设置中是否使用沙盒数据模式
        """
        recover = False
        if is_example_mode != ABuEnv._g_enable_example_env_ipython:
            # 不要使用enable_example_env_ipython改变缓存模式，只临时改变沙盒类型
            ABuEnv._g_enable_example_env_ipython = is_example_mode
            recover = True

        yield
        if recover:
            # 不要使用disable_example_env_ipython改变缓存模式，只恢复沙盒类型
            ABuEnv._g_enable_example_env_ipython = not is_example_mode

    def map_tip_target_label(self, n_target):
        """
        根据n_target要在设置中选择几个symbol映射文字提示
        :param n_target: eg：1，2，－1
        :return: eg：需选择1个分析目标在分析设置中
        """
        if n_target == 1:
            return u'需选择1个分析目标在\'分析设置\'中'
        elif n_target == 2:
            return u'需选择2个分析目标在\'分析设置\'中'
        else:
            return u'需选择多个(>1个)分析目标在\'分析设置\'中'

    def _start_end_n_fold(self):
        """获取设置中的时间获取模式以及具体的start，end，n_folds"""
        n_folds = 1
        start = None
        end = None
        if not self.tool_set.run_years.disabled:
            # 如果使用年回测模式
            n_folds = self.tool_set.run_years.value
        if not self.tool_set.start.disabled:
            # 使用开始回测日期
            start = self.tool_set.start.value
        if not self.tool_set.end.disabled:
            # 使用结束回测日期
            end = self.tool_set.end.value
        return start, end, n_folds

    def info_change_set_mode(self, symbol):
        """
            在上层定参数装饰器single_fetch_symbol_analyse，multi_fetch_symbol_analyse
            中统一处理数据获取失败的case，如果沙盒模式提醒改变设置数据更新
        """
        # noinspection PyProtectedMember
        if ABuEnv._g_enable_example_env_ipython:
            logging.info(
                u'当前数据模式为\'沙盒模式\'无{}数据，'
                u'请在\'分析设置\'中切换数据模式并确认数据可获取！'
                u'非沙盒模式建议先用\'数据下载界面操作\'进行数据下载'
                u'之后设置数据模式为\'开放数据模式\'，联网模式使用\'本地数据模式\''.format(symbol))
        else:
            logging.info(u'{}数据获取失败！'.format(symbol))
        browser_down_csv_zip()

    def _fetch_single_kl(self, symbol):
        """
        通过_start_end_n_fold获取时间参数进行金融序列获取
        :param symbol: eg: usTSLA
        :return: pd.DataFrame
        """
        start, end, n_folds = self._start_end_n_fold()
        kl = ABuSymbolPd.make_kl_df(symbol, n_folds=n_folds, start=start, end=end)
        return kl

    def _fetch_multi_kl(self, choice_symbol):
        """
        多个symbol目标，通过_start_end_n_fold获取时间参数进行金融序列获取
        :param choice_symbol: eg: ['usTSLA', 'usNOAH']
        :return: eg: {'usTSLA': pd.DataFrame, 'usNOAH': pd.DataFrame}
        """
        start, end, n_folds = self._start_end_n_fold()

        kl_dict = {symbol: ABuSymbolPd.make_kl_df(symbol, start=start, end=end, n_folds=n_folds)
                   for symbol in choice_symbol}
        kl_dict = {kl_key: kl_dict[kl_key] for kl_key in kl_dict if kl_dict[kl_key] is not None}
        return kl_dict

    def _fetch_multi_kl_col(self, choice_symbol, col_key, na_val=0):
        """
         多个symbol目标，通过_start_end_n_fold获取时间参数进行金融序列获取,
         多个symbol组成的字典对象通过col_key获取各个元素的金融序列中对应的列
         形成新的pd.DateFrame对象
         :param choice_symbol: eg: ['usTSLA', 'usNOAH']
         :return: pd.DateFrame对象
         """
        start, end, n_folds = self._start_end_n_fold()

        kl_dict = {symbol: ABuSymbolPd.make_kl_df(symbol, start=start, end=end, n_folds=n_folds)
                   for symbol in choice_symbol}
        kl_dict = {kl_key: kl_dict[kl_key] for kl_key in kl_dict if kl_dict[kl_key] is not None}

        if len(kl_dict) > 0:
            kl_col_df = pd.concat({kl_name: kl_dict[kl_name][col_key] for kl_name in kl_dict}, axis=1)
            # noinspection PyUnresolvedReferences
            kl_col_df = kl_col_df.fillna(value=na_val)
            return kl_col_df

    def _choice_symbol_single(self, default=None):
        """单独一个分析目标的函数"""
        choice_symbols = self.tool_set.sc.choice_symbols.options
        if choice_symbols is None or len(choice_symbols) == 0:
            # 如果一个symbol都没有，使用示例
            symbol = 'usTSLA' if default is None else default
            show_msg_func(u'未设置任何symbol将使用示例{}进行分析'.format(symbol))
        elif choice_symbols is not None and len(choice_symbols) > 1:
            symbol = choice_symbols[0]
            show_msg_func(u'分析设置多个symbol目标，只取第一个{}进行分析'.format(symbol))
        else:
            symbol = choice_symbols[-1]
        return symbol

    def _choice_symbol_pair(self, default=None):
        """两个分析目标的函数"""
        choice_symbols = self.tool_set.sc.choice_symbols.options
        if choice_symbols is None or len(choice_symbols) < 2:
            # 如果一个symbol都没有，使用示例
            if default is None:
                symbol1, symbol2 = 'AU0', 'XAU'
            else:
                symbol1, symbol2 = default[0], default[1]
            show_msg_func(u'需要选择两个symbol，将使用示例{} vs {}进行分析'.format(symbol1, symbol2))
        elif choice_symbols is not None and len(choice_symbols) > 2:
            symbol1, symbol2 = choice_symbols[0], choice_symbols[1]
            show_msg_func(u'分析设置多个symbol目标，只取第一个{}，第二个{}进行分析'.format(symbol1, symbol2))
        else:
            symbol1, symbol2 = choice_symbols[-2], choice_symbols[-1]
        return symbol1, symbol2

    def _choice_symbol_multi(self):
        """分析多个目标的函数"""
        choice_symbols = self.tool_set.sc.choice_symbols.options
        if choice_symbols is None or len(choice_symbols) == 0:
            # 如果一个symbol都没有，使用示例
            choice_symbols = ['sh600036', 'sh600809', 'hk00700', 'hk03333']
            show_msg_func(u'未设置任何symbol将使用示例')
        return choice_symbols
