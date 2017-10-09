# -*- encoding:utf-8 -*-
"""量化技术分析工具图形可视化"""

from __future__ import print_function
from __future__ import absolute_import
from __future__ import division

import logging
from contextlib import contextmanager

from IPython.display import display
import ipywidgets as widgets

from ..TLineBu.ABuTL import EShiftDistanceHow, ESkeletonHow
from ..TLineBu.ABuTLExecute import calc_pair_speed
from ..TLineBu.ABuTLJump import calc_jump, calc_jump_line, calc_jump_line_weight
from ..TLineBu.ABuTLGolden import calc_golden
from ..UtilBu import ABuProgress
from ..WidgetBu.ABuWGToolBase import WidgetToolBase, single_fetch_symbol_analyse

__author__ = '阿布'
__weixin__ = 'abu_quant'


# noinspection PyUnusedLocal
class WidgetTLTool(WidgetToolBase):
    """技术分析界面"""

    @contextmanager
    def _init_tip_label_with_step_x(self, callback_analyse, analyse_name, with_step_x=True):
        """step_x需要的地方比较多，统一构建，外部接收赋予名字"""
        if not callable(callback_analyse):
            raise TabError('callback_analyse must callable!')

        tip_label = widgets.Label(self.map_tip_target_label(n_target=1), layout=self.label_layout)
        widget_list = [tip_label]
        step_x = None
        if with_step_x:
            step_x_label = widgets.Label(u'时间步长控制参数step_x，默认1.0',
                                         layout=self.label_layout)
            step_x = widgets.FloatSlider(
                value=1.0,
                min=0.1,
                max=2.6,
                step=0.1,
                description=u'步长',
                disabled=False,
                orientation='horizontal',
                readout=True,
                readout_format='.1f',
            )
            # 返回给需要的ui，命名独有的step_x
        yield widget_list, step_x

        if with_step_x:
            # noinspection PyUnboundLocalVariable
            step_x_box = widgets.VBox([step_x_label, step_x])
            # noinspection PyTypeChecker
            widget_list.append(step_x_box)

        analyse_bt = widgets.Button(description=analyse_name, layout=widgets.Layout(width='98%'),
                                    button_style='info')
        analyse_bt.on_click(callback_analyse)
        widget_list.append(analyse_bt)

    def init_rs_ui(self):
        """阻力支撑分析ui"""

        with self._init_tip_label_with_step_x(
                self._rs_line_analyse, u'支撑阻力线分析', with_step_x=False) as (widget_list, _):
            self.rs_mode = widgets.RadioButtons(
                options={u'只分析支撑线': 0, u'只分析阻力线': 1, u'支撑线和阻力线': 2},
                value=0,
                description=u'分析模式:',
                disabled=False
            )
            widget_list.append(self.rs_mode)
            self.only_last = widgets.RadioButtons(
                options={u'最近的阻力线和支撑线': True, u'所有的阻力线和支撑线': False},
                value=True,
                description=u'最近的阻力线和支撑线',
                disabled=False
            )
            widget_list.append(self.only_last)
        return widgets.VBox(widget_list,  # border='solid 1px',
                            layout=self.tool_layout)

    def init_jump_ui(self):
        """跳空分析ui"""

        with self._init_tip_label_with_step_x(
                self._jump_line_analyse, u'跳空技术分析', with_step_x=False) as (widget_list, _):
            self.jump_mode = widgets.RadioButtons(
                options={u'跳空统计分析': 0, u'跳空缺口筛选': 1, u'缺口时间加权筛选': 2},
                value=0,
                description=u'分析模式:',
                disabled=False
            )
            widget_list.append(self.jump_mode)

            power_threshold_label = widgets.Label(u'缺口能量阀值，默认2.0(只对缺口筛选生效)',
                                                  layout=self.label_layout)
            self.power_threshold = widgets.FloatSlider(
                value=2.0,
                min=1.5,
                max=3.5,
                step=0.1,
                description=u'能量',
                disabled=False,
                orientation='horizontal',
                readout=True,
                readout_format='.1f',
            )
            power_threshold_box = widgets.VBox([power_threshold_label, self.power_threshold])
            widget_list.append(power_threshold_box)

            jump_diff_factor_label = widgets.Label(u'设置调节跳空阀值的大小',
                                                   layout=self.label_layout)
            self.jump_diff_factor = widgets.FloatSlider(
                value=1.0,
                min=0.1,
                max=5.0,
                step=0.1,
                description=u'阀值',
                disabled=False,
                orientation='horizontal',
                readout=True,
                readout_format='.1f',
            )
            jump_diff_factor_box = widgets.VBox([jump_diff_factor_label, self.jump_diff_factor])
            widget_list.append(jump_diff_factor_box)

        return widgets.VBox(widget_list,
                            # border='solid 1px',
                            layout=self.tool_layout)

    def init_shift_distance_ui(self):
        """位移路程比ui"""

        with self._init_tip_label_with_step_x(
                self._shift_distance_analyse, u'位移路程分析') as (widget_list, step_x):
            self.shift_distance_step_x = step_x
            self.shift_distance_mode = widgets.RadioButtons(
                options={u'序列最后的元素做为路程基础': 0, u'极限值做为路程的计算基础': 1,
                         u'序列sum+极值做为路程计算基础': 2},
                value=0,
                description=u'路程模式:',
                disabled=False
            )
            widget_list.append(self.shift_distance_mode)
        return widgets.VBox(widget_list,
                            # border='solid 1px',
                            layout=self.tool_layout)

    def init_regress_ui(self):
        """线性拟合ui"""

        with self._init_tip_label_with_step_x(
                self._regress_analyse, u'线性拟合分析') as (widget_list, step_x):
            self.regress_step_x = step_x
            self.regress_mode_description = widgets.Textarea(
                value=u'1. 技术线最少拟合次数：\n'
                      u'检测至少多少次拟合曲线可以代表原始曲线y的走势，'
                      u'通过度量始y值和均线y_roll_mean的距离和原始y值和拟合回归的趋势曲线y_fit的距离的方法，默认使用metrics_rmse\n'
                      u'2. 技术线最优拟合次数：\n'
                      u'寻找多少次多项式拟合回归的趋势曲线可以完美的代表原始曲线y的走势\n'
                      u'3. 可视化技术线拟合曲线：\n'
                      u'通过步长参数在子金融序列中进行走势拟合，形成拟合曲线及上下拟合通道曲线，返回三条拟合曲线，组成拟合通道',
                disabled=False,
                layout=self.description_layout
            )
            widget_list.append(self.regress_mode_description)
            self.regress_mode = widgets.RadioButtons(
                options={u'技术线最少拟合次数': 0, u'技术线最优拟合次数': 1,
                         u'可视化技术线拟合曲线': 2},
                value=0,
                description=u'拟合模式:',
                disabled=False
            )
            widget_list.append(self.regress_mode)
        return widgets.VBox(widget_list,
                            # border='solid 1px',
                            layout=self.tool_layout)

    def init_golden_line_ui(self):
        """黄金分割ui"""
        with self._init_tip_label_with_step_x(
                self._golden_line_analyse, u'黄金分割分析', with_step_x=False) as (widget_list, _):
            self.golden_line_mode = widgets.RadioButtons(
                options={u'可视化黄金分隔带': 0, u'可视化黄金分隔带＋关键比例': 1,
                         u'可视化关键比例': 2},
                value=0,
                description=u'分隔模式:',
                disabled=False
            )
            widget_list.append(self.golden_line_mode)
            pt_tip_label = widgets.Label(u'比例设置仅对\'可视化关键比例\'生效', layout=self.label_layout)
            self.pt_range = widgets.FloatRangeSlider(
                value=[0.2, 0.8],
                min=0.1,
                max=0.9,
                step=0.1,
                description=u'比例设置:',
                disabled=False,
                continuous_update=False,
                orientation='horizontal',
                readout=True,
                readout_format='.1f',
            )
            pt_box = widgets.VBox([pt_tip_label, self.pt_range])
            widget_list.append(pt_box)

        return widgets.VBox(widget_list,
                            # border='solid 1px',
                            layout=self.tool_layout)

    def init_skeleton_ui(self):
        """价格骨架ui"""
        with self._init_tip_label_with_step_x(
                self._skeleton_analyse, u'价格骨架分析') as (widget_list, step_x):
            self.skeleton_step_x = step_x
            self.skeleton_mode = widgets.RadioButtons(
                options={u'骨架通道点位值': 200, u'三角骨架点位值': 100, u'最小值骨架点位值': 0,
                         u'最大值骨架点位值': 1, u'平均值骨架点位值': 2, u'中位数骨架点位值': 3,
                         u'最后元素骨架点位值': 4},
                value=200,
                description=u'骨架模式:',
                disabled=False
            )
            widget_list.append(self.skeleton_mode)
        return widgets.VBox(widget_list,
                            # border='solid 1px',
                            layout=self.tool_layout)

    def init_pair_speed_ui(self):
        """趋势敏感速度分析ui"""
        with self._init_tip_label_with_step_x(
                self._pair_speed_analyse, u'趋势敏感速度分析', with_step_x=False) as (widget_list, _):
            self.pair_speed_mode = widgets.RadioButtons(
                options={u'对比收盘敏感速度': 'close', u'对比涨跌敏感速度': 'p_change',
                         u'对比最高敏感速度': 'high', u'对比最低敏感速度': 'low'},
                value='close',
                description=u'对比模式:',
                disabled=False
            )
            widget_list.append(self.pair_speed_mode)
            resample_tip_label = widgets.Label(u'趋势敏感速度计算重采样周期', layout=self.label_layout)
            self.pair_resample = widgets.IntSlider(
                value=5,
                min=3,
                max=10,
                step=1,
                description=u'重采样',
                disabled=False,
                orientation='horizontal',
                readout=True,
                readout_format='d'
            )
            resample_box = widgets.VBox([resample_tip_label, self.pair_resample])
            widget_list.append(resample_box)
        return widgets.VBox(widget_list,
                            # border='solid 1px',
                            layout=self.tool_layout)

    def __init__(self, tool_set):
        """初始化技术分析界面"""
        super(WidgetTLTool, self).__init__(tool_set)

        rs_box = self.init_rs_ui()
        jump_box = self.init_jump_ui()
        pair_speed = self.init_pair_speed_ui()
        shift_distance = self.init_shift_distance_ui()
        regress = self.init_regress_ui()
        golden = self.init_golden_line_ui()
        skeleton = self.init_skeleton_ui()

        children = [rs_box, jump_box, pair_speed, shift_distance, regress, golden, skeleton]
        if self.scroll_factor_box:
            tl_box = widgets.Box(children,
                                 layout=self.scroll_widget_layout)
            # 需要再套一层VBox，不然外部的tab显示有问题
            self.widget = widgets.VBox([tl_box])
        else:
            # 一行显示两个，2个为一组，组装sub_children_group序列,
            sub_children_group = self._sub_children(children, len(children) / self._sub_children_group_cnt)
            sub_children_box = [widgets.HBox(sub_children) for sub_children in sub_children_group]
            self.widget = widgets.VBox(sub_children_box)

    def _pair_speed_analyse(self, bt):
        """趋势变化敏感速度分析action"""
        ABuProgress.clear_output()
        symbol1, symbol2 = self._choice_symbol_pair()
        start, end, n_folds = self._start_end_n_fold()
        speed_key = self.pair_speed_mode.value
        resample = self.pair_resample.value
        symbol1_speed, symbol2_speed, corr = calc_pair_speed(symbol1, symbol2, resample=resample, speed_key=speed_key,
                                                             start=start, end=end, n_folds=n_folds, show=True)
        if symbol1_speed is None:
            self.info_change_set_mode('{} and {}'.format(symbol1, symbol2))
        else:
            logging.info(u'{}趋势变化敏感速度{}'.format(symbol1, symbol1_speed))
            logging.info(u'{}趋势变化敏感速度{}'.format(symbol2, symbol2_speed))
            logging.info(u'{}与{}相关度{}'.format(symbol2, symbol2, corr))
            logging.info(u'{}与{}趋势相关敏感速度差{}'.format(symbol1, symbol2, (symbol1_speed - symbol2_speed) * corr))

    @single_fetch_symbol_analyse
    def _jump_line_analyse(self, kl, kl_tl, bt):
        """跳空缺口分析action"""
        # print('正在分析跳空缺口，请稍后...')
        if self.jump_mode.value == 0:
            jumps = calc_jump(kl, jump_diff_factor=self.jump_diff_factor.value)
        elif self.jump_mode.value == 1:
            jumps = calc_jump_line(kl, power_threshold=self.power_threshold.value,
                                   jump_diff_factor=self.jump_diff_factor.value)
        else:
            # 暂时固定加权比例为(0.5, 0.5)
            jumps = calc_jump_line_weight(kl, sw=(0.5, 0.5), power_threshold=self.power_threshold.value,
                                          jump_diff_factor=self.jump_diff_factor.value)
        display(jumps)

    @single_fetch_symbol_analyse
    def _rs_line_analyse(self, kl, kl_tl, bt):
        """支撑阻力线分析action"""
        if self.rs_mode.value == 0:
            # 只绘制支撑线
            kl_tl.show_support_trend(only_last=self.only_last.value, show=True, show_step=False)
        elif self.rs_mode.value == 1:
            # 只绘制阻力线
            kl_tl.show_resistance_trend(only_last=self.only_last.value, show=True, show_step=False)
        else:
            # 支撑线和阻力线都绘制
            kl_tl.show_support_resistance_trend(only_last=self.only_last.value, show=True, show_step=False)

    @single_fetch_symbol_analyse
    def _shift_distance_analyse(self, kl, kl_tl, bt):
        """位移路程比分析action"""
        kl_tl.show_shift_distance(how=EShiftDistanceHow(self.shift_distance_mode.value),
                                  step_x=self.shift_distance_step_x.value)

    @single_fetch_symbol_analyse
    def _regress_analyse(self, kl, kl_tl, bt):
        """走势线性回归分析action"""
        if self.regress_mode.value == 0:
            kl_tl.show_least_valid_poly()
        elif self.regress_mode.value == 1:
            kl_tl.show_best_poly()
        else:
            kl_tl.show_regress_trend_channel(step_x=self.regress_step_x.value)

    @single_fetch_symbol_analyse
    def _golden_line_analyse(self, kl, kl_tl, bt):
        """走势黄金分割分析action"""
        if self.golden_line_mode.value == 0:
            kl_tl.show_golden()
        elif self.golden_line_mode.value == 1:
            calc_golden(kl)
        else:
            kl_tl.show_percents(self.pt_range.value)

    @single_fetch_symbol_analyse
    def _skeleton_analyse(self, kl, kl_tl, bt):
        """走势骨架分析action"""
        step_x = self.skeleton_step_x.value
        skeleton_mode = self.skeleton_mode.value
        if skeleton_mode == 200:
            kl_tl.show_skeleton_channel(step_x=step_x)
        else:
            kl_tl.show_skeleton(how=ESkeletonHow(skeleton_mode), step_x=step_x)
