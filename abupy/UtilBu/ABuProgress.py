# -*- encoding:utf-8 -*-
"""
    辅助进度显示模块，多进程，单进程
"""
from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

import sys
import os
import datetime
import time

from IPython.display import clear_output
from IPython.display import display
from ipywidgets import FloatProgress, Text, Box

from ..CoreBu import ABuEnv
# noinspection PyUnresolvedReferences
from ..CoreBu.ABuFixes import filter
from ..UtilBu.ABuDTUtil import warnings_filter, catch_error
from ..UtilBu import ABuFileUtil, ABuOsUtil
from ..CoreBu.ABuParallel import run_in_subprocess, run_in_thread

__author__ = '阿布'
__weixin__ = 'abu_quant'


def do_clear_output(wait=False):
    """
    模块方法，clear所有的输出，内部针对notebook和命令行输出做区分
    :param wait: 是否同步执行clear操作，透传给IPython.display.clear_output
    """
    if ABuEnv.g_is_ipython:
        # notebook clear
        clear_output(wait=wait)
    else:
        # cmd clear
        cmd = 'clear' if ABuEnv.g_is_mac_os else 'cls'
        os.system(cmd)
        # pass


class UIProgress(object):
    """多进程socket通信下的进度显示类"""

    def __init__(self, a_pid):
        """通过进程pid初始化ui组件"""
        self.progress_widget = FloatProgress(value=0, min=0, max=100)
        self.text_widget = Text('pid={} begin work'.format(a_pid))
        # 通过box容器都放到一个里面
        self.progress_box = Box([self.text_widget, self.progress_widget])
        display(self.progress_box)

    def update(self, p_progress, p_progress_text):
        """进度条更新以及对应文字更新"""
        self.progress_widget.value = p_progress
        self.text_widget.value = p_progress_text

    def close(self):
        """关闭ui显示"""
        self.progress_box.close()


"""多进程下进度条通信socket文件基础名字"""
K_SOCKET_FN_BASE = os.path.join(ABuEnv.g_project_cache_dir, 'abu_socket_progress')
"""多进程下进度条通信socket文件最终名字，这里子进程可以获取g_socket_fn是通过ABuEnvProcess拷贝了主进程全局信息"""
g_socket_fn = None
"""多进程下进度是否显示ui进度，只针对进程间通信类型的进度，有些太频繁的进度显示可以选择关闭"""
g_show_ui_progress = True
"""主进程下用来存贮子进程传递子进程pid为key，进度条对象UIProgress为value"""
ui_progress_dict = {}


def _socket_cmd_handle(socket_cmd):
    """主进程中处理子进程传递的进度条处理信息：创建，进度更新，销毁"""
    socket_cmd = socket_cmd.strip()
    socket_cmd = socket_cmd.strip('\x00')
    socket_cmd = socket_cmd.strip('\0')
    cmd_split = socket_cmd.split('|')
    if len(cmd_split) == 3 and cmd_split[0] in ui_progress_dict:
        # 3个字段的0是：pid
        pid = cmd_split[0]
        # 3个字段的1是：进度，转换float
        progress = float(cmd_split[1])
        # 3个字段的2是：进度文字显示
        progress_text = cmd_split[2]
        # 找到字典中的UIProgress对象开始update
        ui_progress_dict[pid].update(progress, progress_text)
    elif len(cmd_split) == 2 and cmd_split[0] in ui_progress_dict and cmd_split[1] == 'close':
        # 2个字段的0是：pid
        pid = cmd_split[0]
        # 将字典中的UIProgress对象pop出来，后执行close
        pop_progress = ui_progress_dict.pop(pid, None)
        if pop_progress is not None:
            pop_progress.close()
    elif len(cmd_split) == 2 and cmd_split[1] == 'init':
        pid = cmd_split[0]
        if pid in ui_progress_dict:
            ui_progress_dict.pop(pid)
        # 创建的进度条以pid为key放到缓存字典中
        ui_progress_dict[pid] = UIProgress(pid)


# 不管ui进度条有什么问题，也不能影响任务主进程主任务的进度执行，反正有文字进度会始终显示
@catch_error(log=False)
def ui_progress_socket_work():
    """主进程下的子线程函数：子进程传递的进度条处理信息：创建，进度更新，销毁"""
    global g_socket_fn

    # 不管是共享内存实现还是socket都通过当前时间＋pid确定唯一文件名称
    tt = datetime.datetime.now().strftime("%Y_%m_%d_%H_%M_%S_%f")
    if ABuEnv.g_is_mac_os:
        # bsd socket
        g_socket_fn = '{}_{}_{}'.format(K_SOCKET_FN_BASE, os.getpid(), tt)
        if ABuFileUtil.file_exist(g_socket_fn):
            # 如果socket文件存在，删除
            ABuFileUtil.del_file(g_socket_fn)
    else:
        # windows全局共享内存通过pid＋时间确定
        g_socket_fn = 'ABU_PROGRESS_{}_{}'.format(os.getpid(), tt)

    # socket_bind_recv不管是windows还是mac都在这里进行while True阻塞子线程一直运行
    ABuOsUtil.socket_bind_recv(g_socket_fn, _socket_cmd_handle)


# 不管ui进度条有什么问题，也不能影响任务主进程主任务工作的进度执行
@catch_error(log=False)
def check_process_is_dead():
    """主进程下的子线程函数：检测ui_progress_dict中的进程pid是否仍然活着，如果死了，从字典中清除，close ui"""
    while True:
        # 低优先级任务，1分钟执行1次
        time.sleep(60)
        do_check_process_is_dead()


def do_check_process_is_dead():
    """执行检测ui_progress_dict中的进程pid是否仍然活着，如果死了，从字典中清除，close ui"""
    import psutil
    # 获取活着的所有pid序列
    living = psutil.pids()
    clear_arr = list()
    for progress_pid in ui_progress_dict:
        # 需要临时转换一次int，living中进程序列是int
        if int(progress_pid) not in living:
            # 字典中记录的pid如果不在活着的序列中，清除
            clear_arr.append(progress_pid)
    for clear_pid in clear_arr:
        if clear_pid in ui_progress_dict:
            pop_progress = ui_progress_dict.pop(clear_pid, None)
            if pop_progress is not None:
                pop_progress.close()


# 不管ui进度条有什么问题，也不能影响任务主进程主任务工作的进度执行
@catch_error(log=False)
def cache_socket_ready():
    """通信临时文件准备工作"""
    ABuFileUtil.ensure_dir(K_SOCKET_FN_BASE)
    cache_list = os.listdir(ABuEnv.g_project_cache_dir)
    socket_cache_list = list(filter(lambda cache: cache.startswith('abu_socket_progress'), cache_list))
    if len(socket_cache_list) > 300:
        # 如果有超过300个进度socket缓存，进行清理
        for sk_name in socket_cache_list:
            ABuFileUtil.del_file(os.path.join(ABuEnv.g_project_cache_dir, sk_name))

if g_show_ui_progress and ABuEnv.g_main_pid == os.getpid() and ABuEnv.g_is_ipython:
    # 通信临时文件准备工作
    cache_socket_ready()
    # 如果是主进程执行进行子线程函数ui_progress_socket_work：处理子进程传递的进度条处理信息：创建，进度更新，销毁
    run_in_thread(ui_progress_socket_work)
    # 如果是主进程执行进行子线程函数check_process_is_dead：检测ui_progress_dict中的进程是否仍然活着，从字典中清除，close ui
    run_in_thread(check_process_is_dead)


class AbuMulPidProgress(object):
    """多进程进度显示控制类"""

    def __init__(self, total, label, show_progress=True):
        """
        外部使用eg：
        with AbuMulPidProgress(len(self.choice_symbols), 'pick stocks complete') as progress:
            for epoch, target_symbol in enumerate(self.choice_symbols):
                progress.show(epoch + 1)

        :param total: 总任务数量
        :param label: 进度显示label
        """
        self._total = total
        self._label = label
        self.epoch = 0
        self.display_step = 1
        self.progress_widget = None
        self.text_widget = None
        self.progress_box = None
        self.show_progress = show_progress

    # 不管ui进度条有什么问题，也不能影响任务工作的进度执行，反正有文字进度会始终显示
    @catch_error(log=False)
    def init_ui_progress(self):
        """初始化ui进度条"""
        if not self.show_progress:
            return

        if not ABuEnv.g_is_ipython or self._total < 2:
            return

        if ABuEnv.g_main_pid == os.getpid():
            # 如果是在主进程下显示那就直接来
            self.progress_widget = FloatProgress(value=0, min=0, max=100)
            self.text_widget = Text('pid={} begin work'.format(os.getpid()))
            self.progress_box = Box([self.text_widget, self.progress_widget])
            display(self.progress_box)
        else:
            if g_show_ui_progress and g_socket_fn is not None:
                # 子进程下通过socket通信将pid给到主进程，主进程创建ui进度条
                ABuOsUtil.socket_send_msg(g_socket_fn, '{}|init'.format(os.getpid()))

    # 不管ui进度条有什么问题，也不能影响任务工作的进度执行，反正有文字进度会始终显示
    @catch_error(log=False)
    def update_ui_progress(self, ps, ps_text):
        """更新文字进度条"""
        if not self.show_progress:
            return

        if not ABuEnv.g_is_ipython or self._total < 2:
            return

        if ABuEnv.g_main_pid == os.getpid():
            # 如果是在主进程下显示那就直接来
            if self.progress_widget is not None:
                self.progress_widget.value = ps
            if self.text_widget is not None:
                self.text_widget.value = ps_text
        else:
            if g_show_ui_progress and g_socket_fn is not None:
                # 子进程下通过socket通信将pid给到主进程，主进程通过pid查找对应的进度条对象后更新进度
                ABuOsUtil.socket_send_msg(g_socket_fn, '{}|{}|{}'.format(os.getpid(), ps, ps_text))

    # 不管ui进度条有什么问题，也不能影响任务工作的进度执行，反正有文字进度会始终显示
    @catch_error(log=False)
    def close_ui_progress(self):
        """关闭ui进度条显示"""
        if not self.show_progress:
            return

        if not ABuEnv.g_is_ipython or self._total < 2:
            return

        if ABuEnv.g_main_pid == os.getpid():
            # 如果是在主进程下显示那就直接来
            if self.progress_box is not None:
                self.progress_box.close()
        else:
            if g_show_ui_progress and g_socket_fn is not None:
                # 子进程下通过socket通信将pid给到主进程，主进程通过pid查找对应的进度条对象后关闭对象，且弹出
                ABuOsUtil.socket_send_msg(g_socket_fn, '{}|close'.format(os.getpid()))

    def __enter__(self):
        """
        以上下文管理器类方式实现__enter__，针对self._total分配self.display_step
        """
        if self.show_progress:
            self.display_step = 1
            if self._total >= 5000:
                self.display_step = 50
            elif self._total >= 3000:
                self.display_step = 30
            elif self._total >= 2000:
                self.display_step = 20
            elif self._total > 1000:
                self.display_step = 10
            elif self._total >= 600:
                self.display_step = 6
            elif self._total >= 300:
                self.display_step = 3
            elif self._total >= 100:
                self.display_step = 2
            elif self._total >= 20:
                self.display_step = 2
            self.epoch = 0
            self.init_ui_progress()
        return self

    def show(self, epoch=None, clear=True):
        """
        进行进度控制显示主方法
        :param epoch: 默认None, 即使用类内部计算的迭代次数进行进度显示
        :param clear: 默认True, 子进程显示新的进度前，先do_clear_output所有输出
        :return:
        """
        if not self.show_progress:
            return

        self.epoch = epoch if epoch is not None else self.epoch + 1
        if self.epoch % self.display_step == 0:
            ps = round(self.epoch / self._total * 100, 2)
            ps = 100 if ps > 100 else ps
            ps_text = "pid:{} {}:{}%".format(os.getpid(), self._label, ps)
            if not ABuEnv.g_is_ipython or self._total < 2:
                if clear:
                    do_clear_output()
                    # clear_std_output()
                print(ps_text)

            self.update_ui_progress(ps, ps_text)

    def __exit__(self, exc_type, exc_val, exc_tb):
        """
        以上下文管理器类方式实现__exit__，针对在子进城中的输出显示进度进行do_clear_output扫尾工作
        """
        if not self.show_progress:
            return

        clear = False
        if clear:
            # clear在mac上应该打开, 由于windows某些版本浏览器wait=True会有阻塞情况，如果wait＝False, 有clear之后的风险，
            do_clear_output(wait=True)  # wait 需要同步否则会延迟clear
        else:
            # print("pid:{} done!".format(os.getpid()))
            pass

        self.close_ui_progress()


class AbuBlockProgress(object):
    """主进程阻塞任务，启动子单进程任务进度显示控制类"""

    def __init__(self, label, interval=1, max_step=20):
        """
        :param label: 阻塞进度条显示的文字信息
        :param interval: 阻塞进度条显示的时间间隔
        :param max_step: 进度最大显示粒度
        """
        self.label = label
        self.interval = interval
        self.sub_process = None
        self.max_step = max_step

    def __enter__(self):
        """创建子进程做进度显示"""

        def progress_interval(interval, label):
            count = 1
            while True:
                p_str = '*^{}s'.format(int(count * 3))
                end = format('*', p_str)
                progress_str = '{}{}'.format(label, end)
                do_clear_output()
                # clear_std_output()
                print(progress_str)
                count += 1
                if count > self.max_step:
                    count = 1
                time.sleep(interval)

        self.sub_process = run_in_subprocess(progress_interval, self.interval, self.label)

    def __exit__(self, exc_type, exc_val, exc_tb):
        """结束子进程，清理输出"""
        if self.sub_process is not None and self.sub_process.is_alive():
            self.sub_process.terminate()
            do_clear_output()
            # clear_std_output()


class AbuProgress(object):
    """单进程（主进程）进度显示控制类"""

    # 过滤DeprecationWarning: Widget._keys_default is deprecated in traitlets 4.1: use @default decorator instead.
    @warnings_filter
    def __init__(self, total, a_progress, label=None):
        """
        外部使用eg：
            progess = AbuProgress(stock_df.shape[0], 0, 'merging {}'.format(m))
            for i, symbol in enumerate(stock_df['symbol']):
                progess.show(i + 1)
        :param total: 总任务数量
        :param a_progress: 初始进度
        :param label: 进度显示label
        """
        self._total = total
        self._progress = a_progress
        self._label = label
        self.f = sys.stdout
        self.progress_widget = None

    def __enter__(self):
        """创建子进程做进度显示"""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.f.write('\r')
        if self.progress_widget is not None:
            self.progress_widget.close()

    @property
    def progress(self):
        """property获取self._progress"""
        return self._progress

    @progress.setter
    def progress(self, a_progress):
        """rogress.setter设置progress"""
        if a_progress > self._total:
            self._progress = self._total
        elif a_progress < 0:
            self._progress = 0
        else:
            self._progress = a_progress

    def show(self, a_progress=None, ext='', p_format="{}:{}:{}%"):
        """
        进行进度控制显示主方法
        :param ext: 可以添加额外的显示文字，str，默认空字符串
        :param a_progress: 默认None, 即使用类内部计算的迭代次数进行进度显示
        :param p_format: 进度显示格式，默认{}: {}%，即'self._label:round(self._progress / self._total * 100, 2))%'
        """
        self.progress = a_progress if a_progress is not None else self.progress + 1
        ps = round(self._progress / self._total * 100, 2)

        if self._label is not None:
            # 如果初始化label没有就只显示ui进度
            self.f.write('\r')
            self.f.write(p_format.format(self._label, ext, ps))

        if ABuEnv.g_is_ipython:
            if self.progress_widget is None:
                self.progress_widget = FloatProgress(value=0, min=0, max=100)
                display(self.progress_widget)
            self.progress_widget.value = ps

        # 这样会出现余数结束的情况，还是尽量使用上下文管理器控制结束
        if self._progress == self._total:
            self.f.write('\r')
            if self.progress_widget is not None:
                self.progress_widget.close()
