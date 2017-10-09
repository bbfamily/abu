# coding=utf-8
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# noinspection PyUnresolvedReferences
from win32api import *

# Try and use XP features, so we get alpha-blending etc.
try:
    from winxpgui import *
except ImportError:
    # noinspection PyUnresolvedReferences
    from win32gui import *
# noinspection PyUnresolvedReferences
import win32con
import sys
import os
import struct
import time
# noinspection PyUnresolvedReferences
import win32com.client as com
# noinspection PyUnresolvedReferences
import win32event as w32e
# noinspection PyUnresolvedReferences
import mmapfile as mmf
# noinspection PyUnresolvedReferences
import win32api as win_api

from ..UtilBu.ABuStrUtil import to_native_str


def show_msg(title, msg):
    """
    windows pop弹窗，主要用在长时间且耗时的任务中，提示重要问题信息
    :param title: 弹窗标题
    :param msg: 弹窗信息
    """
    MainWindow(to_native_str(title), to_native_str(msg))


def socket_bind_recv(socket_fn, cmd_handler):
    """
    非bsd系统的进程间通信，接受消息，处理消息，使用windows全局共享内存实现，
    函数名称保持与bsd的接口名称一致
    :param socket_fn: 共享内存文件名称
    :param cmd_handler: cmd处理函数，callable类型
    """
    if not callable(cmd_handler):
        print('socket_bind_recv cmd_handler must callable!')

    while True:
        global_fn = 'Global\\{}'.format(socket_fn)
        event = w32e.CreateEvent(None, 0, 0, global_fn)
        event_mmap = mmf.mmapfile(None, socket_fn, 1024)
        w32e.WaitForSingleObject(event, -1)
        socket_cmd = event_mmap.read(1024).decode()
        # 把接收到的socket传递给外部对应的处理函数
        cmd_handler(socket_cmd)
        event_mmap.close()
        win_api.CloseHandle(event)


def socket_send_msg(socket_fn, msg):
    """
    非bsd系统的进程间通信，发送消息，使用windows全局共享内存实现，函数名称保持与bsd的接口名称一致
    :param socket_fn: : 共享内存名称
    :param msg: 字符串类型需要传递的数据，不需要encode，内部进行encode
    """
    global_fn = 'Global\\{}'.format(socket_fn)
    event = w32e.OpenEvent(w32e.EVENT_ALL_ACCESS, 0, global_fn)
    event_mmap = mmf.mmapfile(None, socket_fn, 1024)
    w32e.SetEvent(event)
    event_mmap.write(msg)
    event_mmap.close()
    win_api.CloseHandle(event)


def fold_free_size_mb(folder):
    """
    windows os下剩余磁盘空间获取
    :param folder: 目标目录
    :return: 返回float，单位mb
    """
    return drive_free_space(folder) / 1024 / 1024 / 1024


def drive_free_space(drive):
    # noinspection PyBroadException
    try:
        fso = com.Dispatch("Scripting.FileSystemObject")
        drv = fso.GetDrive(drive)
        return drv.FreeSpace
    except:
        return 0


def max_drive():
    space_array = dict()
    for i in range(65, 91):
        vol = chr(i) + '://'
        if os.path.isdir(vol):
            space_array[vol] = drive_free_space(vol)

    max_v = max(zip(space_array.values(), space_array.keys()))[1]
    if max_v.startswith('c'):
        return os.path.expanduser('~')
    return max_v


# noinspection PyClassHasNoInit
class PyNOTIFYICONDATA:
    _struct_format = (
        "I"  # DWORD cbSize;
        "I"  # HWND hWnd;
        "I"  # UINT uID;
        "I"  # UINT uFlags;
        "I"  # UINT uCallbackMessage;
        "I"  # HICON hIcon;
        "128s"  # TCHAR szTip[128];
        "I"  # DWORD dwState;
        "I"  # DWORD dwStateMask;
        "256s"  # TCHAR szInfo[256];
        "I"  # union {
        #    UINT  uTimeout;
        #    UINT  uVersion;
        # } DUMMYUNIONNAME;
        "64s"  # TCHAR szInfoTitle[64];
        "I"  # DWORD dwInfoFlags;
        #       GUID guidItem;
    )
    _struct = struct.Struct(_struct_format)

    hWnd = 0
    uID = 0
    uFlags = 0
    uCallbackMessage = 0
    hIcon = 0
    szTip = ''
    dwState = 0
    dwStateMask = 0
    szInfo = ''
    uTimeoutOrVersion = 0
    szInfoTitle = ''
    dwInfoFlags = 0

    def pack(self):
        return self._struct.pack(
            self._struct.size,
            self.hWnd,
            self.uID,
            self.uFlags,
            self.uCallbackMessage,
            self.hIcon,
            self.szTip,
            self.dwState,
            self.dwStateMask,
            self.szInfo,
            self.uTimeoutOrVersion,
            self.szInfoTitle,
            self.dwInfoFlags)

    def __setattr__(self, name, value):
        # avoid wrong field names
        if not hasattr(self, name):
            raise NameError(name)
        self.__dict__[name] = value


# noinspection PyUnresolvedReferences,PyUnusedLocal
class MainWindow:
    def __init__(self, title, msg):
        message_map = {
            win32con.WM_DESTROY: self.on_destroy,
        }
        # Register the Window class.
        wc = WNDCLASS()
        hinst = wc.hInstance = GetModuleHandle(None)
        wc.lpszClassName = "PythonTaskbarDemo"
        wc.lpfnWndProc = message_map  # could also specify a wndproc.
        class_atom = RegisterClass(wc)
        # Create the Window.
        style = win32con.WS_OVERLAPPED | win32con.WS_SYSMENU
        self.hwnd = CreateWindow(class_atom, "Taskbar Demo", style,
                                 0, 0, win32con.CW_USEDEFAULT, win32con.CW_USEDEFAULT,
                                 0, 0, hinst, None)
        UpdateWindow(self.hwnd)
        icon_path_name = os.path.abspath(os.path.join(sys.prefix, "pyc.ico"))
        icon_flags = win32con.LR_LOADFROMFILE | win32con.LR_DEFAULTSIZE
        # noinspection PyBroadException
        try:
            hicon = LoadImage(hinst, icon_path_name, win32con.IMAGE_ICON, 0, 0, icon_flags)
        except:
            hicon = LoadIcon(0, win32con.IDI_APPLICATION)
        flags = NIF_ICON | NIF_MESSAGE | NIF_TIP
        nid = (self.hwnd, 0, flags, win32con.WM_USER + 20, hicon, "Balloon  tooltip demo")
        Shell_NotifyIcon(NIM_ADD, nid)
        self.show_balloon(title, msg)
        time.sleep(20)
        DestroyWindow(self.hwnd)

    def show_balloon(self, title, msg):
        # For this message I can't use the win32gui structure because
        # it doesn't declare the new, required fields
        nid = PyNOTIFYICONDATA()
        nid.hWnd = self.hwnd
        nid.uFlags = NIF_INFO
        # type of balloon and text are random
        nid.dwInfoFlags = NIIF_INFO
        nid.szInfo = msg
        nid.szInfoTitle = title
        # Call the Windows function, not the wrapped one
        from ctypes import windll
        shell_notify_icon = windll.shell32.Shell_NotifyIconA
        shell_notify_icon(NIM_MODIFY, nid.pack())

    def on_destroy(self, hwnd, msg, wparam, lparam):
        nid = (self.hwnd, 0)
        Shell_NotifyIcon(NIM_DELETE, nid)
        # Terminate the app.
        PostQuitMessage(0)
