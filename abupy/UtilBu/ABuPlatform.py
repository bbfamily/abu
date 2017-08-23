# -*- encoding:utf-8 -*-
"""
    平台信息工具模块
"""

from __future__ import print_function
from __future__ import absolute_import
from __future__ import division

import os
import platform
import sys
import struct
import logging


def is_32bit():
    """是否32位操作系统"""
    return struct.calcsize("P") * 8 < 64


def is_mac():
    """是否mac os操作系统"""
    return 'Darwin' in platform.system()


def is_windows():
    """是否Windows操作系统"""
    return 'Windows' in platform.system()


def is_linux():
    """是否Linux操作系统"""
    return 'Linux' in platform.system()


def platform_name():
    """操作系统名称字符串"""
    return platform.system()


def get_sys_info():
    """
        平台基本信息收集
        eg:
            [('python', '3.6.0.final.0'),
             ('python-bits', 64),
             ('OS', 'Darwin'),
             ('OS-release', '15.6.0'),
             ('machine', 'x86_64'),
             ('processor', 'i386'),
             ('byteorder', 'little'),
             ('LC_ALL', 'None'),
             ('LANG', 'zh_CN.UTF-8')]
    """
    sys_info = list()
    try:
        sys_name, node_name, release, version, machine, processor = platform.uname(
        )
        # noinspection PyStringFormat
        sys_info.extend([
            ("python", "%d.%d.%d.%s.%s" % sys.version_info[:]),
            ("python-bits", struct.calcsize("P") * 8),
            ("OS", "%s" % sys_name),
            ("OS-release", "%s" % release),
            ("machine", "%s" % machine),
            ("processor", "%s" % processor),
            ("byteorder", "%s" % sys.byteorder),
            ("LC_ALL", "%s" % os.environ.get('LC_ALL', "None")),
            ("LANG", "%s" % os.environ.get('LANG', "None")),
        ])
    except Exception as e:
        logging.exception(e)

    return sys_info


# noinspection PyDeprecation
def show_versions():
    """
        平台基本信息收集以及主要lib版本号信息
        eg.
            INSTALLED VERSIONS
            ------------------
            python: 3.6.0.final.0
            python-bits: 64
            OS: Darwin
            OS-release: 15.6.0
            machine: x86_64
            processor: i386
            byteorder: little
            LC_ALL: None
            LANG: zh_CN.UTF-8

            pandas: 0.19.2
            sklearn: 0.18.1
            numpy: 1.11.3
            scipy: 0.18.1
            statsmodels: 0.6.1
            notebook: 4.3.1
            tables: 3.3.0
            seaborn: 0.7.1
            matplotlib: 2.0.0
            requests: 2.12.4
            bs4: 4.5.3
            numba: 0.30.1
    """
    sys_info = get_sys_info()

    deps_mod = [
        # (MODULE_NAME, f(dep_mod) -> dep_mod version)
        ("pandas", lambda dep_mod: dep_mod.__version__),
        ("sklearn", lambda dep_mod: dep_mod.__version__),
        ("numpy", lambda dep_mod: dep_mod.version.version),
        ("scipy", lambda dep_mod: dep_mod.version.version),
        ("statsmodels", lambda dep_mod: dep_mod.__version__),
        ("notebook", lambda dep_mod: dep_mod.__version__),
        ("tables", lambda dep_mod: dep_mod.__version__),
        ("seaborn", lambda dep_mod: dep_mod.__version__),
        ("matplotlib", lambda dep_mod: dep_mod.__version__),
        ("requests", lambda dep_mod: dep_mod.__version__),
        ("bs4", lambda dep_mod: dep_mod.__version__),
        ("numba", lambda dep_mod: dep_mod.__version__)
    ]

    deps_info = list()
    for (modname, ver_f) in deps_mod:
        try:
            import imp
            try:
                mod = imp.load_module(modname, *imp.find_module(modname))
            except ImportError:
                import importlib
                mod = importlib.import_module(modname)
            ver = ver_f(mod)
            deps_info.append((modname, ver))
        except:
            deps_info.append((modname, None))
    print("\nINSTALLED VERSIONS")
    print("------------------")

    for k, stat in sys_info:
        print("%s: %s" % (k, stat))

    print("")
    for k, stat in deps_info:
        print("%s: %s" % (k, stat))


if __name__ == "__main__":
    show_versions()
