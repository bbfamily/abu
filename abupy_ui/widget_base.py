# -*- encoding:utf-8 -*-
from __future__ import print_function
from __future__ import division

from contextlib import contextmanager
import os
import sys
import warnings

sys.path.insert(0, os.path.abspath('../'))
warnings.filterwarnings('ignore')
warnings.simplefilter('ignore')


@contextmanager
def show_ui_ct():
    print('正在初始化界面元素，请稍后...')
    from abupy import ABuStrUtil
    go_on = True

    try:
        if ABuStrUtil.str_is_cn(str(ABuStrUtil.__file__)):
            # 检测到运行环境路径中含有中文，严重错误，将出错，使用中文警告
            msg = u'严重错误！当前运行环境下有中文路径，abu将无法正常运行！请不要使用中文路径名称, 当前环境为{}'.format(
                ABuStrUtil.to_unicode(str(ABuStrUtil.__file__)))
            import logging
            logging.info(msg)
            go_on = False
    except:
        # 如果是其它编码的字符路径会进到这里
        import logging
        msg = 'error！non English characters in the current running environment,abu will not work properly!'
        logging.info(msg)
        go_on = False
    yield go_on

    if go_on:
        from IPython.display import clear_output
        clear_output()
        # import time
        # 这里sleep(0.3)防止有些版本clear_output把下面要展示的清除了，也不能使用clear_output的wait参数，有些浏览器卡死
        # time.sleep(0.3)
