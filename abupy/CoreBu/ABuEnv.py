# -*- encoding:utf-8 -*-
"""
    全局环境配置模块
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import logging
import os
import re
import platform
import sys
import warnings
from enum import Enum
from os import path

import numpy as np
import pandas as pd

from ..CoreBu.ABuFixes import six

__author__ = '阿布'
__weixin__ = 'abu_quant'

"""暂时支持windows和mac os，不是windows就是mac os（不使用Darwin做判断），linux下没有完整测试"""
g_is_mac_os = platform.system().lower().find("windows") < 0 and sys.platform != "win32"
"""python版本环境，是否python3"""
g_is_py3 = six.PY3
"""ipython，是否ipython运行环境"""
g_is_ipython = True
"""主进程pid，使用并行时由于ABuEnvProcess会拷贝主进程注册了的模块信息，所以可以用g_main_pid来判断是否在主进程"""
g_main_pid = os.getpid()

try:
    # noinspection PyUnresolvedReferences
    __IPYTHON__
except NameError:
    g_is_ipython = False

# noinspection PyBroadException
try:
    # noinspection PyUnresolvedReferences
    import psutil

    """有psutil，使用psutil.cpu_count计算cpu个数"""
    g_cpu_cnt = psutil.cpu_count(logical=True) * 1
except ImportError:
    if g_is_py3:
        # noinspection PyUnresolvedReferences
        g_cpu_cnt = os.cpu_count()
    else:
        import multiprocessing as mp

        g_cpu_cnt = mp.cpu_count()
except:
    # 获取cpu个数失败，默认4个
    g_cpu_cnt = 4

"""pandas忽略赋值警告"""
pd.options.mode.chained_assignment = None

"""numpy，pandas显示控制，默认开启"""
g_display_control = True
if g_display_control:
    # pandas DataFrame表格最大显示行数
    pd.options.display.max_rows = 20
    # pandas DataFrame表格最大显示列数
    pd.options.display.max_columns = 20
    # pandas精度浮点数显示4位
    pd.options.display.precision = 4
    # numpy精度浮点数显示4位，不使用科学计数法
    np.set_printoptions(precision=4, suppress=True)

"""忽略所有警告，默认关闭"""
g_ignore_all_warnings = False
"""忽略库警告，默认打开"""
g_ignore_lib_warnings = True
if g_ignore_lib_warnings:
    # noinspection PyBroadException
    try:
        import matplotlib

        matplotlib.warnings.filterwarnings('ignore')
        matplotlib.warnings.simplefilter('ignore')
        import sklearn

        sklearn.warnings.filterwarnings('ignore')
        sklearn.warnings.simplefilter('ignore')
    except:
        pass
if g_ignore_all_warnings:
    warnings.filterwarnings('ignore')
    warnings.simplefilter('ignore')

# ＊＊＊＊＊＊＊＊＊＊＊＊＊＊＊＊＊＊＊＊ 数据目录 start ＊＊＊＊＊＊＊＊＊＊＊＊＊＊＊＊
"""
    abu 文件目录根目录
    windows应该使用磁盘空间比较充足的盘符，比如：d://, e:/, f:///

    eg:
    root_drive = 'd://'
    root_drive = 'e://'
    root_drive = 'f://'
"""


def str_is_cn(a_str):
    """
        str_is_cn原始位置: UtilBu.ABuStrUtil
        为保持env为最初初始化不引入其它模块，这里临时拷贝使用
        通过正则表达式判断字符串中是否含有中文
        返回结果只判断是否search结果为None, 不返回具体匹配结果
        eg:
            K_CN_RE.search(a_str)('abc') is None
            return False
            K_CN_RE.search(a_str)('abc哈哈') -> <_sre.SRE_Match object; span=(3, 5), match='哈哈'>
            return True
    """

    def to_unicode(text, encoding=None, errors='strict'):
        """
        to_unicode原始位置: UtilBu.ABuStrUtil，为保持env为最初初始化不引入其它模块，这里临时拷贝使用
        """
        if isinstance(text, six.text_type):
            return text
        if not isinstance(text, (bytes, six.text_type)):
            raise TypeError('to_unicode must receive a bytes, str or unicode '
                            'object, got %s' % type(text).__name__)
        if encoding is None:
            encoding = 'utf-8'
        try:
            decode_text = text.decode(encoding, errors)
        except:
            # 切换试一下，不行就需要上层处理
            decode_text = text.decode('gbk' if encoding == 'utf-8' else 'utf-8', errors)
        return decode_text

    cn_re = re.compile(u'[\u4e00-\u9fa5]+')
    try:
        is_cn_path = cn_re.search(to_unicode(a_str)) is not None
    except:
        # 非gbk，utf8的其它编码会进入这里，统一进行处理
        is_cn_path = True
    return is_cn_path


root_drive = path.expanduser('~')
# root_drive = os.path.join(root_drive, u'测试')
# noinspection PyTypeChecker

if str_is_cn(root_drive):
    """
        如果用户根目录使用了中文名称，择放弃使用公共缓存文件夹,
        windows下可以使用中文用户名，这样会导致pandas读取，写入
        csv，hdf5出现问题，所以一旦发现用户路径为中文路径，改变
        缓存路径为abupy根代码路径
    """
    abupy_source_dir = os.path.abspath(os.path.join(os.path.dirname(os.path.realpath(str(__file__))), os.path.pardir))
    # 改变缓存路径为abupy根代码路径
    root_drive = abupy_source_dir
    print('root_drive is change to {}'.format(root_drive))

"""abu数据缓存主目录文件夹"""
g_project_root = path.join(root_drive, 'abu')
"""abu数据文件夹 ~/abu/data"""
g_project_data_dir = path.join(g_project_root, 'data')
"""abu日志文件夹 ~/abu/log"""
g_project_log_dir = path.join(g_project_root, 'log')
"""abu数据库文件夹 ~/abu/db"""
g_project_db_dir = path.join(g_project_root, 'db')
"""abu缓存文件夹 ~/abu/cache"""
g_project_cache_dir = path.join(g_project_data_dir, 'cache')
"""abu项目数据主文件目录，即项目中的RomDataBu位置"""
g_project_rom_data_dir = path.join(path.dirname(path.abspath(path.realpath(__file__))), '../RomDataBu')

"""abu日志文件 ~/abu/log/info.log"""
g_project_log_info = path.join(g_project_log_dir, 'info.log')

"""hdf5做为金融时间序列存储的路径"""
g_project_kl_df_data = path.join(g_project_data_dir, 'df_kl.h5')

_p_dir = os.path.abspath(os.path.join(os.path.dirname(os.path.realpath(__file__)), os.path.pardir))

# 不再使用hdf5做为默认，有windows用户的hdf5环境有问题
"""使用书中相同的沙盒数据环境，RomDataBu/csv内置的金融时间序列文件"""
# g_project_kl_df_data_example = os.path.join(_p_dir, 'RomDataBu/df_kl.h5')
g_project_kl_df_data_example = os.path.join(_p_dir, 'RomDataBu/csv')
# ＊＊＊＊＊＊＊＊＊＊＊＊＊＊＊＊＊＊＊＊ 数据目录 end ＊＊＊＊＊＊＊＊＊＊＊＊＊＊＊＊


#  ＊＊＊＊＊＊＊＊＊＊＊＊＊＊＊＊＊＊＊＊ 数据源 start ＊＊＊＊＊＊＊＊＊＊＊＊＊＊＊＊

# ＊＊＊＊＊＊＊＊＊＊＊＊＊＊＊＊＊＊＊＊ CrawlBu end ＊＊＊＊＊＊＊＊＊＊＊＊＊＊＊＊
"""
chrome 驱动
"""
g_crawl_chrome_driver = None


# ＊＊＊＊＊＊＊＊＊＊＊＊＊＊＊＊＊＊＊＊ CrawlBu start ＊＊＊＊＊＊＊＊＊＊＊＊＊＊＊＊


# TODO 缩短 E_MARKET_SOURCE_bd－>BD
class EMarketSourceType(Enum):
    """
        数据源，当数据获取不可靠时，可尝试切换数据源，更可连接私有的数据源
    """
    """百度 a股，美股，港股"""
    E_MARKET_SOURCE_bd = 0
    """腾讯 a股，美股，港股"""
    E_MARKET_SOURCE_tx = 1
    """网易 a股，美股，港股"""
    E_MARKET_SOURCE_nt = 2
    """新浪 美股"""
    E_MARKET_SOURCE_sn_us = 3

    """新浪 国内期货"""
    E_MARKET_SOURCE_sn_futures = 100
    """新浪 国际期货"""
    E_MARKET_SOURCE_sn_futures_gb = 101

    """火币 比特币，莱特币"""
    E_MARKET_SOURCE_hb_tc = 200


"""默认设置数据源使用E_MARKET_SOURCE_bd"""
g_market_source = EMarketSourceType.E_MARKET_SOURCE_bd

"""自定义的私有数据源类，默认None"""
g_private_data_source = None


# TODO 缩短 E_MARKET_TARGET_US－>US
class EMarketTargetType(Enum):
    """
        交易品种类型，即市场类型，
        eg. 美股市场, A股市场, 港股市场, 国内期货市场,
            美股期权市场, TC币市场（比特币等
    """
    """美股市场"""
    E_MARKET_TARGET_US = 'us'
    """A股市场"""
    E_MARKET_TARGET_CN = 'hs'
    """港股市场"""
    E_MARKET_TARGET_HK = 'hk'

    """国内期货市场"""
    E_MARKET_TARGET_FUTURES_CN = 'futures_cn'
    """国际期货市场"""
    E_MARKET_TARGET_FUTURES_GLOBAL = 'futures_global'
    """美股期权市场"""
    E_MARKET_TARGET_OPTIONS_US = 'options_us'

    """TC币市场（比特币等）"""
    E_MARKET_TARGET_TC = 'tc'


class EMarketSubType(Enum):
    """
        子市场（交易所）类型定义
    """

    """美股纽交所NYSE"""
    US_N = 'NYSE'
    """美股纳斯达克NASDAQ"""
    US_OQ = 'NASDAQ'
    """美股粉单市场"""
    US_PINK = 'PINK'
    """美股OTCMKTS"""
    US_OTC = 'OTCMKTS'
    """美国证券交易所"""
    US_AMEX = 'AMEX'
    """未上市"""
    US_PREIPO = 'PREIPO'

    """港股hk"""
    HK = 'hk'

    """上证交易所sh"""
    SH = 'sh'
    """深圳交易所sz"""
    SZ = 'sz'

    """大连商品交易所DCE'"""
    DCE = 'DCE'
    """郑州商品交易所ZZCE'"""
    ZZCE = 'ZZCE'
    """上海期货交易所SHFE'"""
    SHFE = 'SHFE'

    """伦敦金属交易所"""
    LME = 'LME'
    """芝加哥商品交易所"""
    CBOT = 'CBOT'
    """纽约商品交易所"""
    NYMEX = 'NYMEX'

    """币类子市场COIN'"""
    COIN = 'COIN'


"""切换目标操作市场，美股，A股，港股，期货，比特币等，默认美股市场"""
g_market_target = EMarketTargetType.E_MARKET_TARGET_US

"""市场中1年交易日，默认250日"""
g_market_trade_year = 250
if g_market_target == EMarketTargetType.E_MARKET_TARGET_US:
    # 美股252天
    g_market_trade_year = 252
if g_market_target == EMarketTargetType.E_MARKET_TARGET_TC:
    # 默认设置币类每天都可以交易
    g_market_trade_year = 365


# TODO EMarketDataSplitMode移动到市场请求相关对应的模块中
class EMarketDataSplitMode(Enum):
    """
        ABuSymbolPd中请求参数，关于是否需要与基准数据对齐切割
    """
    """直接取出所有data，不切割，即外部需要切割"""
    E_DATA_SPLIT_UNDO = 0
    """内部根据start，end取切割data"""
    E_DATA_SPLIT_SE = 1


# TODO 缩短 E_DATA_FETCH_NORMAL－>NORMAL
class EMarketDataFetchMode(Enum):
    """
        金融时间数据获取模式
    """
    """普通模式，尽量从本地获取数据，本地数据不满足的情况下进行网络请求"""
    E_DATA_FETCH_NORMAL = 0
    """强制从本地获取数据，本地数据不满足的情况下，返回None"""
    E_DATA_FETCH_FORCE_LOCAL = 1
    """强制从网络获取数据，不管本地数据是否满足"""
    E_DATA_FETCH_FORCE_NET = 2


"""
    金融时间数据获取模式模块设置g_data_fetch_mode，默认为E_DATA_FETCH_NORMAL，实际上默认值建议
    为E_DATA_FETCH_FORCE_LOCAL，所有数据提前使用ABu.run_kl_update完成updtae，之后使用本地数据回测，
    原因：
    1. mac os 10.9 later 多进程 ＋ numpy有系统bug
    2. hdf5并行容易写坏文件
    3. 执行效率更高
    4. 分开数据获取与回测流程，更容易问题分析
"""
g_data_fetch_mode = EMarketDataFetchMode.E_DATA_FETCH_NORMAL

"""是否开启ipython example 环境，默认关闭False"""
_g_enable_example_env_ipython = False


def enable_example_env_ipython(show_log=True, check_cn=True):
    """
    只为在ipython example 环境中运行与书中一样的数据，即读取RomDataBu/csv下的数据

    初始内置在RomDataBu/csv.zip下的数据只有zip压缩包，因为git上面的文件最好不要超过50m，
    内置测试数据，包括美股，a股，期货，比特币，港股数据初始化在csv.zip中，通过解压zip
    之后将测试数据为csv(老版本都是使用hdf5，但windows用户有些hdf5环境有问题)
    show_log: 是否显示enable example env will only read RomDataBu/df_kl.h5
    check_cn: 是否检测运行环境有中文路径
    """

    if not os.path.exists(g_project_kl_df_data_example):
        # 如果还没有进行解压，开始解压csv.zip
        data_example_zip = os.path.join(_p_dir, 'RomDataBu/csv.zip')
        try:
            from zipfile import ZipFile
            zip_csv = ZipFile(data_example_zip, 'r')
            unzip_dir = os.path.join(_p_dir, 'RomDataBu/')
            for csv in zip_csv.namelist():
                zip_csv.extract(csv, unzip_dir)
            zip_csv.close()
        except Exception as e:
            # 解压测试数据zip失败，就不开启测试数据模式了
            print('example env failed! e={}'.format(e))
            return

    global _g_enable_example_env_ipython, g_data_fetch_mode
    _g_enable_example_env_ipython = True
    g_data_fetch_mode = EMarketDataFetchMode.E_DATA_FETCH_FORCE_LOCAL
    if check_cn:
        try:
            from ..UtilBu.ABuStrUtil import str_is_cn, to_unicode
            if str_is_cn(str(__file__)):
                # 检测到运行环境路径中含有中文，严重错误，将出错，使用中文警告
                msg = u'严重错误！当前运行环境下有中文路径，abu将无法正常运行！请不要使用中文路径名称, 当前环境为{}'.format(
                    to_unicode(str(__file__)))
                logging.info(msg)
                return
        except:
            # 没有必要显示log给用户，如果是其它编码的字符路径会进到这里
            # logging.exception(e)
            msg = 'error！non English characters in the current running environment,abu will not work properly!'
            logging.info(msg)
    if show_log:
        logging.info('enable example env will only read RomDataBu/csv')


def disable_example_env_ipython(show_log=True):
    """
    只为在ipython example 环境中运行与书中一样的数据。，即读取RomDataBu/df_kl.h5下的数据
    show_log: 是否显示disable example env
    """
    global _g_enable_example_env_ipython, g_data_fetch_mode
    _g_enable_example_env_ipython = False
    g_data_fetch_mode = EMarketDataFetchMode.E_DATA_FETCH_NORMAL
    if show_log:
        logging.info('disable example env')


class EDataCacheType(Enum):
    """
        金融时间序列数据缓存类型
    """

    """读取及写入最快 但非固态硬盘写入慢，存贮空间需要大"""
    E_DATA_CACHE_HDF5 = 0
    """读取及写入最慢 但非固态硬盘写速度还可以，存贮空间需要小"""
    E_DATA_CACHE_CSV = 1
    """适合分布式扩展，存贮空间需要大"""
    E_DATA_CACHE_MONGODB = 2


# """默认金融时间序列数据缓存类型为HDF5，单机固态硬盘推荐HDF5，非固态硬盘使用CSV，否则量大后hdf5写入速度无法接受"""
# g_data_cache_type = EDataCacheType.E_DATA_CACHE_HDF5
"""对外版本由于用户电脑性能，存储空间且winodws用户，python2用户多，所以更改默认存储类型为csv"""
g_data_cache_type = EDataCacheType.E_DATA_CACHE_CSV

"""csv模式下的存储路径"""
g_project_kl_df_data_csv = path.join(g_project_data_dir, 'csv')

# ＊＊＊＊＊＊＊＊＊＊＊＊＊＊＊＊＊＊＊＊ 数据源 end   ＊＊＊＊＊＊＊＊＊＊＊＊＊＊＊＊

# ＊＊＊＊＊＊＊＊＊＊＊＊＊＊＊＊＊＊＊＊＊＊＊特征快照切割 start ＊＊＊＊＊＊＊＊＊＊＊＊＊＊＊＊＊＊＊＊＊＊＊＊＊＊＊＊＊
"""是否开启机器学习特征收集, 开启后速度会慢，默认关闭False"""
g_enable_ml_feature = False

"""是否开启买入订单前生成k线图快照，默认关闭False"""
g_enable_take_kl_snapshot = False

"""是否开启选股切割训练集股票数据与测试集股票数据，默认关闭False"""
g_enable_train_test_split = False

"""是否开启选股使用上一次切割完成的测试集股票数据，默认关闭False"""
g_enable_last_split_test = False

"""是否开启选股使用上一次切割完成的训练集股票数据，默认关闭False"""
g_enable_last_split_train = False

"""选股切割训练集股票数据与测试集股票数据切割参数n_folds，默认10"""
g_split_tt_n_folds = 10
# ＊＊＊＊＊＊＊＊＊＊＊＊＊＊＊＊＊＊＊＊＊＊＊特征快照切割 end ＊＊＊＊＊＊＊＊＊＊＊＊＊＊＊＊＊＊＊＊＊＊＊＊＊＊＊＊＊


# ＊＊＊＊＊＊＊＊＊＊＊＊＊＊＊＊＊＊＊＊＊＊＊主裁 start ＊＊＊＊＊＊＊＊＊＊＊＊＊＊＊＊＊＊＊＊＊＊＊＊＊＊＊＊＊
# TODO 内置ump的设置move到ABuUmpManager中

"""是否开启裁判拦截机制: 主裁deg，默认关闭False"""
g_enable_ump_main_deg_block = False
"""是否开启裁判拦截机制: 主裁jump，默认关闭False"""
g_enable_ump_main_jump_block = False
"""是否开启裁判拦截机制: 主裁price，默认关闭False"""
g_enable_ump_main_price_block = False
"""是否开启裁判拦截机制: 主裁wave，默认关闭False"""
g_enable_ump_main_wave_block = False
# ＊＊＊＊＊＊＊＊＊＊＊＊＊＊＊＊＊＊＊＊＊＊＊主裁 end ＊＊＊＊＊＊＊＊＊＊＊＊＊＊＊＊＊＊＊＊＊＊＊＊＊＊＊＊＊

# ＊＊＊＊＊＊＊＊＊＊＊＊＊＊＊＊＊＊＊＊＊＊＊边裁 start ＊＊＊＊＊＊＊＊＊＊＊＊＊＊＊＊＊＊＊＊＊＊＊＊＊＊＊＊＊

"""是否开启裁判拦截机制: 边裁deg，默认关闭False"""
g_enable_ump_edge_deg_block = False

"""是否开启裁判拦截机制: 边裁price，默认关闭False"""
g_enable_ump_edge_price_block = False

"""是否开启裁判拦截机制: 边裁wave，默认关闭False"""
g_enable_ump_edge_wave_block = False
"""是否开启裁判拦截机制: 边裁full，默认关闭False"""
g_enable_ump_edge_full_block = False


# ＊＊＊＊＊＊＊＊＊＊＊＊＊＊＊＊＊＊＊＊＊＊＊边裁 end ＊＊＊＊＊＊＊＊＊＊＊＊＊＊＊＊＊＊＊＊＊＊＊＊＊＊＊＊＊


#  ＊＊＊＊＊＊＊＊＊＊＊＊＊＊＊＊＊＊＊＊ 日志 start ＊＊＊＊＊＊＊＊＊＊＊＊＊＊＊＊
# TODO 将log抽出来从env中
def init_logging():
    """
    logging相关初始化工作，配置log级别，默认写入路径，输出格式
    """
    if g_is_ipython and not g_is_py3:
        """ipython在python2的一些版本需要reload logging模块，否则不显示log信息"""
        # noinspection PyUnresolvedReferences, PyCompatibility
        reload(logging)
        # pass

    if not os.path.exists(g_project_log_dir):
        # 创建log文件夹
        os.makedirs(g_project_log_dir)

    # 输出格式规范
    # file_handler = logging.FileHandler(g_project_log_info, 'a', 'utf-8')
    logging.basicConfig(level=logging.DEBUG,
                        format='%(asctime)s %(filename)s[line:%(lineno)d] %(levelname)s %(message)s',
                        datefmt='%a, %d %b %Y %H:%M:%S',
                        filename=g_project_log_info,
                        filemode='a'
                        # handlers=[file_handler]
                        )

    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    # 屏幕打印只显示message
    formatter = logging.Formatter('%(message)s')
    console.setFormatter(formatter)
    logging.getLogger('').addHandler(console)


init_logging()

#  ＊＊＊＊＊＊＊＊＊＊＊＊＊＊＊＊＊＊＊＊ 日志 end ＊＊＊＊＊＊＊＊＊＊＊＊＊＊＊＊

g_plt_figsize = (14, 7)


def init_plot_set():
    """全局plot设置"""
    import seaborn as sns
    sns.set_context('notebook', rc={'figure.figsize': g_plt_figsize})
    sns.set_style("darkgrid")

    import matplotlib
    # conda 5.0后需要添加单独matplotlib的figure设置否则pandas的plot size不生效
    matplotlib.rcParams['figure.figsize'] = g_plt_figsize


init_plot_set()
