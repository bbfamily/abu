# coding=utf-8

import os
import sys
from os import path
import logging
import shutil
from setuptools import setup, find_packages

import abupy

DIST_NAME = 'abupy'
LICENSE = 'GPL'
AUTHOR = u"阿布"
EMAIL = "service@abuquant.com"
URL = "http://abuquant.com/"
DOWNLOAD_URL = 'https://github.com/bbfamily/abu'
CLASSIFIERS = []

DESCRIPTION = "阿布量化系统"
LONG_DESCRIPTION = """
**abu追求的是一句话就能够说明的智能策略**
abu能够帮助用户自动完善策略，主动分析策略产生的交易行为，智能拦截策略生成的容易失败的交易单。

现阶段的量化策略还是人工编写的代码，abu量化交易系统的设计将会向着由计算机自动实现整套流程的方向迈进，包括编写量化策略本身。

我们对未来的期望是：abupy用户只需要提供一些简单的种子策略，计算机在这些种子基础上不断自我学习、自我成长，创造出新的策略，并且随着时间序列数据不断智能调整策略的参数。

### 特点

* 使用多种机器学习技术智能优化策略
* 在实盘中指导策略进行交易，提高策略的实盘效果，战胜市场

### 支持的投资市场:

* 美股，A股，港股
* 期货，期权
* 比特币，莱特币

### 工程设计目标：

* 分离基础策略和策略优化监督模块
* 提高灵活度和适配性

"""
KEY_WORDS = ['阿布', 'abu', 'quant', 'quantization',
             'crawler', 'spider', 'scrapy', 'stock', 'machine learning',
             '股票', '机器学习', '量化', '爬虫']


def init_mete_data():
    mete_data = dict()
    mete_data['name'] = DIST_NAME
    mete_data['version'] = abupy.__version__
    mete_data['url'] = URL
    mete_data['download_url'] = DOWNLOAD_URL
    mete_data['description'] = DESCRIPTION
    mete_data['long_description'] = LONG_DESCRIPTION
    mete_data['author'] = AUTHOR
    mete_data['license'] = LICENSE
    mete_data['author_email'] = EMAIL
    mete_data['platforms'] = 'any'
    mete_data['keywords'] = KEY_WORDS

    mete_data['packages'] = list(filter(lambda pack: pack.startswith(DIST_NAME), find_packages()))
    mete_data['package_data'] = {'abupy': ['RomDataBu/*.txt',
                                           'RomDataBu/*.db',
                                           'RomDataBu/*.csv',
                                           'RomDataBu/*.zip',
                                           'RomDataBu/us_industries']}

    mete_data['install_requires'] = ['numpy',
                                     'pandas',
                                     'scipy',
                                     'scikit-learn',
                                     'matplotlib',
                                     'seaborn',
                                     'statsmodels',
                                     'requests']

    return mete_data


def clear_build():
    from abupy.UtilBu.ABuFileUtil import file_exist
    cur_path = path.dirname(path.abspath(path.realpath(__file__)))
    if file_exist(path.join(cur_path, 'dist')):
        shutil.rmtree('dist')
    if file_exist(path.join(cur_path, 'build')):
        shutil.rmtree('build')
    if file_exist(path.join(cur_path, 'abupy.egg-info')):
        shutil.rmtree('abupy.egg-info')


def uninstall():
    pass


def install():
    uninstall()
    clear_build()
    sys.argv.append('install')
    setup(**init_mete_data())


def deploy_2_pypi():
    """
    cmd: python setup.py sdist bdist_wheel && twine upload dist/*
    """
    clear_build()
    # 源码方式
    # e.g:  dist/abupy-xxxx-.tar.gz
    sys.argv.append('sdist')
    # python wheel 方式
    sys.argv.append('bdist_wheel')
    # wheel方式 通用包，即生成py2.py3的个平台包
    # e.g: dist/abupy-xxx-py2.py3-none-any.whl
    sys.argv.append('--universal')
    setup(**init_mete_data())
    """
    upload 需要在当前用户目录创建.pypirc文件，并写入
    ###########################################
    [distutils]
    index-servers=pypi

    [pypi]
    repository = https://upload.pypi.org/legacy/
    username = 你的用户名
    password = 你的密码
    ###########################################
    """

    def deploy():
        from twine.commands import upload
        upload.main(['dist/*'])

    try:
        deploy()
    except ImportError:
        if os.system('pip install twine') != 0:
            deploy()
        else:
            logging.warning('install twine error!! must be need root permission.')


if __name__ == '__main__':
    if '2pypi' in sys.argv:
        sys.argv.remove('2pypi')
        deploy_2_pypi()
    else:
        install()
