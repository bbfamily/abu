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

# TODO  start
DESCRIPTION = "强大的股票量化库"
LONG_DESCRIPTION = "abu是以时间驱动来选股，生成订单。"
KEY_WORDS = ['阿布', 'abu', 'quant', 'quantization',
             'crawler', 'spider', 'scrapy', 'stock', 'machine learning',
             '股票', '机器学习', '量化', '爬虫']


# TODO end


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
