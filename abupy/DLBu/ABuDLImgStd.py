# -*- encoding:utf-8 -*-
"""
    深度学习工具模块，为caffe等工具库标准化图片格式
"""

from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import glob
import imghdr
import os

import PIL.Image
from PIL import ImageFile

# noinspection PyUnresolvedReferences
from ..CoreBu.ABuFixes import map

from ..UtilBu import ABuFileUtil

__all__ = ['std_img_from_root_dir',
           'covert_to_jpeg',
           'find_img_by_ext',
           'change_to_real_type'
           ]

__author__ = '阿布'
__weixin__ = 'abu_quant'

# 为了im.convert('RGB')的异常错误，需要设置ImageFile.LOAD_TRUNCATED_IMAGES = True
ImageFile.LOAD_TRUNCATED_IMAGES = True


def std_img_from_root_dir(root_dir, a_ext):
    """
    找到root_dir目录下所有.ext的图片，都转换其真实类型后，
    将不是jpeg的全部转换为jpeg，即标准化图像格式
    :param root_dir: str对象，文件夹路径信息
    :param a_ext: 图像文件后缀，eg. png，jpg
    """

    # root_dir目录下及一级子目录下所有.ext后缀的文件全路径返回
    img_list = find_img_by_ext(a_ext, root_dir)
    # img_list中所有的转换为其真实类型
    all_type = change_to_real_type(img_list)
    for ext in all_type:
        # 迭代change_to_real_type返回的集合
        if ext != 'jpeg':
            # 标准化所有输入图片，将不是jpeg的都转换为jpeg
            if ext is None:
                ext = a_ext
            # 找到非标准类型子序列sub_img_list
            sub_img_list = find_img_by_ext(ext, root_dir)
            # 一个个进行转换
            _ = list(map(lambda img: covert_to_jpeg(img), sub_img_list))
            # 然后再次转换其真实后缀名
            change_to_real_type(sub_img_list)


def covert_to_jpeg(org_img, dst_img=None):
    """
    将输入img转换为RGB的jpeg格式图像
    :param org_img: 原始图像路径，str对象
    :param dst_img: 转换后的图像路径输出路径，str对象，默认=None将覆盖org_img路径
    """
    im = PIL.Image.open(org_img)
    if dst_img is None:
        # 不设置输出路径，即在输入文件上直接覆盖
        dst_img = org_img
    im.convert('RGB').save(dst_img, 'JPEG')


def find_img_by_ext(ext, root_dir):
    """
    将root_dir目录下及一级子目录下所有.ext后缀的文件全路径返回,
    注意只遍历根目录及一级子目录
    :param ext: str对象 图像文件后缀，eg. png，jpg
    :param root_dir: str对象，文件夹路径信息
    :return: list序列对象
    """

    # 遍历一级子目录
    dirs = [root_dir + name for name in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, name))]
    # 形成根目录及一级子目录
    dirs.append(root_dir)
    img_list = list()
    for dr in dirs:
        # 遍历所有目录，通过glob模糊查询所有.ext文件
        sub_list = glob.glob('{}/*.{}'.format(dr, ext))
        img_list.extend(sub_list)
    return img_list


def change_to_real_type(img_list):
    """
    将img的后缀名转换为其真实类型
        eg. a.png 如果 a.png实际上是jpeg，则将后缀修改－> a.jpeg
    :param img_list:
    :return:
    """
    # 类型记录集合
    record_type = set()
    for img in img_list:
        if not ABuFileUtil.file_exist(img):
            # 过滤实际不存在的文件
            continue

        # 使用imghdr识别图像真实类型
        real_type = imghdr.what(img)
        # 将img_list有的类型做记录，add到集合中
        record_type.add(real_type)
        if real_type is None:
            # 识别不出来，可能就是残图，过滤
            continue
        # 修改成真实的类型
        real_name = img[:img.rfind('.')] + '.' + real_type
        os.rename(img, real_name)
    return record_type
