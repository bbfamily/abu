# -*- encoding:utf-8 -*-
"""
    深度学习工具模块，为caffe工具库做数据集准备，
    切割训练集
"""

from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

from collections import defaultdict
import itertools
import os

__author__ = '阿布'
__weixin__ = 'abu_quant'

__all__ = ['train_val_split']


def train_val_split(train_path, n_folds=10):
    """将caffe返回的数据集文件进行切割工作，切割为训练集，交叉测试集，测试集数据文件"""

    if n_folds <= 1:
        raise ValueError('n_folds must > 1')

    with open(train_path, 'r') as f:
        lines = f.readlines()
        class_dict = defaultdict(list)
        for line in lines:
            cs = line[line.rfind(' '):]
            class_dict[cs].append(line)

    train = list()
    val = list()
    for cs in class_dict:
        cs_len = len(class_dict[cs])
        val_cnt = int(cs_len / n_folds)
        val.append(class_dict[cs][:val_cnt])
        train.append(class_dict[cs][val_cnt:])
    val = list(itertools.chain.from_iterable(val))
    train = list(itertools.chain.from_iterable(train))
    test = [t.split(' ')[0] + '\n' for t in val]

    # 在参数的train_path同目录下写train_split.txt
    fn = os.path.dirname(train_path) + '/train_split.txt'
    with open(fn, 'wb') as f:
        f.writelines(train)
    # 在参数的train_path同目录下写val_split.txt
    fn = os.path.dirname(train_path) + '/val_split.txt'
    with open(fn, 'wb') as f:
        f.writelines(val)
    # 在参数的train_path同目录下写/test_split.txt
    fn = os.path.dirname(train_path) + '/test_split.txt'
    with open(fn, 'wb') as f:
        f.writelines(test)
