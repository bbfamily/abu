from __future__ import print_function
from __future__ import absolute_import
from __future__ import division

import os

import pandas as pd

from ..CoreBu import ABuEnv
from ..UtilBu import ABuFileUtil

SIMILAR_CACHE_PATH = os.path.join(ABuEnv.g_project_cache_dir, 'similar.hdf5')


def similar_key(symbol, cmp_cnt=None, n_folds=None, start=None, end=None, corr_type=None):
    return '{}_{}_{}_{}_{}_{}'.format(symbol, cmp_cnt, n_folds, start, end, corr_type)


def dump_2_hdh5(key, obj):
    ABuFileUtil.dump_hdf5(SIMILAR_CACHE_PATH, obj, key)


def load_2_hdh5(key):
    return ABuFileUtil.load_hdf5(SIMILAR_CACHE_PATH, key)


def show_keys():
    with pd.HDFStore(SIMILAR_CACHE_PATH) as h5s:
        return h5s.keys()


def clear_cache(key=None):
    if key is not None:
        ABuFileUtil.del_hdf5(SIMILAR_CACHE_PATH, key)
    else:
        ABuFileUtil.del_file(SIMILAR_CACHE_PATH)
