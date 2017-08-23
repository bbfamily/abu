# -*- encoding:utf-8 -*-
"""
    ump基础模块
"""

from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

import functools
import weakref
from abc import ABCMeta, abstractmethod

import numpy as np
import sklearn.preprocessing as preprocessing

from ..UtilBu import ABuFileUtil
from ..CoreBu.ABuFixes import six

__author__ = '阿布'
__weixin__ = 'abu_quant'


class BuyUmpMixin(object):
    """
        买入ump混入, 与BuyFeatureMixin不同，每一个具体的ump只能属于一个ump类别
        即不是BuyUmpMixin就应该是SellUmpMixin
    """
    _ump_type_prefix = 'buy_'


class SellUmpMixin(object):
    """
        卖出ump混入, 与SellFeatureMixin不同，每一个具体的ump只能属于一个ump类别
        即不是BuyUmpMixin就应该是SellUmpMixin
    """
    _ump_type_prefix = 'sell_'


class UmpDict(dict):
    """Several built-in types such as list and dict do not directly support weak references
     but can add support through subclassing:"""
    pass


class CachedUmpManager:
    """ump拦截缓存实体，分别在主裁和边裁类中"""

    """不对外开发的设置，仅针对源代码修改，默认使用dict不使用WeakValueDictionary"""
    s_use_weak = False

    def __init__(self):
        """初始化_cache本体，根据s_use_weak决定使用WeakValueDictionary或者dict"""
        self._cache = weakref.WeakValueDictionary() if CachedUmpManager.s_use_weak else dict()

    def get_ump(self, ump):
        """
        主要在具体裁判类的predict方法中获取裁判本体使用，如果
        不在_cache中load_pickle，否则使用catch中的ump返回
        :param ump: 具体裁判对象，AbuUmpBase对象
        :return: 每个裁判需要的决策数据，每类裁判主体的数据形式不同，且使用方法不同
                 eg：主裁中的使用
                    def predict(self, x, need_hit_cnt=1):
                        dump_clf_with_ind = AbuUmpMainBase.dump_clf_manager.get_ump(self)
                        count_hit = 0
                        for clf, ind in dump_clf_with_ind.values():
                            ss = clf.predict(x)
                            if ss == ind:
                                count_hit += 1
                                if need_hit_cnt == count_hit:
                                    return 1
                        return 0
        """
        # dump_file_fn是每一个具体裁判需要复写的方法，声明自己缓存的存放路径
        name = ump.dump_file_fn()
        if name not in self._cache:
            # 不在缓存字典中load_pickle
            dump_clf = ABuFileUtil.load_pickle(name)
            if dump_clf is None:
                # 没有对ump进行训练，就直接进行拦截了，抛异常
                raise RuntimeError('{}: you must first fit orders, or {} is not exist!!'.format(
                    ump.__class__.__name__, name))
            if CachedUmpManager.s_use_weak:
                # 如果使用WeakValueDictionary模式，需要进一步使用UmpDict包一层
                dump_clf = UmpDict(**dump_clf)
            self._cache[name] = dump_clf
        else:
            # 有缓存直接拿缓存
            dump_clf = self._cache[name]
        return dump_clf

    def clear(self):
        """清除缓存中所有cache ump"""
        self._cache.clear()


def ump_main_make_xy(func):
    """
    主裁中对应fiter class中make_xy的装饰器，
    使用eg：详阅ABuUmpMainDeg或其它主裁子类实现

        class AbuUmpMainDeg(AbuUmpMainBase, BuyUmpMixin):
            class UmpDegFiter(AbuMLPd):
                @ump_main_make_xy
                def make_xy(self, **kwarg):
                    regex = 'result|{}'.format(
                        '|'.join(AbuFeatureDeg().get_feature_ump_keys(ump_cls=AbuUmpMainDeg)))
                    # noinspection PyUnresolvedReferences
                    deg_df = self.order_has_ret.filter(regex=regex)
                    return deg_df
    """
    @functools.wraps(func)
    def wrapper(self, *args, **kwargs):
        """
        kwargs中必须的参数为：kwargs['orders_pd']，在AbuUmpMainBase初始化__init__中
        将orders_pd=self.orders_pd，eg：
            self.fiter = self.fiter_cls(orders_pd=self.orders_pd, **kwarg)

        kwargs中可选的参数为：kwargs['scaler']，bool类型，默认不传递为false，控制在
        make_xy中返回的特征矩阵数据是否进行标准化处理
        """
        if kwargs is None or 'orders_pd' not in kwargs:
            raise ValueError('kwarg is None or not kwarg.has_key orders_pd')

        orders_pd = kwargs['orders_pd']
        # 从orders_pd中筛选有交易结果形成order_has_ret
        order_has_ret = orders_pd[orders_pd['result'] != 0]

        # 之前的交易结果－1为loss，1为win，0为keep状态，order_has_ret没有keep状态，所以转换为loss：0，win：1
        # noinspection PyTypeChecker
        order_has_ret['result'] = np.where(order_has_ret['result'] == -1, 0, 1)
        self.order_has_ret = order_has_ret
        # 通过被装饰的make_xy方法，筛选具体裁判需要的特征形成特征矩阵ump_df
        ump_df = func(self, *args, **kwargs)

        if 'scaler' in kwargs and kwargs['scaler'] is True:
            # 控制在make_xy中返回的特征矩阵数据是否进行标准化处理
            scaler = preprocessing.StandardScaler()
            for col in ump_df.columns[1:]:
                ump_df[col] = scaler.fit_transform(ump_df[col].values.reshape(-1, 1))
        # 转换为matrix，形成x，y
        ump_np = ump_df.as_matrix()
        self.y = ump_np[:, 0]
        self.x = ump_np[:, 1:]
        # 将pd.DataFrame对象ump_df也保留一份
        self.df = ump_df
        self.np = ump_np

    return wrapper


def ump_edge_make_xy(func):
    """
    边裁中对应fiter class中make_xy的装饰器，
    使用eg：详阅AbuUmpEdgeDeg或其它边裁子类实现

        class AbuUmpEdgeDeg(AbuUmpEdgeBase, BuyUmpMixin):
            class UmpDegFiter(AbuMLPd):
                @ump_edge_make_xy
                def make_xy(self, **kwarg):
                    filter_list = ['profit', 'profit_cg']
                    cols = AbuFeatureDeg().get_feature_ump_keys(ump_cls=AbuUmpEdgeDeg)
                    filter_list.extend(cols)
                    # noinspection PyUnresolvedReferences
                    deg_df = self.order_has_ret.filter(filter_list)
                    return deg_df
    """
    @functools.wraps(func)
    def wrapper(self, *args, **kwargs):
        """
        kwargs中必须的参数为：kwargs['orders_pd']，在AbuUmpEdgeBase初始化__init__中
        将orders_pd=self.orders_pd，eg：
            self.fiter = self.fiter_cls(orders_pd=orders_pd, **kwarg)
        """
        if kwargs is None or 'orders_pd' not in kwargs:
            raise ValueError('kwarg is None or not kwarg.has_key orders_pd')

        orders_pd = kwargs['orders_pd']
        # 从orders_pd中筛选有交易结果形成orders_pd_tmp
        orders_pd_tmp = orders_pd[orders_pd['result'] != 0]
        # 从orders_pd_tmp进行二次筛选必须profit != 0
        order_has_ret = orders_pd_tmp[orders_pd_tmp['profit'] != 0]
        self.order_has_ret = order_has_ret
        # 通过被装饰的make_xy方法，筛选具体裁判需要的特征形成特征矩阵ump_df
        ump_df = func(self, *args, **kwargs)
        # 转换为matrix，形成x，y
        ump_np = ump_df.as_matrix()
        # 边裁特征中filter_list = ['profit', 'profit_cg']都设定为y
        self.y = ump_np[:, :2]
        self.x = ump_np[:, 2:]
        # 将pd.DataFrame对象ump_df也保留一份
        self.df = ump_df
        self.np = ump_np

    return wrapper


class AbuUmpBase(six.with_metaclass(ABCMeta, object)):
    """ump拦截缓存，在AbuUmpBase类中"""
    dump_clf_manager = CachedUmpManager()

    @abstractmethod
    def get_fiter_class(self):
        """abstractmethod子类必须实现，声明具体子类裁判使用的筛选特征形成特征的类"""
        pass

    @abstractmethod
    def get_predict_col(self):
        """abstractmethod子类必须实现，获取具体子类裁判需要的特征keys"""
        pass

    @classmethod
    def is_buy_ump(cls):
        """
        返回裁判本身是否是买入拦截裁判，类方法
        :return: bool，是否是买入拦截裁判
        """
        return getattr(cls, "_ump_type_prefix") == 'buy_'

    @classmethod
    @abstractmethod
    def class_unique_id(cls):
        """
        具体ump类关键字唯一名称，类方法，abstractmethod子类必须实现
        主要针对外部user设置自定义ump使用, 需要user自己保证class_unique_id的唯一性，内部不做检测
        具体使用见ABuUmpManager中extend_ump_block方法
        """
        pass

    def __str__(self):
        """打印对象显示：class name, is_buy_ump,  predict_col"""
        return '{}: is_buy_ump:{} predict_col:{}'.format(self.__class__.__name__,
                                                         self.__class__.is_buy_ump(), self.get_predict_col())

    __repr__ = __str__
