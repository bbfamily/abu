# -*- encoding:utf-8 -*-
"""
    买入卖出因子与ump进行组织管理通信模块
"""
from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

from ..UtilBu.ABuLazyUtil import LazyFunc
from ..UtilBu.ABuFileUtil import file_exist
from ..UtilBu.ABuDelegateUtil import first_delegate_has_method, replace_word_delegate_has_method
from ..UmpBu.ABuUmpEdgeBase import EEdgeType
from ..UmpBu.ABuUmpEdgeDeg import AbuUmpEdgeDeg
from ..UmpBu.ABuUmpEdgeFull import AbuUmpEdgeFull
from ..UmpBu.ABuUmpEdgePrice import AbuUmpEdgePrice
from ..UmpBu.ABuUmpEdgeWave import AbuUmpEdgeWave
from ..UmpBu.ABuUmpMainDeg import AbuUmpMainDeg
from ..UmpBu.ABuUmpMainJump import AbuUmpMainJump
from ..UmpBu.ABuUmpMainPrice import AbuUmpMainPrice
from ..UmpBu.ABuUmpMainWave import AbuUmpMainWave
from ..UmpBu.ABuUmpMainBase import AbuUmpMainBase
from ..CoreBu import ABuEnv
# noinspection PyUnresolvedReferences
from ..CoreBu.ABuFixes import filter, six

__author__ = '阿布'
__weixin__ = 'abu_quant'

"""外部用户使用append_user_ump添加到外部ump list容器中"""
_g_extend_ump_list = list()
"""是否启用外部用户使用append_user_ump添加的ump对交易进行拦截决策"""
g_enable_user_ump = False


def append_user_ump(ump, check=True):
    """
    外部用户设置扩展ump接口, 参数ump可以是ump class类型，也可以是实例化后的ump object
    :param ump: 可以是ump class类型，也可以是实例化后的ump object
    :param check: 是否检测ump是否以及训练好，且生成了cache文件
    """
    if check:
        # 检测ump训练后的本地物理文件是否存在
        if isinstance(ump, six.class_types):
            ump_cache_path = ump(predict=True).dump_file_fn()
        else:
            ump_cache_path = ump.dump_file_fn()
        if not file_exist(ump_cache_path):
            # 如果用户添加的ump没有经过训练集训练，提升用户进行训练
            raise RuntimeError('you must first fit orders, {} is not exist!!'.format(ump_cache_path))
    _g_extend_ump_list.append(ump)


def clear_user_ump():
    """
    将外部用户设置的ump队列清空
    :return:
    """
    global _g_extend_ump_list
    _g_extend_ump_list = list()


class AbuUmpManager(object):
    """ump组织管理类"""

    def __init__(self, factor):
        """
        在AbuFactorBuyBase __init__和AbuFactorSellBase __init__函数中构造AbuUmpManager，
        外部用户不应有构造

        :param factor: 买入或者卖出因子对象，AbuFactorBuyBase or AbuFactorSellBase类对象实例
        """

        # 在买入或卖出因子__init__ 中self.ump_manger = AbuUmpManager(self)，即因子和AbuUmpManager互持op
        self.factor = factor
        from ..FactorBuyBu import AbuFactorBuyBase
        self.is_buy_factor = isinstance(self.factor, AbuFactorBuyBase)

        self.extend_ump_list = list()
        if g_enable_user_ump and len(_g_extend_ump_list) > 0:
            # 筛选与因子对应的ump，即买入因子AbuFactorBuyBase对象只筛选买入BuyUmpMixin类型的ump
            filter_ump = list(filter(lambda ump: self.is_buy_factor == ump.is_buy_ump(), _g_extend_ump_list))
            self.extend_ump_list.extend(filter_ump)
        self._fix_ump_env()

    def __str__(self):
        """打印对象显示：class name, factor, extend_ump_list"""
        return '{}: factor={}, self.extend_ump_list={}'.format(self.__class__.__name__, self.factor,
                                                               self.extend_ump_list)

    __repr__ = __str__

    @classmethod
    def _fix_ump_env(cls):
        """
        只为修复在ABuEnv中对ump的设置和manger中的ump设置不同步问题，即ABuEnv.g_enable_ml_feature
        设置不正确，将env中ump的设置迁移到本模块中后便不需要fix了，外部user不应主动使用，只在manger构造使用
        """
        if ABuEnv.g_enable_ump_edge_deg_block or ABuEnv.g_enable_ump_edge_price_block or \
                ABuEnv.g_enable_ump_edge_wave_block or ABuEnv.g_enable_ump_edge_full_block:
            ABuEnv.g_enable_ml_feature = True
        if ABuEnv.g_enable_ump_main_deg_block or ABuEnv.g_enable_ump_main_jump_block or \
                ABuEnv.g_enable_ump_main_price_block or ABuEnv.g_enable_ump_main_wave_block:
            ABuEnv.g_enable_ml_feature = True
        # TODO 将env中ump的设置迁移到本模块

        if g_enable_user_ump:
            ABuEnv.g_enable_ml_feature = True

    @LazyFunc
    def ump_main_deg(self):
        """
            根据ABuEnv.g_enable_ump_main_deg_block设置决定是否构造AbuUmpMainDeg，
            使用LazyFunc装饰器，构造后即使用缓存AbuUmpMainDeg(predict=True)实例
        """
        return AbuUmpMainDeg(predict=True)

    @LazyFunc
    def ump_main_jump(self):
        """
            根据ABuEnv.g_enable_ump_main_jump_block设置决定是否构造AbuUmpMainJump，
            使用LazyFunc装饰器，构造后即使用缓存AbuUmpMainJump(predict=True)实例
        """
        return AbuUmpMainJump(predict=True)

    @LazyFunc
    def ump_main_price(self):
        """
            根据ABuEnv.g_enable_ump_main_price_block设置决定是否构造AbuUmpMainPrice，
            使用LazyFunc装饰器，构造后即使用缓存AbuUmpMainPrice(predict=True)实例
        """
        return AbuUmpMainPrice(predict=True)

    @LazyFunc
    def ump_main_wave(self):
        """
            根据ABuEnv.g_enable_ump_main_wave_block设置决定是否构造AbuUmpMainWave，
            使用LazyFunc装饰器，构造后即使用缓存AbuUmpMainWave(predict=True)实例
        """
        return AbuUmpMainWave(predict=True)

    @LazyFunc
    def ump_edge_deg(self):
        """
            根据ABuEnv.g_enable_ump_edge_deg_block设置决定是否构造AbuUmpEdgeDeg，
            使用LazyFunc装饰器，构造后即使用缓存AbuUmpEdgeDeg(predict=True)实例
        """
        return AbuUmpEdgeDeg(predict=True)

    @LazyFunc
    def ump_edge_price(self):
        """
            根据ABuEnv.g_enable_ump_edge_price_block设置决定是否构造AbuUmpEdgePrice，
            使用LazyFunc装饰器，构造后即使用缓存AbuUmpEdgePrice(predict=True)实例
        """
        return AbuUmpEdgePrice(predict=True)

    @LazyFunc
    def ump_edge_wave(self):
        """
            根据ABuEnv.g_enable_ump_edge_wave_block设置决定是否构造AbuUmpEdgeWave，
            使用LazyFunc装饰器，构造后即使用缓存AbuUmpEdgeWave(predict=True)实例
        """
        return AbuUmpEdgeWave(predict=True)

    @LazyFunc
    def ump_edge_full(self):
        """
            根据ABuEnv.g_enable_ump_edge_full_block设置决定是否构造AbuUmpEdgeFull，
            使用LazyFunc装饰器，构造后即使用缓存AbuUmpEdgeFull(predict=True)实例
        """
        return AbuUmpEdgeFull(predict=True)

    # noinspection PyMethodMayBeStatic
    def _default_main_hit_cnt(self):
        # noinspection PyProtectedMember
        default_hit_cnt = 1 if ABuEnv._g_enable_example_env_ipython else 2
        return default_hit_cnt

    @first_delegate_has_method(delegate='factor')
    def ump_main_deg_hit_cnt(self):
        """
        角度主裁使用predict_kwargs的参数need_hit_cnt值，即:
                predict_kwargs(need_hit_cnt=self.ump_main_deg_hit_cnt(), **ml_feature_dict)

        被装饰器first_delegate_has_method(delegate='factor')装饰，当被委托的因子，即self.factor中
        有对应实现的ump_main_deg_hit_cnt方法时，返回self.factor.ump_main_deg_hit_cnt()的返回值
        :return: int
        """
        return self._default_main_hit_cnt()

    @first_delegate_has_method(delegate='factor')
    def ump_main_jump_hit_cnt(self):
        """
        跳空主裁使用predict_kwargs的参数need_hit_cnt值，即:
                predict_kwargs(need_hit_cnt=self.ump_main_jump_hit_cnt, **ml_feature_dict)

        被装饰器first_delegate_has_method(delegate='factor')装饰，当被委托的因子，即self.factor中
        有对应实现的ump_main_jump_hit_cnt方法时，返回self.factor.ump_main_jump_hit_cnt()的返回值
        :return: int
        """
        return self._default_main_hit_cnt()

    @first_delegate_has_method(delegate='factor')
    def ump_main_price_hit_cnt(self):
        """
        价格主裁使用predict_kwargs的参数need_hit_cnt值，即:
                predict_kwargs(need_hit_cnt=self.ump_main_price_hit_cnt, **ml_feature_dict)

        被装饰器first_delegate_has_method(delegate='factor')装饰，当被委托的因子，即self.factor中
        有对应实现的ump_main_price_hit_cnt方法时，返回self.factor.ump_main_price_hit_cnt()的返回值
        :return: int
        """
        return self._default_main_hit_cnt()

    @first_delegate_has_method(delegate='factor')
    def ump_main_wave_hit_cnt(self):
        """
        价格波动主裁使用predict_kwargs的参数need_hit_cnt值，即:
                predict_kwargs(need_hit_cnt=self.ump_main_wave_hit_cnt, **ml_feature_dict)

        被装饰器first_delegate_has_method(delegate='factor')装饰，当被委托的因子，即self.factor中
        有对应实现的ump_main_wave_hit_cnt方法时，返回self.factor.ump_main_wave_hit_cnt()的返回值
        :return: int
        """
        return self._default_main_hit_cnt()

    # noinspection PyMethodMayBeStatic
    def ump_main_user_hit_cnt(self):
        """
        用户自定义的主裁ump类的使用predict_kwargs的参数need_hit_cnt值，
        用户在因子中可通过实现特点的方法名称来替换ump_main_user_hit_cnt值，

        使用时使用replace_word_delegate_has_method检测self.factor中有没有对应的方法，
        eg:
            replace_hit_cnt = replace_word_delegate_has_method(delegate='factor', key_word='user',
                                                                       replace_word=class_unique_id)
            hit_cnt = replace_hit_cnt(self.ump_main_user_hit_cnt)()

            即如果用户编写的主裁ump中class_unique_id方法返回'extend_main_test'
                @classmethod
                def class_unique_id(cls):
                    return 'extend_main_test'
            则在因子中对应自定义hit_cnt的方法名称应为：
                def ump_main_extend_main_test_hit_cnt(self)
                    return 1
            更多具体实现阅extend_ump_block以及replace_word_delegate_has_method方法的实现

        :return: int
        """
        return self._default_main_hit_cnt()

    def ump_block(self, ml_feature_dict):
        """
        在买入或者卖出因子中make_ump_block_decision方法中使用，决策特定交易是否被拦截，
        ump_block中首先使用内置ump进行拦截决策，如果不被拦截，使用外部定义的ump进行拦截决策

        :param ml_feature_dict: 交易所形成的特征字典
                eg: ml_feature_dict
                    {'buy_deg_ang42': -0.45400000000000001, 'buy_deg_ang252': 5.532,
                    'buy_deg_ang60': 2.1419999999999999, 'buy_deg_ang21': 0.93100000000000005,
                    'buy_price_rank120': 1.0, 'buy_price_rank90': 1.0, 'buy_price_rank60': 1.0,
                    'buy_price_rank252': 1.0, 'buy_wave_score1': 1.2470000000000001, 'buy_wave_score2': 1.286,
                    'buy_wave_score3': 1.2849999999999999, 'buy_atr_std': 0.19400000000000001,
                    'buy_jump_down_power': -13.57, 'buy_diff_down_days': 136, 'buy_jump_up_power': 1.038,
                    'buy_diff_up_days': 2}
        :return: bool, 对ml_feature_dict所描述的交易特征是否进行拦截
        """
        # 内置ump进行拦截决策
        if self.builtin_ump_block(ml_feature_dict):
            return True

        # 外部定义的ump进行拦截决策
        if self.extend_ump_block(ml_feature_dict):
            return True
        return False

    def extend_ump_block(self, ml_feature_dict):
        """
        外部用户设置的ump进行拦截决策，迭代self.extend_ump_list中外部设置的ump，
        由于对外添加ump的接口append_user_ump中参数ump可以是ump class类型，
        也可以是实例化后的ump object，所以需要把class类型的ump进行实例构造，且将
        实例的ump对象缓存在类变量中（通过class_unique_id为类变量构造唯一名称），
        ump对象构造好后根据主裁还是边裁选择决策方法：
                主裁使用：predict_kwargs(need_hit_cnt=need_hit_cnt, **ml_feature_dict)
                边裁使用：predict(**ml_feature_dict) == EEdgeType.E_EEdge_TOP_LOSS
        对交易进行拦截决策

        :param ml_feature_dict: 交易所形成的特征字典
        eg: ml_feature_dict
            {'buy_deg_ang42': -0.45400000000000001, 'buy_deg_ang252': 5.532,
            'buy_deg_ang60': 2.1419999999999999, 'buy_deg_ang21': 0.93100000000000005,
            'buy_price_rank120': 1.0, 'buy_price_rank90': 1.0, 'buy_price_rank60': 1.0,
            'buy_price_rank252': 1.0, 'buy_wave_score1': 1.2470000000000001, 'buy_wave_score2': 1.286,
            'buy_wave_score3': 1.2849999999999999, 'buy_atr_std': 0.19400000000000001,
            'buy_jump_down_power': -13.57, 'buy_diff_down_days': 136, 'buy_jump_up_power': 1.038,
            'buy_diff_up_days': 2}
        :return: bool, 对ml_feature_dict所描述的交易特征是否进行拦截
        """
        for extend_ump in self.extend_ump_list:
            class_unique_id = extend_ump.class_unique_id()
            # 由于对外添加ump的接口append_user_ump中参数ump可以是ump class类型，也可以是实例化后的ump object
            if isinstance(extend_ump, six.class_types):
                # 把class类型的ump进行实例构造

                is_main_ump = issubclass(extend_ump, AbuUmpMainBase)
                main_ump_key = 'main' if is_main_ump else 'edge'

                # 通过class_unique_id和issubclass(extend_ump, AbuUmpMainBase)为类变量构造唯一名称
                extend_ump_attr_str = 'ump_{}_{}'.format(main_ump_key, class_unique_id)
                if hasattr(self, extend_ump_attr_str):
                    # 将类变量中的实例代替类
                    extend_ump = getattr(self, extend_ump_attr_str)
                else:
                    # 内置ump通过LazyFunc进行效率提升，外部设置的ump通过手动setattr，将实例的ump对象缓存在类变量中
                    extend_ump_obj = extend_ump(predict=True)
                    setattr(self, extend_ump_attr_str, extend_ump_obj)
                    # 将实例化后的实例代替类
                    extend_ump = extend_ump_obj

            is_main_ump = isinstance(extend_ump, AbuUmpMainBase)
            if is_main_ump:
                # replace_word_delegate_has_method不做装饰器修饰ump_main_user_hit_cnt，因为需要动态获取replace_word
                try:
                    replace_hit_cnt = replace_word_delegate_has_method(delegate='factor', key_word='user',
                                                                       replace_word=class_unique_id)
                    hit_cnt = replace_hit_cnt(self.ump_main_user_hit_cnt)()
                except:
                    # 忽略用户自定义factor中关于hit_cnt的任何错误
                    hit_cnt = self.ump_main_user_hit_cnt()
                if extend_ump.predict_kwargs(need_hit_cnt=hit_cnt, **ml_feature_dict):
                    return True
            else:
                if extend_ump.predict(**ml_feature_dict) == EEdgeType.E_EEdge_TOP_LOSS:
                    return True
        return False

    def builtin_ump_block(self, ml_feature_dict):
        """
        内置ump进行拦截决策，通过ABuEnv中的拦截设置以及因子的买入卖出类型是否和ump类型匹配，
        来决定是否使用特定的ump进行拦截决策，如需要决策：

                主裁使用：predict_kwargs(need_hit_cnt=need_hit_cnt, **ml_feature_dict)
                边裁使用：predict(**ml_feature_dict) == EEdgeType.E_EEdge_TOP_LOSS
        对交易进行拦截决策

                :param ml_feature_dict: 交易所形成的特征字典
                eg: ml_feature_dict
                    {'buy_deg_ang42': -0.45400000000000001, 'buy_deg_ang252': 5.532,
                    'buy_deg_ang60': 2.1419999999999999, 'buy_deg_ang21': 0.93100000000000005,
                    'buy_price_rank120': 1.0, 'buy_price_rank90': 1.0, 'buy_price_rank60': 1.0,
                    'buy_price_rank252': 1.0, 'buy_wave_score1': 1.2470000000000001, 'buy_wave_score2': 1.286,
                    'buy_wave_score3': 1.2849999999999999, 'buy_atr_std': 0.19400000000000001,
                    'buy_jump_down_power': -13.57, 'buy_diff_down_days': 136, 'buy_jump_up_power': 1.038,
                    'buy_diff_up_days': 2}

        :return: bool, 对ml_feature_dict所描述的交易特征是否进行拦截
        """

        """内置主裁开始"""
        if ABuEnv.g_enable_ump_main_deg_block and self.is_buy_factor == self.ump_main_deg.is_buy_ump() \
                and self.ump_main_deg.predict_kwargs(need_hit_cnt=self.ump_main_deg_hit_cnt(), **ml_feature_dict):
            return True

        if ABuEnv.g_enable_ump_main_jump_block and self.is_buy_factor == self.ump_main_jump.is_buy_ump() \
                and self.ump_main_jump.predict_kwargs(need_hit_cnt=self.ump_main_jump_hit_cnt(), **ml_feature_dict):
            return True

        if ABuEnv.g_enable_ump_main_price_block and self.is_buy_factor == self.ump_main_price.is_buy_ump() \
                and self.ump_main_price.predict_kwargs(need_hit_cnt=self.ump_main_price_hit_cnt(), **ml_feature_dict):
            return True

        if ABuEnv.g_enable_ump_main_wave_block and self.is_buy_factor == self.ump_main_wave.is_buy_ump() \
                and self.ump_main_wave.predict_kwargs(need_hit_cnt=self.ump_main_wave_hit_cnt(), **ml_feature_dict):
            return True

        """内置边裁开始"""
        if ABuEnv.g_enable_ump_edge_deg_block and self.is_buy_factor == self.ump_edge_deg.is_buy_ump() \
                and self.ump_edge_deg.predict(**ml_feature_dict) == EEdgeType.E_EEdge_TOP_LOSS:
            return True

        if ABuEnv.g_enable_ump_edge_price_block and self.is_buy_factor == self.ump_edge_price.is_buy_ump() \
                and self.ump_edge_price.predict(**ml_feature_dict) == EEdgeType.E_EEdge_TOP_LOSS:
            return True

        if ABuEnv.g_enable_ump_edge_wave_block and self.is_buy_factor == self.ump_edge_wave.is_buy_ump() \
                and self.ump_edge_wave.predict(**ml_feature_dict) == EEdgeType.E_EEdge_TOP_LOSS:
            return True

        if ABuEnv.g_enable_ump_edge_full_block and self.is_buy_factor == self.ump_edge_full.is_buy_ump() \
                and self.ump_edge_full.predict(**ml_feature_dict) == EEdgeType.E_EEdge_TOP_LOSS:
            return True

        return False
