# -*- encoding:utf-8 -*-
"""
    边裁基础实现模块
"""

from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

import logging
import os
from abc import abstractmethod

import numpy as np
import sklearn.preprocessing as preprocessing
from enum import Enum
from sklearn.metrics.pairwise import pairwise_distances
from ..CoreBu import ABuEnv
from ..UtilBu import ABuFileUtil
from ..SimilarBu.ABuCorrcoef import ECoreCorrType, corr_xy
from .ABuUmpBase import AbuUmpBase
# noinspection PyUnresolvedReferences
from ..CoreBu.ABuFixes import filter

__author__ = '阿布'
__weixin__ = 'abu_quant'

"""在predict中度量输入的x和矩阵中其它矢量的pairwise_distances后，通过if distances_cx.min() > K_DISTANCE_THRESHOLD过滤"""
K_DISTANCE_THRESHOLD = 0.668
"""从第一轮pairwise_distances的结果使用argsort后取K_N_TOP_SEED个做为第二轮相似匹配的种子"""
K_N_TOP_SEED = 100
"""完成第二轮相似度匹配后使用K_SIMILAR_THRESHOLD做为阀值过滤后得到有投票权的向量"""
K_SIMILAR_THRESHOLD = 0.91

"""
    K_CG_TOP_RATE做为计算win_top和loss_top
    win_top = len(self.fiter.df['profit_cg']) - len(self.fiter.df['profit_cg']) * K_CG_TOP_RATE
    eg:
      len(self.fiter.df['profit_cg']) == 100
        -> win_top = 100 - 100 * 0.236
        -> win_top = 100 - 23.6
        -> win_top = 76.4
    loss_top = len(self.fiter.df['profit_cg']) * K_CG_TOP_RATE
    eg:
        len(self.fiter.df['profit_cg']) == 100
        -> loss_top = 100 * 0.236
        -> loss_top = 23.6
"""
K_CG_TOP_RATE = 0.236

"""在predict中最后的投票结果需要大于一定比例才被认可, 即对有争议的投票需要一方拥有相对优势才认可"""
K_EDGE_JUDGE_RATE = 0.618


class EEdgeType(Enum):
    """对交易的利润亏损进行rank后的分类结果"""

    """损失最多的一类交易，可理解为最底端"""
    E_EEdge_TOP_LOSS = -1
    """其它的普通收益亏损的交易，在整个训练集交易中占最多数"""
    E_EEdge_NORMAL = 0
    """盈利最多的一类交易，可理解为最顶端"""
    E_STORE_TOP_WIN = 1


"""在第二轮的相似度匹配中使用的方法，传递给ABuCorrcoef.corr_xy函数"""
g_similar_type = ECoreCorrType.E_CORE_TYPE_PEARS


class AbuUmpEdgeBase(AbuUmpBase):
    """边裁基类"""

    @classmethod
    def ump_edge_clf_dump(cls, orders_pd_train, show_info=False, market_name=None):
        """
        类方法，通过交易训练集orders_pd_train构造AbuUmpEdgeBase子类对象, 使用fit方法对训练集进行特征采集，后进行dump_clf即
        本地序列化存贮等工作
        :param orders_pd_train: 交易训练集，pd.DataFrame对象
        :param show_info: 是否显示edge.fiter.df.head()，默认False
        :param market_name: 主裁训练或者获取裁判对应的存贮唯一名称，默认None, 根据env中的当前市场设置存储名称
        :return: AbuUmpEdgeBase子类对象实例
        """
        edge = cls(orders_pd_train, market_name=market_name)
        edge.fit()
        edge.dump_clf()
        if show_info:
            print('edge.fiter.df.head():\n', edge.fiter.df.head())
        return edge

    @abstractmethod
    def get_fiter_class(self):
        """abstractmethod子类必须实现，声明具体子类裁判使用的筛选特征形成特征的类"""
        pass

    @abstractmethod
    def get_predict_col(self):
        """abstractmethod子类必须实现，获取具体子类裁判需要的特征keys"""
        pass

    @classmethod
    @abstractmethod
    def class_unique_id(cls):
        """
        具体ump类关键字唯一名称，类方法，abstractmethod子类必须实现
        主要针对外部user设置自定义ump使用, 需要user自己保证class_unique_id的唯一性，内部不做检测
        具体使用见ABuUmpManager中extend_ump_block方法
        """
        pass

    def __init__(self, orders_pd=None, predict=False, market_name=None, **kwarg):
        """
        :param orders_pd: 回测结果生成的交易订单构成的pd.DataFrame对象, 最好是经过度量类
                          AbuMetricsBase对象进行度量fit_metrics之后的orders_pd
        :param predict: 是否构造的裁判类型为预测，非训练裁判
        :param market_name: 主裁训练或者获取裁判对应的存贮唯一名称，默认None, 根据env中的当前市场设置存储名称
        :param kwarg: 将kwarg参数透传给fiter_cls的构造：
                        self.fiter = self.fiter_cls(orders_pd=orders_pd, **kwarg)
        """
        # 特征筛选类fiter_cls
        self.fiter_cls = self.get_fiter_class()

        # 对交易特征进行统一标准化的scaler对象
        self.scaler = preprocessing.StandardScaler()

        if isinstance(market_name, ABuEnv.EMarketTargetType):
            market_name = market_name.value
        # predict或者训练的情况都需要对应裁判的唯一名称, 默认使用对应市场的字符串名字 eg，'us'， 'cn'
        self.market_name = ABuEnv.g_market_target.value if market_name is None else market_name

        if not predict:
            # TODO 拆开predict和训练数据逻辑，不要纠缠在一起
            if orders_pd is not None and 'profit_cg' not in orders_pd.columns:
                # profit_cg等度量参数是要在AbuMetricsBase结束后才会有
                logging.info('you do better AbuMetricsBase.fit_metrics in orders_pd!!!!')
                from ..MetricsBu.ABuMetricsBase import AbuMetricsBase
                # 这里只做fit_metrics_order，没做fit_metrics因为比如期货，比特币会有自己的度量类，使用通用的fit_metrics_order
                AbuMetricsBase(orders_pd, None, None, None).fit_metrics_order()
            # 实例化特征构造对象self.fiter
            self.fiter = self.fiter_cls(orders_pd=orders_pd, **kwarg)
            """
                通过self.fiter_cls构造形成self.fiter后self.fiter.df中以存在特征

                eg：self.fiter.df
                            profit	profit_cg	buy_deg_ang42	buy_deg_ang252	buy_deg_ang60	buy_deg_ang21
                2014-09-24	-22618.04	-0.0566	3.378	3.458	3.458	1.818
                2014-10-24	-29690.28	-0.0742	0.191	2.889	2.809	-1.089
                2014-10-29	18959.19	0.0542	-2.026	16.689	-0.761	1.980
                2014-10-29	148209.36	0.5022	-3.427	-11.956	-8.296	6.507
                2014-10-29	24867.60	0.0952	-2.915	39.469	-6.043	7.046
            """
            # 默认使用svm，这里需要参数可设置
            self.fiter().estimator.svc()

    def fit(self):
        """
        边裁训练集拟合存储函数，相对主裁的训练fit函数，边裁的fit很简单
        self.fiter.df经过fit后添加了新列p_rk_cg和rk形式如下所示

            eg：self.fiter.df
                           profit  profit_cg  buy_deg_ang42  buy_deg_ang252  \
            2014-09-24  -22618.04    -0.0566          3.378           3.458
            2014-10-24  -29690.28    -0.0742          0.191           2.889
            2014-10-29   18959.19     0.0542         -2.026          16.689
            2014-10-29  148209.36     0.5022         -3.427         -11.956
            2014-10-29   24867.60     0.0952         -2.915          39.469
            2014-10-29   18959.19     0.0542         -2.026          16.689
            2014-11-03    1250.80     0.0045          0.103          39.202
            2014-11-11   59888.21     0.1857          8.341          -9.450
            2014-11-12   -3578.78    -0.0140          3.963           6.595
            2014-11-26  -29085.19    -0.0946         14.052           6.061
            ...               ...        ...            ...             ...
            2016-03-14   16220.57     0.0559          4.002         -10.559
            2016-03-14  -25328.12    -0.1218          0.129          -6.649
            2016-03-30  -29858.44    -0.0863         13.121          -8.461
            2016-04-04    5373.76     0.0244          4.409         -33.097
            2016-04-13  -28044.40    -0.1159          6.603         -31.459
            2016-04-14  -18645.93    -0.0467          4.611          18.428
            2016-04-15  -32484.79    -0.1149          4.238         -13.247
            2016-04-15  -32484.79    -0.1149          4.238         -13.247
            2016-04-29     290.96     0.0007          1.445          16.266
            2016-04-29     290.96     0.0007          1.445          16.266

                        buy_deg_ang60  buy_deg_ang21  p_rk_cg  rk
            2014-09-24          3.458          1.818     19.0   0
            2014-10-24          2.809         -1.089     13.0  -1
            2014-10-29         -0.761          1.980     35.5   0
            2014-10-29         -8.296          6.507     56.0   1
            2014-10-29         -6.043          7.046     43.0   1
            2014-10-29         -0.761          1.980     35.5   0
            2014-11-03         -4.614         10.125     28.0   0
            2014-11-11          0.730         12.397     48.0   1
            2014-11-12         -7.524          6.671     23.0   0
            2014-11-26          7.566         12.494      9.0  -1
            ...                   ...            ...      ...  ..
            2016-03-14         -7.992          9.324     37.0   0
            2016-03-14        -10.880          5.201      2.0  -1
            2016-03-30          4.498          4.070     12.0  -1
            2016-04-04         -6.281          5.618     33.0   0
            2016-04-13          0.191          4.457      4.0  -1
            2016-04-14          3.134          0.733     20.0   0
            2016-04-15          4.693          1.162      5.5  -1
            2016-04-15          4.693          1.162      5.5  -1
            2016-04-29          4.615         -1.115     24.5   0
            2016-04-29          4.615         -1.115     24.5   0

        边裁裁决方式多次使用非均衡技术对最后的结果概率进行干预，目的是使最终的裁决正确率达成非均衡的目标，
        非均衡技术思想是量化中很很重要的一种设计思路，因为我们量化的目标结果就是非均衡（我们想要赢的钱比输的多）
        """

        # 对训练特征fiter.df中的profit_cg进行rank，即针对训练集中的交易盈利亏损值进行rank排序, rank结果添加到self.fiter.df新列
        # TODO 暂时只使用profit_cg不使用profit做为训练参数，需要整合profit为训练的rank等综合权重处理
        self.fiter.df['p_rk_cg'] = self.fiter.df['profit_cg'].rank()
        """
            eg: self.fiter.df['p_rk_cg']
            2014-09-24    19.0
            2014-10-24    13.0
            2014-10-29    35.5
            2014-10-29    56.0
            2014-10-29    43.0
            2014-10-29    35.5
            2014-11-03    28.0
            2014-11-11    48.0
            2014-11-12    23.0
            2014-11-26     9.0
                          ...
            2016-03-14    37.0
            2016-03-14     2.0
            2016-03-30    12.0
            2016-04-04    33.0
            2016-04-13     4.0
            2016-04-14    20.0
            2016-04-15     5.5
            2016-04-15     5.5
            2016-04-29    24.5
            2016-04-29    24.5
        """

        # K_CG_TOP_RATE=0.236, 由于策略的胜负的非均衡，win_top的位置实际比较loss_top为非均衡，为后续制造概率优势
        win_top = len(self.fiter.df['profit_cg']) - len(self.fiter.df['profit_cg']) * K_CG_TOP_RATE
        """
            eg:
                len(self.fiter.df['profit_cg']) == 100
                -> win_top = 100 - 100 * 0.236
                -> win_top = 100 - 23.6
                -> win_top = 76.4
        """
        loss_top = len(self.fiter.df['profit_cg']) * K_CG_TOP_RATE
        """
            eg:
                len(self.fiter.df['profit_cg']) == 100
                -> loss_top = 100 * 0.236
                -> loss_top = 23.6
        """

        # self.fiter.df添加新列'rk'，初始值都为EEdgeType.E_EEdge_NORMAL.value，即0
        self.fiter.df['rk'] = EEdgeType.E_EEdge_NORMAL.value
        """
            根据win_top, loss_top将整体切分为三段，rk：-1, 0, 1

                        rk  profit_cg  p_rk_cg
            2011-09-21   0   0.036216  58816.0
            2011-09-21   1   0.046784  61581.0
            2011-09-21  -1  -0.191184   1276.0
            2011-09-21   0  -0.000428  43850.0
            2011-09-21   0   0.001724  44956.0

        """
        # noinspection PyTypeChecker
        self.fiter.df['rk'] = np.where(self.fiter.df['p_rk_cg'] > win_top, EEdgeType.E_STORE_TOP_WIN.value,
                                       self.fiter.df['rk'])
        # noinspection PyTypeChecker
        self.fiter.df['rk'] = np.where(self.fiter.df['p_rk_cg'] < loss_top, EEdgeType.E_EEdge_TOP_LOSS.value,
                                       self.fiter.df['rk'])

    def dump_file_fn(self):
        """
            边裁本地缓存的存储路径规则：
            ABuEnv.g_project_data_dir ＋ 'ump/ump_edge_' ＋ market_name + self.class_unique_id()
        """

        # TODO 如果有裁判覆盖，保留备份，显示通知
        unique_ump_name = 'ump/ump_edge_{}_{}'.format(self.market_name, self.class_unique_id())
        return os.path.join(ABuEnv.g_project_data_dir, unique_ump_name)

    def dump_clf(self):
        """
            边裁的本地序列化相对主裁的dump_clf也简单很多，
            将self.fiter.df和self.fiter.x打包成一个字典对象df_x_dict
            通过ABuFileUtil.dump_pickle进行保存
        """
        df_x_dict = {'fiter_df': self.fiter.df, 'fiter_x': self.fiter.x}
        """
            eg：df_x_dict
            array([[  3.378,   3.458,   3.458,   1.818],
                   [  0.191,   2.889,   2.809,  -1.089],
                   [ -2.026,  16.689,  -0.761,   1.98 ],
                   [ -3.427, -11.956,  -8.296,   6.507],
                   [ -2.915,  39.469,  -6.043,   7.046],
                   [ -2.026,  16.689,  -0.761,   1.98 ],
                   [  0.103,  39.202,  -4.614,  10.125],
                   [  8.341,  -9.45 ,   0.73 ,  12.397],
                   [  3.963,   6.595,  -7.524,   6.671],
                   ....................................
                   [  4.002, -10.559,  -7.992,   9.324],
                   [  0.129,  -6.649, -10.88 ,   5.201],
                   [ 13.121,  -8.461,   4.498,   4.07 ],
                   [  4.409, -33.097,  -6.281,   5.618],
                   [  6.603, -31.459,   0.191,   4.457],
                   [  4.611,  18.428,   3.134,   0.733],
                   [  4.238, -13.247,   4.693,   1.162],
                   [  4.238, -13.247,   4.693,   1.162],
                   [  1.445,  16.266,   4.615,  -1.115],
                   [  1.445,  16.266,   4.615,  -1.115]])
        """
        ABuFileUtil.dump_pickle(df_x_dict, self.dump_file_fn(), how='zero')

    def predict(self, **kwargs):
        """
        边裁交易决策函数，从CachedUmpManager中获取缓存df_x_dict，对kwargs关键字参数所描述的交易特征进行拦截决策
        边裁的predict()实现相对主裁来说比较复杂，大致思路如下：

        1. 从输入的新交易中挑选需要的特征组成x
        2. 将x和之前保存的训练集数据组合concatenate()，一起做数据标准化scaler
        3. 使用sklearn.metrics.pairwise.pairwise_distances()度量输入特征和训练集矩阵中的距离序列
        4. 取pairwise_distances() TOP个作为种子，继续匹配相似度
        5. 相似度由大到小排序，保留大于保留阀值的相似度交易数据做为最终有投票权利的
        6. 保留的交易认为是与新交易最相似的交易，保留的交易使用之前非均衡的rk对新交易进行投票
        7. 最后的判断需要大于一定比例才被结果认可，即再次启动非均衡


        :param kwargs: 需要和子类对象实现的虚方法get_predict_col中获取特征列对应的
                       关键字参数，eg: buy_deg_ang42=3.378, buy_deg_ang60=3.458
                                     buy_deg_ang21=3.191, buy_deg_ang252=1.818
        :return: 是否对kwargs关键字参数所描述的交易特征进行拦截，
                 EEdgeType: 不拦截: EEdgeType.E_EEdge_NORMAL or EEdgeType.E_STORE_TOP_WIN
                            拦截: EEdgeType.E_EEdge_TOP_LOSS
        """

        # 统一从CachedUmpManager中获取缓存ump，没有缓存的情况下load_pickle
        df_x_dict = AbuUmpBase.dump_clf_manager.get_ump(self)

        # 从df_x_dict['fiter_df'].columns中筛选特征列
        feature_columns = df_x_dict['fiter_df'].columns.drop(['profit', 'profit_cg', 'p_rk_cg', 'rk'])
        """
            eg: df_x_dict['fiter_df'].columns
            Index(['profit', 'profit_cg', 'buy_deg_ang42', 'buy_deg_ang252',
                   'buy_deg_ang60', 'buy_deg_ang21', 'p_rk_cg', 'rk'], dtype='object')

            drop(['profit', 'profit_cg', 'p_rk_cg', 'rk']
            -> ['buy_deg_ang42', 'buy_deg_ang252', 'buy_deg_ang60', 'buy_deg_ang21']
        """

        # eg, x: array([ 3.378,  3.458,  3.458,  1.818])
        x = np.array([kwargs[col] for col in feature_columns])
        x = x.reshape(1, -1)

        # 把新的x concatenate到之前保存的矩阵中
        con_x = np.concatenate((x, df_x_dict['fiter_x']), axis=0)
        # 将输入的x和原始矩阵组装好的新矩阵con_x一起标准化
        con_x = self.scaler.fit_transform(con_x)
        # 使用输入的x即con_x[0]和矩阵中其它的进行pairwise_distances比较
        distances_cx = pairwise_distances(con_x[0].reshape(1, -1), con_x[1:],
                                          metric='euclidean')
        distances_cx = distances_cx[0]
        """
            eg: distances_cx
            array([[ 0.    ,  0.8432,  1.4371,  2.4178,  3.1302,  1.4371,  3.1774,
                     2.5422,  1.7465,  3.0011,  0.7233,  2.264 ,  0.8279,  0.8279,
                     2.309 ,  1.4878,  1.9396,  0.7438,  0.9731,  0.4494,  2.0755,
                     2.9762,  4.5869,  5.2029,  0.7362,  0.7362,  3.623 ,  0.6105,
                     0.6105,  1.2288,  2.0991,  2.0991,  3.2272,  0.8599,  0.7419,
                     0.7419,  0.7804,  2.5241,  1.8116,  2.5373,  2.2742,  2.1726,
                     3.2738,  1.293 ,  2.4555,  2.4555,  2.3358,  2.1673,  2.0187,
                     2.8637,  2.5066,  1.052 ,  1.1481,  1.1481,  1.1175,  1.1175]])

        """

        # 如果最小距离大于阀值，认为无效，K_DISTANCE_THRESHOLD = 0.668
        if distances_cx.min() > K_DISTANCE_THRESHOLD:
            return EEdgeType.E_EEdge_NORMAL
        distances_sort = distances_cx.argsort()
        """
            eg: distances_sort
            array([ 0, 19, 28, 27, 10, 24, 25, 35, 34, 17, 36, 13, 12,  1, 33, 18, 51,
                   54, 55, 52, 53, 29, 43,  5,  2, 15,  8, 38, 16, 48, 20, 30, 31, 47,
                   41, 11, 40, 14, 46,  3, 45, 44, 50, 37, 39,  7, 49, 21,  9,  4,  6,
                   32, 42, 26, 22, 23])
        """

        n_top = K_N_TOP_SEED if len(distances_cx) > K_N_TOP_SEED else len(distances_cx)
        # 取前100个作为种子继续匹配相似度做数据准备
        distances_sort = distances_sort[:n_top]

        # 进行第二轮的相似度匹配，使用输入的x即con_x[0]和distances_sort中记录的其它矩阵矢量进行corr_xy
        similar_cx = {arg: corr_xy(con_x[0], con_x[arg + 1], g_similar_type) for arg in distances_sort}
        """
            eg: similar_cx

            {0: 1.0, 19: 0.9197507467964976, 28: 0.57289288329659238, 27: 0.57289288329659238,
            10: 0.44603792013583493, 24: 0.4103293780402798, 25: 0.4103293780402798,
            35: 0.22026514236282496, 34: 0.22026514236282496, 17: -0.24170074544552811,
            36: 0.43863838382081699, 13: 0.16234971594751921, 12: 0.16234971594751921, 1: 0.92424298737490296,
            33: 0.47818723914034433, 18: -0.17734957863273493, 51: 0.63704694680797502, 54: 0.75395818997353681,
            55: 0.75395818997353681, 52: 0.6485413094804453, 53: 0.6485413094804453,
            29: 0.89796883127042837, 43: 0.86342390437553329, 5: 0.12738173851484677,
            2: 0.12738173851484677, 15: 0.53496775815355813, 8: -0.92624283913287053,
            38: -0.52046967255944876, 16: -0.65837858483393186, 48: 0.26241267262766549,
            20: 0.45007515315947716, 30: -0.78037071039800843, 31: -0.78037071039800843,
            47: -0.99196576241088685, 41: 0.71286817166895511, 11: -0.57565781272205685,
            40: -0.089683927257343574, 14: -0.49743962329463148, 46: -0.84622925585859421, 3: -0.82066914234853283,
            45: 0.30735926720691314, 44: 0.30735926720691314, 50: 0.010871213734502339, 37: -0.65150765047066517,
            39: -0.38809703338219459, 7: -0.57947244493007666, 49: -0.33103296960584466, 21: 0.69444344588208717,
            9: -0.3435188573004419, 4: -0.39204446380766983, 6: -0.54996919528831723, 32: -0.9481034251744791,
            42: 0.20829094732022327, 26: 0.9936229414412624, 22: -0.35972456962349542, 23: -0.085747705364200594}
        """
        # 相似度大到小排序
        similar_sorted = sorted(zip(similar_cx.values(), similar_cx.keys()))[::-1]
        """
            eg: similar_sorted
            [(1.0, 0), (0.9936229414412624, 26), (0.92424298737490296, 1), (0.9197507467964976, 19), (
            0.89796883127042837, 29), (0.86342390437553329, 43), (0.75395818997353681, 55), (0.75395818997353681, 54),
            (0.71286817166895511, 41), (0.69444344588208717, 21), (0.6485413094804453, 53), (0.6485413094804453, 52),
            (0.63704694680797502, 51), (0.57289288329659238, 28), (0.57289288329659238, 27), (0.53496775815355813, 15),
            (0.47818723914034433, 33), (0.45007515315947716, 20), (0.44603792013583493, 10), (0.43863838382081699, 36),
            (0.4103293780402798, 25), (0.4103293780402798, 24), (0.30735926720691314, 45), (0.30735926720691314, 44),
            (0.26241267262766549, 48), (0.22026514236282496, 35), (0.22026514236282496, 34), (0.20829094732022327, 42),
             (0.16234971594751921, 13), (0.16234971594751921, 12), (0.12738173851484677, 5), (0.12738173851484677, 2),
             (0.010871213734502339, 50), (-0.085747705364200594, 23), (-0.089683927257343574, 40),
             (-0.17734957863273493, 18), (-0.24170074544552811, 17), (-0.33103296960584466, 49),
             (-0.3435188573004419, 9), (-0.35972456962349542, 22), (-0.38809703338219459, 39),
             (-0.39204446380766983, 4), (-0.49743962329463148, 14), (-0.52046967255944876, 38),
             (-0.54996919528831723, 6), (-0.57565781272205685, 11), (-0.57947244493007666, 7),
             (-0.65150765047066517, 37), (-0.65837858483393186, 16), (-0.78037071039800843, 31),
             (-0.78037071039800843, 30), (-0.82066914234853283, 3), (-0.84622925585859421, 46),
             (-0.92624283913287053, 8), (-0.9481034251744791, 32), (-0.99196576241088685, 47)]
        """
        # 只取大于阀值相似度K_SIMILAR_THRESHOLD的做为最终有投票权利的
        similar_filters = list(filter(lambda sm: sm[0] > K_SIMILAR_THRESHOLD, similar_sorted))
        """
            eg: similar_filters
            [(1.0, 0), (0.9936229414412624, 26), (0.92424298737490296, 1), (0.9197507467964976, 19)]
        """
        if len(similar_filters) < int(n_top * 0.1):
            # 投票的太少，初始相似种子n_top的0.1为阀值，认为无效，eg：int(100 * 0.1) == 10
            return EEdgeType.E_EEdge_NORMAL

        top_loss_cluster_cnt = 0
        top_win_cluster_cnt = 0
        # 由于gmm_component_filter中win_top的非均衡，导致top_win_cluster_cnt > top_loss_cluster_cnt概率大
        for similar in similar_filters:
            """
                eg:
                    similar: (0.9936229414412624, 26)
                    order_ind = similar[1] = 26
                    similar_val = similar[0] = 0.9936229414412624
            """
            order_ind = similar[1]
            similar_val = similar[0]
            # 通过order_ind获取有投票权利的交易的rk值
            rk = df_x_dict['fiter_df'].iloc[order_ind]['rk']
            # 对应这个最相似的在哪一个分类中，判断edge
            if rk == -1:
                # 需要 * similar_val eg： top_loss_cluster_cnt += 1 * 0.9936229414412624
                top_loss_cluster_cnt += 1 * similar_val
            elif rk == 1:
                top_win_cluster_cnt += 1 * similar_val

        # 最后的投票结果需要大于一定比例才被认可, 即对有争议的投票需要一方拥有相对优势才认可
        if int(top_win_cluster_cnt * K_EDGE_JUDGE_RATE) > top_loss_cluster_cnt:
            """
                eg: top_win_cluster_cnt = 100
                    top_loss_cluster_cnt ＝ 50

                    int(top_win_cluster_cnt * K_EDGE_JUDGE_RATE) == 62
                    62 > 50 -> EEdgeType.E_STORE_TOP_WIN
            """
            return EEdgeType.E_STORE_TOP_WIN
        elif int(top_loss_cluster_cnt * K_EDGE_JUDGE_RATE) > top_win_cluster_cnt:
            """
                eg: top_loss_cluster_cnt = 100
                    top_win_cluster_cnt ＝ 50

                    int(top_loss_cluster_cnt * K_EDGE_JUDGE_RATE) == 62
                    62 > 50 -> EEdgeType.E_EEdge_TOP_LOSS
            """
            # 由于top_win_cluster_cnt > top_loss_cluster_cnt的非均衡本来就有概率优势，＊ K_EDGE_JUDGE_RATE进一步扩大概率优势
            return EEdgeType.E_EEdge_TOP_LOSS
        return EEdgeType.E_EEdge_NORMAL
