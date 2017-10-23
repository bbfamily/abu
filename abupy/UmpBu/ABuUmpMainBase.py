# -*- encoding:utf-8 -*-
"""
    主裁基础实现模块
"""

from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

import os
import copy
from abc import abstractmethod
import math

from ..MarketBu import ABuMarketDrawing

from ..CoreBu import ABuEnv
import logging
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.optimize as sco
# noinspection PyUnresolvedReferences
from mpl_toolkits.mplot3d import Axes3D

from ..UtilBu import ABuFileUtil
from ..UtilBu.ABuProgress import AbuProgress
from .ABuUmpBase import AbuUmpBase
from ..CoreBu.ABuFixes import GMM
from ..UtilBu.ABuProgress import AbuMulPidProgress
from ..CoreBu.ABuParallel import delayed, Parallel
from ..UtilBu.ABuDTUtil import plt_show

__author__ = '阿布'
__weixin__ = 'abu_quant'

"""主裁使用的gmm默认分类从40-85个分类"""
K_DEFAULT_NCS_RANG = slice(40, 85, 1)

"""主裁使用的gmm默认分类K_DEFAULT_NCS_RANG最大对应的分类数量，详阅读使用"""
K_DEFAULT_NCS_MAX = 50000

"""在brust_min中计算lps_step, lms_step, lrs_step中使用的默认step参数，默认50，详见brust_min函数实现"""
g_brust_min_step_cnt = 50

"""在不计算全局最优参数brust_min组合情况下的直接使用的默认参数摘取cprs形成llps"""
g_brust_min_default = (0, 0, 0.65)

"""代表在ump_main_clf_dump中show_order或者save_order为True的情况下最多绘制和保存的交易快照数量"""
g_plot_order_max_cnt = 100


def _do_gmm_cluster(sub_ncs, x, df, threshold):
    """
    在AbuUmpMainBase中fit并行启动的子进程执行的gmm cluster函数，进程函数，
    在子进程中迭代sub_ncs中的component值代入gmm，通过threshold对gmm cluster
    结果进行筛选过滤
    :param sub_ncs: 子进程中gmm分类的范围, eg： [10, 11, 12, 13, 14, 15]
    :param x: 主裁训练集特征x矩阵，numpy矩阵对象
    :param df: 主裁训练集特征pd.DataFrame对象, 包括x，y
    :param threshold:  分类簇中失败率选择的阀值（默认0.65），即大于threshold值的gmm分类簇做为主裁学习目标分类簇
    :return: clf_component_dict, cluster_df_dict
    """
    clf_component_dict = {}
    cluster_df_dict = {}

    # 启动多进程进度显示AbuMulPidProgress
    with AbuMulPidProgress(len(sub_ncs), 'gmm fit') as progress:
        for epoch, component in enumerate(sub_ncs):
            progress.show(epoch + 1)
            clf = GMM(component, random_state=3).fit(x)
            cluster = clf.predict(x)
            """
                eg：component=14, cluster形式如：

                      [ 4,  4, 13,  2,  3, 13,  3,  7,  1,  8,  4, 13,  4,  4,  2, 13, 13,
                        ....... 1,  1,  5,  7,  5,  7,  0,  8, 13, 10, 10,  2,  2,  7, 12,
                      12, 13,  7,  7, 13, 13]
            """
            # 只是临时保存一下分类簇序号，因为要使用pd.crosstab，下一个迭代中新的cluster又会生成
            df['cluster'] = cluster
            xt = pd.crosstab(df['cluster'], df['result'])
            """
                xt形如: 即gmm分类簇中每一个子分类的失败交易和盈利交易的数量统计
                result       0      1
                cluster
                0          290    279
                2         1156    766
                3          160    137
                .....................
            """
            # 进行一次cluster数量的淘汰，无法准确量化某一个分类数量少于多少为不正常，这里范定义总交易数的1/1000
            xt = xt[xt.sum(axis=1) > (df.shape[0] / 1000)]

            xt_pct = xt.div(xt.sum(1).astype(float), axis=0)
            """
                xt_pct形如: 即gmm分类簇中每一个子分类的失败交易和盈利交易的数量占比例
                            目的是方便和参数中的threshold（默认0.65）进行阀值对比
                result          0         1
                cluster
                0        0.509666  0.490334
                2        0.601457  0.398543
                3        0.538721  0.461279
                ............................
            """

            # xt_pct[0]即为分类簇中每一个子分类的失败率
            cluster_ind = xt_pct[xt_pct[0] > threshold].index
            """
                eg：xt_pct[xt_pct[0] > threshold].index的返回结果rt形式如下：
                Int64Index([7, 8, 9, 10], dtype='int64', name='cluster')

                cluster_ind的类型为pd.Index序列对象，值序列[7, 8, 9, 10]即代表使用GMM(component)分类
                下分成component个分类中，满足子分类的失败率大于阀值threshold的子分类序号
            """
            if len(cluster_ind) > 0:
                # 把失败概率大于阀值的component的保存clf以及cluster_ind
                clf_component_dict[component] = (clf, cluster_ind)
                """
                    eg：clf_component_dict字典形式如下：
                        key=component, value=(GaussianMixtured对象，cluster_ind: GMM(component)分component个类中，
                                              满足子分类的失败率大于阀值threshold的子分类序号)

                    {14: (GaussianMixture(max_iter=100, n_components=14, n_init=1),
                        Int64Index([7, 8, 9, 10], dtype='int64', name='cluster')),

                    15: (GaussianMixture(max_iter=100, n_components=15, n_init=1),
                        Int64Index([7, 8, 9, 10], dtype='int64', name='cluster'))}
                """
                # component下的大于阀值的子分类cluster_ind进行迭代
                for cluster in cluster_ind:
                    # cluster_df_key = component + cluster, eg: '14_7'
                    cluster_df_key = '{0}_{1}'.format(component, cluster)
                    # 从df中取出cluster对于的子pd.DataFrame对象cluster_df
                    cluster_df = df[df['cluster'] == cluster]
                    """
                        eg: cluster_df
                                    result  buy_deg_ang42  buy_deg_ang252  buy_deg_ang60  \
                        2014-11-11       1          8.341          -9.450          0.730
                        2015-10-28       0          7.144          -9.818         -3.886
                        2015-11-04       0         12.442         -10.353          3.313
                        2016-03-30       0         13.121          -8.461          4.498
                        2016-04-15       0          4.238         -13.247          4.693
                        2016-04-15       0          4.238         -13.247          4.693

                                    buy_deg_ang21  ind  cluster
                        2014-11-11         12.397    7        7
                        2015-10-28          6.955   39        7
                        2015-11-04          7.840   41        7
                        2016-03-30          4.070   49        7
                        2016-04-15          1.162   53        7
                        2016-04-15          1.162   54        7
                    """
                    # 以cluster_df_key做为key， value=cluster_df保存在cluster_df_dict中
                    cluster_df_dict[cluster_df_key] = cluster_df

    return clf_component_dict, cluster_df_dict


# noinspection PyAttributeOutsideInit
class AbuUmpMainBase(AbuUmpBase):
    """主裁基类"""

    @classmethod
    def ump_main_clf_dump(cls, orders_pd_train, p_ncs=None, brust_min=True, market_name=None,
                          show_component=False, show_info=False, save_order=False, show_order=False):
        """
        类方法，通过交易训练集orders_pd_train构造AbuUmpMainBase子类对象，透传brust_min，p_ncs等参数
        使用fit方法对训练集进行分类簇筛选，使用dump_clf本地序列化训练结果
        :param orders_pd_train: 交易训练集，pd.DataFrame对象
        :param p_ncs: gmm分类的范围, 可以为具体序列对象如[10, 11, 12....80], 也可以为生成器对象，
                      如xrange(10, 80, 10)，还支持slice对象，eg：slice(10, 80, 10)
        :param brust_min: bool类型，代表是否进行全局最优参数计算brust_min，如果否则直接使用默认g_brust_min_default(0, 0, 0.65)
                          如果不使用brust_min，即brust_min=False可大大提高训练运行效率
        :param market_name: 主裁训练或者获取裁判对应的存贮唯一名称，默认None, 根据env中的当前市场设置存储名称
        :param show_component: 是否可视化lcs，lrs，lps，lms数据(2d, 3d)，即透传给fit函数的show参数
        :param show_info: 是否显示brust_min计算的最优lrs，lps，lms组合结果以及best_hit_cnt_info，choose_cprs_component
                          等辅助函数的辅助输出信息
        :param save_order: 是否保存失败概率最大的分类簇的交易快照图片到本地
        :param show_order: 是否绘制失败概率最大的分类簇的交易快照图片
        :return: AbuUmpMainBase子类对象实例
        """
        ump = cls(orders_pd_train, market_name=market_name)
        # 训练抽取分类簇，根据brust_min
        ump.fit(show=show_component, p_ncs=p_ncs, brust_min=brust_min)
        # 本地序列化
        ump.dump_clf()

        if show_info:
            ump.log_func('全局最优:{}'.format(ump.llps_brust_min))
            # 统计拦截hit best参数
            ump.best_hit_cnt_info(ump.llps)
            # 统计滤去llps的提升情况
            ump.choose_cprs_component(ump.llps)

        # 保存或者绘制失败概率最大的分类簇的交易快照图片
        if save_order or show_order:
            # 获取失败概率最大的分类簇
            max_failed_cluster_orders = ump.nts[ump.cprs.lrs.argmax()]
            # 获取失败概率最大的分类簇中的总交易数量order_len
            order_len = max_failed_cluster_orders.shape[0]
            # 构建绘制和保存的交易快照进度条
            with AbuProgress(g_plot_order_max_cnt, 0, 'save or plot order progress') as order_progress:
                for ind in np.arange(0, order_len):
                    order_progress.show(ind)
                    if ind > g_plot_order_max_cnt:
                        # g_plot_order_max_cnt最多绘制和保存的交易快照数量
                        break
                    # 获取max_failed_cluster_orders中交易的order_ind
                    order_ind = int(max_failed_cluster_orders.iloc[ind].ind)
                    # 通过order_ind从原始交易单子ump.fiter.order_has_ret中获取单子使用ABuMarketDrawing.plot_candle_from_order
                    ABuMarketDrawing.plot_candle_from_order(ump.fiter.order_has_ret.iloc[order_ind], save=save_order)
                    if save_order and show_order:
                        # FIXME 即绘制又保存的参数情况下又重新绘制了一遍
                        ABuMarketDrawing.plot_candle_from_order(ump.fiter.order_has_ret.iloc[order_ind])
        # 最终返回AbuUmpMainBase子类对象实例
        return ump

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
                        self.fiter = self.fiter_cls(orders_pd=self.orders_pd, **kwarg)
        """
        self.orders_pd = orders_pd
        # 特征筛选类fiter_cls
        self.fiter_cls = self.get_fiter_class()
        # ipython notebook下使用logging.info
        self.log_func = logging.info if ABuEnv.g_is_ipython else print
        if isinstance(market_name, ABuEnv.EMarketTargetType):
            market_name = market_name.value
        # predict或者训练的情况都需要对应裁判的唯一名称, 默认使用对应市场的字符串名字 eg，'us'， 'cn'
        self.market_name = ABuEnv.g_market_target.value if market_name is None else market_name
        if predict:
            # 如果是predict非训练目的，这里直接返回
            # TODO 拆开predict和训练数据逻辑，不要纠缠在一起
            return

        if orders_pd is not None and 'profit_cg' not in orders_pd.columns:
            # profit_cg等度量参数是要在AbuMetricsBase结束后才会有
            self.log_func('you do better AbuMetricsBase.fit_metrics in orders_pd!!!!')
            from ..MetricsBu.ABuMetricsBase import AbuMetricsBase
            # 这里只做fit_metrics_order，没做fit_metrics因为比如期货，比特币会有自己的度量类，使用通用的fit_metrics_order
            AbuMetricsBase(orders_pd, None, None, None).fit_metrics_order()
        # 实例化特征构造对象self.fiter
        self.fiter = self.fiter_cls(orders_pd=self.orders_pd, **kwarg)
        """
            self.fiter是AbuMLPd子类对象，在init中即通过make_xy筛选出orders_pd中需要的训练集特征，在子类
            实现的make_xy函数都被主裁装饰器函数ump_main_make_xy装饰对筛选出的训练集特征进行转换矩阵，提取x,
            y序列等统一操作，详阅AbuUmpMainDeg等示例主裁具体子类

            这里构造好的self.fiter中df对象已经存在特定的特征
            eg: self.fiter.df
                        result	buy_deg_ang42	buy_deg_ang252	buy_deg_ang60	buy_deg_ang21
            2014-09-24	0	    3.378	        3.458	        3.458	        1.818
            2014-10-24	0	    0.191	        2.889	        2.809	        -1.089
            2014-10-29	1	    -2.026	        16.689      	-0.761	        1.980
            2014-10-29	1	    -3.427      	-11.956	        -8.296	        6.507
            2014-10-29	1	    -2.915	        39.469      	-6.043	        7.046
        """
        # 默认使用svm，这里需要参数可设置
        self.fiter().estimator.svc()

    # noinspection PyMethodMayBeStatic
    def _sub_ncs_split(self, ncs, n_jobs):
        sub_ncs_cnt = int(len(ncs) / n_jobs)
        if sub_ncs_cnt == 0:
            sub_ncs_cnt = 1
        group_adjacent = lambda a, k: zip(*([iter(a)] * k))
        ncs_group = list(group_adjacent(ncs, sub_ncs_cnt))
        residue_ind = -(len(ncs) % sub_ncs_cnt) if sub_ncs_cnt > 0 else 0
        if residue_ind < 0:
            # 所以如果不能除尽，最终切割的子序列数量为k_split+1, 外部如果需要进行多认为并行，可根据最终切割好的数量重分配任务数
            ncs_group.append(ncs[residue_ind:])
        return ncs_group

    def fit(self, p_ncs=None, threshold=0.65, brust_min=True, show=True):
        """
        交易训练集训练拟合函数，根据规则计算gmm分类的范围，使用gmm对训练集交易进行阀值筛选，
        根据brust_min参数决定是否进行全局最优参数计算，最终生成self.rts，self.nts，self.cprs

        eg：self.rts字典形式如下：
            key=component, value=(GaussianMixtured对象，cluster_ind: GMM(component)分component个类中，
            满足子分类的失败率大于阀值threshold的子分类序号)

            {14: (GaussianMixture(max_iter=100, n_components=14, n_init=1),
                Int64Index([7, 8, 9, 10], dtype='int64', name='cluster')),

            15: (GaussianMixture(max_iter=100, n_components=15, n_init=1),
                Int64Index([7, 8, 9, 10], dtype='int64', name='cluster'))}

        eg: self.nts字典对象形式如下所示：
                {
                '14-7':
                            result	buy_deg_ang42	buy_deg_ang252	buy_deg_ang60	buy_deg_ang21	ind	cluster
                2014-11-11	1	    8.341	        -9.450	        0.730	        12.397	        7	7
                2015-10-28	0	    7.144	        -9.818	        -3.886	        6.955	        39	7
                2015-11-04	0	    12.442	        -10.353	        3.313	        7.840	        41	7
                2016-03-30	0	    13.121	        -8.461	        4.498	        4.070	        49	7
                2016-04-15	0	    4.238	        -13.247	        4.693	        1.162	        53	7
                2016-04-15	0	    4.238	        -13.247	        4.693	        1.162	        54	7
                ...............................................................................................

                '14-8':
                            result	buy_deg_ang42	buy_deg_ang252	buy_deg_ang60	buy_deg_ang21	ind	cluster
                2014-11-26	0	    14.052	        6.061	        7.566	        12.494	        9	8
                2015-04-22	1	    20.640	        2.751	        20.436	        18.781	        23	8
                2015-12-16	0	    12.906	        6.312	        16.638	        12.379	        43	8

                ................................................................................................
                }

        eg: self.cprs pd.DataFrame对象形式如下
                   lcs     lms     lps     lrs
            14_7     6 -0.0552 -0.3310  0.8333
            14_8     3 -0.0622 -0.1866  0.6667
            14_9     1 -0.1128 -0.1128  1.0000
            14_10    2 -0.0327 -0.0654  1.0000
            15_7     4 -0.1116 -0.4465  1.0000
            15_8     3 -0.0622 -0.1866  0.6667
            ...    ...     ...     ...     ...
            27_8     1 -0.1128 -0.1128  1.0000
            27_9     3 -0.0103 -0.0309  0.6667
            27_14    1 -0.0946 -0.0946  1.0000
            27_15    2 -0.1149 -0.2299  1.0000
            27_17    1 -0.1041 -0.1041  1.0000
            27_18    2 -0.0327 -0.0654  1.0000
            27_24    1 -0.0140 -0.0140  1.0000

        :param p_ncs: gmm分类的范围, 可以为具体序列对象如[10, 11, 12....80], 也可以为生成器对象，
                      如xrange(10, 80, 10)，还支持slice对象，eg：slice(10, 80, 10)
        :param threshold: 分类簇中失败率选择的阀值（默认0.65），即大于threshold值的gmm分类簇做为主裁学习目标分类簇
        :param brust_min: bool类型，代表是否进行全局最优参数计算brust_min，如果否则直接使用默认g_brust_min_default(0, 0, 0.65)
                          如果不使用brust_min，即brust_min=False可大大提高训练运行效率
        :param show: 是否可视化lcs，lrs，lps，lms数据(2d, 3d)
        """
        if self.fiter().df.shape[0] < 50:
            # 有结果的交易最少要有50个，否则直接返回
            self.log_func('order count at least more than 50!')
            return

        if p_ncs is None:
            if self.fiter().df.shape[0] < K_DEFAULT_NCS_MAX:
                # 如果没有设置gmm分类的范围，且总交易单数量小于K_DEFAULT_NCS_MAX，使用默认gmm分类簇范围slice(40, 85, 1)
                ncs = np.arange(K_DEFAULT_NCS_RANG.start, K_DEFAULT_NCS_RANG.stop, K_DEFAULT_NCS_RANG.step)
            else:
                # 交易单数量 > K_DEFAULT_NCS_MAX时，重新计算start，stop，step
                ncs_stop = int(math.ceil(self.fiter().df.shape[0] * K_DEFAULT_NCS_RANG.stop / K_DEFAULT_NCS_MAX))
                # 根据计算出的stop和默认的start重新计算start
                ncs_start = int(ncs_stop - K_DEFAULT_NCS_RANG.start)
                ncs_step = K_DEFAULT_NCS_RANG.step
                ncs = np.arange(ncs_start, ncs_stop, ncs_step)

        elif p_ncs is not None and isinstance(p_ncs, slice):
            # 外部设置的gmm分类的范围为slice切片对象，构造np.arange
            ncs = np.arange(p_ncs.start, p_ncs.stop, p_ncs.step)
        else:
            # 把外部设置的生成器对象如xrange(10, 80, 10)，转换为序列对象，或者tuple, set类型的序列统一到list
            ncs = list(p_ncs)

        if ncs[-1] >= self.fiter().df.shape[0] / 2:
            # 如果分类簇最大分类数ncs[-1]大于总训练集交易量的一半，重新分配gmm分类的范围，使用交易量1/4-1/2
            ncs = np.arange(int(self.fiter().df.shape[0] / 4), int(self.fiter().df.shape[0] / 2))

        # return ncs
        # 这里copy训练集特征对象self.fiter().df，因为要修改df，保持原始训练集特征对象不变
        df = copy.deepcopy(self.fiter().df)
        # 添加一个索引序列，方便之后快速查找原始单据, 即从self.fiter.order_has_ret.iloc[df.ind.values]
        df['ind'] = np.arange(0, df.shape[0])
        """
            eg: df添加ind列
                            result  buy_deg_ang42  buy_deg_ang252  buy_deg_ang60  \
            2014-09-24       0          3.378           3.458          3.458
            2014-10-24       0          0.191           2.889          2.809
            2014-10-29       1         -2.026          16.689         -0.761
            2014-10-29       1         -3.427         -11.956         -8.296
            2014-10-29       1         -2.915          39.469         -6.043
                        buy_deg_ang21  ind
            2014-09-24          1.818    0
            2014-10-24         -1.089    1
            2014-10-29          1.980    2
            2014-10-29          6.507    3
            2014-10-29          7.046    4
        """
        clf_component_dict = {}
        cluster_df_dict = {}

        if self.fiter().df.shape[0] < 1000:
            # 交易单总数小于1000个，单进程
            clf_component_dict, cluster_df_dict = _do_gmm_cluster(ncs, self.fiter().x, df, threshold)
        else:
            n_jobs = ABuEnv.g_cpu_cnt
            # 根据进程数切割ncs为n_jobs个子ncs形成ncs_group
            ncs_group = self._sub_ncs_split(ncs, n_jobs)
            parallel = Parallel(
                n_jobs=len(ncs_group), verbose=0, pre_dispatch='2*n_jobs')
            out = parallel(delayed(_do_gmm_cluster)(sub_ncs, self.fiter().x, df, threshold) for sub_ncs in ncs_group)

            for sub_out in out:
                # 将每一个进程返回的结果进行合并
                clf_component_dict.update(sub_out[0])
                cluster_df_dict.update(sub_out[1])

        self.rts = clf_component_dict
        self.nts = cluster_df_dict
        self.cprs = self._fit_cprs(show=show)
        """
         eg: self.cprs形式如
                   lcs     lms     lps     lrs
            14_7     6 -0.0552 -0.3310  0.8333
            14_8     3 -0.0622 -0.1866  0.6667
            14_9     1 -0.1128 -0.1128  1.0000
            14_10    2 -0.0327 -0.0654  1.0000
            15_7     4 -0.1116 -0.4465  1.0000
            15_8     3 -0.0622 -0.1866  0.6667
            ...    ...     ...     ...     ...
            27_8     1 -0.1128 -0.1128  1.0000
            27_9     3 -0.0103 -0.0309  0.6667
            27_14    1 -0.0946 -0.0946  1.0000
            27_15    2 -0.1149 -0.2299  1.0000
            27_17    1 -0.1041 -0.1041  1.0000
            27_18    2 -0.0327 -0.0654  1.0000
            27_24    1 -0.0140 -0.0140  1.0000
        """
        # noinspection PyTypeChecker
        self._fit_brust_min(brust_min)

    def _fit_cprs(self, show):
        """
        通过self.nts，eg: self.nts字典对象形式如下所示：
                {
                '14-7':
                            result	buy_deg_ang42	buy_deg_ang252	buy_deg_ang60	buy_deg_ang21	ind	cluster
                2014-11-11	1	    8.341	        -9.450	        0.730	        12.397	        7	7
                2015-10-28	0	    7.144	        -9.818	        -3.886	        6.955	        39	7
                2015-11-04	0	    12.442	        -10.353	        3.313	        7.840	        41	7
                2016-03-30	0	    13.121	        -8.461	        4.498	        4.070	        49	7
                2016-04-15	0	    4.238	        -13.247	        4.693	        1.162	        53	7
                2016-04-15	0	    4.238	        -13.247	        4.693	        1.162	        54	7
                ...............................................................................................

                '14-8':
                            result	buy_deg_ang42	buy_deg_ang252	buy_deg_ang60	buy_deg_ang21	ind	cluster
                2014-11-26	0	    14.052	        6.061	        7.566	        12.494	        9	8
                2015-04-22	1	    20.640	        2.751	        20.436	        18.781	        23	8
                2015-12-16	0	    12.906	        6.312	        16.638	        12.379	        43	8

                ................................................................................................
                }
        计算分类簇中的交易个数，分类簇中的交易的平均收益，分类簇中的交易的总收益比例，
        以及分类簇中的胜率(分类簇中失败亏损的交易数量/类醋中总交易数量), 以上构成字典对象：cprs_dict
            eg: cprs_dict = {'lcs': [], 'lrs': [], 'lps': [], 'lms': []}
                lcs：分类簇中的交易个数
                lrs：代表分类簇中失败亏损的交易数量/类醋中总交易数量
                lps：通过cluster_order_df.profit_cg计算分类簇中的交易的总收益比例
                lms：通过cluster_order_df.profit_cg计算分类簇中的交易的平均收益
        构造pd.DataFrame(cprs_dict, index=cprs_index)提供数据初始

        :param show: 是否可视化lcs，lrs，lps，lms数据(2d, 3d)
        :return: pd.DataFrame对象
                eg: cprs形式如
                       lcs     lms     lps     lrs
                14_7     6 -0.0552 -0.3310  0.8333
                14_8     3 -0.0622 -0.1866  0.6667
                14_9     1 -0.1128 -0.1128  1.0000
                14_10    2 -0.0327 -0.0654  1.0000
                15_7     4 -0.1116 -0.4465  1.0000
                15_8     3 -0.0622 -0.1866  0.6667
                15_9     1 -0.1128 -0.1128  1.0000
                15_10    2 -0.0327 -0.0654  1.0000
                16_7     4 -0.1116 -0.4465  1.0000
                16_8     3 -0.0622 -0.1866  0.6667
                ...    ...     ...     ...     ...
                26_17    1 -0.1041 -0.1041  1.0000
                26_18    2 -0.0327 -0.0654  1.0000
                26_24    1 -0.0140 -0.0140  1.0000
                27_8     1 -0.1128 -0.1128  1.0000
                27_9     3 -0.0103 -0.0309  0.6667
                27_14    1 -0.0946 -0.0946  1.0000
                27_15    2 -0.1149 -0.2299  1.0000
                27_17    1 -0.1041 -0.1041  1.0000
                27_18    2 -0.0327 -0.0654  1.0000
                27_24    1 -0.0140 -0.0140  1.0000

        """

        """
            cprs_dict和cprs_index为了构造pd.DataFrame(cprs_dict, index=cprs_index)提供数据初始
            lcs：分类簇中的交易个数
            lrs：代表分类簇中失败亏损的交易数量/类醋中总交易数量
            lps：通过cluster_order_df.profit_cg计算分类簇中的交易的总收益比例
            lms：通过cluster_order_df.profit_cg计算分类簇中的交易的平均收益
        """
        cprs_dict = {'lcs': [], 'lrs': [], 'lps': [], 'lms': []}
        cprs_index = list()
        for cluster_df_key in self.nts:
            """
                eg: self.nts字典对象形式如下所示
                {
                '14-7':
                            result	buy_deg_ang42	buy_deg_ang252	buy_deg_ang60	buy_deg_ang21	ind	cluster
                2014-11-11	1	    8.341	        -9.450	        0.730	        12.397	        7	7
                2015-10-28	0	    7.144	        -9.818	        -3.886	        6.955	        39	7
                2015-11-04	0	    12.442	        -10.353	        3.313	        7.840	        41	7
                2016-03-30	0	    13.121	        -8.461	        4.498	        4.070	        49	7
                2016-04-15	0	    4.238	        -13.247	        4.693	        1.162	        53	7
                2016-04-15	0	    4.238	        -13.247	        4.693	        1.162	        54	7
                ...............................................................................................

                '14-8':
                            result	buy_deg_ang42	buy_deg_ang252	buy_deg_ang60	buy_deg_ang21	ind	cluster
                2014-11-26	0	    14.052	        6.061	        7.566	        12.494	        9	8
                2015-04-22	1	    20.640	        2.751	        20.436	        18.781	        23	8
                2015-12-16	0	    12.906	        6.312	        16.638	        12.379	        43	8

                ................................................................................................
                }
            """
            # eg: cluster_df_key = '14-7' 拿出对应的pd.DataFrame对象
            df_cluster = self.nts[cluster_df_key]
            # 分类簇中的交易个数
            cluster_cnt = df_cluster.shape[0]
            """
                eg:
                    df_cluster['result']:

                        2014-11-11    1
                        2015-10-28    0
                        2015-11-04    0
                        2016-03-30    0
                        2016-04-15    0
                        2016-04-15    0

                    df_cluster['result'].value_counts()

                        0    5
                        1    1

                    ->: df_cluster['result'].value_counts()[0] == 5
                    ->: df_cluster['result'].value_counts().sum() == 6

                －> loss_rate = 5 / 6: 代表分类簇中失败亏损的交易数量/类醋中总交易数量
            """
            loss_rate = df_cluster['result'].value_counts()[0] / df_cluster['result'].value_counts().sum()
            # 在fit开始时使用df['ind'] = np.arange(0, df.shape[0])就是为了这里与原始交易单子保留对应关系
            cluster_order_df = self.fiter.order_has_ret.iloc[df_cluster.ind.values]
            # 原始交易单子orders_pd中profit_cg数据，代表每笔交易中盈利金额与这笔交易买入金融的比例
            # 通过cluster_order_df.profit_cg计算分类簇中的交易的平均收益
            cluster_profit_mean = cluster_order_df.profit_cg.mean()
            # 通过cluster_order_df.profit_cg计算分类簇中的交易的总收益比例
            cluster_profit_sum = cluster_order_df.profit_cg.sum()

            cprs_dict['lcs'].append(cluster_cnt)
            cprs_dict['lrs'].append(loss_rate)
            cprs_dict['lms'].append(cluster_profit_mean)
            cprs_dict['lps'].append(cluster_profit_sum)
            # cluster_df_key为最终生成pd.DataFrame对象index序列
            cprs_index.append(cluster_df_key)

        cprs = pd.DataFrame(cprs_dict, index=cprs_index)
        """
            eg: cprs形式如
                   lcs     lms     lps     lrs
            14_7     6 -0.0552 -0.3310  0.8333
            14_8     3 -0.0622 -0.1866  0.6667
            14_9     1 -0.1128 -0.1128  1.0000
            14_10    2 -0.0327 -0.0654  1.0000
            15_7     4 -0.1116 -0.4465  1.0000
            15_8     3 -0.0622 -0.1866  0.6667
            15_9     1 -0.1128 -0.1128  1.0000
            15_10    2 -0.0327 -0.0654  1.0000
            16_7     4 -0.1116 -0.4465  1.0000
            16_8     3 -0.0622 -0.1866  0.6667
            ...    ...     ...     ...     ...
            26_17    1 -0.1041 -0.1041  1.0000
            26_18    2 -0.0327 -0.0654  1.0000
            26_24    1 -0.0140 -0.0140  1.0000
            27_8     1 -0.1128 -0.1128  1.0000
            27_9     3 -0.0103 -0.0309  0.6667
            27_14    1 -0.0946 -0.0946  1.0000
            27_15    2 -0.1149 -0.2299  1.0000
            27_17    1 -0.1041 -0.1041  1.0000
            27_18    2 -0.0327 -0.0654  1.0000
            27_24    1 -0.0140 -0.0140  1.0000
        """
        if show and len(cprs_dict) > 0:
            # show 2d
            cmap = plt.get_cmap('jet', 20)
            cmap.set_under('gray')
            fig, ax = plt.subplots(figsize=(9, 6))
            # x轴lrs，y轴，使用scatter绘制点，点的颜色由lps决定
            cax = ax.scatter(cprs_dict['lrs'], cprs_dict['lcs'], c=cprs_dict['lps'], cmap=cmap,
                             vmin=np.min(cprs_dict['lps']),
                             vmax=np.max(cprs_dict['lps']))
            fig.colorbar(cax, label='lps', extend='min')
            plt.grid(True)
            plt.xlabel('lrs')
            plt.ylabel('lcs')
            plt.show()

            # show 3d
            fig = plt.figure(figsize=(9, 6))
            ax = fig.gca(projection='3d')
            ax.view_init(30, 60)
            # x轴lrs，y轴lcs, z轴lps，scatter3D绘制3d点，点的颜色由lms决定
            ax.scatter3D(cprs_dict['lrs'], cprs_dict['lcs'], cprs_dict['lps'],
                         c=cprs_dict['lms'], s=50, cmap='spring')
            ax.set_xlabel('lrs')
            ax.set_ylabel('lcs')
            ax.set_zlabel('lms')
            plt.show()

        return cprs

    def _fit_brust_min(self, brust_min):
        """
        通过全局最优函数brust_min计算lrs，lps，lms的最优组合，针对最优结果effect_num
        进行如果effect_num == 0使用默认g_brust_min_default(0, 0, 0.65)，通过最优组合
        值对self.cprs对分类簇集合进行筛选形成self.llps
        :param brust_min: bool类型，代表是否进行全局最优参数计算brust_min，如果否则直接使用默认g_brust_min_default(0, 0, 0.65)
                          如果不使用brust_min，即brust_min=False可大大提高训练运行效率
        """
        if self.cprs.shape[0] < 50:
            self.llps = self.cprs[self.cprs.lps < 0]
            self.llps_brust_min = g_brust_min_default
            self.log_func('{}: cprs shape < 50!'.format(self.__class__.__name__))
            return

        effect_num = 0
        if brust_min:
            # 使用全局最优参数计算brust_min
            brust_min = self.brust_min()
            _, effect_num = self.min_func(brust_min)

        if effect_num == 0:
            # 如不使用全局最优参数计算brust_min，或者计算出的effect_num＝0使用g_brust_min_default
            brust_min = g_brust_min_default
        # 通过最优参数对分类簇集合进行筛选
        self.llps = self.cprs[(self.cprs['lps'] <= brust_min[0]) & (self.cprs['lms'] <= brust_min[1]) &
                              (self.cprs['lrs'] >= brust_min[2])]
        # 将最终从cprs筛选llps使用的brust_min值保存起来
        self.llps_brust_min = brust_min

    def brust_min(self):
        """
        在fit操作之后对self.cprs进行分析可以发现，这些分类中存在很多分类簇中的交易胜率不高，
        但是交易获利比例总和却为正值，也就是说这个分类簇由于非均衡赔率使得非均衡胜率得以保持平衡，
        并且最终获利，那么我们将所有分类簇保存在本地，对之后的交易进行裁决显然是不妥当的，所以需要使用
        全局最优技术对分类簇集合进行筛选

        1. 将lps的范围选定(分类簇中交易获利比例总和最小值-0), 即醋中交易获胜比例组合大于0的过滤
        2. 将lms的范围选定(分类簇中交易获利比例平均值最小值-0), 即醋中交易获利比例平均值大于0的过滤
        3. 将lrs交易醋失败率从最小值－最大值
        4. 计算最小值-0(最大值)之间合适的step值
        5. bnds = (slice(lps_min, 0, lps_step), slice(lms_min, 0, lms_step), slice(lrs_min, lrs_max, lrs_step))
        6. sco.brute中设定最优函数self.min_func_improved使用bnds进行计算
        更多详见brust_min实现函数
        """
        # noinspection PyProtectedMember
        factor_step = 1 if ABuEnv._g_enable_example_env_ipython else 3
        factor_step *= g_brust_min_step_cnt
        # 分类簇中交易获利比例总和最小值
        lps_min = round(self.cprs['lps'].min(), 3)
        # 默认从最小值－0之间step g_brust_min_step_cnt个：eg: (0 - -13.5) / 100 = 0.135
        lps_step = (0 - lps_min) / factor_step
        # 调整step的粒度保持在合理的范围内，不然太慢，或者粒度太粗
        if lps_step < 0.01 and lps_min + 0.01 <= 0:
            lps_step = 0.01

        # 分类簇中交易获利比例平均值最小值
        lms_min = round(self.cprs['lms'].min(), 3)
        # 默认从最小值－0之间step g_brust_min_step_cnt个：eg: (0 - -0.07) / 100 = 0.0007
        lms_step = (0 - lms_min) / factor_step
        # 调整step的粒度保持在合理的范围内，eg: 0.0007就会被修正到0.01
        if lms_step < 0.01 and lms_step + 0.01 <= 0:
            lms_step = 0.01

        # 交易醋失败率从最小值－最大值
        lrs_min = round(self.cprs['lrs'].min(), 3)
        lrs_max = round(self.cprs['lrs'].max(), 3)
        # eg: (0.87 - 0.65) / 100 = 0.002
        lrs_step = (lrs_max - lrs_min) / factor_step
        if lrs_step < 0.05 < lrs_max - lrs_min:
            lrs_step = 0.05

        bnds = (slice(lps_min, 0, lps_step), slice(lms_min, 0, lms_step), slice(lrs_min, lrs_max, lrs_step))
        """
            eg: bnds形如
            slice(-13.75, 0, 0.5), slice(-0.07, 0, 0.01), slice(0.65, 0.78, 0.1)

            bnds在最优函数sco.brute内部会使用如np.arange形式展开，即
            eg:
                slice(-13.75, 0, 0.5) －> np.arange(-13.75, 0, 0.5):

                array([-13.75, -13.25, -12.75, -12.25, -11.75, -11.25, -10.75, -10.25,
                        -9.75,  -9.25,  -8.75,  -8.25,  -7.75,  -7.25,  -6.75,  -6.25,
                        -5.75,  -5.25,  -4.75,  -4.25,  -3.75,  -3.25,  -2.75,  -2.25,
                        -1.75,  -1.25,  -0.75,  -0.25])
        """

        progress = 1
        for bnds_pos in (0, 1, 2):
            progress *= len(np.arange(bnds[bnds_pos].start, bnds[bnds_pos].stop, bnds[bnds_pos].step))
        # 进行最优时使用的进度条
        self.brust_progress = AbuProgress(progress, 0, '{}: brute min progress'.format(self.__class__.__name__))
        # 为提高运行效率，不用每次都使用_calc_llps_improved计算
        self.brust_cache = dict()
        brust_result = sco.brute(self.min_func_improved, bnds, full_output=False, finish=None)
        return brust_result

    def min_func(self, l_pmr):
        """
        使用lps，lms，lrs的特点组合值获取crps的子pd.DataFrame对象：
                llps = self.cprs[(self.cprs['lps'] <= l_pmr[0]) & (self.cprs['lms'] <= l_pmr[1]) &
                         (self.cprs['lrs'] >= l_pmr[2])]

            eg: llps形式如

                  lcs     lms     lps  lrs
            15_9    4 -0.1116 -0.4465  0.65
            16_7    12 -0.1786 -0.8122  0.72
            ................................
        通过llps从self.nts字典中获取原始训练集特征数据：
                                result  deg_ang21  deg_ang42  deg_ang60  deg_ang252    ind        cluster
                2011-09-21       0      8.645     -3.294     -8.426      -8.758     40     48      -0.083791
                2011-09-30       0     11.123     -1.887     -2.775       1.585    158     49      -0.066309

        计算llps组合的情况下拦截的单子的数量，以及胜率，失败率，通过：
                improved = (effect_num / self.fiter.order_has_ret.shape[0]) * (loss_rate - win_rate)

                eg：(effect_num / self.fiter.order_has_ret.shape[0]) * (loss_rate - win_rate)
                effect_num ＝ 10, self.fiter.order_has_ret.shape[0] ＝ 100
                loss_rate ＝ 0.8， win_rate ＝0.2
                －> (10 / 100) * (0.8 - 0.2) = 0.06

                effect_num ＝ 50, self.fiter.order_has_ret.shape[0] ＝ 100
                loss_rate ＝ 0.8， win_rate ＝0.2
                －> (50 / 100) * (0.8 - 0.2) = 0.3

                effect_num ＝ 50, self.fiter.order_has_ret.shape[0] ＝ 100
                loss_rate ＝ 0.9， win_rate ＝0.1
                －> (50 / 100) * (0.9 - 0.1) = 0.4

                即最终提升的比例和llps下被拦截的交易数量成正比，和llps下被拦截的交易失败率成正比
        即计算llps组合的情况下，即参数l_pmr给定的lps，lms，lrs的特点组合下的交易提升值
        :param l_pmr: 可迭代序列，eg：(-0.45, -0.11, 0.77), 由三个float值组成的序列
                      l_pmr[0]: 切取self.cprs使用的分类簇交易获利和值，即lps阀值：self.cprs['lps'] <= l_pmr[0]
                      l_pmr[1]: 切取self.cprs使用的分类簇交易获利平均值，即lms阀值：self.cprs['lms'] <= l_pmr[1]
                      l_pmr[2]: 切取self.cprs使用的分类簇交易失败比例，即lrs阀值：self.cprs['lrs'] >= l_pmr[2]
        :return: [improved, effect_num]
        """

        # 组合参数对训练集交易带来的提高默认=-np.inf，因为min_func_improved中的-self.min_func(l_pmr)[0]
        improved = -np.inf
        # 组合参数对训练集影响交易数量默认=0
        effect_num = 0
        # 只有失败率条件是取 lrs >=，获利和lps与平均获利lms取 <=，选取cprs子集llps
        llps = self.cprs[(self.cprs['lps'] <= l_pmr[0]) & (self.cprs['lms'] <= l_pmr[1]) &
                         (self.cprs['lrs'] >= l_pmr[2])]
        """
            llps与cprs形式完全一致，llps是cprs的子集pd.DataFrame对象
            eg: llps形式如

                      lcs     lms     lps  lrs
                15_9    4 -0.1116 -0.4465  0.65
                16_7    12 -0.1786 -0.8122  0.72
                ................................
        """
        if llps.empty:
            # 如果筛选出来的子集是空的，说明对整体效果没有提高np.array([0, 0])
            return [improved, effect_num]

        hash_llps = str(llps.index.tolist())
        if hash_llps in self.brust_cache:
            # 为提高运行效率，不用每次都使用_calc_llps_improved计算
            return self.brust_cache[hash_llps]

        # 通过self._calc_llps_improved(llps)继续计算improved和effect_num
        improved, effect_num, _, _ = self._calc_llps_improved(llps)
        brust = [improved, effect_num]
        # 使用缓存字典，llps.index做为key, _calc_llps_improved返回的[improved, effect_num]做为value
        self.brust_cache[hash_llps] = brust
        return brust

    def min_func_improved(self, l_pmr):
        """
        包装min_func具体现实函数，在brust_min函数中使用：
        sco.brute(self.min_func_improved, bnds, full_output=False, finish=None)
        计算最优参数组合，因为在self.min_func中计算的是提高improved值，要想得到最大
        提高参数组合，即self.min_func返回值最大的组合，使用最优sco.brute的目标是
        最小值，所以使用 -self.min_func(l_pmr)[0]，找到最小值，结果的参数组合即为
        最大提高improved值的参数组合
        :param l_pmr: 可迭代序列，eg：(-0.45, -0.11, 0.77), 由三个float值组成的序列
                      l_pmr[0]: 切取self.cprs使用的分类簇交易获利和值，即lps阀值：self.cprs['lps'] <= l_pmr[0]
                      l_pmr[1]: 切取self.cprs使用的分类簇交易获利平均值，即lms阀值：self.cprs['lms'] <= l_pmr[1]
                      l_pmr[2]: 切取self.cprs使用的分类簇交易失败比例，即lrs阀值：self.cprs['lrs'] >= l_pmr[2]
        :return: -self.min_func(l_pmr)[0]，即[improved, effect_num][0], 即improved值，float
        """
        self.brust_progress.show()
        return -self.min_func(l_pmr)[0]

    def _calc_llps_improved(self, llps):
        """
        通过llps从self.nts字典中获取原始训练集特征数据，计算llps组合的情况下拦截的单子的数量，以及胜率，失败率
        即计算llps组合的情况下，即参数l_pmr给定的lps，lms，lrs的特点组合下的交易提升值
        :param llps: eg: llps形式如

                          lcs     lms     lps  lrs
                    15_9    4 -0.1116 -0.4465  0.65
                    16_7    12 -0.1786 -0.8122  0.72
                    ..............................
        :return: improved, effect_num, loss_rate, nts_pd
        """
        nts_pd = pd.DataFrame()
        for component_cluster in llps.index:
            # component_cluster eg:  '14-7', self.nts[component_cluster]即对应的pd.DataFrame对象
            nts_pd = nts_pd.append(self.nts[component_cluster])
            """
                eg: self.nts字典中元素如下所示：
                '14-7':
                            result	buy_deg_ang42	buy_deg_ang252	buy_deg_ang60	buy_deg_ang21	ind	cluster
                2014-11-11	1	    8.341	        -9.450	        0.730	        12.397	        7	7
                2015-10-28	0	    7.144	        -9.818	        -3.886	        6.955	        39	7
                2015-11-04	0	    12.442	        -10.353	        3.313	        7.840	        41	7
                2016-03-30	0	    13.121	        -8.461	        4.498	        4.070	        49	7
                2016-04-15	0	    4.238	        -13.247	        4.693	        1.162	        53	7
                2016-04-15	0	    4.238	        -13.247	        4.693	        1.162	        54	7
            """
        """
            所有的分类簇交易通过nts_pd.append(self.nts[component_cluster])组成必然有重复
            在所有component_cluster中通过nts_pd.drop_duplicates去除重复的单子，注意使用'
            subset='ind'进行去重复，因为self.nts中的pd.DataFrame对象中cluster是不一样的
        """
        nts_pd = nts_pd.drop_duplicates(subset='ind', keep='last')
        """
        nts_pd如下形式，实际上就是所有可能被阻拦的component_cluster对应的交易去除重复的结果
                         result  deg_ang21  deg_ang42  deg_ang60  deg_ang252    ind        cluster
        2011-09-21       0      8.645     -3.294     -8.426      -8.758     40     48      -0.083791
        2011-09-30       0     11.123     -1.887     -2.775       1.585    158     49      -0.066309
        """

        # llps下所有可能被拦截的交易个数
        effect_num = nts_pd.shape[0]
        # llps下被拦截的交易失败率
        loss_rate = nts_pd.result.value_counts()[0] / nts_pd.result.value_counts().sum()
        win_rate = 1 - loss_rate
        # 按照比例有可能提升的效果
        improved = (effect_num / self.fiter.order_has_ret.shape[0]) * (loss_rate - win_rate)
        """
            eg：(effect_num / self.fiter.order_has_ret.shape[0]) * (loss_rate - win_rate)
                effect_num ＝ 10, self.fiter.order_has_ret.shape[0] ＝ 100
                loss_rate ＝ 0.8， win_rate ＝0.2
                －> (10 / 100) * (0.8 - 0.2) = 0.06

                effect_num ＝ 50, self.fiter.order_has_ret.shape[0] ＝ 100
                loss_rate ＝ 0.8， win_rate ＝0.2
                －> (50 / 100) * (0.8 - 0.2) = 0.3

                effect_num ＝ 50, self.fiter.order_has_ret.shape[0] ＝ 100
                loss_rate ＝ 0.9， win_rate ＝0.1
                －> (50 / 100) * (0.9 - 0.1) = 0.4

                即最终提升的比例和llps下被拦截的交易数量成正比，和llps下被拦截的交易失败率成正比
        """
        # TODO 添加其它计算improved的方式，或者权重胜率等多因素决策improved值
        return improved, effect_num, loss_rate, nts_pd

    def dump_file_fn(self):
        """
            主裁本地缓存的存储路径规则：
            ABuEnv.g_project_data_dir ＋ 'ump/ump_main_' ＋ market_name + self.class_unique_id()
        """

        # TODO 如果有裁判覆盖，保留备份，显示通知
        unique_ump_name = 'ump/ump_main_{}_{}'.format(self.market_name, self.class_unique_id())
        return os.path.join(ABuEnv.g_project_data_dir, unique_ump_name)

    def dump_clf(self, llps=None):
        """
        1. 通过llps.index，从self.rts中获取GaussianMixture对象，即self.rts[clf][0]

        eg: llps.index
        Index(['14_7', '14_8', '14_9', '14_10', '15_7', '15_8', '15_9', '15_10',
               '16_7', '16_8', '16_9', '16_10', '17_7', '17_8', '17_9', '17_10',
                ................................................................
               '27_14', '27_15', '27_17', '27_18', '27_24'],
               dtype='object')

        2. 将cluster和GaussianMixture对象做为value，llps.index中的元素做为key形成最终序列化的dict对象形式如下所示
        eg:
            {
            '14_7': (GaussianMixture(covariance_type='full', init_params='kmeans', max_iter=100,
                                    means_init=None, n_components=14, n_init=1, precisions_init=None,
                                    random_state=3, reg_covar=1e-06, tol=0.001, verbose=0,
                                    verbose_interval=10, warm_start=False, weights_init=None), 7),
            '14_8': (GaussianMixture(covariance_type='full', init_params='kmeans', max_iter=100,
                                    means_init=None, n_components=14, n_init=1, precisions_init=None,
                                    random_state=3, reg_covar=1e-06, tol=0.001, verbose=0,
                                    verbose_interval=10, warm_start=False, weights_init=None), 8)
            ....................................................................................................
            }

        3. 通过ABuFileUtil.dump_pickle最终将元素序列化保存在本地
        :param llps: 需要最终保存的llps对象，默认参数llps=None将使用llps = self.llps
                    eg: llps形式如

                          lcs     lms     lps  lrs
                    15_9    4 -0.1116 -0.4465  0.65
                    16_7    12 -0.1786 -0.8122  0.72
                    ................................
        """
        if llps is None:
            if not hasattr(self, 'llps'):
                # fit中订单不足量等终止情况
                return
            llps = self.llps

        clf_cluster_dict = {}
        for clf_cluster in llps.index:
            """

                eg: llps.index
                Index(['14_7', '14_8', '14_9', '14_10', '15_7', '15_8', '15_9', '15_10',
                       '16_7', '16_8', '16_9', '16_10', '17_7', '17_8', '17_9', '17_10',
                        ................................................................
                       '27_14', '27_15', '27_17', '27_18', '27_24'],
                       dtype='object')
            """
            #  clf, cluster = ('14', '7')
            clf, cluster = clf_cluster.split('_')
            clf = int(clf)
            cluster = int(cluster)
            """
                eg:
                    self.rts[clf][0]
                    GaussianMixture(covariance_type='full', init_params='kmeans', max_iter=100,
                                    means_init=None, n_components=14, n_init=1, precisions_init=None,
                                    random_state=3, reg_covar=1e-06, tol=0.001, verbose=0,
                                    verbose_interval=10, warm_start=False, weights_init=None)
            """
            clf_cluster_dict[clf_cluster] = (self.rts[clf][0], cluster)
            """
                eg: clf_cluster_dict形式如下所示：
                {
                '14_7': (GaussianMixture(covariance_type='full', init_params='kmeans', max_iter=100,
                                        means_init=None, n_components=14, n_init=1, precisions_init=None,
                                        random_state=3, reg_covar=1e-06, tol=0.001, verbose=0,
                                        verbose_interval=10, warm_start=False, weights_init=None), 7),
                '14_8': (GaussianMixture(covariance_type='full', init_params='kmeans', max_iter=100,
                                        means_init=None, n_components=14, n_init=1, precisions_init=None,
                                        random_state=3, reg_covar=1e-06, tol=0.001, verbose=0,
                                        verbose_interval=10, warm_start=False, weights_init=None), 8)
                ....................................................................................................
                }
            """
        # 通过ABuFileUtil.dump_pickle将clf_cluster_dict进行序列化
        ABuFileUtil.dump_pickle(clf_cluster_dict, self.dump_file_fn(), how='zero')

    def predict(self, x, need_hit_cnt=1):
        """
        主交易决策函数，从CachedUmpManager中获取缓存clf_cluster_dict，迭代ump字典对象，
        对交易特征形成的x使用字典中元素(clf, cluster)中的clf进行predict
        结果和 (clf, cluster)中存储的cluster一致的情况下代表hit，最终对交易
        做出决策

        :param x: 交易特征形成的x eg: array([[  8.341,  -9.45 ,   0.73 ,  12.397]])
        :param need_hit_cnt: 对交易特征形成的x使用字典中元素(clf, cluster)中的clf进行predict结果和
                            (clf, cluster)中存储的cluster一致的情况下代表hit一次：count_hit += 1
                            只有当need_hit_cnt == count_hit才最终做出决策对交易进行拦截

        :return: 最终做出决策对交易是否进行拦截，拦截即返回1，放行即返回0
        """

        # 统一从CachedUmpManager中获取缓存ump，没有缓存的情况下load_pickle
        clf_cluster_dict = AbuUmpMainBase.dump_clf_manager.get_ump(self)
        """
            eg: clf_cluster_dict
                {
                '14_7': (GaussianMixture(covariance_type='full', init_params='kmeans', max_iter=100,
                                        means_init=None, n_components=14, n_init=1, precisions_init=None,
                                        random_state=3, reg_covar=1e-06, tol=0.001, verbose=0,
                                        verbose_interval=10, warm_start=False, weights_init=None), 7),
                '14_8': (GaussianMixture(covariance_type='full', init_params='kmeans', max_iter=100,
                                        means_init=None, n_components=14, n_init=1, precisions_init=None,
                                        random_state=3, reg_covar=1e-06, tol=0.001, verbose=0,
                                        verbose_interval=10, warm_start=False, weights_init=None), 8)
                ....................................................................................................
                }
        """
        count_hit = 0
        # if need_hit_cnt > 1 and len(clf_cluster_dict) < 50:
        #     need_hit_cnt = 1
        for clf, cluster in clf_cluster_dict.values():
            """
                eg: (clf, cluster)
                    (GaussianMixture(covariance_type='full', init_params='kmeans', max_iter=100,
                                        means_init=None, n_components=14, n_init=1, precisions_init=None,
                                        random_state=3, reg_covar=1e-06, tol=0.001, verbose=0,
                                        verbose_interval=10, warm_start=False, weights_init=None), 7)
            """
            predict_cluster = clf.predict(x)
            if predict_cluster == cluster:
                # 即使用clf对x进行predict结果和 (clf, cluster)中存储的cluster一致的情况下代表hit
                count_hit += 1
                if need_hit_cnt == count_hit:
                    # 只有当need_hit_cnt == count_hit才最终做出决策对交易进行拦截，即返回1
                    return 1
        return 0

    def predict_kwargs(self, w_col=None, need_hit_cnt=1, **kwargs):
        """
        主裁交易决策函数，对kwargs关键字参数所描述的交易特征进行拦截决策，从子类对象必须实现的虚方法get_predict_col中获取特征列，
        将kwargs中的特征值转换为x，套接self.predict进行核心裁决判定

        :param w_col: 自定义特征列，一般不会使用，正常情况从子类对象必须实现的虚方法get_predict_col中获取特征列
        :param need_hit_cnt: 透传给self.predict中need_hit_cnt参数，做为分类簇匹配拦截阀值
        :param kwargs: 需要和子类对象实现的虚方法get_predict_col中获取特征列对应的
                       关键字参数，eg: buy_deg_ang42=3.378, buy_deg_ang60=3.458
                                     buy_deg_ang21=3.191, buy_deg_ang252=1.818
        :return: 是否对kwargs关键字参数所描述的交易特征进行拦截，int，不拦截: 0，拦截: 1
        """
        if w_col is None:
            # 如果不自定义特征列，即从子类对象必须实现的虚方法get_predict_col中获取特征列
            w_col = self.get_predict_col()
            """eg, w_col: ['buy_deg_ang42', 'buy_deg_ang252', 'buy_deg_ang60', 'buy_deg_ang21']"""

        for col in w_col:
            if col not in kwargs:
                # 如果kwargs中没有某一个特征值，raise ValueError
                raise ValueError('col not in kwargs!')
        # 将kwargs中的特征值转换为x
        x = np.array([kwargs[col] for col in w_col])
        """eg, x: array([ 3.378,  3.458,  3.191,  1.818])"""
        x = x.reshape(1, -1)
        # 套接self.predict进行核心裁决判定
        return self.predict(x, need_hit_cnt)

    def show_parse_rt(self, rt):
        """
        辅助工具函数，使用柱状图可视化分类簇中cluster，result的数据关系
        :param rt: 在fit函数中保存在字典中的元素：
                    clf_component_dict[component] = (clf, cluster_ind)

                    eg：rt，(GaussianMixtured对象，cluster_ind: GMM(component)分component个类中，
                                                  满足子分类的失败率大于阀值threshold的子分类序号)
        """
        # GaussianMixtured对象
        clf = rt[0]
        cluster = clf.predict(self.fiter().x)
        # copy一个因为要添加cluster列以便于使用crosstab
        df = self.fiter.df.copy()
        df['cluster'] = cluster
        # 交叉表组织cluster和result的交叉数据xt
        xt = pd.crosstab(df['cluster'], df['result'])
        xt_pct = xt.div(xt.sum(1).astype(float), axis=0)
        # 通过kind='bar'绘制柱状图显示分类簇中cluster，result的数据关系
        xt_pct.plot(
            figsize=(16, 8),
            kind='bar',
            stacked=True,
            title=str('cluster') + ' -> ' + str('result'))
        plt.xlabel(str('cluster'))
        plt.ylabel(str('result'))

    def choose_cprs_component(self, llps=None):
        """
        辅助工具函数，输出显示在llps下训练集交易中的生效数量，和训练集可提升数据比例
        以及可视化所有训练集中被拦截的交易profit_cg的cumsum

        :param llps: llps对象，默认参数llps=None将使用llps = self.llps

                eg: llps形式如

                      lcs     lms     lps  lrs
                15_9    4 -0.1116 -0.4465  0.65
                16_7    12 -0.1786 -0.8122  0.72
                ................................
        """

        if llps is None:
            llps = self.llps

        improved, effect_num, loss_rate, nts_pd = self._calc_llps_improved(llps)

        self.log_func('拦截的交易中正确拦截比例:{0:.3f}'.format(loss_rate))
        self.log_func('训练集中拦截生效后可提升比例:{0:.3f}'.format(improved))
        self.log_func('训练集中拦截生效数量{0}， 训练集中拦截生效占总训练集比例{1:.3f}%'.format(
            effect_num, effect_num / self.fiter.df.shape[0] * 100))

        """
        nts_pd如下形式，实际上就是所有可能被阻拦的component_cluster对应的交易去除重复的结果
                         result  deg_ang21  deg_ang42  deg_ang60  deg_ang252    ind        cluster
        2011-09-21       0      8.645     -3.294     -8.426      -8.758     40     48      -0.083791
        2011-09-30       0     11.123     -1.887     -2.775       1.585    158     49      -0.066309
        """
        # 通过nts_pd通过apply迭代每一行，即每一笔交易的ind映射的原始交易单self.fiter.order_has_ret中的profit_cg值
        nts_pd['profit_cg'] = nts_pd.apply(lambda x: self.fiter.order_has_ret.ix[int(x.ind)].profit_cg, axis=1)
        """
            eg：nts_pd添加了新列profit_cg后如下所示：
                        result  buy_deg_ang42  buy_deg_ang252  buy_deg_ang60  \
            2015-10-28       0          7.144          -9.818         -3.886
            2015-04-22       1         20.640           2.751         20.436
            2015-06-09       0         10.741          16.352         28.340
            2014-11-11       1          8.341          -9.450          0.730
            2015-11-04       0         12.442         -10.353          3.313
            2016-03-30       0         13.121          -8.461          4.498
            2014-11-26       0         14.052           6.061          7.566
            2016-04-15       0          4.238         -13.247          4.693
            2016-04-15       0          4.238         -13.247          4.693
            2015-12-16       0         12.906           6.312         16.638
            2016-01-29       0         -5.578          16.161         -5.167
            2016-01-29       0         -5.578          16.161         -5.167
            2014-11-12       0          3.963           6.595         -7.524

                        buy_deg_ang21  ind  cluster  profit_cg
            2015-10-28          6.955   39        7    -0.0702
            2015-04-22         18.781   23        8     0.0121
            2015-06-09         -0.937   26        8    -0.1128
            2014-11-11         12.397    7        9     0.1857
            2015-11-04          7.840   41        9    -0.1303
            2016-03-30          4.070   49        9    -0.0863
            2014-11-26         12.494    9       14    -0.0946
            2016-04-15          1.162   53       15    -0.1149
            2016-04-15          1.162   54       15    -0.1149
            2015-12-16         12.379   43       17    -0.1041
            2016-01-29         -3.855   45       18    -0.0327
            2016-01-29         -3.855   46       18    -0.0327
            2014-11-12          6.671    8       24    -0.0140
        """

        with plt_show():
            # 可视化所有训练集中被拦截的交易profit_cg的cumsum
            nts_pd.sort_index()['profit_cg'].cumsum().plot()

    def best_hit_cnt_info(self, llps=None):
        """
        辅助工具函数，通过统计在各个分类簇中重复出现的交易单子，计算出
        平均值

        :param llps: llps对象，默认参数llps=None将使用llps = self.llps

                eg: llps形式如
                      lcs     lms     lps  lrs
                15_9    4 -0.1116 -0.4465  0.65
                16_7    12 -0.1786 -0.8122  0.72
                ................................
        """
        if llps is None:
            llps = self.llps

        nts_pd = pd.DataFrame()
        for component_cluster in llps.index:
            # component_cluster eg:  '14-7', self.nts[component_cluster]即对应的pd.DataFrame对象
            nts_pd = nts_pd.append(self.nts[component_cluster])
            """
                eg: self.nts字典中元素如下所示：
                '14-7':
                            result	buy_deg_ang42	buy_deg_ang252	buy_deg_ang60	buy_deg_ang21	ind	cluster
                2014-11-11	1	    8.341	        -9.450	        0.730	        12.397	        7	7
                2015-10-28	0	    7.144	        -9.818	        -3.886	        6.955	        39	7
                2015-11-04	0	    12.442	        -10.353	        3.313	        7.840	        41	7
                2016-03-30	0	    13.121	        -8.461	        4.498	        4.070	        49	7
                2016-04-15	0	    4.238	        -13.247	        4.693	        1.162	        53	7
                2016-04-15	0	    4.238	        -13.247	        4.693	        1.162	        54	7
            """

        """
            这里不去除重复nts_pd，就是需要统计nts_pd重复ind在各个分类簇中出现的频率
            使用： ind_result_xt = pd.crosstab(nts_pd.ind, nts_pd.result)

            eg：nts_pd.ind
                            2014-11-11     7
                            2015-10-28    39
                            2015-11-04    41
                            2016-03-30    49
                            2016-04-15    53
                            2016-04-15    54
                            2014-11-26     9
                            2015-04-22    23
                            2015-12-16    43
                            2015-06-09    26
                                          ..
                            2014-11-11     7
                            2015-11-04    41
                            2016-03-30    49
                            2014-11-26     9
                            2016-04-15    53
                            2016-04-15    54
                            2015-12-16    43
                            2016-01-29    45
                            2016-01-29    46
                            2014-11-12     8
            eg：nts_pd.result
                            2014-11-11    1
                            2015-10-28    0
                            2015-11-04    0
                            2016-03-30    0
                            2016-04-15    0
                            2016-04-15    0
                            2014-11-26    0
                            2015-04-22    1
                            2015-12-16    0
                            2015-06-09    0
                                         ..
                            2014-11-11    1
                            2015-11-04    0
                            2016-03-30    0
                            2014-11-26    0
                            2016-04-15    0
                            2016-04-15    0
                            2015-12-16    0
                            2016-01-29    0
                            2016-01-29    0
                            2014-11-12    0
            eg：ind_result_xt
                            ind_result_pd
                            result   0  1
                            ind
                            7        0  8
                            8        6  0
                            9       14  0
                            23       0  3
                            26      14  0
                            39       1  0
                            41      14  0
                            43      14  0
                            45      14  0
                            46      14  0
                            49      14  0
                            53      14  0
                            54      14  0
            即ind_result_xt统计出各个分类簇中重复的拦截次数
        """
        ind_result_xt = pd.crosstab(nts_pd.ind, nts_pd.result)

        # 得到失败的平均数, 亦可以使用中位数等其它统计函数
        mean_hit_failed = ind_result_xt[0].mean()
        # 向上取整数，如果是2.1则3
        mean_hit_failed = np.ceil(mean_hit_failed)
        self.log_func('mean_hit_failed = {}'.format(mean_hit_failed))

    def hit_cnt(self, x):
        """
        辅助统计工具函数，从CachedUmpManager中获取缓存clf_cluster_dict，迭代ump字典对象，
        对交易特征形成的x使用字典中元素(clf, cluster)中的clf进行predict
        结果和 (clf, cluster)中存储的cluster一致的情况下代表hit

        :param x: 交易特征形成的x eg: array([[  8.341,  -9.45 ,   0.73 ,  12.397]])
        :return: kwargs关键字参数所描述的交易特征进行分类簇命中统计，返回int值
        """

        # 统一从CachedUmpManager中获取缓存ump，没有缓存的情况下load_pickle
        clf_cluster_dict = AbuUmpBase.dump_clf_manager.get_ump(self)
        hit_cnt = 0
        for clf, cluster in clf_cluster_dict.values():
            """
                eg: (clf, cluster)
                    (GaussianMixture(covariance_type='full', init_params='kmeans', max_iter=100,
                                        means_init=None, n_components=14, n_init=1, precisions_init=None,
                                        random_state=3, reg_covar=1e-06, tol=0.001, verbose=0,
                                        verbose_interval=10, warm_start=False, weights_init=None), 7)
            """
            predict_cluster = clf.predict(x)
            if predict_cluster == cluster:
                # 即使用clf对x进行predict结果和 (clf, cluster)中存储的cluster一致的情况下代表hit
                hit_cnt += 1
        # 最终返回交易特征进行分类簇命中统计结果，int值
        return hit_cnt

    def predict_hit_kwargs(self, w_col=None, **kwargs):
        """
        辅助统计工具函数，对kwargs关键字参数所描述的交易特征进行ump分类簇命中统计，从子类对象必须实现的虚方法get_predict_col中获取特征列，
        将kwargs中的特征值转换为x，套接self.hit_cnt进行分类簇命中统计

        :param w_col: 自定义特征列，一般不会使用，正常情况从子类对象必须实现的虚方法get_predict_col中获取特征列
        :param kwargs: 需要和子类对象实现的虚方法get_predict_col中获取特征列对应的
                       关键字参数，eg: buy_deg_ang42=3.378, buy_deg_ang60=3.458
                                     buy_deg_ang21=3.191, buy_deg_ang252=1.818
        :return: kwargs关键字参数所描述的交易特征进行分类簇命中统计，返回int值
        """
        if w_col is None:
            # 如果不自定义特征列，即从子类对象必须实现的虚方法get_predict_col中获取特征列
            w_col = self.get_predict_col()
            """eg, w_col: ['buy_deg_ang42', 'buy_deg_ang252', 'buy_deg_ang60', 'buy_deg_ang21']"""

        for col in w_col:
            if col not in kwargs:
                # 如果kwargs中没有某一个特征值，raise ValueError
                raise ValueError('col not in kwargs!')
        # 将kwargs中的特征值转换为x
        x = np.array([kwargs[col] for col in w_col])
        """eg, x: array([ 3.378,  3.458,  3.191,  1.818])"""
        x = x.reshape(1, -1)
        # 与predict_kwargs不同的地方主要就是在这里，使用hit_cnt进行分类簇命中统计
        return self.hit_cnt(x)
