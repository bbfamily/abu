# -*- encoding:utf-8 -*-
"""
    封装常用学习器的初始化流程的模块
"""

from __future__ import print_function
from __future__ import absolute_import
from __future__ import division

from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.ensemble import AdaBoostRegressor
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.ensemble import BaggingRegressor
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import RandomForestRegressor
# noinspection PyPep8Naming
from sklearn.ensemble import GradientBoostingClassifier as GBC
# noinspection PyPep8Naming
from sklearn.ensemble import GradientBoostingRegressor as GBR

from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsClassifier
from sklearn.multiclass import OneVsOneClassifier, OneVsRestClassifier

from ..CoreBu.ABuFixes import GMM
from ..MLBu import ABuMLGrid

# TODO 对外的版本暂时全部都使用sklearn不从其它第三方库import，增加可选开关等设置
# try:
#     # noinspection PyPep8Naming
#     from xgboost.sklearn import XGBClassifier as GBC
#     # noinspection PyPep8Naming
#     from xgboost.sklearn import XGBRegressor as GBR
# except ImportError:
#     # noinspection PyPep8Naming
#     from sklearn.ensemble import GradientBoostingClassifier as GBC
#     # noinspection PyPep8Naming
#     from sklearn.ensemble import GradientBoostingRegressor as GBR
# try:
#     # noinspection PyPep8Naming
#     from hmmlearn.hmm import GaussianHMM as GMM
# except ImportError:
#     from ..CoreBu.ABuFixes import GMM

__author__ = '阿布'
__weixin__ = 'abu_quant'


class AbuMLCreater(object):
    """封装常用有简单和无监督学习器实例化类"""

    def __init__(self):
        """
        默认使用线性回归初始化回归器:
            self.reg = self.linear_regression()
        默认使用带概率估计的svm初始化分类器:
            self.clf = self.svc(probability=True)

        默认无简单学习器：hmm，pca，keman全部初始None值
        """

        # 有简单机器学习，分类回归
        self.reg = self.linear_regression()
        self.clf = self.svc(probability=True)
        # 无监督机器学习，pca，聚类，hmm, 默认不初始化
        self.hmm = None
        self.pca = None
        self.kmean = None

    def __str__(self):
        """打印对象显示：reg, clf, hmm, pca, kmean"""
        return 'reg: {}\nclf: {}\nhmm: {}\npca: {}\nkmean: {}\n'.format(self.reg, self.clf, self.hmm, self.pca,
                                                                        self.kmean)

    __repr__ = __str__

    def pca_decomposition(self, assign=True, **kwargs):
        """
        无监督学习器，实例化PCA，默认使用pca = PCA(0.95)，通过**kwargs即
        关键字参数透传PCA，即PCA(**kwargs)

        :param assign: 是否保存实例后的PCA对象，默认True，self.pca = pca
        :param kwargs: 有参数情况下初始化: PCA(**kwargs)
                       无参数情况下初始化: pca = PCA(0.95)
        :return: 实例化的PCA对象
        """
        if kwargs is not None and len(kwargs) > 0:
            pca = PCA(**kwargs)
        else:
            # 没参数直接要保留95%
            pca = PCA(0.95)
        if assign:
            self.pca = pca

        return pca

    def kmean_cluster(self, assign=True, **kwargs):
        """
        无监督学习器，实例化KMeans，默认使用KMeans(n_clusters=2, random_state=0)，
        通过**kwargs即关键字参数透传KMeans，即KMeans(**kwargs)

        :param assign: 是否保存实例后的kmean对象，默认True，self.kmean = kmean
        :param kwargs: 有参数情况下初始化: KMeans(**kwargs)
                       无参数情况下初始化: KMeans(n_clusters=2, random_state=0)
        :return: 实例化的KMeans对象
        """
        if kwargs is not None and len(kwargs) > 0:
            kmean = KMeans(**kwargs)
        else:
            # 默认也只有两个n_clusters
            kmean = KMeans(n_clusters=2, random_state=0)
        if assign:
            self.kmean = kmean
        return kmean

    def hmm_gaussian(self, assign=True, **kwargs):
        """
        无监督学习器，实例化GMM，默认使用GMM(n_components=2)，通过**kwargs即
        关键字参数透传GMM，即GMM(**kwargs)

        导入模块使用
            try:
                from hmmlearn.hmm import GaussianHMM as GMM
            except ImportError:
                from ..CoreBu.ABuFixes import GMM
        即优先选用hmmlearn中的GaussianHMM，没有安装的情况下使用sklearn中的GMM

        :param assign: 是否保存实例后的hmm对象，默认True，self.hmm = hmm
        :param kwargs: 有参数情况下初始化: GMM(**kwargs)
                       无参数情况下初始化: GMM(n_components=2)
        :return: 实例化的GMM对象
        """
        if kwargs is not None and len(kwargs) > 0:
            hmm = GMM(**kwargs)
        else:
            # 默认只有n_components=2, 两个分类
            hmm = GMM(n_components=2)
        if assign:
            self.hmm = hmm
        return hmm

    # noinspection PyMethodMayBeStatic
    def _estimators_prarms_best(self, create_func, x, y, param_grid, assign, n_jobs, show,
                                grid_callback=ABuMLGrid.grid_search_init_n_estimators):
        """
        封装使用ABuMLGrid寻找针对学习器的最优参数值，针对不同学习器，选择不同的
        关键字参数做最优搜索，将寻找到的最优参数做为**kwargs用来重新构造学习器

        :param create_func: callable, 学习器函数构造器，eg：self.adaboost_classifier
        :param x: 训练集x矩阵，numpy矩阵
        :param y: 训练集y序列，numpy序列
        :param param_grid: 最优字典关键字参数，使用grid_search_mul_init_kwargs寻找参数最优值：
                        eg: _, best_params = ABuMLGrid.grid_search_mul_init_kwargs(estimator, x, y,
                                                       param_grid=param_grid, n_jobs=n_jobs, show=show)
        :param assign: 是否保存实例化后最优参数的学习器对象，默认True，透传create_func，用来根据最优参数重新构造学习器保存在类变量
                        eg: create_func(assign=assign, **best_params)
        :param show: 是否可视化最优参数搜索结果
        :param grid_callback: 如果没有有传递最优字典关键字参数param_grid，使用学习器对应的grid_callback函数，搜索特定的最优参数
                              默认ABuMLGrid.grid_search_init_n_estimators
        :return: 通过最优参数构造的学习器对象，eg: create_func(assign=assign, **best_params)
        """
        # 通过create_func创建一个示例学习器，assign=False
        estimator = create_func(assign=False)
        if param_grid is not None and isinstance(param_grid, dict):
            # 如果有传递最优字典关键字参数，使用grid_search_mul_init_kwargs寻找参数最优值
            _, best_params = ABuMLGrid.grid_search_mul_init_kwargs(estimator, x, y,
                                                                   param_grid=param_grid, n_jobs=n_jobs, show=show)
        else:
            # 如果没有有传递最优字典关键字参数，使用学习器对应的grid_callback函数，默认ABuMLGrid.grid_search_init_n_estimators
            _, best_params = grid_callback(estimator, x, y, show=show)

        if best_params is not None:
            # 将寻找到的最优参数best_params，做为参数重新传递create_func(assign=assign, **best_params)
            return create_func(assign=assign, **best_params)

    def bagging_classifier(self, assign=True, base_estimator=DecisionTreeClassifier(), **kwargs):
        """
        有监督学习分类器，实例化BaggingClassifier，默认使用：
            BaggingClassifier(base_estimator=base_estimator, n_estimators=200,
                              bootstrap=True, oob_score=True, random_state=1)

        通过**kwargs即关键字参数透传BaggingClassifier，即:
            BaggingClassifier(**kwargs)

        :param base_estimator: 默认使用DecisionTreeClassifier()
        :param assign: 是否保存实例后的BaggingClassifier对象，默认True，self.clf = clf
        :param kwargs: 有参数情况下初始化: BaggingClassifier(**kwargs)
                       无参数情况下初始化: BaggingClassifier(base_estimator=base_estimator, n_estimators=200,
                                                           bootstrap=True, oob_score=True, random_state=1)
        :return: 实例化的BaggingClassifier对象
        """
        if kwargs is not None and len(kwargs) > 0:
            if 'base_estimator' not in kwargs:
                kwargs['base_estimator'] = base_estimator
            clf = BaggingClassifier(**kwargs)
        else:
            clf = BaggingClassifier(base_estimator=base_estimator, n_estimators=200,
                                    bootstrap=True, oob_score=True, random_state=1)
        if assign:
            self.clf = clf
        return clf

    def bagging_classifier_best(self, x, y, param_grid=None, assign=True, n_jobs=-1, show=True):
        """
        寻找BaggingClassifier构造器的最优参数
        上层AbuML中bagging_classifier_best函数，直接使用AbuML中的x，y数据调用
        eg：
            bagging_classifier_best无param_grid参数调用：

            from abupy import AbuML, ml
            ttn_abu = AbuML.create_test_more_fiter()
            ttn_abu.bagging_classifier_best()

            bagging_classifier_best有param_grid参数调用：

            param_grid = {'max_samples': np.arange(1, 5), 'n_estimators': np.arange(100, 300, 50)}
            ttn_abu.bagging_classifier_best(param_grid=param_grid, n_jobs=-1)

            out: BaggingClassifier(max_samples=4, n_estimators=100)


        :param x: 训练集x矩阵，numpy矩阵
        :param y: 训练集y序列，numpy序列
        :param param_grid: 最优字典关键字参数，
                        eg：param_grid = {'max_samples': np.arange(1, 5), 'n_estimators': np.arange(100, 300, 50)}
        :param assign: 是否保存实例化后最优参数的学习器对象，默认True
        :param n_jobs: 并行执行的进程任务数量，默认-1, 开启与cpu相同数量的进程数
        :param show: 是否可视化最优参数搜索结果
        :return: 通过最优参数构造的BaggingClassifier对象
        """
        return self._estimators_prarms_best(self.bagging_classifier, x, y, param_grid, assign, n_jobs, show)

    def bagging_regressor(self, assign=True, base_estimator=DecisionTreeRegressor(), **kwargs):
        """
        有监督学习回归器，实例化BaggingRegressor，默认使用：
            BaggingRegressor(base_estimator=base_estimator, n_estimators=200,
                             bootstrap=True, oob_score=True, random_state=1)

        通过**kwargs即关键字参数透传BaggingRegressor，即:
            BaggingRegressor(**kwargs)

        :param base_estimator: 默认使用DecisionTreeRegressor()
        :param assign: 是否保存实例后的BaggingRegressor对象，默认True，self.reg = reg
        :param kwargs: 有参数情况下初始化: BaggingRegressor(**kwargs)
                       无参数情况下初始化: BaggingRegressor(base_estimator=base_estimator, reg_core, n_estimators=200,
                                                          bootstrap=True, oob_score=True, random_state=1)
        :return: 实例化的BaggingRegressor对象
        """
        if kwargs is not None and len(kwargs) > 0:
            if 'base_estimator' not in kwargs:
                kwargs['base_estimator'] = base_estimator
            reg = BaggingRegressor(**kwargs)
        else:
            reg = BaggingRegressor(base_estimator=base_estimator, n_estimators=200,
                                   bootstrap=True, oob_score=True, random_state=1)

        if assign:
            self.reg = reg
        return reg

    def bagging_regressor_best(self, x, y, param_grid=None, assign=True, n_jobs=-1, show=True):
        """
        寻找BaggingRegressor构造器的最优参数
        上层AbuML中bagging_regressor_best函数，直接使用AbuML中的x，y数据调用
        eg：
            bagging_regressor_best无param_grid参数调用：

            from abupy import AbuML, ml
            ttn_abu = AbuML.create_test_more_fiter()
            ttn_abu.bagging_regressor_best()

            bagging_regressor_best有param_grid参数调用：

            param_grid = {'max_samples': np.arange(1, 5), 'n_estimators': np.arange(100, 300, 50)}
            ttn_abu.bagging_regressor_best(param_grid=param_grid, n_jobs=-1)

            out: BaggingRegressor(max_samples=4, n_estimators=250)


        :param x: 训练集x矩阵，numpy矩阵
        :param y: 训练集y序列，numpy序列
        :param param_grid: 最优字典关键字参数，
                        eg：param_grid = {'max_samples': np.arange(1, 5), 'n_estimators': np.arange(100, 300, 50)}
        :param assign: 是否保存实例化后最优参数的学习器对象，默认True
        :param n_jobs: 并行执行的进程任务数量，默认-1, 开启与cpu相同数量的进程数
        :param show: 是否可视化最优参数搜索结果
        :return: 通过最优参数构造的BaggingRegressor对象
        """
        return self._estimators_prarms_best(self.bagging_regressor, x, y, param_grid, assign, n_jobs, show)

    def adaboost_regressor(self, assign=True, base_estimator=DecisionTreeRegressor(), **kwargs):
        """
        有监督学习回归器，实例化AdaBoostRegressor，默认使用：
            AdaBoostRegressor(base_estimator=base_estimator, n_estimators=100, random_state=1)

        通过**kwargs即关键字参数透传AdaBoostRegressor，即:
            AdaBoostRegressor(**kwargs)

        :param base_estimator: 默认使用DecisionTreeRegressor()
        :param assign: 是否保存实例后的AdaBoostRegressor对象，默认True，self.reg = reg
        :param kwargs: 有参数情况下初始化: AdaBoostRegressor(**kwargs)
                       无参数情况下初始化: AdaBoostRegressor(n_estimators=100, random_state=1)

        :return: 实例化的AdaBoostRegressor对象
        """
        if kwargs is not None and len(kwargs) > 0:
            if 'base_estimator' not in kwargs:
                kwargs['base_estimator'] = base_estimator
            reg = AdaBoostRegressor(**kwargs)
        else:
            reg = AdaBoostRegressor(base_estimator=base_estimator, n_estimators=100, random_state=1)

        if assign:
            self.reg = reg

        return reg

    def adaboost_regressor_best(self, x, y, param_grid=None, assign=True, n_jobs=-1, show=True):
        """
        寻找AdaBoostRegressor构造器的最优参数

        上层AbuML中adaboost_regressor_best函数，直接使用AbuML中的x，y数据调用
        eg：
            adaboost_regressor_best无param_grid参数调用：

            from abupy import AbuML, ml
            ttn_abu = AbuML.create_test_more_fiter()
            ttn_abu.adaboost_regressor_best()

            adaboost_classifier_best有param_grid参数调用：

            param_grid = {'learning_rate': np.arange(0.2, 1.2, 0.2), 'n_estimators': np.arange(10, 100, 10)}
            ttn_abu.adaboost_regressor_best(param_grid=param_grid, n_jobs=-1)

            out: AdaBoostRegressor(learning_rate=0.8, n_estimators=40)


        :param x: 训练集x矩阵，numpy矩阵
        :param y: 训练集y序列，numpy序列
        :param param_grid: 最优字典关键字参数，
                        eg：param_grid = {'learning_rate': np.arange(0.2, 1.2, 0.2),
                                         'n_estimators': np.arange(10, 100, 10)}
        :param assign: 是否保存实例化后最优参数的学习器对象，默认True
        :param n_jobs: 并行执行的进程任务数量，默认-1, 开启与cpu相同数量的进程数
        :param show: 是否可视化最优参数搜索结果
        :return: 通过最优参数构造的AdaBoostRegressor对象
        """
        return self._estimators_prarms_best(self.adaboost_regressor, x, y, param_grid, assign, n_jobs, show)

    def adaboost_classifier(self, assign=True, base_estimator=DecisionTreeClassifier(), **kwargs):
        """
        有监督学习分类器，实例化AdaBoostClassifier，默认使用：
            AdaBoostClassifier(base_estimator=base_estimator, n_estimators=100, random_state=1)

        通过**kwargs即关键字参数透传AdaBoostClassifier，即:
            AdaBoostClassifier(**kwargs)

        :param base_estimator: 默认使用DecisionTreeClassifier()
        :param assign: 是否保存实例后的AdaBoostClassifier对象，默认True，self.clf = clf
        :param kwargs: 有参数情况下初始化: AdaBoostClassifier(**kwargs)
                       无参数情况下初始化: AdaBoostClassifier(n_estimators=100, random_state=1)

        :return: 实例化的AdaBoostClassifier对象
        """
        if kwargs is not None and len(kwargs) > 0:
            if 'base_estimator' not in kwargs:
                kwargs['base_estimator'] = base_estimator
            clf = AdaBoostClassifier(**kwargs)
        else:
            clf = AdaBoostClassifier(base_estimator=base_estimator, n_estimators=100, random_state=1)
        if assign:
            self.clf = clf
        return clf

    def adaboost_classifier_best(self, x, y, param_grid=None, assign=True, n_jobs=-1, show=True):
        """
        寻找AdaBoostClassifier构造器的最优参数

        上层AbuML中adaboost_classifier_best函数，直接使用AbuML中的x，y数据调用
        eg：
             adaboost_classifier_best无param_grid参数调用：

             from abupy import AbuML, ml
             ttn_abu = AbuML.create_test_more_fiter()
             ttn_abu.adaboost_classifier_best()

             adaboost_classifier_best有param_grid参数调用：

             param_grid = {'learning_rate': np.arange(0.2, 1.2, 0.2), 'n_estimators': np.arange(10, 100, 10)}
             ttn_abu.adaboost_classifier_best(param_grid=param_grid, n_jobs=-1)

             out: AdaBoostClassifier(learning_rate=0.6, n_estimators=70)


        :param x: 训练集x矩阵，numpy矩阵
        :param y: 训练集y序列，numpy序列
        :param param_grid: 最优字典关键字参数，
                     eg：param_grid = {'learning_rate': np.arange(0.2, 1.2, 0.2),
                                       'n_estimators': np.arange(10, 100, 10)}
        :param assign: 是否保存实例化后最优参数的学习器对象，默认True
        :param n_jobs: 并行执行的进程任务数量，默认-1, 开启与cpu相同数量的进程数
        :param show: 是否可视化最优参数搜索结果
        :return: 通过最优参数构造的AdaBoostClassifier对象
        """
        return self._estimators_prarms_best(self.adaboost_classifier, x, y, param_grid, assign, n_jobs, show)

    def xgb_regressor(self, assign=True, **kwargs):
        """
        有监督学习回归器，默认使用：
                        GBR(n_estimators=100)
        通过**kwargs即关键字参数透传GBR(**kwargs)，即:
                        GBR(**kwargs)

        注意导入使用：
            try:
                from xgboost.sklearn import XGBRegressor as GBR
            except ImportError:
                from sklearn.ensemble import GradientBoostingRegressor as GBR

        :param assign: 是否保存实例后的回归器对象，默认True，self.reg = reg
        :param kwargs: 有参数情况下初始化: GBR(n_estimators=100)
                       无参数情况下初始化: GBR(**kwargs)

        :return: 实例化的GBR对象
        """
        if kwargs is not None and len(kwargs) > 0:
            reg = GBR(**kwargs)
        else:
            reg = GBR(n_estimators=100)
        if assign:
            self.reg = reg
        return reg

    def xgb_regressor_best(self, x, y, param_grid=None, assign=True, n_jobs=-1, show=True):
        """
        寻找GradientBoostingRegressor构造器的最优参数

        上层AbuML中xgb_regressor_best函数，直接使用AbuML中的x，y数据调用
        eg：
             xgb_regressor_best无param_grid参数调用：

             from abupy import AbuML, ml
             ttn_abu = AbuML.create_test_more_fiter()
             ttn_abu.xgb_regressor_best()

             xgb_regressor_best有param_grid参数调用：

             param_grid = {'learning_rate': np.arange(0.1, 0.5, 0.05), 'n_estimators': np.arange(10, 100, 10)}
             ttn_abu.xgb_regressor_best(param_grid=param_grid, n_jobs=-1)

             out: GradientBoostingRegressor(learning_rate=0.2, n_estimators=70)


        :param x: 训练集x矩阵，numpy矩阵
        :param y: 训练集y序列，numpy序列
        :param param_grid: 最优字典关键字参数，
                     eg：param_grid = {'learning_rate': np.arange(0.1, 0.5, 0.05),
                                      'n_estimators': np.arange(10, 100, 10)}
        :param assign: 是否保存实例化后最优参数的学习器对象，默认True
        :param n_jobs: 并行执行的进程任务数量，默认-1, 开启与cpu相同数量的进程数
        :param show: 是否可视化最优参数搜索结果
        :return: 通过最优参数构造的GradientBoostingRegressor对象
        """
        return self._estimators_prarms_best(self.xgb_regressor, x, y, param_grid, assign, n_jobs, show)

    def xgb_classifier(self, assign=True, **kwargs):
        """
        有监督学习分类器，默认使用：
                        GBC(n_estimators=100)

        通过**kwargs即关键字参数透传GBC(**kwargs)，即:
                        GBC(**kwargs)

        注意导入使用：
            try:
                from xgboost.sklearn import XGBClassifier as GBC
            except ImportError:
                from sklearn.ensemble import GradientBoostingClassifier as GBC


        :param assign: 是否保存实例后的分类器对象，默认True，self.clf = clf
        :param kwargs: 有参数情况下初始化: GBC(n_estimators=100)
                       无参数情况下初始化: GBC(**kwargs)

        :return: 实例化的GBC对象
        """
        if kwargs is not None and len(kwargs) > 0:
            clf = GBC(**kwargs)
        else:
            clf = GBC(n_estimators=100)
        if assign:
            self.clf = clf
        return clf

    def xgb_classifier_best(self, x, y, param_grid=None, assign=True, n_jobs=-1, show=True):
        """
        寻找GradientBoostingClassifier构造器的最优参数

        上层AbuML中xgb_classifier_best函数，直接使用AbuML中的x，y数据调用
        eg：
             xgb_classifier_best无param_grid参数调用：

             from abupy import AbuML, ml
             ttn_abu = AbuML.create_test_more_fiter()
             ttn_abu.xgb_classifier_best()

             xgb_classifier_best有param_grid参数调用：

             param_grid = {'learning_rate': np.arange(0.1, 0.5, 0.05), 'n_estimators': np.arange(50, 200, 10)}
             ttn_abu.xgb_classifier_best(param_grid=param_grid, n_jobs=-1)

             out: GradientBoostingClassifier(learning_rate=0.1, n_estimators=160)

        :param x: 训练集x矩阵，numpy矩阵
        :param y: 训练集y序列，numpy序列
        :param param_grid: 最优字典关键字参数，
                     eg：param_grid = {'learning_rate': np.arange(0.1, 0.5, 0.05),
                                      'n_estimators': np.arange(50, 200, 10)}
        :param assign: 是否保存实例化后最优参数的学习器对象，默认True
        :param n_jobs: 并行执行的进程任务数量，默认-1, 开启与cpu相同数量的进程数
        :param show: 是否可视化最优参数搜索结果
        :return: 通过最优参数构造的GradientBoostingClassifier对象
        """
        return self._estimators_prarms_best(self.xgb_classifier, x, y, param_grid, assign, n_jobs, show)

    def random_forest_regressor(self, assign=True, **kwargs):
        """
        有监督学习回归器，实例化RandomForestRegressor，默认使用：
            RandomForestRegressor(n_estimators=100)

        通过**kwargs即关键字参数透传RandomForestRegressor，即:
            RandomForestRegressor(**kwargs)

        :param assign: 是否保存实例后的RandomForestRegressor对象，默认True，self.reg = reg
        :param kwargs: 有参数情况下初始化: RandomForestRegressor(**kwargs)
                       无参数情况下初始化: RandomForestRegressor(n_estimators=100)

        :return: 实例化的RandomForestRegressor对象
        """
        if kwargs is not None and len(kwargs) > 0:
            reg = RandomForestRegressor(**kwargs)
        else:
            reg = RandomForestRegressor(n_estimators=100)
        if assign:
            self.reg = reg
        return reg

    def random_forest_regressor_best(self, x, y, param_grid=None, assign=True, n_jobs=-1, show=True):
        """
        寻找RandomForestRegressor构造器的最优参数

        上层AbuML中random_forest_regressor_best函数，直接使用AbuML中的x，y数据调用
        eg：
             random_forest_regressor_best无param_grid参数调用：

             from abupy import AbuML, ml
             ttn_abu = AbuML.create_test_more_fiter()
             ttn_abu.random_forest_regressor_best()

             random_forest_regressor_best有param_grid参数调用：

            param_grid = {'max_features': ['sqrt', 'log2', ], 'n_estimators': np.arange(10, 150, 15)}
            ttn_abu.random_forest_regressor_best(param_grid=param_grid, n_jobs=-1)

             out: RandomForestRegressor(max_features='sqrt', n_estimators=115)

        :param x: 训练集x矩阵，numpy矩阵
        :param y: 训练集y序列，numpy序列
        :param param_grid: 最优字典关键字参数，
                     eg：param_grid = {'max_features': ['sqrt', 'log2', ],
                                      'n_estimators': np.arange(10, 150, 15)}
        :param assign: 是否保存实例化后最优参数的学习器对象，默认True
        :param n_jobs: 并行执行的进程任务数量，默认-1, 开启与cpu相同数量的进程数
        :param show: 是否可视化最优参数搜索结果
        :return: 通过最优参数构造的RandomForestRegressor对象
        """
        return self._estimators_prarms_best(self.random_forest_regressor, x, y, param_grid, assign, n_jobs, show)

    def random_forest_classifier(self, assign=True, **kwargs):
        """
        有监督学习分类器，实例化RandomForestClassifier，默认使用：
            RandomForestRegressor(n_estimators=100)

        通过**kwargs即关键字参数透传RandomForestRegressor，即:
            RandomForestRegressor(**kwargs)

        :param assign: 是否保存实例后的RandomForestRegressor对象，默认True，self.reg = reg
        :param kwargs: 有参数情况下初始化: RandomForestRegressor(**kwargs)
                       无参数情况下初始化: RandomForestRegressor(n_estimators=100)

        :return: 实例化的RandomForestRegressor对象
        """
        if kwargs is not None and len(kwargs) > 0:
            clf = RandomForestClassifier(**kwargs)
        else:
            clf = RandomForestClassifier(n_estimators=100)
        if assign:
            self.clf = clf
        return clf

    def random_forest_classifier_best(self, x, y, param_grid=None, assign=True, n_jobs=-1, show=True):
        """
        寻找RandomForestClassifier构造器的最优参数

        上层AbuML中random_forest_classifier_best函数，直接使用AbuML中的x，y数据调用
        eg：
            random_forest_classifier_best无param_grid参数调用：

            from abupy import AbuML, ml
            ttn_abu = AbuML.create_test_more_fiter()
            ttn_abu.random_forest_classifier_best()

            random_forest_classifier_best有param_grid参数调用：

            param_grid = {'max_features': ['sqrt', 'log2', ], 'n_estimators': np.arange(50, 200, 20)}
            ttn_abu.random_forest_classifier_best(param_grid=param_grid, n_jobs=-1)

            out: RandomForestClassifier(max_features='sqrt', n_estimators=190)

        :param x: 训练集x矩阵，numpy矩阵
        :param y: 训练集y序列，numpy序列
        :param param_grid: 最优字典关键字参数，
                     eg：param_grid = {'max_features': ['sqrt', 'log2', ],
                                      'n_estimators': np.arange(50, 200, 20)}
        :param assign: 是否保存实例化后最优参数的学习器对象，默认True
        :param n_jobs: 并行执行的进程任务数量，默认-1, 开启与cpu相同数量的进程数
        :param show: 是否可视化最优参数搜索结果
        :return: 通过最优参数构造的RandomForestClassifier对象
        """
        return self._estimators_prarms_best(self.random_forest_classifier, x, y, param_grid, assign, n_jobs, show)

    def svc(self, assign=True, **kwargs):
        """
        有监督学习分类器，实例化SVC，默认使用：
            SVC(kernel='rbf', probability=True)

        通过**kwargs即关键字参数透传SVC，即:
            SVC(**kwargs)

        :param assign: 是否保存实例后的RandomForestRegressor对象，默认True，self.clf = clf
        :param kwargs: 有参数情况下初始化: SVC(**kwargs)
                       无参数情况下初始化: SVC(kernel='rbf', probability=True)

        :return: 实例化的SVC对象
        """
        if kwargs is not None and len(kwargs) > 0:
            clf = SVC(**kwargs)
        else:
            clf = SVC(kernel='rbf', probability=True)
        if assign:
            self.clf = clf
        return clf

    def decision_tree_regressor(self, assign=True, **kwargs):
        """
        有监督学习回归器，实例化DecisionTreeRegressor，默认使用：
            DecisionTreeRegressor(max_depth=2, random_state=1)

        通过**kwargs即关键字参数透传DecisionTreeRegressor，即:
            DecisionTreeRegressor(**kwargs)

        :param assign: 是否保存实例后的DecisionTreeRegressor对象，默认True，self.reg = reg
        :param kwargs: 有参数情况下初始化: DecisionTreeRegressor(**kwargs)
                       无参数情况下初始化: DecisionTreeRegressor(max_depth=2, random_state=1)

        :return: 实例化的DecisionTreeRegressor对象
        """

        if kwargs is not None and len(kwargs) > 0:
            reg = DecisionTreeRegressor(**kwargs)
        else:
            reg = DecisionTreeRegressor(max_depth=2, random_state=1)
        if assign:
            self.reg = reg
        return reg

    def decision_tree_regressor_best(self, x, y, param_grid=None, assign=True, n_jobs=-1, show=True):
        """
        寻找DecisionTreeRegressor构造器的最优参数

        上层AbuML中decision_tree_regressor_best函数，直接使用AbuML中的x，y数据调用
        eg：
            decision_tree_regressor_best无param_grid参数调用：

            from abupy import AbuML, ml
            ttn_abu = AbuML.create_test_more_fiter()
            ttn_abu.decision_tree_regressor_best()

            decision_tree_regressor_best有param_grid参数调用：

            param_grid = {'max_features': ['sqrt', 'log2', ], 'max_depth': np.arange(1, 10, 1)}
            ttn_abu.decision_tree_regressor_best(param_grid=param_grid, n_jobs=-1)

            out: DecisionTreeRegressor(max_features='sqrt', max_depth=3)

        :param x: 训练集x矩阵，numpy矩阵
        :param y: 训练集y序列，numpy序列
        :param param_grid: 最优字典关键字参数，
                     eg：param_grid = {'max_features': ['sqrt', 'log2', ],
                                      'n_estimators': np.arange(50, 200, 20)}
        :param assign: 是否保存实例化后最优参数的学习器对象，默认True
        :param n_jobs: 并行执行的进程任务数量，默认-1, 开启与cpu相同数量的进程数
        :param show: 是否可视化最优参数搜索结果
        :return: 通过最优参数构造的DecisionTreeRegressor对象
        """
        return self._estimators_prarms_best(self.decision_tree_regressor, x, y, param_grid, assign, n_jobs, show,
                                            grid_callback=ABuMLGrid.grid_search_init_max_depth)

    def decision_tree_classifier(self, assign=True, **kwargs):
        """
        有监督学习分类器，实例化DecisionTreeClassifier，默认使用：
            DecisionTreeClassifier(max_depth=2, random_state=1)

        通过**kwargs即关键字参数透传DecisionTreeClassifier，即:
            DecisionTreeClassifier(**kwargs)

        :param assign: 是否保存实例后的DecisionTreeClassifier对象，默认True，self.clf = clf
        :param kwargs: 有参数情况下初始化: DecisionTreeClassifier(**kwargs)
                       无参数情况下初始化: DecisionTreeClassifier(max_depth=2, random_state=1)

        :return: 实例化的DecisionTreeClassifier对象
        """

        if kwargs is not None and len(kwargs) > 0:
            clf = DecisionTreeClassifier(**kwargs)
        else:
            clf = DecisionTreeClassifier(max_depth=2, random_state=1)
        if assign:
            self.clf = clf
        return clf

    def decision_tree_classifier_best(self, x, y, param_grid=None, assign=True, n_jobs=-1, show=True):
        """
        寻找DecisionTreeClassifier构造器的最优参数

        上层AbuML中decision_tree_classifier_best函数，直接使用AbuML中的x，y数据调用
        eg：
            decision_tree_classifier_best无param_grid参数调用：

            from abupy import AbuML, ml
            ttn_abu = AbuML.create_test_more_fiter()
            ttn_abu.decision_tree_classifier_best()

            decision_tree_classifier_best有param_grid参数调用：

            param_grid = {'max_features': ['sqrt', 'log2', ], 'max_depth': np.arange(1, 10, 1)}
            ttn_abu.decision_tree_classifier_best(param_grid=param_grid, n_jobs=-1)

            out: DecisionTreeClassifier(max_features='sqrt', max_depth=7)

        :param x: 训练集x矩阵，numpy矩阵
        :param y: 训练集y序列，numpy序列
        :param param_grid: 最优字典关键字参数，
                     eg：param_grid = {'max_features': ['sqrt', 'log2', ],
                                      'n_estimators': np.arange(50, 200, 20)}
        :param assign: 是否保存实例化后最优参数的学习器对象，默认True
        :param n_jobs: 并行执行的进程任务数量，默认-1, 开启与cpu相同数量的进程数
        :param show: 是否可视化最优参数搜索结果
        :return: 通过最优参数构造的DecisionTreeClassifier对象
        """
        return self._estimators_prarms_best(self.decision_tree_classifier, x, y, param_grid, assign, n_jobs, show,
                                            grid_callback=ABuMLGrid.grid_search_init_max_depth)

    def knn_classifier(self, assign=True, **kwargs):
        """
        有监督学习分类器，实例化KNeighborsClassifier，默认使用：
            KNeighborsClassifier(n_neighbors=1)

        通过**kwargs即关键字参数透传KNeighborsClassifier，即:
            KNeighborsClassifier(**kwargs)

        :param assign: 是否保存实例后的KNeighborsClassifier对象，默认True，self.clf = clf
        :param kwargs: 有参数情况下初始化: KNeighborsClassifier(**kwargs)
                       无参数情况下初始化: KNeighborsClassifier(n_neighbors=1)

        :return: 实例化的KNeighborsClassifier对象
        """

        if kwargs is not None and len(kwargs) > 0:
            clf = KNeighborsClassifier(**kwargs)
        else:
            clf = KNeighborsClassifier(n_neighbors=1)
        if assign:
            self.clf = clf
        return clf

    def knn_classifier_best(self, x, y, param_grid=None, assign=True, n_jobs=-1, show=True):
        """
        寻找KNeighborsClassifier构造器的最优参数

        上层AbuML中knn_classifier_best函数，直接使用AbuML中的x，y数据调用
        eg：
          knn_classifier_best无param_grid参数调用：

          from abupy import AbuML, ml
          ttn_abu = AbuML.create_test_more_fiter()
          ttn_abu.knn_classifier_best()

          knn_classifier_best有param_grid参数调用：

          param_grid = {'algorithm': ['ball_tree', 'kd_tree', 'brute'], 'n_neighbors': np.arange(1, 26, 1)}
          ttn_abu.knn_classifier_best(param_grid=param_grid, n_jobs=-1)

          out: KNeighborsClassifier(algorithm='ball_tree', n_neighbors=14)

        :param x: 训练集x矩阵，numpy矩阵
        :param y: 训练集y序列，numpy序列
        :param param_grid: 最优字典关键字参数，
                   eg：param_grid = {'algorithm': ['ball_tree', 'kd_tree', 'brute'],
                                    'n_neighbors': np.arange(1, 26, 1)}
        :param assign: 是否保存实例化后最优参数的学习器对象，默认True
        :param n_jobs: 并行执行的进程任务数量，默认-1, 开启与cpu相同数量的进程数
        :param show: 是否可视化最优参数搜索结果
        :return: 通过最优参数构造的KNeighborsClassifier对象
        """
        return self._estimators_prarms_best(self.knn_classifier, x, y, param_grid, assign, n_jobs, show,
                                            grid_callback=ABuMLGrid.grid_search_init_n_neighbors)

    def logistic_classifier(self, assign=True, **kwargs):
        """
        有监督学习分类器，实例化LogisticRegression，默认使用：
            LogisticRegression(C=1.0, penalty='l1', tol=1e-6)

        通过**kwargs即关键字参数透传LogisticRegression，即:
            LogisticRegression(**kwargs)

        :param assign: 是否保存实例后的LogisticRegression对象，默认True，self.clf = clf
        :param kwargs: 有参数情况下初始化: LogisticRegression(**kwargs)
                       无参数情况下初始化: LogisticRegression(C=1.0, penalty='l1', tol=1e-6)

        :return: 实例化的LogisticRegression对象
        """
        if kwargs is not None and len(kwargs) > 0:
            clf = LogisticRegression(**kwargs)
        else:
            clf = LogisticRegression(C=1.0, penalty='l1', tol=1e-6)
        if assign:
            self.clf = clf
        return clf

    def linear_regression(self, assign=True, **kwargs):
        """
        有监督学习回归器，实例化LinearRegression，默认使用：
            LinearRegression()

        通过**kwargs即关键字参数透传LinearRegression，即:
            LinearRegression(**kwargs)

        :param assign: 是否保存实例后的LinearRegression对象，默认True，self.reg = reg
        :param kwargs: 有参数情况下初始化: LinearRegression(**kwargs)
                       无参数情况下初始化: LinearRegression()

        :return: 实例化的LinearRegression对象
        """
        if kwargs is not None and len(kwargs) > 0:
            reg = LinearRegression(**kwargs)
        else:
            reg = LinearRegression()
        if assign:
            self.reg = reg
        return reg

    def polynomial_regression(self, assign=True, degree=2, **kwargs):
        """
        有监督学习回归器，使用：
            make_pipeline(PolynomialFeatures(degree), LinearRegression(**kwargs))

        :param assign: 是否保存实例后的LinearRegression对象，默认True，self.reg = reg
        :param degree: 多项式拟合参数，默认2
        :param kwargs: 由make_pipeline(PolynomialFeatures(degree), LinearRegression(**kwargs))
                       即关键字参数**kwargs全部传递给LinearRegression做为构造参数

        :return: 实例化的回归对象
        """
        reg = make_pipeline(PolynomialFeatures(degree), LinearRegression(**kwargs))
        if assign:
            self.reg = reg
        return reg

    def onevsone_classifier(self, assign=False, **kwargs):
        """
        封装有监督学习分类器，使用OneVsOneClassifier进行多label的
        分类器二次封装，即：
             OneVsOneClassifier(self.clf, **kwargs)

        :param assign: 是否保存实例后的二次封装分类器对象，与其它构造器不同，
                       默认False，即默认不保存在类中替换原始分类器
        :param kwargs: 透传OneVsOneClassifier做为构造关键字参数
        :return: OneVsOneClassifier对象
        """
        onevsone = OneVsOneClassifier(self.clf, **kwargs)
        if assign:
            self.clf = onevsone
        return onevsone

    def onevsreset_classifier(self, assign=False, **kwargs):
        """
        封装有监督学习分类器，使用OneVsRestClassifier进行多label的
        分类器二次封装，即：
             OneVsRestClassifier(self.clf, **kwargs)

        :param assign: 是否保存实例后的二次封装分类器对象，与其它构造器不同，
                       默认False，即默认不保存在类中替换原始分类器
        :param kwargs: 透传OneVsRestClassifier做为构造关键字参数
        :return: OneVsRestClassifier对象
        """
        onevsreset = OneVsRestClassifier(self.clf, **kwargs)
        if assign:
            self.clf = onevsreset
        return onevsreset
