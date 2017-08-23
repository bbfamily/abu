# -*- encoding:utf-8 -*-
"""
中间层，从上层拿到x，y，df
拥有create estimator

"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import logging
import os
import functools
from enum import Enum

import numpy as np
import pandas as pd
from sklearn.base import TransformerMixin, ClassifierMixin, RegressorMixin, clone
from sklearn import metrics
from sklearn.datasets import load_iris
from sklearn.feature_selection import RFE, VarianceThreshold
from sklearn.preprocessing import label_binarize, StandardScaler, binarize

from . import ABuMLExecute
from .ABuMLCreater import AbuMLCreater
from ..CoreBu import ABuEnv
from ..CoreBu.ABuFixes import train_test_split, cross_val_score, mean_squared_error_scorer, six
from ..UtilBu import ABuFileUtil
from ..UtilBu.ABuProgress import AbuProgress
from ..UtilBu.ABuDTUtil import warnings_filter
from ..UtilBu.ABuDTUtil import params_to_numpy
from ..CoreBu.ABuFixes import signature

__author__ = '阿布'
__weixin__ = 'abu_quant'

p_dir = os.path.abspath(os.path.join(os.path.dirname(os.path.realpath(__file__)), os.path.pardir))
ML_TEST_FILE = os.path.join(p_dir, 'RomDataBu/ml_test.csv')


class _EMLScoreType(Enum):
    """针对有监督学习的度量支持enum"""

    """有监督学习度量准确率"""
    E_SCORE_ACCURACY = 'accuracy'
    """有监督学习度量mse"""
    E_SCORE_MSE = mean_squared_error_scorer
    """有监督学习度量roc_auc"""
    E_SCORE_ROC_AUC = 'roc_auc'


class EMLFitType(Enum):
    """支持常使用的学习器类别enum"""

    """有监督学习：自动选择，根据y的label数量，> 10使用回归否则使用分类"""
    E_FIT_AUTO = 'auto'
    """有监督学习：回归"""
    E_FIT_REG = 'reg'
    """有监督学习：分类"""
    E_FIT_CLF = 'clf'

    """无监督学习：HMM"""
    E_FIT_HMM = 'hmm'
    """无监督学习：PCA"""
    E_FIT_PCA = 'pca'
    """无监督学习：KMEAN"""
    E_FIT_KMEAN = 'kmean'


def entry_wrapper(support=(EMLFitType.E_FIT_CLF, EMLFitType.E_FIT_REG, EMLFitType.E_FIT_HMM,
                           EMLFitType.E_FIT_PCA, EMLFitType.E_FIT_KMEAN)):
    """
    类装饰器函数，对关键字参数中的fiter_type进行标准化，eg，fiter_type参数是'clf'， 转换为EMLFitType(fiter_type)
    赋予self.fiter_type，检测当前使用的具体学习器不在support参数中不执行被装饰的func函数了，打个log返回

    :param support: 默认 support=(EMLFitType.E_FIT_CLF, EMLFitType.E_FIT_REG, EMLFitType.E_FIT_HMM,
                           EMLFitType.E_FIT_PCA, EMLFitType.E_FIT_KMEAN)
                    即支持所有，被装饰的函数根据自身特性选择装饰参数
    """

    def decorate(func):
        @functools.wraps(func)
        def wrapper(self, *args, **kwargs):
            org_fiter_type = self.fiter_type
            if 'fiter_type' in kwargs:
                # 如果传递了fiter_type参数，pop出来
                fiter_type = kwargs.pop('fiter_type')
                # 如果传递的fiter_type参数是str，eg：'clf'， 转换为EMLFitType(fiter_type)
                if isinstance(fiter_type, six.string_types):
                    fiter_type = EMLFitType(fiter_type)
                self.fiter_type = fiter_type

            check_support = self.fiter_type
            if self.fiter_type == EMLFitType.E_FIT_AUTO:
                # 把auto的归到具体的分类或者回归
                check_y = self.y
                if 'y' in kwargs:
                    check_y = kwargs['y']
                check_support = EMLFitType.E_FIT_CLF if len(np.unique(check_y)) <= 10 else EMLFitType.E_FIT_REG
            if check_support not in support:
                # 当前使用的具体学习器不在support参数中不执行被装饰的func函数了，打个log返回
                self.log_func('{} not support {}!'.format(func.__name__, check_support.value))
                # 如果没能成功执行把类型再切换回来
                self.fiter_type = org_fiter_type
                return

            return func(self, *args, **kwargs)

        return wrapper

    return decorate


# noinspection PyUnresolvedReferences
class AbuML(object):
    """封装有简单学习及无监督学习方法以及相关操作类"""

    @classmethod
    def create_test_fiter(cls):
        """
        类方法：使用iris数据构造AbuML对象，测试接口，通过简单iris数据对方法以及策略进行验证
        iris数据量小，如需要更多数据进行接口测试可使用create_test_more_fiter接口

        eg: iris_abu = AbuML.create_test_fiter()

        :return: AbuML(x, y, df)，
                    eg: df
                         y   x0   x1   x2   x3
                    0    0  5.1  3.5  1.4  0.2
                    1    0  4.9  3.0  1.4  0.2
                    2    0  4.7  3.2  1.3  0.2
                    3    0  4.6  3.1  1.5  0.2
                    4    0  5.0  3.6  1.4  0.2
                    ..  ..  ...  ...  ...  ...
                    145  2  6.7  3.0  5.2  2.3
                    146  2  6.3  2.5  5.0  1.9
                    147  2  6.5  3.0  5.2  2.0
                    148  2  6.2  3.4  5.4  2.3
                    149  2  5.9  3.0  5.1  1.8
        """
        iris = load_iris()
        x = iris.data
        """
            eg: iris.data
            array([[ 5.1,  3.5,  1.4,  0.2],
                   [ 4.9,  3. ,  1.4,  0.2],
                   [ 4.7,  3.2,  1.3,  0.2],
                   [ 4.6,  3.1,  1.5,  0.2],
                   [ 5. ,  3.6,  1.4,  0.2],
                    ....... ....... .......
                   [ 6.7,  3. ,  5.2,  2.3],
                   [ 6.3,  2.5,  5. ,  1.9],
                   [ 6.5,  3. ,  5.2,  2. ],
                   [ 6.2,  3.4,  5.4,  2.3],
                   [ 5.9,  3. ,  5.1,  1.8]])
        """
        y = iris.target
        """
            eg: y
            array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                   0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                   0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                   1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                   1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2,
                   2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2,
                   2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2])
        """
        x_df = pd.DataFrame(x, columns=['x0', 'x1', 'x2', 'x3'])
        y_df = pd.DataFrame(y, columns=['y'])
        df = y_df.join(x_df)
        return AbuML(x, y, df)

    @classmethod
    def load_ttn_raw_df(cls):
        """
        读取泰坦尼克测试数据
        :return: pd.DataFrame对象，from接口pd.read_csv(train_csv_path)
        """
        train_csv_path = ML_TEST_FILE
        if not ABuFileUtil.file_exist(train_csv_path):
            # 泰坦尼克数据文件如果不存在RuntimeError
            raise RuntimeError('{} not exist, please down a ml_test.csv!'.format(train_csv_path))
        # 训练文件使用read_csv从文件读取
        return pd.read_csv(train_csv_path)

    @classmethod
    @warnings_filter
    def create_test_more_fiter(cls):
        """
        类方法：使用泰坦尼克数据构造AbuML对象，测试接口，对方法以及策略进行验证 比iris数据多
                eg: ttn_abu = AbuML.create_test_more_fiter()

        :return: AbuML(x, y, df)，构造AbuML最终的泰坦尼克数据形式如：

                eg: df
                                 Survived  SibSp  Parch  Cabin_No  Cabin_Yes  Embarked_C  Embarked_Q  \
                    0           0      1      0         1          0           0           0
                    1           1      1      0         0          1           1           0
                    2           1      0      0         1          0           0           0
                    3           1      1      0         0          1           0           0
                    4           0      0      0         1          0           0           0
                    5           0      0      0         1          0           0           1
                    6           0      0      0         0          1           0           0
                    7           0      3      1         1          0           0           0
                    8           1      0      2         1          0           0           0
                    9           1      1      0         1          0           1           0
                    ..        ...    ...    ...       ...        ...         ...         ...

                         Embarked_S  Sex_female  Sex_male  Pclass_1  Pclass_2  Pclass_3  \
                    0             1           0         1         0         0         1
                    1             0           1         0         1         0         0
                    2             1           1         0         0         0         1
                    3             1           1         0         1         0         0
                    4             1           0         1         0         0         1
                    5             0           0         1         0         0         1
                    6             1           0         1         1         0         0
                    7             1           0         1         0         0         1
                    8             1           1         0         0         0         1
                    9             0           1         0         0         1         0
                    ..          ...         ...       ...       ...       ...       ...
                         Age_scaled  Fare_scaled
                    0       -0.5614      -0.5024
                    1        0.6132       0.7868
                    2       -0.2677      -0.4889
                    3        0.3930       0.4207
                    4        0.3930      -0.4863
                    5       -0.4271      -0.4781
                    6        1.7877       0.3958
                    7       -2.0295      -0.2241
                    8       -0.1943      -0.4243
                    ..          ...         ...
        """
        raw_df = cls.load_ttn_raw_df()

        def set_missing_ages(p_df):
            """
            对数据中缺失的年龄使用RandomForestRegressor进行填充
            """
            from sklearn.ensemble import RandomForestRegressor
            age_df = p_df[['Age', 'Fare', 'Parch', 'SibSp', 'Pclass']]
            known_age = age_df[age_df.Age.notnull()].as_matrix()
            unknown_age = age_df[age_df.Age.isnull()].as_matrix()
            y_inner = known_age[:, 0]
            x_inner = known_age[:, 1:]
            rfr_inner = RandomForestRegressor(random_state=0, n_estimators=2000, n_jobs=-1)
            rfr_inner.fit(x_inner, y_inner)
            predicted_ages = rfr_inner.predict(unknown_age[:, 1::])
            p_df.loc[(p_df.Age.isnull()), 'Age'] = predicted_ages
            return p_df, rfr_inner

        def set_cabin_type(p_df):
            """
            对数据中缺失的Cabin处理
            """
            p_df.loc[(p_df.Cabin.notnull()), 'Cabin'] = "Yes"
            p_df.loc[(p_df.Cabin.isnull()), 'Cabin'] = "No"
            return p_df

        raw_df, rfr = set_missing_ages(raw_df)
        raw_df = set_cabin_type(raw_df)

        # 对多label使用get_dummies进行离散二值化处理
        dummies_cabin = pd.get_dummies(raw_df['Cabin'], prefix='Cabin')
        """
            eg:
                data_train['Cabin']:
                        0       No
                        1      Yes
                        2       No
                        3      Yes
                        4       No
                        5       No
                        6      Yes
                        7       No
                        8       No
                        9       No
                              ...
                dummies_cabin:
                                Cabin_No  Cabin_Yes
                        0           1          0
                        1           0          1
                        2           1          0
                        3           0          1
                        4           1          0
                        5           1          0
                        6           0          1
                        7           1          0
                        8           1          0
                        9           1          0
                        ..        ...        ...
        """
        dummies__embarked = pd.get_dummies(raw_df['Embarked'], prefix='Embarked')
        dummies__sex = pd.get_dummies(raw_df['Sex'], prefix='Sex')
        dummies__pclass = pd.get_dummies(raw_df['Pclass'], prefix='Pclass')
        # 将离散二值化处理生成的dummies和data进行拼接
        df = pd.concat([raw_df, dummies_cabin, dummies__embarked, dummies__sex, dummies__pclass], axis=1)
        # 删除之前非离散二值的数据
        # noinspection PyUnresolvedReferences
        df.drop(['Pclass', 'Name', 'Sex', 'Ticket', 'Cabin', 'Embarked'], axis=1, inplace=True)
        # 使用StandardScaler对数据进行标准化处理
        scaler = StandardScaler()
        # noinspection PyUnresolvedReferences
        df['Age_scaled'] = scaler.fit_transform(df['Age'].values.reshape(-1, 1))
        """
            eg:
                df['Age']
                    0      22.000
                    1      38.000
                    2      26.000
                    3      35.000
                    4      35.000
                    5      23.829
                    6      54.000
                    7       2.000
                    8      27.000
                    9      14.000
                            ...

                df['Age_scaled']
                    0     -0.5614
                    1      0.6132
                    2     -0.2677
                    3      0.3930
                    4      0.3930
                    5     -0.4271
                    6      1.7877
                    7     -2.0295
                    8     -0.1943
                    9     -1.1486
                            ...
        """
        # noinspection PyUnresolvedReferences
        df['Fare_scaled'] = scaler.fit_transform(df['Fare'].values.reshape(-1, 1))
        # noinspection PyUnresolvedReferences
        df = df.filter(regex='Survived|Age_.*|SibSp|Parch|Fare_.*|Cabin_.*|Embarked_.*|Sex_.*|Pclass_.*')
        train_np = df.as_matrix()
        y = train_np[:, 0]
        x = train_np[:, 1:]
        return AbuML(x, y, df)

    def __init__(self, x, y, df, fiter_type=EMLFitType.E_FIT_AUTO):
        """
        AbuML属于中间层需要所有原料都配齐，x, y, df，构造方式参考
        create_test_fiter方法中的实行流程

        :param x: 训练集x矩阵，numpy矩阵
        :param y: 训练集y序列，numpy序列
        :param df: 拆分x，y使用的pd.DataFrame对象
        :param fiter_type: 使用的学习器类型，默认使用E_FIT_AUTO即根据y的label数量选择分类或者回归
        """
        self.estimator = AbuMLCreater()
        # 如果传递进来的是字符串类型，转换为EMLFitType
        if isinstance(fiter_type, six.string_types):
            fiter_type = EMLFitType(fiter_type)
        self.x = x
        self.y = y
        self.df = df
        # ipython notebook下使用logging.info
        self.log_func = logging.info if ABuEnv.g_is_ipython else print
        self.fiter_type = fiter_type

    def is_supervised_learning(self):
        """
        返回self.fiter_type所使用的是有监督学习还是无监督学习
        :return: bool，True: 有监督，False: 无监督
        """
        return self.fiter_type == EMLFitType.E_FIT_REG or self.fiter_type == EMLFitType.E_FIT_CLF or \
            self.fiter_type == EMLFitType.E_FIT_AUTO

    def echo_info(self, fiter=None):
        """
        显示fiter class信息，self.df信息包括，head，tail，describe
        eg：
            fiter class is: DecisionTreeClassifier(class_weight=None, criterion='gini', max_depth=None,
                max_features=None, max_leaf_nodes=None,
                min_impurity_split=1e-07, min_samples_leaf=1,
                min_samples_split=2, min_weight_fraction_leaf=0.0,
                presort=False, random_state=None, splitter='best')
            describe:
                          y        x0        x1        x2        x3
            count  150.0000  150.0000  150.0000  150.0000  150.0000
            mean     1.0000    5.8433    3.0540    3.7587    1.1987
            std      0.8192    0.8281    0.4336    1.7644    0.7632
            min      0.0000    4.3000    2.0000    1.0000    0.1000
            25%      0.0000    5.1000    2.8000    1.6000    0.3000
            50%      1.0000    5.8000    3.0000    4.3500    1.3000
            75%      2.0000    6.4000    3.3000    5.1000    1.8000
            max      2.0000    7.9000    4.4000    6.9000    2.5000
        :param fiter:
        :return:
        """
        if fiter is None:
            fiter = self.get_fiter()
        self.log_func('fiter class is: {}'.format(fiter))
        self.log_func('describe:\n{}'.format(self.df.describe()))
        self.log_func('head:\n{}'.format(self.df.head()))
        self.log_func('tail:\n{}'.format(self.df.tail()))

    def get_fiter(self):
        """
        根据self.fiter_type的类型选择从self.estimator返回学习器对象

        self.fiter_type == EMLFitType.E_FIT_AUTO：
            自动选择有简单学习，当y的label数量 < 10个使用分类self.estimator.clf，否则回归self.estimator.reg
        self.fiter_type == EMLFitType.E_FIT_REG:
            使用有监督学习回归self.estimator.reg
        self.fiter_type == EMLFitType.E_FIT_CLF:
            使用有监督学习分类self.estimator.clf
        self.fiter_type == EMLFitType.E_FIT_HMM:
            使用无监督学习hmm，self.estimator.hmm
        self.fiter_type == EMLFitType.E_FIT_PCA:
            使用无监督学习pca，self.estimator.pca
        self.fiter_type == EMLFitType.E_FIT_KMEAN:
            使用无监督学习kmean，self.estimator.kmean
        :return: 返回学习器对象
        """
        if self.fiter_type == EMLFitType.E_FIT_AUTO:
            if len(np.unique(self.y)) <= 10:
                # 小于等于10个class的y就认为是要用分类了
                fiter = self.estimator.clf
            else:
                fiter = self.estimator.reg
        elif self.fiter_type == EMLFitType.E_FIT_REG:
            fiter = self.estimator.reg
        elif self.fiter_type == EMLFitType.E_FIT_CLF:
            fiter = self.estimator.clf
        elif self.fiter_type == EMLFitType.E_FIT_HMM:
            if self.estimator.hmm is None:
                self.estimator.hmm_gaussian()
            fiter = self.estimator.hmm
        elif self.fiter_type == EMLFitType.E_FIT_PCA:
            if self.estimator.pca is None:
                self.estimator.pca_decomposition()
            fiter = self.estimator.pca
        elif self.fiter_type == EMLFitType.E_FIT_KMEAN:
            if self.estimator.kmean is None:
                self.estimator.kmean_cluster()
            fiter = self.estimator.kmean
        else:
            raise TypeError('self.fiter_type = {}, is error type'.format(self.fiter_type))

        return fiter

    @entry_wrapper(support=(EMLFitType.E_FIT_CLF,))
    def cross_val_accuracy_score(self, cv=10, **kwargs):
        """
        被装饰器entry_wrapper(support=(EMLFitType.E_FIT_CLF,))装饰，
        即只支持有监督学习分类，使用cross_val_score对数据进行accuracy度量
        :param cv: 透传cross_val_score的参数，默认10
        :param kwargs: 外部可以传递x, y, 通过
                                x = kwargs.pop('x', self.x)
                                y = kwargs.pop('y', self.y)
                       确定传递self._do_cross_val_score中参数x，y，
                       装饰器使用的fiter_type，eg：ttn_abu.cross_val_accuracy_score(fiter_type=ml.EMLFitType.E_FIT_CLF)
        :return: cross_val_score返回的score序列，
                 eg: array([ 1.  ,  0.9 ,  1.  ,  0.9 ,  1.  ,  0.9 ,  1.  ,  0.9 ,  0.95,  1.  ])
        """
        x = kwargs.pop('x', self.x)
        y = kwargs.pop('y', self.y)
        return self._do_cross_val_score(x, y, cv, _EMLScoreType.E_SCORE_ACCURACY.value)

    @entry_wrapper(support=(EMLFitType.E_FIT_CLF,))
    def cross_val_prob_accuracy_score(self, pb_threshold, cv=10, show=True, **kwargs):
        """
        被装饰器entry_wrapper(support=(EMLFitType.E_FIT_CLF,))装饰，
        即只支持有监督学习分类，拆分训练集，测试集，对所有数据进行一次predict_proba
        获取分类的概率（具体阅读ABuMLExecute.run_prob_cv_estimator），之后根据
        pb_threshold的值对结果概率进行二值转换，pb_threshold的均衡对影响数据的度量
        准确度等
                eg:
                    in: ttn_abu = AbuML.create_test_more_fiter()
                        ttn_abu.estimator.svc(probability=True)
                        ttn_abu.cross_val_prob_accuracy_score(pb_threshold=0.60)
                    out:
                        threshold=0.6 prob accuracy=0.83, effect cnt=870, effect rate=0.98, score=0.81
                    阀值0.6，准确率0.83，生效比例0.98，分数0.81

                    in:
                        ttn_abu.cross_val_prob_accuracy_score(pb_threshold=0.80)
                    out:
                        threshold=0.8 prob accuracy=0.87, effect cnt=718, effect rate=0.81, score=0.70
                    阀值0.8，准确率0.87 提高，生效比例0.81 降低，分数0.70 降低

                    in:
                        ttn_abu.cross_val_prob_accuracy_score(pb_threshold=0.85)
                    out:
                        threshold=0.85 prob accuracy=0.89, effect cnt=337, effect rate=0.38, score=0.34
                    阀值0.85，准确率0.89 再次提高，生效比例0.38 很低，分数0.34 降低

        即通过训练集数据寻找合适的prob值对数据的predict_prob进行非均衡处理，必然对交易的拦截进行非均衡
        处理，只对有很大概率的的交易进行拦截

        :param pb_threshold: binarize(y_prob, threshold=pb_threshold)中使用的二分阀值，float（0-1）
        :param cv: 透传ABuMLExecute.run_prob_cv_estimator中的cv参数，默认10，int
        :param show: 是否显示输出结果信息，默认显示
        :param kwargs: 外部可以传递x, y, 通过
                                x = kwargs.pop('x', self.x)
                                y = kwargs.pop('y', self.y)
                       确定传递self._do_cross_val_score中参数x，y
                       装饰器使用的fiter_type
        :return: accuracy, effect_cnt, effect_rate, score
        """
        if pb_threshold < 0.0 or pb_threshold > 1:
            self.log_func('pb_threshold must > 0 and < 1! now={}'.format(pb_threshold))
            return

        x = kwargs.pop('x', self.x)
        y = kwargs.pop('y', self.y)
        fiter = self.get_fiter()

        y_prob = ABuMLExecute.run_prob_cv_estimator(fiter, x, y, n_folds=cv)
        """
            eg: y_prob
            array([[ 0.8726,  0.1274],
                   [ 0.0925,  0.9075],
                   [ 0.2485,  0.7515],
                   ...,
                   [ 0.3881,  0.6119],
                   [ 0.7472,  0.2528],
                   [ 0.8555,  0.1445]])
        """
        y_prob_binarize = binarize(y_prob, threshold=pb_threshold)
        """
            根据参数中pb_threshold的值对y_prob进行二值化，二值化的结果中有一行全是1，或者全是0的，
            pb_threshold的值越是非均衡，结果中全是1，或者全是0的的数据越多，度量准确性的时候这些都
            是期权票
            eg: y_prob_binarize
            array([[ 1.,  0.],
                   [ 0.,  1.],
                   [ 1.,  1.],
                   ...,
                   [ 0.,  0.],
                   [ 1.,  0.],
                   [ 1.,  0.]])
        """
        # y_unique eg: array([ 0.,  1.])
        y_unique = np.unique(y)
        y_label_binarize_df = pd.get_dummies(y, prefix='true')
        """
            eg: y_label_binarize_df
                 true_0.0  true_1.0
            0           1         0
            1           0         1
            2           0         1
            3           0         1
            4           1         0
            5           1         0
            6           1         0
            7           1         0
            8           0         1
            9           0         1
            ..        ...       ...
            881         1         0
            882         1         0
            883         1         0
            884         1         0
            885         1         0
            886         1         0
            887         0         1
            888         1         0
            889         0         1
            890         1         0
        """
        y_prob_df = pd.DataFrame(y_prob_binarize, columns=['prob_{}'.format(y_label) for y_label in y_unique])
        """
            eg: y_prob_df
                 prob_0.0  prob_1.0
            0         1.0       0.0
            1         0.0       1.0
            2         0.0       1.0
            3         0.0       1.0
            4         1.0       0.0
            5         1.0       0.0
            6         1.0       0.0
            7         1.0       0.0
            8         0.0       0.0
            9         0.0       1.0
            ..        ...       ...
            881       1.0       0.0
            882       0.0       1.0
            883       1.0       0.0
            884       1.0       0.0
            885       1.0       0.0
            886       1.0       0.0
            887       0.0       1.0
            888       0.0       0.0
            889       1.0       0.0
            890       1.0       0.0
        """
        # 把两个df合并起来
        true_prob_df = pd.concat([y_label_binarize_df, y_prob_df], axis=1)
        """
            eg: true_prob_df
                 true_0.0  true_1.0  prob_0.0  prob_1.0
            0           1         0       1.0       0.0
            1           0         1       0.0       1.0
            2           0         1       0.0       1.0
            3           0         1       0.0       1.0
            4           1         0       1.0       0.0
            5           1         0       1.0       0.0
            6           1         0       1.0       0.0
            7           1         0       1.0       0.0
            8           0         1       0.0       0.0
            9           0         1       0.0       1.0
            ..        ...       ...       ...       ...
            881         1         0       1.0       0.0
            882         1         0       0.0       1.0
            883         1         0       1.0       0.0
            884         1         0       1.0       0.0
            885         1         0       1.0       0.0
            886         1         0       1.0       0.0
            887         0         1       0.0       1.0
            888         1         0       0.0       0.0
            889         0         1       1.0       0.0
            890         1         0       1.0       0.0
        """

        # 即筛选出非均衡阀值情况下有效的投票行index
        vote_index = (y_prob_df.sum(axis=1) > 0) & (y_prob_df.sum(axis=1) < 2)
        """
            需要过滤非均衡阀值情况下如pb_threshold = 0.1，都投了1
            和如pb_threshold = 0.9，都不进行投票（全是0）的情况
            eg: pb_threshold = 0.1
                 prob_0.0  prob_1.0
            0         1.0       1.0
            1         1.0       1.0
            2         1.0       1.0
            3         1.0       1.0
            4         1.0       1.0
            5         1.0       1.0
            ..        ...       ...
            eg: pb_threshold = 0.9
                 prob_0.0  prob_1.0
            0         0.0       0.0
            1         0.0       0.0
            2         0.0       0.0
            3         0.0       0.0
            4         0.0       0.0
            5         0.0       0.0
            ..        ...       ...
        """

        # 再次进行拆开，根据vote_index
        # noinspection PyUnresolvedReferences
        true_df = true_prob_df[vote_index].filter(regex='true*')
        # noinspection PyUnresolvedReferences
        prob_df = true_prob_df[vote_index].filter(regex='prob*')
        """
            prob_df即是y_prob_df中拥有有效投票的序列，true_df对应prob_df的index
            eg：prob_df
                 prob_0.0  prob_1.0
            0         1.0       0.0
            1         0.0       1.0
            2         0.0       1.0
            3         0.0       1.0
            4         1.0       0.0
            5         1.0       0.0
            6         1.0       0.0
            7         1.0       0.0
            9         0.0       1.0
            10        0.0       1.0
            ..        ...       ...
            880       0.0       1.0
            881       1.0       0.0
            882       0.0       1.0
            883       1.0       0.0
            884       1.0       0.0
            885       1.0       0.0
            886       1.0       0.0
            887       0.0       1.0
            889       1.0       0.0
            890       1.0       0.0
        """

        # 生效数量，投票不合格的不做准确率统计
        effect_cnt = prob_df.shape[0]
        # 生效率：effect_cnt / y.shape[0]
        effect_rate = effect_cnt / y.shape[0]
        # 生效的数据准确率
        accuracy = 0.0
        if effect_cnt > 0:
            accuracy = metrics.accuracy_score(true_df, prob_df)
        # 分数：生效比例 ＊ 生效准确率（0-1）
        score = effect_rate * accuracy
        if show:
            self.log_func(
                'threshold={} prob accuracy={:.2f}, effect cnt={}, effect rate={:.2f}, score={:.2f}'.format(
                    pb_threshold,
                    accuracy,
                    effect_cnt,
                    effect_rate,
                    score))
        return accuracy, effect_cnt, effect_rate, score

    @entry_wrapper(support=(EMLFitType.E_FIT_KMEAN,))
    def cross_val_silhouette_score(self, cv=10, **kwargs):
        """
        被装饰器entry_wrapper(support=(EMLFitType.E_FIT_KMEAN, ))装饰，即
        只支持无监督学习kmean的cv验证，使用silhouette_score对聚类后的结果labels_
        进行度量使用silhouette_score
        :param cv: 透传run_silhouette_cv_estimator的参数，默认10
        :param kwargs: 外部可以传递x 通过
                                x = kwargs.pop('x', self.x)
                       确定传递ABuMLExecute.run_silhouette_cv_estimator中参数x
                       装饰器使用的fiter_type，
                            eg：ttn_abu.cross_val_silhouette_score(fiter_type=ml.EMLFitType.E_FIT_KMEAN)
        :return: run_silhouette_cv_estimator返回的score序列，
                 eg: array([ 0.6322,  0.6935,  0.7187,  0.6887,  0.6699,  0.703 ,  0.6922,
                            0.7049,  0.6462,  0.6755])
        """
        x = kwargs.pop('x', self.x)
        fiter = self.get_fiter()

        scores = ABuMLExecute.run_silhouette_cv_estimator(fiter, x, n_folds=cv)
        scores = np.array(scores)
        self.log_func('{} score mean: {}'.format(fiter.__class__.__name__, scores.mean()))
        return scores

    @entry_wrapper(support=(EMLFitType.E_FIT_CLF, EMLFitType.E_FIT_REG))
    def cross_val_mean_squared_score(self, cv=10, **kwargs):
        """
        被装饰器entry_wrapper(support=(EMLFitType.E_FIT_CLF, EMLFitType.E_FIT_REG))装饰，
        即支持有监督学习回归和分类，使用cross_val_score对数据进行rmse度量
        :param cv: 透传cross_val_score的参数，默认10
        :param kwargs: 外部可以传递x, y, 通过
                                x = kwargs.pop('x', self.x)
                                y = kwargs.pop('y', self.y)
                       确定传递self._do_cross_val_score中参数x，y，
                       以及装饰器使用的fiter_type，eg：ttn_abu.cross_val_roc_auc_score(fiter_type=ml.EMLFitType.E_FIT_CLF)
        :return: cross_val_score返回的score序列，
                 eg: array([-0.1889, -0.1667, -0.2135, -0.1348, -0.1573, -0.2022, -0.1798,
                            -0.2022, -0.1348, -0.1705])
        """
        x = kwargs.pop('x', self.x)
        y = kwargs.pop('y', self.y)
        return self._do_cross_val_score(x, y, cv, _EMLScoreType.E_SCORE_MSE.value)

    @entry_wrapper(support=(EMLFitType.E_FIT_CLF,))
    def cross_val_roc_auc_score(self, cv=10, **kwargs):
        """
        被装饰器entry_wrapper(support=(EMLFitType.E_FIT_CLF,))装饰，
        即支持有监督学习分类，使用cross_val_score对数据进行roc_auc度量，如果数据的y的
        label标签 > 2，通过label_binarize将label标签进行二值化处理，
        依次计算二值化的列的roc_auc，结果返回score最好的数据度量
        :param cv: 透传cross_val_score的参数，默认10
        :param kwargs: 外部可以传递x, y, 通过
                                x = kwargs.pop('x', self.x)
                                y = kwargs.pop('y', self.y)
                       确定传递self._do_cross_val_score中参数x，y，
                       以及装饰器使用的fiter_type，eg：ttn_abu.cross_val_roc_auc_score(fiter_type=ml.EMLFitType.E_FIT_REG)
        :return: cross_val_score返回的score序列，
                 eg: array([ 1.  ,  0.9 ,  1.  ,  0.9 ,  1.  ,  0.9 ,  1.  ,  0.9 ,  0.95,  1.  ])
        """
        x = kwargs.pop('x', self.x)
        y = kwargs.pop('y', self.y)
        return self._do_cross_val_score(x, y, cv, _EMLScoreType.E_SCORE_ROC_AUC.value)

    @entry_wrapper()
    def feature_selection(self, show=True, **kwargs):
        """
            被装饰器entry_wrapper()装饰，默认参数即支持有监督和无监督学习
            监督学习通过sklern中的RFE包装拟合器进行fit(x, y)，对数据的特征进行ranking和support评定

            eg：
                RFE selection
                             ranking support
                SibSp              1    True
                Parch              1    True
                Cabin_No           1    True
                Cabin_Yes          7   False
                Embarked_C         2   False
                Embarked_Q         3   False
                Embarked_S         5   False
                Sex_female         8   False
                Sex_male           1    True
                Pclass_1           4   False
                Pclass_2           6   False
                Pclass_3           1    True
                Age_scaled         1    True
                Fare_scaled        1    True

            无监督学习通过sklern中的VarianceThreshold进行fit(x)，根据x的方差进行特征评定
            eg:
                unsupervised VarianceThreshold
                            support
                SibSp          True
                Parch          True
                Cabin_No       True
                Cabin_Yes      True
                Embarked_C     True
                Embarked_Q     True
                Embarked_S     True
                Sex_female     True
                Sex_male       True
                Pclass_1       True
                Pclass_2       True
                Pclass_3       True
                Age_scaled     True
                Fare_scaled    True

        :param show: 是否在内部输出打印结果
        :param kwargs: 外部可以传递x, y, 通过
                                x = kwargs.pop('x', self.x)
                                y = kwargs.pop('y', self.y)
                       确定传递self._do_cross_val_score中参数x，y，
                       以及装饰器使用的fiter_type，eg：ttn_abu.feature_selection(fiter_type=ml.EMLFitType.E_FIT_REG)
        """
        x = kwargs.pop('x', self.x)
        y = kwargs.pop('y', self.y)
        fiter = self.get_fiter()

        if self.is_supervised_learning():
            selector = RFE(fiter)
            selector.fit(x, y)
            feature_df = pd.DataFrame({
                'support': selector.support_, 'ranking': selector.ranking_}, index=self.df.columns[1:])
            if show:
                self.log_func('RFE selection')
                self.log_func(feature_df)
        else:
            selector = VarianceThreshold()
            selector.fit(x)
            feature_df = pd.DataFrame({
                'support': selector.get_support()}, index=self.df.columns[1:])
            if show:
                self.log_func('unsupervised VarianceThreshold')
                self.log_func(feature_df)
        return feature_df

    @entry_wrapper(support=(EMLFitType.E_FIT_CLF, EMLFitType.E_FIT_REG))
    def importances_coef_pd(self, **kwargs):
        """
        被装饰器entry_wrapper(support=(EMLFitType.E_FIT_CLF, EMLFitType.E_FIT_REG))装饰，
        即支持有监督学习回归和分类，根据fit后的feature_importances_或者coef_和原始self.df.columns
        组成pd.DataFrame对象返回
            eg：
                    feature	    importance
                7	Sex_female	0.0000
                10	Pclass_2	0.0018
                3	Cabin_Yes	0.0033
                5	Embarked_Q	0.0045
                9	Pclass_1	0.0048
                4	Embarked_C	0.0098
                6	Embarked_S	0.0105
                1	Parch	0.0154
                2	Cabin_No	0.0396
                0	SibSp	0.0506
                11	Pclass_3	0.0790
                13	Fare_scaled	0.1877
                12	Age_scaled	0.2870
                8	Sex_male	0.3060

                    coef	            columns
                0	[-0.344229036121]	SibSp
                1	[-0.1049314305]	Parch
                2	[0.0]	Cabin_No
                3	[0.902140498996]	Cabin_Yes
                4	[0.0]	Embarked_C
                5	[0.0]	Embarked_Q
                6	[-0.417254399259]	Embarked_S
                7	[1.95656682017]	Sex_female
                8	[-0.677432099492]	Sex_male
                9	[0.3411515052]	Pclass_1
                10	[0.0]	Pclass_2
                11	[-1.19413332987]	Pclass_3
                12	[-0.523782082975]	Age_scaled
                13	[0.0844326510536]	Fare_scaled
        :param kwargs: 外部可以传递x, y, 通过
                                x = kwargs.pop('x', self.x)
                                y = kwargs.pop('y', self.y)
                       以及装饰器使用的fiter_type，eg：ttn_abu.importances_coef_pd(fiter_type=ml.EMLFitType.E_FIT_REG)
        :return: pd.DataFrame对象
        """
        if not hasattr(self, 'df'):
            raise ValueError('please make a df func first!')
        x = kwargs.pop('x', self.x)
        y = kwargs.pop('y', self.y)

        fiter = self.get_fiter()
        # 训练前进行clone(fiter)
        fiter = clone(fiter)
        fiter.fit(x, y)

        # self.echo_info(fiter)
        if hasattr(fiter, 'feature_importances_'):
            return pd.DataFrame(
                {'feature': list(self.df.columns)[1:], 'importance': fiter.feature_importances_}).sort_values(
                'importance')
        elif hasattr(fiter, 'coef_'):
            return pd.DataFrame({"columns": list(self.df.columns)[1:], "coef": list(fiter.coef_.T)})

        else:
            self.log_func('fiter not hasattr feature_importances_ or coef_!')

    @entry_wrapper(support=(EMLFitType.E_FIT_CLF,))
    def train_test_split_xy(self, test_size=0.1, random_state=0, **kwargs):
        """
            被entry_wrapper(support=(EMLFitType.E_FIT_CLF, ))装饰，即只支持分类
            首先使用train_test_split将x，y根据参数test_size切割训练集和测试集，
            显示数据集，训练集，测试集的数量，针对训练集数据进行训练，使用训练好的
            分类器对测试集x进行predict，对结果分别使用metrics.accuracy_score,
            metrics.precision_score, metrics.recall_score度量准确率，查准率，
            和召回率，多label的的情况下使用average = 'macro'对precision_score和
            recall_score进行度量，最后显示分类结果混淆矩阵以及metrics.classification_report
            情况

            eg:
                x-y:(891, 14)-(891,)
                train_x-train_y:(801, 14)-(801,)
                test_x-test_y:(90, 14)-(90,)
                accuracy = 0.77
                precision_score = 0.74
                recall_score = 0.72
                          Predicted
                         |  0  |  1  |
                         |-----|-----|
                       0 |  41 |  10 |
                Actual   |-----|-----|
                       1 |  11 |  28 |
                         |-----|-----|
                             precision    recall  f1-score   support

                        0.0       0.79      0.80      0.80        51
                        1.0       0.74      0.72      0.73        39

                avg / total       0.77      0.77      0.77        90

        :param test_size: 测试集占比例，float，默认0.1，即将数据分10份，一份做为测试集
        :param random_state: 透传给train_test_split的随机参数
        :param kwargs: 外部可以传递x, y, 通过
                                x = kwargs.pop('x', self.x)
                                y = kwargs.pop('y', self.y)
                       以及装饰器使用的fiter_type，eg：ttn_abu.train_test_split_xy(fiter_type=ml.EMLFitType.E_FIT_CLF)
        """
        x = kwargs.pop('x', self.x)
        y = kwargs.pop('y', self.y)
        train_x, test_x, train_y, test_y = train_test_split(x,
                                                            y,
                                                            test_size=test_size,
                                                            random_state=random_state)
        self.log_func('x-y:{}-{}'.format(x.shape, y.shape))
        self.log_func('train_x-train_y:{}-{}'.format(train_x.shape, train_y.shape))
        self.log_func('test_x-test_y:{}-{}'.format(test_x.shape, test_y.shape))

        fiter = self.get_fiter()
        # 训练前进行clone(fiter)
        fiter = clone(fiter)
        # 针对训练集数据进行训练
        clf = fiter.fit(train_x, train_y)
        # 使用训练好的分类器对测试集x进行predict，结果y_predict
        y_predict = clf.predict(test_x)

        # 度量分类准确率
        self.log_func("accuracy = %.2f" % (metrics.accuracy_score(test_y, y_predict)))
        # precision_score和predictions在二分类的情况下使用binary
        average = 'binary'
        if len(np.unique(y)) != 2:
            # “micro表示在多分类中的对所有label进行averaging计算平均precision，recall以及F值等度量
            average = 'macro'
        # 度量分类查准率
        self.log_func("precision_score = %.2f" % (metrics.precision_score(test_y, y_predict, average=average)))
        # 度量分类召回率
        self.log_func("recall_score = %.2f" % (metrics.recall_score(test_y, y_predict, average=average)))
        # 混淆矩阵以及metrics.classification_report
        self._confusion_matrix_with_report(test_y, y_predict, labels=np.unique(y))

    def train_test_split_df(self, test_size=0.1, random_state=0, **kwargs):
        """
        套接封装train_test_split_xy，外部传递pd.DataFrame参数时使用
        :param test_size: 透传参数train_test_split_xy
        :param random_state: 透传参数train_test_split_xy
        :param kwargs: 通过 df = kwargs.pop('df', self.df)弹出传递的pd.DataFrame对象进行x，y分解
                       y = matrix[:, 0]，即硬编码分类y在第一列，外部传递的df对象需要遵循
                       以及装饰器使用的fiter_type，eg：ttn_abu.train_test_split_df(fiter_type=ml.EMLFitType.E_FIT_CLF)
        """
        df = kwargs.pop('df', self.df)
        matrix = df.as_matrix()
        y = matrix[:, 0]
        x = matrix[:, 1:]
        self.train_test_split_xy(test_size=test_size, random_state=random_state, x=x, y=y, **kwargs)

    @entry_wrapper()
    def fit(self, **kwargs):
        """
        包装fit操作，根据是否是有监督学习来区别
        使用fit(x, y)还是fit(x)

            eg:
               in:   iris_abu.estimator.random_forest_classifier()
                     iris_abu.fit()
               out:
                     RandomForestClassifier(bootstrap=True, class_weight=None, criterion='gini',
                                            max_depth=None, max_features='auto', max_leaf_nodes=None,
                                            min_impurity_split=1e-07, min_samples_leaf=1,
                                            min_samples_split=2, min_weight_fraction_leaf=0.0,
                                            n_estimators=100, n_jobs=1, oob_score=False, random_state=None,
                                            verbose=0, warm_start=False)
        :param kwargs:
        :return: fit(x, y)或者fit(x)操作后返回
        """
        x = kwargs.pop('x', self.x)
        y = kwargs.pop('y', self.y)
        fiter = self.get_fiter()
        if hasattr(fiter, 'fit'):
            if self.is_supervised_learning():
                fit_self = fiter.fit(x, y)
            else:
                fit_self = fiter.fit(x)
            return fit_self
        else:
            self.log_func('{} not support fit'.format(fiter))

    @entry_wrapper()
    def fit_transform(self, **kwargs):
        """
        被装饰器@entry_wrapper()装饰，默认参数即支持有监督和无监督学习，
        内部通过检测isinstance(fiter, TransformerMixin) or hasattr(fiter, 'fit_transform')
        来判定是否可以fit_transform

        eg：
            input:  ttn_abu.x.shape
            output: (891, 14)

            input:  ttn_abu.fit_transform(fiter_type=ml.EMLFitType.E_FIT_PCA).shape
            output: (891, 4)

            input:  ttn_abu.fit_transform(fiter_type=ml.EMLFitType.E_FIT_KMEAN).shape
            output: (891, 2)

        :param kwargs: 外部可以传递x, y, 通过
                                x = kwargs.pop('x', self.x)
                                y = kwargs.pop('y', self.y)
                       以及装饰器使用的fiter_type，eg：ttn_abu.fit_transform(fiter_type=ml.EMLFitType.E_FIT_CLF)
        :return: fit_transform后的转换结果矩阵
        """
        fiter = self.get_fiter()
        if isinstance(fiter, TransformerMixin) or hasattr(fiter, 'fit_transform'):
            x = kwargs.pop('x', self.x)
            y = kwargs.pop('y', self.y)
            if self.is_supervised_learning():
                trans = fiter.fit_transform(x, y)
            else:
                trans = fiter.fit_transform(x)
            return trans
        else:
            self.log_func('{} not support fit_transform'.format(fiter))

    def predict(self, x, pre_fit=True, **kwargs):
        """
        call有监督和无监督学习的predict，套接self._predict_callback方法，其
        被装饰器params_to_numpy装饰，将predict参数中所有可迭代序列变成numpy，
        即统一将x转换numpy序列,
            eg:
                test_input = pd.DataFrame.from_dict({'x0': 6.8, 'x1': 3.5,  'x2': 5.4, 'x3': 2.2}, orient='index')
                test_input:
                                0
                            x0	6.8
                            x1	3.5
                            x2	5.4
                            x3	2.2

                iris_abu.predict(test_input)

                params_to_numpy装饰后参数中的x转换为：array([[ 6.8,  3.5,  5.4,  2.2]]) numpy array

        被装饰器entry_wrapper()装饰，默认参数即支持有监督和无监督学习
        :param x: 可迭代序列，通过params_to_numpy装饰统一转换numpy
        :param pre_fit: bool，默认pre_fit True, 代表每次predict前先fit，
                        如在乎效率的情况下，需要在外部先fit后，参数pre_fit置False
        :param kwargs: 装饰器使用的fiter_type，eg：ttn_abu.predict(test_input, fiter_type=ml.EMLFitType.E_FIT_KMEAN)
        :return: eg: array([2])
        """
        return self._predict_callback('predict', x, pre_fit=pre_fit, **kwargs)

    def predict_proba(self, x, pre_fit=True, **kwargs):
        """
        call有监督和无监督学习的predict_proba，套接self._predict_callback方法，其
        被装饰器params_to_numpy装饰，将predict参数中所有可迭代序列变成numpy，
        即统一将x转换numpy序列,
            eg:
                test_input = pd.DataFrame.from_dict({'x0': 6.8, 'x1': 3.5,  'x2': 5.4, 'x3': 2.2}, orient='index')
                test_input:
                                0
                            x0	6.8
                            x1	3.5
                            x2	5.4
                            x3	2.2

                iris_abu.predict_proba(test_input, fiter_type=ml.EMLFitType.E_FIT_CLF)

                params_to_numpy装饰后参数中的x转换为：array([[ 6.8,  3.5,  5.4,  2.2]]) numpy array

        被装饰器entry_wrapper()装饰，默认参数即支持有监督和无监督学习
        :param x: 通过params_to_numpy装饰统一转换numpy
        :param pre_fit: bool，默认pre_fit True, 代表每次predict前先fit，
                        如在乎效率的情况下，需要在外部先fit后，参数pre_fit置False
        :param kwargs: 装饰器使用的fiter_type，eg：iris_abu.predict_proba(test_input, fiter_type=ml.EMLFitType.E_FIT_CLF)
        :return: eg: array([[ 0.2,  0.3,  0.5]])
        """
        return self._predict_callback('predict_proba', x, pre_fit=pre_fit, **kwargs)

    def predict_proba_threshold(self, x, threshold, default_ret, pre_fit=True, **kwargs):
        """
        eg：
            in:  ttn_abu = AbuML.create_test_more_fiter()
            in:  ttn_abu.estimator.svc(probability=True)
            in： ttn_abu.search_match_pos_threshold(0.85, 0.80, fiter_type=ml.EMLFitType.E_FIT_CLF)
            out: 0.770 satisfy require, accuracy:0.850, effect_rate:0.854

            即通过search_match_pos_threshold获取了predict_proba的二分阀值为0.77时，分类的正确率达到0.85， 且覆盖80%样本

            in:  test = np.array([ 1.    ,  0.    ,  0.    ,  1.    ,  1.    ,  0.    ,  0.    ,1.    ,  0.    ,  1.
                                ,  0.    ,  0.    ,  0.8132,  0.5868])
            in:  ttn_abu.predict_proba(test)
            out: array([[ 0.106,  0.894]])
            使用predict_proba得到的是分类的概率

            in:  ttn_abu.predict_proba_threshold(test, threshold=0.77 ,default_ret=0)
            out: 1
            使用predict_proba_threshold将阀值threshold设定0.77后，可以得到输出为1，即概率 0.894 > 0.77, 标签1分类成立

            in:  test2 = np.array([ 0.    ,  1.    ,  1.    ,  0.    ,  1.    ,  1.    ,  0.    ,1.    ,  0.    ,  0.
                                ,  0.    ,  1.    ,  0.7832,  0.2868])
            in:  ttn_abu.predict_proba(test2)
            out: array([[ 0.2372,  0.7628]])

            in:  ttn_abu.predict_proba_threshold(test2, threshold=0.77 ,default_ret=0)
            out: 0
            对test2做predict_proba_threshold返回0，因为0.7628 < 0.77, 标签1的分类不成立，返回default_ret

            应用场景：比如对交易进行拦截，实行高放行率，低拦截率，0代表放行，1代表拦截，
                    上述predict_proba_threshold(test2, threshold=0.77 ,default_ret=0)
                    即可实行对较大概率的交易进行拦截，即把握大的进行拦截，把握不大的默认选择放行
        :param x: 在predict_proba中通过params_to_numpy装饰统一转换numpy
        :param threshold: 对predict_proba结果进行二值化的阀值 eg： threshold=0.77
        :param default_ret: 使用predict_proba返回的矢量和不等于1时，即没有进行有效投票时返回的值：
                            eg：
                                    proba = np.where(proba >= threshold, 1, 0)
                                    if proba.sum() != 1:
                                        # eg: proba = array([[ 0.2328,  0.7672]])->array([[0, 0]])
                                        return default_ret
        :param pre_fit: bool，默认pre_fit True, 代表每次predict前先fit，
                        如在乎效率的情况下，需要在外部先fit后，参数pre_fit置False
        :param kwargs: 装饰器使用的fiter_type，
                        eg：iris_abu.predict_proba_threshold(test_input, , threshold=0.77 ,default_ret=0
                                                            fiter_type=ml.EMLFitType.E_FIT_CLF)
        :return: int，default_ret or proba.argmax()
        """
        # 套接self.predict_proba对x所描述的特征进行概率proba
        proba = self.predict_proba(x, pre_fit=pre_fit, **kwargs)
        # eg：array([[ 0.1063,  0.8937]]) -> array([[0, 1]])
        # noinspection PyTypeChecker
        proba = np.where(proba >= threshold, 1, 0)
        if proba.sum() != 1:
            # eg: proba = array([[ 0.2328,  0.7672]])->array([[0, 0]])
            return default_ret
        # 唯一最大值就是序列值为1的，通过argmax获取index，即y label
        return proba.argmax()

    @params_to_numpy
    @entry_wrapper()
    def _predict_callback(self, callback, x, pre_fit=True, **kwargs):
        """
        统一封装predict和predict需要的流程，使用callback做为具体实现
        :param callback: str字符类型，不是callable类型
        :param x: 可迭代序列，通过params_to_numpy装饰统一转换numpy
        :param pre_fit: bool，默认pre_fit True, 代表每次predict前先fit，
                        如在乎效率的情况下，需要在外部先fit后，参数pre_fit置False
        :param kwargs: 装饰器使用的fiter_type，eg：ttn_abu.predict(test_input, fiter_type=ml.EMLFitType.E_FIT_KMEAN)
        :return: eg: array([2])
        """
        # 标准化输入x
        x = x.reshape(1, -1)
        if self.x[0].reshape(1, -1).shape != x.shape:
            # predict中有check_input=True也check，返回训练集中的一个数据做为input x的示例
            self.log_func('input x must similar with {}'.format(self.x[0]))
            return

        if pre_fit:
            # 默认pre_fit True, 代表每次predict前先fit，如在乎效率的情况下，需要在外部先fit后，参数pre_fit置False
            self.fit(**kwargs)

        fiter = self.get_fiter()
        if not isinstance(callback, six.string_types):
            # callback必须是字符串类型
            self.log_func('callback must str, not {}'.format(type(callback)))
            return

        if hasattr(fiter, callback):
            if 'check_input' in list(signature(fiter.predict).parameters.keys()):
                # 针对有check_input参数的，check_input True, 因为前面X.dtype np.float32等格式化问题
                return getattr(fiter, callback)(x, check_input=True)
            else:
                return getattr(fiter, callback)(x)
        else:
            self.log_func('{} not support {}'.format(fiter, callback))

    # TODO 需要重构这个类，太长了

    def search_match_neg_threshold(self, accuracy_match=0, effect_rate_match=0, neg_num=50, **kwargs):
        """
        套接self.cross_val_prob_accuracy_score，通过np.linspace(0.01, 0.50, num=neg_num)[::-1]生成
        pb_threshold参数序列，这里linspace的start从0.01至0.50后[::-1]倒序，依次迭代生成的阀值参数
        ，当cross_val_prob_accuracy_score返回的正确率大于 参数中accuracy_match且返回的生效率大于参数中的effect_rate_match，
        匹配寻找成功，中断迭代操作，返回寻找到的满足条件的阀值，返回的阀值应用场景阅读predict_proba_threshold函数

            eg:
                in:  ttn_abu.search_match_neg_threshold(0.85, 0.80, fiter_type=ml.EMLFitType.E_FIT_CLF)
                out: 0.220 satisfy require, accuracy:0.852, effect_rate:0.844

        :param accuracy_match: 寻找阀值条件，需要当cross_val_prob_accuracy_score返回的正确率大于accuracy_match，
                               float， 范围（0-1），默认值0
        :param effect_rate_match: 寻找阀值条件，需要当cross_val_prob_accuracy_score返回的生效率大于effect_rate_match，
                               float， 范围（0-1），默认值0
        :param neg_num: 透传neg_thresholds = np.linspace(0.01, 0.50, num=neg_num)[::-1]的参数，默认50
        :param kwargs: 装饰器使用的fiter_type，
                    eg: iris_abu.search_match_neg_threshold(0.85, 0.80, fiter_type=ml.EMLFitType.E_FIT_CLF)
        :return: 返回寻找到的满足条件的阀值，float
        """
        neg_thresholds = np.linspace(0.01, 0.50, num=neg_num)[::-1]
        """
            eg: neg_thresholds
                array([ 0.5 ,  0.49,  0.48,  0.47,  0.46,  0.45,  0.44,  0.43,  0.42,
            0.41,  0.4 ,  0.39,  0.38,  0.37,  0.36,  0.35,  0.34,  0.33,
            0.32,  0.31,  0.3 ,  0.29,  0.28,  0.27,  0.26,  0.25,  0.24,
            0.23,  0.22,  0.21,  0.2 ,  0.19,  0.18,  0.17,  0.16,  0.15,
            0.14,  0.13,  0.12,  0.11,  0.1 ,  0.09,  0.08,  0.07,  0.06,
            0.05,  0.04,  0.03,  0.02,  0.01])
        """
        with AbuProgress(len(neg_thresholds), 0, 'search neg threshold') as search_neg_progress:
            for neg in neg_thresholds:
                accuracy, _, effect_rate, _ = self.cross_val_prob_accuracy_score(neg, show=False, **kwargs)
                search_neg_progress.show(ext='threshold:{:.2f} accuracy:{:.2f}, effect_rate:{:.2f}'.format(
                    neg, accuracy, effect_rate))
                if accuracy >= accuracy_match and effect_rate >= effect_rate_match:
                    # eg: 0.500 satisfy require, accuracy:0.940, effect_rate:1.000
                    self.log_func('{:.3f} satisfy require, accuracy:{:.3f}, effect_rate:{:.3f}'.format(
                        neg, accuracy, effect_rate))
                    # 返回寻找到的满足条件的阀值
                    return neg
        # 迭代完成所有neg_thresholds，没有找到符合参数需求的二分阀值
        self.log_func('neg_thresholds no satisfy require, search failed!')

    def search_match_pos_threshold(self, accuracy_match=0, effect_rate_match=0, pos_num=50, **kwargs):
        """
        套接self.cross_val_prob_accuracy_score，通过np.linspace(0.50, 0.99, num=neg_num)生成
        pb_threshold参数序列，这里linspace的start从0.50至0.99正序，依次迭代生成的阀值参数
        ，当cross_val_prob_accuracy_score返回的正确率大于 参数中accuracy_match且返回的生效率大于参数中的effect_rate_match，
        匹配寻找成功，中断迭代操作，返回寻找到的满足条件的阀值，返回的阀值应用场景阅读predict_proba_threshold函数

                eg:
                    in: ttn_abu.search_match_pos_threshold(0.85, 0.80, fiter_type=ml.EMLFitType.E_FIT_CLF)
                    out: 0.770 satisfy require, accuracy:0.850, effect_rate:0.854

        :param accuracy_match: 寻找阀值条件，需要当cross_val_prob_accuracy_score返回的正确率大于accuracy_match，
                               float， 范围（0-1），默认值0
        :param effect_rate_match: 寻找阀值条件，需要当cross_val_prob_accuracy_score返回的生效率大于effect_rate_match，
                               float， 范围（0-1），默认值0
        :param pos_num: 透传neg_thresholds = np.linspace(0.50, 0.99, num=neg_num)的参数，默认50
        :param kwargs: 装饰器使用的fiter_type，
                    eg: iris_abu.search_match_pos_threshold(0.85, 0.80, fiter_type=ml.EMLFitType.E_FIT_CLF)
        :return: 返回寻找到的满足条件的阀值，float
        """
        pos_thresholds = np.linspace(0.50, 0.99, num=pos_num)
        """
            eg: array([ 0.5 ,  0.51,  0.52,  0.53,  0.54,  0.55,  0.56,  0.57,  0.58,
                0.59,  0.6 ,  0.61,  0.62,  0.63,  0.64,  0.65,  0.66,  0.67,
                0.68,  0.69,  0.7 ,  0.71,  0.72,  0.73,  0.74,  0.75,  0.76,
                0.77,  0.78,  0.79,  0.8 ,  0.81,  0.82,  0.83,  0.84,  0.85,
                0.86,  0.87,  0.88,  0.89,  0.9 ,  0.91,  0.92,  0.93,  0.94,
                0.95,  0.96,  0.97,  0.98,  0.99])
        """
        with AbuProgress(len(pos_thresholds), 0, 'search pos threshold') as search_pos_progress:
            for neg in pos_thresholds:
                accuracy, _, effect_rate, _ = self.cross_val_prob_accuracy_score(neg, show=False, **kwargs)
                search_pos_progress.show(ext='threshold:{:.2f} accuracy:{:.2f}, effect_rate:{:.2f}'.format(
                    neg, accuracy, effect_rate))
                if accuracy >= accuracy_match and effect_rate >= effect_rate_match:
                    # eg: 0.500 satisfy require, accuracy:0.940, effect_rate:1.000
                    self.log_func('{:.3f} satisfy require, accuracy:{:.3f}, effect_rate:{:.3f}'.format(
                        neg, accuracy, effect_rate))
                    # 返回寻找到的满足条件的阀值
                    return neg
        # 迭代完成所有pos_thresholds，没有找到符合参数需求的二分阀值
        self.log_func('pos_thresholds no satisfy require, search failed!')

    @entry_wrapper(support=(EMLFitType.E_FIT_CLF, EMLFitType.E_FIT_REG))
    def plot_learning_curve(self, **kwargs):
        """
        被装饰器entry_wrapper(support=(EMLFitType.E_FIT_CLF, EMLFitType.E_FIT_REG))装饰，
        即支持有监督学习回归和分类，绘制训练集数据的学习曲线，当训练集的y标签label非2分问题，
        使用OneVsOneClassifier进行包装
        :param kwargs:
                    外部可以传递x, y, 通过
                                x = kwargs.pop('x', self.x)
                                y = kwargs.pop('y', self.y)
                    装饰器使用的fiter_type，
                    eg:
                        ttn_abu = AbuML.create_test_more_fiter()
                        ttn_abu.plot_learning_curve(fiter_type=ml.EMLFitType.E_FIT_CLF)
        """
        x = kwargs.pop('x', self.x)
        y = kwargs.pop('y', self.y)

        fiter = self.get_fiter()
        if self.fiter_type == EMLFitType.E_FIT_CLF and len(np.unique(y)) != 2:
            # 多标签，使用OneVsOneClassifier进行包装，onevsreset_classifier参数assign默认是false
            fiter = self.estimator.onevsreset_classifier(fiter)
        ABuMLExecute.plot_learning_curve(fiter, x, y)

    @entry_wrapper(support=(EMLFitType.E_FIT_CLF, EMLFitType.E_FIT_REG))
    def plot_graphviz_tree(self, **kwargs):
        """
        被装饰器entry_wrapper(support=(EMLFitType.E_FIT_CLF, EMLFitType.E_FIT_REG))装饰，
        即支持有监督学习回归和分类，绘制决策树或者core基于树的分类回归算法的决策示意图绘制，查看
        学习器本身hasattr(fiter, 'tree_')是否有tree_属性，如果没有使用决策树替换

        :param kwargs:  外部可以传递x, y, 通过
                                x = kwargs.pop('x', self.x)
                                y = kwargs.pop('y', self.y)
                        装饰器使用的fiter_type，
                        eg:
                            ttn_abu = AbuML.create_test_more_fiter()
                            ttn_abu.plot_graphviz_tree(fiter_type=ml.EMLFitType.E_FIT_CLF)
        """
        x = kwargs.pop('x', self.x)
        y = kwargs.pop('y', self.y)
        fiter = self.get_fiter()

        if not hasattr(fiter, 'tree_'):
            self.log_func('{} not hasattr tree_, use decision tree replace'.format(
                fiter.__class__.__name__))

            if isinstance(fiter, ClassifierMixin):
                # FIXME 最好不要使用ClassifierMixin判定学习器类型，因为限定了sklearn
                fiter = self.estimator.decision_tree_classifier(assign=False)
            elif isinstance(fiter, RegressorMixin):
                # # FIXME 最好不要使用RegressorMixin, AbuMLCreater中引用了hmmlearn，xgboost等第三方库
                fiter = self.estimator.decision_tree_regressor(assign=False)
            else:
                fiter = self.estimator.decision_tree_classifier(assign=False)
        # 这里需要将self.df.columns做为名字传入
        return ABuMLExecute.graphviz_tree(fiter, self.df.columns, x, y)

    @entry_wrapper(support=(EMLFitType.E_FIT_CLF, EMLFitType.E_FIT_REG))
    def plot_visualize_tree(self, use_pca=True, **kwargs):
        """
        被装饰器entry_wrapper(support=(EMLFitType.E_FIT_CLF, EMLFitType.E_FIT_REG))装饰，
        即支持有监督学习回归和分类, 绘制特征平面，由于特征平面需要x的维度只能是2维，所以对
        x的特征列多于两个的进行降维操作，默认使用pca，还可以选择根据特征的重要, 程度选择两个importances
        最重要的特征进行特征平面绘制

        :param use_pca: 是否使用pca进行降维，bool，默认True
        :param kwargs: 外部可以传递x, y, 通过
                        x = kwargs.pop('x', self.x)
                        y = kwargs.pop('y', self.y)
            装饰器使用的fiter_type，
            eg:
                ttn_abu = AbuML.create_test_more_fiter()
                ttn_abu.plot_visualize_tree(fiter_type=ml.EMLFitType.E_FIT_CLF)
        """
        x = kwargs.pop('x', self.x)
        y = kwargs.pop('y', self.y)
        fiter = self.get_fiter()
        # 进行降维
        x = self._decomposition_2x(x, use_pca=use_pca)
        ABuMLExecute.visualize_tree(fiter, x, y)

    @entry_wrapper(support=(EMLFitType.E_FIT_CLF, EMLFitType.E_FIT_REG))
    def plot_decision_function(self, use_pca=True, **kwargs):
        """
        被装饰器entry_wrapper(support=(EMLFitType.E_FIT_CLF, EMLFitType.E_FIT_REG))装饰，
        即支持有监督学习回归和分类
        :param use_pca: 是否使用pca进行降维，bool，默认True
        :param kwargs: 外部可以传递x, y, 通过
                        x = kwargs.pop('x', self.x)
                        y = kwargs.pop('y', self.y)
            装饰器使用的fiter_type，
            eg:
                ttn_abu = AbuML.create_test_more_fiter()
                ttn_abu.plot_decision_function(fiter_type=ml.EMLFitType.E_FIT_CLF)
        """
        x = kwargs.pop('x', self.x)
        y = kwargs.pop('y', self.y)
        fiter = self.get_fiter()
        # 进行降维
        x = self._decomposition_2x(x, use_pca=use_pca)
        # 训练前进行clone(fiter)
        fiter = clone(fiter)
        fiter.fit(x, y)
        ABuMLExecute.plot_decision_boundary(lambda p_x: fiter.predict(p_x), x, y)

    @entry_wrapper(support=(EMLFitType.E_FIT_CLF,))
    def plot_roc_estimator(self, pos_label=None, **kwargs):
        """
        被entry_wrapper(support=(EMLFitType.E_FIT_CLF, ))装饰，即只支持分类
        计算fpr, tpr, thresholds，最后绘制roc_auc曲线进行可视化操作

        :param pos_label:
        :param kwargs: 外部可以传递x, y, 通过
                        x = kwargs.pop('x', self.x)
                        y = kwargs.pop('y', self.y)
            装饰器使用的fiter_type，
            eg:
                ttn_abu = AbuML.create_test_more_fiter()
                ttn_abu.plot_roc_estimator(fiter_type=ml.EMLFitType.E_FIT_CLF)
        """
        x = kwargs.pop('x', self.x)
        y = kwargs.pop('y', self.y)
        fiter = self.get_fiter()
        ABuMLExecute.plot_roc_estimator(fiter, x, y, pos_label=pos_label)

    @entry_wrapper(support=(EMLFitType.E_FIT_CLF,))
    def plot_confusion_matrices(self, **kwargs):
        """
        被entry_wrapper(support=(EMLFitType.E_FIT_CLF, ))装饰，即只支持分类
        套接plot_confusion_matrices进行训练集测试集拆封分混淆矩阵计算且可视化
        混淆矩阵

        :param 外部可以传递x, y, 通过
                        x = kwargs.pop('x', self.x)
                        y = kwargs.pop('y', self.y)
            装饰器使用的fiter_type，
            eg:
                ttn_abu = AbuML.create_test_more_fiter()
                ttn_abu.plot_confusion_matrices(fiter_type=ml.EMLFitType.E_FIT_CLF)
        """
        x = kwargs.pop('x', self.x)
        y = kwargs.pop('y', self.y)

        fiter = self.get_fiter()
        ABuMLExecute.plot_confusion_matrices(fiter, x, y)

    def bagging_classifier_best(self, **kwargs):
        """
        eg：
            bagging_classifier_best有param_grid参数调用：

            param_grid = {'max_samples': np.arange(1, 5), 'n_estimators': np.arange(100, 300, 50)}
            ttn_abu.bagging_classifier_best(param_grid=param_grid, n_jobs=-1)

            out: BaggingClassifier(max_samples=4, n_estimators=100)

            bagging_classifier_best无param_grid参数调用：

            from abupy import AbuML, ml
            ttn_abu = AbuML.create_test_more_fiter()
            ttn_abu.bagging_classifier_best()

        :param kwargs: 关键字可选参数param_grid: 最优字典关键字参数
                        eg：param_grid = {'max_samples': np.arange(1, 5), 'n_estimators': np.arange(100, 300, 50)}
                       关键字可选参数assign: 是否保存实例化后最优参数的学习器对象，默认True
                       关键字可选参数n_jobs: 并行执行的进程任务数量，默认-1, 开启与cpu相同数量的进程数
                       关键字可选参数show: 是否可视化最优参数搜索结果
        :return: 通过最优参数构造的BaggingClassifier对象
        """

        return self.estimator.bagging_classifier_best(self.x, self.y, **kwargs)

    def bagging_regressor_best(self, **kwargs):
        """
        eg：
            bagging_regressor_best有param_grid参数调用：

            param_grid = {'max_samples': np.arange(1, 5), 'n_estimators': np.arange(100, 300, 50)}
            ttn_abu.bagging_regressor_best(param_grid=param_grid, n_jobs=-1)

            out: BaggingRegressor(max_samples=4, n_estimators=250)

            bagging_regressor_best无param_grid参数调用：

            from abupy import AbuML, ml
            ttn_abu = AbuML.create_test_more_fiter()
            ttn_abu.bagging_regressor_best()

        :param kwargs: 关键字可选参数param_grid: 最优字典关键字参数
                        eg：param_grid = {'max_samples': np.arange(1, 5), 'n_estimators': np.arange(100, 300, 50)}
                       关键字可选参数assign: 是否保存实例化后最优参数的学习器对象，默认True
                       关键字可选参数n_jobs: 并行执行的进程任务数量，默认-1, 开启与cpu相同数量的进程数
                       关键字可选参数show: 是否可视化最优参数搜索结果
        :return: 通过最优参数构造的BaggingRegressor对象
        """
        return self.estimator.bagging_regressor_best(self.x, self.y, **kwargs)

    def adaboost_regressor_best(self, **kwargs):
        """
        eg：
            adaboost_regressor_best有param_grid参数调用：

            param_grid = {'learning_rate': np.arange(0.2, 1.2, 0.2), 'n_estimators': np.arange(10, 100, 10)}
            ttn_abu.adaboost_regressor_best(param_grid=param_grid, n_jobs=-1)

            out: AdaBoostRegressor(learning_rate=0.8, n_estimators=40)

            adaboost_regressor_best无param_grid参数调用：

            from abupy import AbuML, ml
            ttn_abu = AbuML.create_test_more_fiter()
            ttn_abu.adaboost_regressor_best()

        :param kwargs: 关键字可选参数param_grid: 最优字典关键字参数
                        eg：param_grid = {'learning_rate': np.arange(0.2, 1.2, 0.2),
                                         'n_estimators': np.arange(10, 100, 10)}
                       关键字可选参数assign: 是否保存实例化后最优参数的学习器对象，默认True
                       关键字可选参数n_jobs: 并行执行的进程任务数量，默认-1, 开启与cpu相同数量的进程数
                       关键字可选参数show: 是否可视化最优参数搜索结果
        :return: 通过最优参数构造的AdaBoostRegressor对象
        """
        return self.estimator.adaboost_regressor_best(self.x, self.y, **kwargs)

    def adaboost_classifier_best(self, **kwargs):
        """
        eg：
             adaboost_classifier_best有param_grid参数调用：

             param_grid = {'learning_rate': np.arange(0.2, 1.2, 0.2), 'n_estimators': np.arange(10, 100, 10)}
             ttn_abu.adaboost_classifier_best(param_grid=param_grid, n_jobs=-1)

             out: AdaBoostClassifier(learning_rate=0.6, n_estimators=70)

             adaboost_classifier_best无param_grid参数调用：

             from abupy import AbuML, ml
             ttn_abu = AbuML.create_test_more_fiter()
             ttn_abu.adaboost_classifier_best()


        :param kwargs: 关键字可选参数param_grid: 最优字典关键字参数
                        eg：param_grid = {'learning_rate': np.arange(0.2, 1.2, 0.2),
                                          'n_estimators': np.arange(10, 100, 10)}
                       关键字可选参数assign: 是否保存实例化后最优参数的学习器对象，默认True
                       关键字可选参数n_jobs: 并行执行的进程任务数量，默认-1, 开启与cpu相同数量的进程数
                       关键字可选参数show: 是否可视化最优参数搜索结果
        :return: 通过最优参数构造的AdaBoostClassifier对象
        """
        return self.estimator.adaboost_classifier_best(self.x, self.y, **kwargs)

    def random_forest_classifier_best(self, **kwargs):
        """
        eg：
            random_forest_classifier_best有param_grid参数调用：

            param_grid = {'max_features': ['sqrt', 'log2', ], 'n_estimators': np.arange(50, 200, 20)}
            ttn_abu.random_forest_classifier_best(param_grid=param_grid, n_jobs=-1)

            out: RandomForestClassifier(max_features='sqrt', n_estimators=190)

            random_forest_classifier_best无param_grid参数调用：

            from abupy import AbuML, ml
            ttn_abu = AbuML.create_test_more_fiter()
            ttn_abu.random_forest_classifier_best()

        :param kwargs: 关键字可选参数param_grid: 最优字典关键字参数
                        eg：param_grid = {'max_features': ['sqrt', 'log2', ],
                                         'n_estimators': np.arange(10, 150, 15)}

                       关键字可选参数assign: 是否保存实例化后最优参数的学习器对象，默认True
                       关键字可选参数n_jobs: 并行执行的进程任务数量，默认-1, 开启与cpu相同数量的进程数
                       关键字可选参数show: 是否可视化最优参数搜索结果
        :return: 通过最优参数构造的RandomForestClassifier对象
        """
        return self.estimator.random_forest_classifier_best(self.x, self.y, **kwargs)

    def random_forest_regressor_best(self, **kwargs):
        """
        eg：
            random_forest_regressor_best无param_grid参数调用：

            from abupy import AbuML, ml
            ttn_abu = AbuML.create_test_more_fiter()
            ttn_abu.random_forest_regressor_best()

            random_forest_regressor_best有param_grid参数调用：

            param_grid = {'max_features': ['sqrt', 'log2', ], 'n_estimators': np.arange(10, 150, 15)}
            ttn_abu.random_forest_regressor_best(param_grid=param_grid, n_jobs=-1)

            out: RandomForestRegressor(max_features='log2', n_estimators=115)


        :param kwargs: 关键字可选参数param_grid: 最优字典关键字参数
                        eg：param_grid = {'max_features': ['sqrt', 'log2', ],
                                         'n_estimators': np.arange(10, 150, 15)}

                       关键字可选参数assign: 是否保存实例化后最优参数的学习器对象，默认True
                       关键字可选参数n_jobs: 并行执行的进程任务数量，默认-1, 开启与cpu相同数量的进程数
                       关键字可选参数show: 是否可视化最优参数搜索结果
        :return: 通过最优参数构造的RandomForestRegressor对象
        """
        return self.estimator.random_forest_regressor_best(self.x, self.y, **kwargs)

    def xgb_classifier_best(self, **kwargs):
        """
        eg：
             xgb_classifier_best有param_grid参数调用：

             param_grid = {'learning_rate': np.arange(0.1, 0.5, 0.05), 'n_estimators': np.arange(50, 200, 10)}
             ttn_abu.xgb_classifier_best(param_grid=param_grid, n_jobs=-1)

             out: GradientBoostingClassifier(learning_rate=0.1, n_estimators=160)
             xgb_classifier_best无param_grid参数调用：

             from abupy import AbuML, ml
             ttn_abu = AbuML.create_test_more_fiter()
             ttn_abu.xgb_classifier_best()

        :param kwargs: 关键字可选参数param_grid: 最优字典关键字参数
                        eg：param_grid = {'learning_rate': np.arange(0.1, 0.5, 0.05),
                                          'n_estimators': np.arange(50, 200, 10)}

                       关键字可选参数assign: 是否保存实例化后最优参数的学习器对象，默认True
                       关键字可选参数n_jobs: 并行执行的进程任务数量，默认-1, 开启与cpu相同数量的进程数
                       关键字可选参数show: 是否可视化最优参数搜索结果
        :return: 通过最优参数构造的GradientBoostingClassifier对象
        """
        return self.estimator.xgb_classifier_best(self.x, self.y, **kwargs)

    def xgb_regressor_best(self, **kwargs):
        """
        eg：
             xgb_regressor_best有param_grid参数调用：

             param_grid = {'learning_rate': np.arange(0.1, 0.5, 0.05), 'n_estimators': np.arange(10, 100, 10)}
             ttn_abu.xgb_regressor_best(param_grid=param_grid, n_jobs=-1)

             out: GradientBoostingRegressor(learning_rate=0.2, n_estimators=70)


             xgb_regressor_best无param_grid参数调用：

             from abupy import AbuML, ml
             ttn_abu = AbuML.create_test_more_fiter()
             ttn_abu.xgb_regressor_best()


        :param kwargs: 关键字可选参数param_grid: 最优字典关键字参数
                        eg：param_grid = {'learning_rate': np.arange(0.1, 0.5, 0.05),
                                          'n_estimators': np.arange(10, 100, 10)}
                       关键字可选参数assign: 是否保存实例化后最优参数的学习器对象，默认True
                       关键字可选参数n_jobs: 并行执行的进程任务数量，默认-1, 开启与cpu相同数量的进程数
                       关键字可选参数show: 是否可视化最优参数搜索结果
        :return: 通过最优参数构造的GradientBoostingRegressor对象
        """
        return self.estimator.xgb_regressor_best(self.x, self.y, **kwargs)

    def decision_tree_classifier_best(self, **kwargs):
        """
        eg：
            decision_tree_classifier_best无param_grid参数调用：

            from abupy import AbuML, ml
            ttn_abu = AbuML.create_test_more_fiter()
            ttn_abu.decision_tree_classifier_best()

            decision_tree_classifier_best有param_grid参数调用：

            param_grid = {'max_features': ['sqrt', 'log2', ], 'max_depth': np.arange(1, 10, 1)}
            ttn_abu.decision_tree_classifier_best(param_grid=param_grid, n_jobs=-1)

            out: DecisionTreeClassifier(max_features='sqrt', max_depth=7)

        :param kwargs: 关键字可选参数param_grid: 最优字典关键字参数
                        eg：param_grid = {'max_features': ['sqrt', 'log2', ],
                                         'max_depth': np.arange(1, 10, 1)}
                       关键字可选参数assign: 是否保存实例化后最优参数的学习器对象，默认True
                       关键字可选参数n_jobs: 并行执行的进程任务数量，默认-1, 开启与cpu相同数量的进程数
                       关键字可选参数show: 是否可视化最优参数搜索结果
        :return: 通过最优参数构造的DecisionTreeClassifier对象
        """
        return self.estimator.decision_tree_classifier_best(self.x, self.y, **kwargs)

    def decision_tree_regressor_best(self, **kwargs):
        """
        eg：
            decision_tree_regressor_best无param_grid参数调用：

            from abupy import AbuML, ml
            ttn_abu = AbuML.create_test_more_fiter()
            ttn_abu.decision_tree_regressor_best()

            decision_tree_regressor_best有param_grid参数调用：

            param_grid = {'max_features': ['sqrt', 'log2', ], 'max_depth': np.arange(1, 10, 1)}
            ttn_abu.decision_tree_regressor_best(param_grid=param_grid, n_jobs=-1)

            out: DecisionTreeRegressor(max_features='sqrt', max_depth=3)

        :param kwargs: 关键字可选参数param_grid: 最优字典关键字参数
                        eg：param_grid = {'max_features': ['sqrt', 'log2', ],
                                          'max_depth': np.arange(1, 10, 1)}
                       关键字可选参数assign: 是否保存实例化后最优参数的学习器对象，默认True
                       关键字可选参数n_jobs: 并行执行的进程任务数量，默认-1, 开启与cpu相同数量的进程数
                       关键字可选参数show: 是否可视化最优参数搜索结果
        :return: 通过最优参数构造的DecisionTreeRegressor对象
        """
        return self.estimator.decision_tree_regressor_best(self.x, self.y, **kwargs)

    def knn_classifier_best(self, **kwargs):
        """
        eg：
          knn_classifier_best有param_grid参数调用：

          param_grid = {'algorithm': ['ball_tree', 'kd_tree', 'brute'], 'n_neighbors': np.arange(1, 26, 1)}
          ttn_abu.knn_classifier_best(param_grid=param_grid, n_jobs=-1)

          out: KNeighborsClassifier(algorithm='ball_tree', n_neighbors=14)

          knn_classifier_best无param_grid参数调用：

          from abupy import AbuML, ml
          ttn_abu = AbuML.create_test_more_fiter()
          ttn_abu.knn_classifier_best()

        :param kwargs: 关键字可选参数param_grid: 最优字典关键字参数
                        eg：param_grid = {'algorithm': ['ball_tree', 'kd_tree', 'brute'],
                                         'n_neighbors': np.arange(1, 26, 1)}
                       关键字可选参数assign: 是否保存实例化后最优参数的学习器对象，默认True
                       关键字可选参数n_jobs: 并行执行的进程任务数量，默认-1, 开启与cpu相同数量的进程数
                       关键字可选参数show: 是否可视化最优参数搜索结果
        :return: 通过最优参数构造的KNeighborsClassifier对象
        """
        return self.estimator.knn_classifier_best(self.x, self.y, **kwargs)

    def _confusion_matrix_with_report(self, y_true, y_predict, labels=None):
        """
        封装metrics.confusion_matrix与metrics.classification_report，对分类
        结果进行度量
        :param y_true: 原始数据中的y序列值
        :param y_predict: 使用分类器predict的y序列值
        :param labels: y序列中的lable序列
        """
        confusion_matrix = metrics.confusion_matrix(y_true, y_predict, labels=labels)
        if len(labels) == 2:
            # 只有二分问题Actual/Predicted
            self.log_func("          Predicted")
            self.log_func("         |  0  |  1  |")
            self.log_func("         |-----|-----|")
            self.log_func("       0 | %3d | %3d |" % (confusion_matrix[0, 0],
                                                      confusion_matrix[0, 1]))
            self.log_func("Actual   |-----|-----|")
            self.log_func("       1 | %3d | %3d |" % (confusion_matrix[1, 0],
                                                      confusion_matrix[1, 1]))
            self.log_func("         |-----|-----|")
        else:
            # 非二分问题直接显示confusion_matrix
            self.log_func("Confusion Matrix: \n{}".format(confusion_matrix))

        self.log_func(metrics.classification_report(y_true, y_predict))

    def _decomposition_2x(self, x, use_pca):
        """
        通过pca进行降维或者选中两个importances最大的特征将
        x变成只有两个维度的特征矩阵
        :param x: 进行降维的特征矩阵
        :param use_pca: 是否使用pca进行特征降维
        :return: 降维后的特征矩阵，降维后的矩阵只有两个维度
        """
        if use_pca:
            # 构造一个临时的pca进行降维，n_components=2只保留两个维度
            pca_2n = self.estimator.pca_decomposition(n_components=2, assign=False)
            x = pca_2n.fit_transform(x)
            """
                eg:
                x before fit_transform:
                array([[ 1.    ,  0.    ,  1.    , ...,  1.    , -0.5614, -0.5024],
                       [ 1.    ,  0.    ,  0.    , ...,  0.    ,  0.6132,  0.7868],
                       [ 0.    ,  0.    ,  1.    , ...,  1.    , -0.2677, -0.4889],
                       ...,
                       [ 1.    ,  2.    ,  1.    , ...,  1.    , -0.9924, -0.1763],
                       [ 0.    ,  0.    ,  0.    , ...,  0.    , -0.2677, -0.0444],
                       [ 0.    ,  0.    ,  1.    , ...,  1.    ,  0.1727, -0.4924]])
                x fit_transform:
                array([[ 0.3805, -1.0005],
                       [ 0.0586,  1.7903],
                       [-0.3162, -0.7404],
                       ...,
                       [ 1.6132, -0.5185],
                       [-0.5952,  0.6252],
                       [-0.7428, -0.7119]])
            """
        else:
            # 选中两个importances最大的两个去画决策边界
            importances = self.importances_coef_pd()
            """
                        feature  importance
                0         SibSp      0.0000
                1         Parch      0.0000
                3     Cabin_Yes      0.0000
                4    Embarked_C      0.0000
                5    Embarked_Q      0.0000
                6    Embarked_S      0.0000
                7    Sex_female      0.0000
                9      Pclass_1      0.0000
                10     Pclass_2      0.0000
                12   Age_scaled      0.0000
                13  Fare_scaled      0.0000
                2      Cabin_No      0.0831
                11     Pclass_3      0.1836
                8      Sex_male      0.7333
            """
            if importances is None:
                self.log_func('self.importances_coef_pd() importances is None!!!')
                return
            # 根据importance排序特征重要程度，拿出最重要的两个维度index序列, eg: most_two=[8, 11]
            most_two = sorted(importances.sort_values('importance').index[-2:].tolist())
            # 从x中根据most_two拼接出一个新的x矩阵，只有两个维度
            x = np.concatenate((x[:, most_two[0]][:, np.newaxis],
                                x[:, most_two[1]][:, np.newaxis]), axis=1)

        return x

    def _do_cross_val_score(self, x, y, cv, scoring):
        """
        封装sklearn中cross_val_score方法， 参数x, y, cv, scoring透传cross_val_score
        :param x: 训练集x矩阵，numpy矩阵
        :param y: 训练集y序列，numpy序列
        :param cv: 透传cross_val_score，cv参数，int
        :param scoring: 透传cross_val_score, 使用的度量方法
        :return: cross_val_score返回的score序列，
                 eg: array([ 1.  ,  0.9 ,  1.  ,  0.9 ,  1.  ,  0.9 ,  1.  ,  0.9 ,  0.95,  1.  ])
        """
        fiter = self.get_fiter()
        """
            eg: fiter
            DecisionTreeClassifier(class_weight=None, criterion='gini', max_depth=None,
                                    max_features=None, max_leaf_nodes=None,
                                    min_impurity_split=1e-07, min_samples_leaf=1,
                                    min_samples_split=2, min_weight_fraction_leaf=0.0,
                                    presort=False, random_state=None, splitter='best')
        """
        if scoring == _EMLScoreType.E_SCORE_ROC_AUC.value and len(np.unique(y)) != 2:
            # roc auc的度量下且y的label数量不是2项分类，首先使用label_binarize进行处理
            y_label_binarize = label_binarize(y, classes=np.unique(y))
            """
                eg：
                    np.unique(y) ＝ array([0, 1, 2])
                    y_label_binarize:
                    array([[1, 0, 0],
                           [1, 0, 0],
                           [1, 0, 0],
                           [1, 0, 0],
                           [1, 0, 0],
                           .........
                           [0, 1, 0],
                           [0, 1, 0],
                           [0, 1, 0],
                           [0, 1, 0],
                           [0, 1, 0],
                           [0, 1, 0],
                           [0, 1, 0],
                           [0, 1, 0],
                           .........
                           [0, 0, 1],
                           [0, 0, 1],
                           [0, 0, 1],
                           [0, 0, 1],
                           [0, 0, 1],
                           [0, 0, 1]])
            """
            label_cnt = len(np.unique(y))
            # one vs rest的score平均值的和
            mean_sum = 0
            # one vs rest中的最好score平均值
            best_mean = 0
            # 最好score平均值(best_mean)的score序列，做为结果返回
            scores = list()
            for ind in np.arange(0, label_cnt):
                # 开始 one vs rest
                _y = y_label_binarize[:, ind]
                """
                    eg: _y
                    array([1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                           1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                           1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                           0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                           0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                           0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                           0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
                """
                tmp_scores = cross_val_score(fiter, x, _y, cv=cv, scoring=scoring)
                tmp_mean = np.mean(tmp_scores)
                # one vs rest的score平均值进行叠加sum
                mean_sum += tmp_mean
                if len(scores) == 0 or tmp_mean > best_mean:
                    scores = tmp_scores
            # one vs rest的score平均值的和 / label_cnt
            mean_sc = mean_sum / label_cnt
        else:
            scores = cross_val_score(fiter, x, y, cv=cv, scoring=scoring)
            # 计算度量的score平均值，做为log输出，结果返回的仍然是scores
            mean_sc = -np.mean(np.sqrt(-scores)) if scoring == mean_squared_error_scorer \
                else np.mean(scores)
        self.log_func('{} score mean: {}'.format(fiter.__class__.__name__, mean_sc))

        return scores
