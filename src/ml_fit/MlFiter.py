# -*- encoding:utf-8 -*-
"""
中间层，从上层拿到x，y，df
拥有create estimator

由于现在在写一本关于股票量化方面的书，会涉及到相关文章里的一些内容，
出版社方面不希望我现在开源全部代码，但是一定会开源，最晚等书出版发行
以后，现在只能开源文章中涉及的部分代码，整个系统的开源会稍后，请谅解
我其实也觉着有人能看你的代码就已经很给面子了，但是。。。再次抱歉！！

"""
from __future__ import division
from __future__ import print_function

import functools
from collections import defaultdict

import ZEnv
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn import cross_validation
from sklearn import metrics
from sklearn.cross_validation import train_test_split
from sklearn.feature_selection import RFE
from sklearn.feature_selection import RFECV
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.preprocessing import binarize

from sklearn.datasets import load_iris
import sklearn.preprocessing as preprocessing

import MlFiterExcute
import MlFiterGrid
import ZLog
from Decorator import warnings_filter
from MlFiterCreater import MlFiterCreaterClass

__author__ = 'BBFamily'

K_ACCURACY_SCORE = 'accuracy'
K_MEAN_SQ_SCORE = 'mean_squared_error'
K_ROC_AUC_SCORE = 'roc_auc'


def decorator_xy(func):
    """
    封装proxy_xy的装饰起
    """
    @functools.wraps(func)
    def wrapper(self, *args, **kwargs):
        """
        置换kwargs中的x, y
        :param self:
        :param args:
        :param kwargs:
        :return:
        """
        if 'x' in kwargs and 'y' in kwargs:
            x, y = self.proxy_xy(kwargs['x'], kwargs['y'])
        else:
            x, y = self.proxy_xy(None, None)
        kwargs['x'] = x
        kwargs['y'] = y
        return func(self, *args, **kwargs)

    return wrapper


class MlFiterClass(object):
    @classmethod
    def create_test_fiter(cls):
        iris = load_iris()
        x = iris.data
        y = iris.target

        x_df = pd.DataFrame(x, columns=['x0', 'x1', 'x2', 'x3'])
        y_df = pd.DataFrame(y, columns=['y'])
        df = y_df.join(x_df)
        return MlFiterClass(x, y, df)

    @classmethod
    @warnings_filter
    def create_test_more_fiter(cls):
        data_train = pd.read_csv(ZEnv.g_project_root + "/data/Train.csv")
        from sklearn.ensemble import RandomForestRegressor

        def set_missing_ages(p_df):
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
            p_df.loc[(p_df.Cabin.notnull()), 'Cabin'] = "Yes"
            p_df.loc[(p_df.Cabin.isnull()), 'Cabin'] = "No"
            return p_df

        data_train, rfr = set_missing_ages(data_train)
        data_train = set_cabin_type(data_train)
        dummies__cabin = pd.get_dummies(data_train['Cabin'], prefix='Cabin')
        dummies__embarked = pd.get_dummies(data_train['Embarked'], prefix='Embarked')
        dummies__sex = pd.get_dummies(data_train['Sex'], prefix='Sex')
        dummies__pclass = pd.get_dummies(data_train['Pclass'], prefix='Pclass')
        df = pd.concat([data_train, dummies__cabin, dummies__embarked, dummies__sex, dummies__pclass], axis=1)
        # noinspection PyUnresolvedReferences
        df.drop(['Pclass', 'Name', 'Sex', 'Ticket', 'Cabin', 'Embarked'], axis=1, inplace=True)
        scaler = preprocessing.StandardScaler()
        age_scale_param = scaler.fit(df['Age'])
        # noinspection PyUnresolvedReferences
        df['Age_scaled'] = scaler.fit_transform(df['Age'], age_scale_param)
        # noinspection PyUnresolvedReferences
        fare_scale_param = scaler.fit(df['Fare'])
        # noinspection PyUnresolvedReferences
        df['Fare_scaled'] = scaler.fit_transform(df['Fare'], fare_scale_param)
        # noinspection PyUnresolvedReferences
        train_df = df.filter(regex='Survived|Age_.*|SibSp|Parch|Fare_.*|Cabin_.*|Embarked_.*|Sex_.*|Pclass_.*')
        train_np = train_df.as_matrix()
        y = train_np[:, 0]
        x = train_np[:, 1:]
        return MlFiterClass(x, y, train_df)

    def __init__(self, x, y, df, force_clf=False):
        """
            中间层需要所有原料都配齐
        """
        self.estimator = MlFiterCreaterClass()

        self.x = x
        self.y = y
        self.df = df
        self.force_clf = force_clf

    def proxy_xy(self, x, y):
        if x is None or y is None:
            return self.x, self.y
        return x, y

    @decorator_xy
    def cross_val_accuracy_score(self, cv=10, **kwargs):
        return self._do_cross_val_score(kwargs['x'], kwargs['y'], cv, K_ACCURACY_SCORE)

    @decorator_xy
    def cross_val_mean_squared_score(self, cv=10, **kwargs):
        return self._do_cross_val_score(kwargs['x'], kwargs['y'], cv, K_MEAN_SQ_SCORE)

    @decorator_xy
    def cross_val_roc_auc_score(self, cv=10, **kwargs):
        return self._do_cross_val_score(kwargs['x'], kwargs['y'], cv, K_ROC_AUC_SCORE)

    @decorator_xy
    def feature_selection(self, **kwargs):
        x, y = kwargs['x'], kwargs['y']
        fiter = self.get_fiter()

        selector = RFE(fiter)
        selector.fit(x, y)

        ZLog.info('RFE selection')
        ZLog.info(pd.DataFrame({'support': selector.support_, 'ranking': selector.ranking_},
                               index=self.df.columns[1:]))

        selector = RFECV(fiter, cv=3, scoring='mean_squared_error')
        selector.fit(x, y)
        ZLog.newline()
        ZLog.info('RFECV selection')
        ZLog.info(pd.DataFrame({'support': selector.support_, 'ranking': selector.ranking_},
                               index=self.df.columns[1:]))

    def _confusion_matrix_with_report(self, test_y, predictions):
        confusion_matrix = metrics.confusion_matrix(test_y, predictions)
        print("Confusion Matrix ", confusion_matrix)
        print("          Predicted")
        print("         |  0  |  1  |")
        print("         |-----|-----|")
        print("       0 | %3d | %3d |" % (confusion_matrix[0, 0],
                                          confusion_matrix[0, 1]))
        print("Actual   |-----|-----|")
        print("       1 | %3d | %3d |" % (confusion_matrix[1, 0],
                                          confusion_matrix[1, 1]))
        print("         |-----|-----|")

        '''
            一般情况下loss， win可以概括
        '''
        ZLog.info(classification_report(test_y,
                                        predictions))

    def echo_info(self, fiter=None):
        if fiter is None:
            fiter = self.get_fiter()
        ZLog.info(format(fiter.__class__.__name__, '*^58s'))

    def fit(self):
        fiter = self.get_fiter()
        fiter.fit(self.x, self.y)
        return fiter

    @decorator_xy
    def importances_coef_pd(self, **kwargs):
        if not hasattr(self, 'df'):
            raise ValueError('please make a df func first!')

        x, y = kwargs['x'], kwargs['y']
        fiter = self.get_fiter()
        fiter.fit(x, y)

        self.echo_info(fiter)
        if hasattr(fiter, 'feature_importances_'):
            return pd.DataFrame(
                {'feature': list(self.df.columns)[1:], 'importance': fiter.feature_importances_}).sort_values(
                'importance')
        elif hasattr(fiter, 'coef_'):
            return pd.DataFrame({"columns": list(self.df.columns)[1:], "coef": list(fiter.coef_.T)})
        else:
            ZLog.info('fiter not hasattr feature_importances_ or coef_!')

    def _do_prob_maximum_score(self, pb_st, x, y, how, threshold, show):
        x, y = self.proxy_xy(x, y)
        fiter = self.get_fiter()
        mi = MlFiterExcute.run_prob_cv_estimator(fiter, x, y, n_folds=10)
        prob_dict = {pb: binarize(mi.reshape(1, -1), pb).T for pb in pb_st}

        pmean_dict = {}
        for pb, y_pre in prob_dict.items():

            confusion = metrics.confusion_matrix(y, y_pre)

            t_p = confusion[1, 1]
            t_n = confusion[0, 0]
            f_p = confusion[0, 1]
            f_n = confusion[1, 0]

            # ac = metrics.accuracy_score(y, y_pre)

            ac = t_n + f_n + t_p + f_p
            sc_mean = 0
            if how == 'nc' and t_n > ac * threshold:
                '''
                    在数量海好的前提下：
                    关心对本来是1误判为0不容易接受
                         本来是0判断为1还凑合
                    需要平衡1，3，4�现

                            Predicted
                             |  0  |  1  |
                             |-----|-----|
                           0 | 556 | 3387 |
                    Actual   |-----|-----|
                           1 | 332 | 2671 |
                             |-----|-----|
                '''
                nc = t_n / (t_n + f_n)  # 1,3
                # nc = TN / (TN + FN + TP)
                # rs = TP / (TN + FN + TP) 
                rs = metrics.recall_score(y, y_pre)  # 3,4
                sc_mean = (nc + rs) / 2
            elif how == 'np' and f_p > ac * threshold:
                '''
                    在数量海好的前提下：
                    关心对本来是0误判为1不容易接受
                    本来是1判断为0还凑合
                    需要平衡1，2，4�现
                '''
                # np = TN / (TN + FP + TP)
                # ps = TP / (TN + FP + TP) 
                tnp = t_n / (t_n + f_p)
                ps = metrics.precision_score(y, y_pre)
                sc_mean = (tnp + ps) / 2

            pmean_dict[pb] = (sc_mean, t_n)

        max_v = max(zip(pmean_dict.values(), pmean_dict.keys()))

        if show:
            pmean_dict_st = sorted(zip(pmean_dict.keys(), pmean_dict.values()))
            plt.plot([st[0] for st in pmean_dict_st], [st[1][0] for st in pmean_dict_st])
            plt.plot(max_v[1], max_v[0][0], 'ro', markersize=12, markeredgewidth=1.5,
                     markerfacecolor='None', markeredgecolor='r')
            plt.title('maxmin score')

            self.scores(prob_dict[max_v[1]], y)
        return max_v

    def prob_maximum_recall_score(self, x=None, y=None, pb_st=None, threshold=0.05, show=True):
        """
            向low recall score 方向寻找最优概率

            thresholds = np.linspace(0.05, 0.4, 8)
            for threshold in thresholds:
                fiter.prob_maximum_recall_score(threshold=threshold)
        """
        if pb_st is None:
            pb_st = np.linspace(0.10, 0.50)
        return self._do_prob_maximum_score(pb_st, x, y, 'nc', threshold, show)

    def prob_maximum_low(self, x=None, y=None, show=True):
        x, y = self.proxy_xy(x, y)
        fiter = self.get_fiter()
        y_prob = MlFiterExcute.run_prob_cv_estimator(fiter, x, y, n_folds=10)
        l_pb = y_prob[y_prob < y_prob.mean()].mean()
        y_prob_l = binarize(y_prob.reshape(-1, 1), l_pb)
        if show:
            self.scores(y_prob_l, y)
        return l_pb

    def prob_maximum_precision_score(self, x=None, y=None, pb_st=None, threshold=0.05, show=True):
        """
            向low precision_score 方向寻找最优概率
        """
        if pb_st is None:
            pb_st = np.linspace(0.50, 0.90)
        return self._do_prob_maximum_score(pb_st, x, y, 'np', threshold, show)

    def prob_binarize_y(self, x=None, y=None, pb=0.5):
        x, y = self.proxy_xy(x, y)
        fiter = self.get_fiter()
        mi = MlFiterExcute.run_prob_cv_estimator(fiter, x, y, n_folds=10)
        ypb = binarize(mi.reshape(1, -1), pb).T
        return ypb

    def scores(self, y_pre, y=None):
        ZLog.info('scores(self, y_pre, y=None)')
        _, y = self.proxy_xy(None, y)
        ZLog.info("accuracy = %.2f" % (accuracy_score(y, y_pre)))
        ZLog.info("precision_score = %.2f" % (metrics.precision_score(y, y_pre)))
        ZLog.info("recall_score = %.2f" % (metrics.recall_score(y, y_pre)))

        self._confusion_matrix_with_report(y, y_pre)

    def train_test_split_xy(self, x=None, y=None, test_size=0.1, random_state=0):
        x, y = self.proxy_xy(x, y)
        train_x, test_x, train_y, test_y = train_test_split(x,
                                                            y,
                                                            test_size=test_size,
                                                            random_state=random_state)
        ZLog.info(x.shape, y.shape)
        ZLog.info(train_x.shape, train_y.shape)
        ZLog.info(test_x.shape, test_y.shape)

        fiter = self.get_fiter()
        clf = fiter.fit(train_x, train_y)
        predictions = clf.predict(test_x)

        ZLog.info("accuracy = %.2f" % (accuracy_score(test_y, predictions)))
        ZLog.info("precision_score = %.2f" % (metrics.precision_score(test_y, predictions)))
        ZLog.info("recall_score = %.2f" % (metrics.recall_score(test_y, predictions)))

        self._confusion_matrix_with_report(test_y, predictions)

    def train_test_split_df(self, df=None, test_size=0.1, random_state=0):
        if df is None:
            df = self.df

        train_df, cv_df = train_test_split(df, test_size=test_size, random_state=random_state)

        fiter = self.get_fiter()

        fiter.fit(train_df.as_matrix()[:, 1:], train_df.as_matrix()[:, 0])

        predictions = fiter.predict(cv_df.as_matrix()[:, 1:])

        ZLog.info("accuracy = %.2f" % (accuracy_score(cv_df.as_matrix()[:, 0], predictions)))
        ZLog.info("precision_score = %.2f" % (metrics.precision_score(cv_df.as_matrix()[:, 0], predictions)))
        ZLog.info("recall_score = %.2f" % (metrics.recall_score(cv_df.as_matrix()[:, 0], predictions)))

        self._confusion_matrix_with_report(cv_df.as_matrix()[:, 0], predictions)

    def plot_graphviz_tree(self, x=None, y=None):
        x, y = self.proxy_xy(x, y)
        fiter = self.get_fiter()

        return MlFiterExcute.graphviz_tree(fiter, self.df.columns, x, y)

    def plot_visualize_tree(self, x=None, y=None, use_pca=True):
        x, y = self.proxy_xy(x, y)
        fiter = self.get_fiter()
        '''
            只能选中两个importances最大的两个去画决策边界
        '''
        if use_pca:
            pca_2n = MlFiterCreaterClass().pca_func(n_components=2)
            pca_2n.fit(x)
            x = pca_2n.fit_transform(x)
        else:
            importances = self.importances_coef_pd()
            if importances is None:
                return
            most_two = sorted(importances.sort_values('importance').index[-2:].tolist())
            # X = X[:, most_two[0]:most_two[1] + 1]
            x = np.concatenate((x[:, most_two[0]][:, np.newaxis],
                                x[:, most_two[1]][:, np.newaxis]), axis=1)

        return MlFiterExcute.visualize_tree(fiter, x, y)

    def plot_learning_curve(self, x=None, y=None, use_reg=True):
        x, y = self.proxy_xy(x, y)
        if use_reg:
            fiter = self.estimator.reg
        else:
            fiter = self.get_fiter()
        return MlFiterExcute.plot_learning_curve(fiter, x, y)

    def plot_learning_curve_degree(self, x=None, y=None, degree=1):
        x, y = self.proxy_xy(x, y)
        return MlFiterExcute.plot_learning_curve(self.estimator.polynomial_regression(degree=degree), x, y)

    def plot_learning_curve_degree_error(self, x=None, y=None, degree=1):
        x, y = self.proxy_xy(x, y)
        return MlFiterExcute.plot_learning_curve_error(self.estimator.polynomial_regression(degree=degree), x, y)

    def plot_decision_function(self, x=None, y=None, use_pca=True):
        x, y = self.proxy_xy(x, y)

        fiter = self.get_fiter()

        '''
            只能选中两个importances最大的两个去画决策边界
        '''
        if use_pca:
            pca_2n = MlFiterCreaterClass().pca_func(n_components=2)
            pca_2n.fit(x)
            x = pca_2n.fit_transform(x)
        else:
            importances = self.importances_coef_pd()
            most_two = sorted(importances.sort_values('importance').index[-2:].tolist())
            x = np.concatenate((x[:, most_two[0]][:, np.newaxis],
                                x[:, most_two[1]][:, np.newaxis]), axis=1)

        fiter.fit(x, y)
        plt.scatter(x[:, 0], x[:, 1], c=y, s=50, cmap='spring')
        MlFiterExcute.plot_decision_function(fiter)
        MlFiterExcute.plot_decision_boundary(lambda p_x: fiter.predict(p_x), x, y)

    def plot_roc_estimator(self, x=None, y=None):
        x, y = self.proxy_xy(x, y)
        fiter = self.get_fiter()
        ZLog.info(fiter.__class__.__name__ + ' :roc')
        MlFiterExcute.plot_roc_estimator(fiter, x, y)

    def plot_confusion_matrices(self, x=None, y=None):
        x, y = self.proxy_xy(x, y)
        fiter = self.get_fiter()
        # y_clf = np.random.binomial(1, 0.5, len(X))
        MlFiterExcute.plot_confusion_matrices(fiter, x, y)

    def make_prob_cv_estimator(self, x=None, y=None):
        x, y = self.proxy_xy(x, y)
        fiter = self.get_fiter()

        pred_prob = MlFiterExcute.run_prob_cv_estimator(fiter, x, y)

        is_one = (y == 1)  # 实际上真正的y
        counts = pd.value_counts(pred_prob)

        true_prob = defaultdict(float)
        for prob in counts.index:
            '''
                按照个数应有的比例算true_prob
                其实不是真实的
            '''
            true_prob[prob] = np.mean(is_one[pred_prob == prob])
        true_prob = pd.Series(true_prob)
        counts = pd.concat([counts, true_prob], axis=1).reset_index()
        counts.columns = ['pred_prob', 'count', 'true_prob']
        return counts

    def plot_degree_fit(self, x=None, y=None, degree=1):
        x, y = self.proxy_xy(x, y)
        model = self.estimator.polynomialRegression(degree=degree)
        model.fit(x, y)

        x_test = np.linspace(np.floor(x.min()), np.ceil(x.max()), len(x) * 5)[:, None]
        y_test = model.predict(x_test)

        plt.scatter(x, y)
        plt.plot(x_test.ravel(), y_test)
        plt.show()

    def get_fiter(self):
        fiter = self.estimator.reg

        if len(np.unique(self.y)) <= 10:
            ''' 
                小于等于10个class的y就认为是要用分类了，这里自动判别，
                不再通过参数等方式，大于10个的要绕开
            '''
            fiter = self.estimator.clf
        elif self.force_clf:
            fiter = self.estimator.clf

        return fiter

    def _do_cross_val_score(self, x, y, cv, scoring):
        fiter = self.get_fiter()

        scores = cross_validation.cross_val_score(fiter, x, y, cv=cv, scoring=scoring)

        mean_sc = np.mean(np.sqrt(-scores)) if scoring == 'mean_squared_error' \
            else np.mean(scores)

        ZLog.info(scoring + ' mean: ' + str(mean_sc))

        return scores

    def grid_search_estimators_clf(self, x=None, y=None):
        x, y = self.proxy_xy(x, y)
        fiter = self.get_fiter()
        MlFiterGrid.grid_search_estimators_clf(fiter, x, y)
