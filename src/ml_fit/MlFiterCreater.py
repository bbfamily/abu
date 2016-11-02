# -*- encoding:utf-8 -*-
"""

封装ml分类回归器的Creater

由于现在在写一本关于股票量化方面的书，会涉及到相关文章里的一些内容，
出版社方面不希望我现在开源全部代码，但是一定会开源，最晚等书出版发行
以后，现在只能开源文章中涉及的部分代码，整个系统的开源会稍后，请谅解
我其实也觉着有人能看你的代码就已经很给面子了，但是。。。再次抱歉！！

"""
from __future__ import print_function

from hmmlearn.hmm import GaussianHMM
from sklearn import linear_model
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import AdaBoostRegressor
from sklearn.ensemble import BaggingClassifier
from sklearn.ensemble import BaggingRegressor
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsClassifier

__author__ = 'BBFamily'


class MlFiterCreaterClass(object):
    def __init__(self):
        """
            默认使用简单的回归，分类器
        """
        self.reg = DecisionTreeRegressor()
        self.clf = DecisionTreeClassifier()
        self.hmm = None
        self.gmm = None
        self.pca = None
        self.kmean = None
        # raise RuntimeError('CANT __init__ MUST CALL A FACTOR FUNC!')

    def __str__(self):
        return 'reg: ' + str(self.reg) + '\n' + \
               'clf: ' + str(self.clf)

    __repr__ = __str__

    def gaussian_hmm(self, **kwargs):
        if kwargs is not None and len(kwargs) > 0:
            self.hmm = GaussianHMM(**kwargs)
        else:
            self.hmm = GaussianHMM(n_components=5, covariance_type='diag', n_iter=5000)
        return self.hmm

    def pca_func(self, **kwargs):
        if kwargs is not None and len(kwargs) > 0:
            self.pca = PCA(**kwargs)
        else:
            """
                没参数直接要保留95%的尾数
            """
            self.pca = PCA(0.95)
        return self.pca

    def kmean(self, **kwargs):
        if kwargs is not None and len(kwargs) > 0:
            self.kmean = KMeans(**kwargs)
        else:
            self.kmean = KMeans(n_clusters=3, random_state=0)
        return self.kmean

    def bagging_classifier(self, clf_core=DecisionTreeClassifier(), **kwargs):
        if kwargs is not None and len(kwargs) > 0:
            self.clf = BaggingClassifier(clf_core, **kwargs)
        else:
            self.clf = BaggingClassifier(clf_core, n_estimators=200, bootstrap=True, oob_score=True, random_state=1)
        return self.clf

    def bagging_regressor(self, reg_core=DecisionTreeRegressor(), **kwargs):
        if kwargs is not None and len(kwargs) > 0:
            self.reg = BaggingRegressor(reg_core, **kwargs)
        else:
            self.reg = BaggingRegressor(reg_core, n_estimators=200, bootstrap=True, oob_score=True, random_state=1)
        return self.reg

    def adaboost_regressor(self, **kwargs):
        if kwargs is not None and len(kwargs) > 0:
            self.reg = AdaBoostRegressor(**kwargs)
        else:
            self.reg = AdaBoostRegressor(n_estimators=100, random_state=1)
        return self.reg

    def random_forest_regressor(self, **kwargs):
        if kwargs is not None and len(kwargs) > 0:
            self.clf = RandomForestRegressor(**kwargs)
        else:
            self.clf = RandomForestRegressor(n_estimators=100)
        return self.clf

    def adaboost_classifier(self, **kwargs):
        if kwargs is not None and len(kwargs) > 0:
            self.clf = AdaBoostClassifier(**kwargs)
        else:
            self.clf = AdaBoostClassifier(n_estimators=100, random_state=1)
        return self.clf

    def random_forest_classifier(self, **kwargs):
        if kwargs is not None and len(kwargs) > 0:
            self.clf = RandomForestClassifier(**kwargs)
        else:
            self.clf = RandomForestClassifier(n_estimators=100)
        return self.clf

    def svc(self, **kwargs):
        if kwargs is not None and len(kwargs) > 0:
            self.clf = SVC(**kwargs)
        else:
            self.clf = SVC(kernel='rbf')
        return self.clf

    def decision_tree_regressor(self, **kwargs):
        if kwargs is not None and len(kwargs) > 0:
            self.reg = linear_model.DecisionTreeRegressor(**kwargs)
        else:
            self.reg = linear_model.DecisionTreeRegressor(max_depth=2, random_state=1)
        return self.reg

    def decision_tree_classifier(self, **kwargs):
        if kwargs is not None and len(kwargs) > 0:
            self.clf = DecisionTreeClassifier(**kwargs)
        else:
            self.clf = DecisionTreeClassifier(max_depth=2, random_state=1)
        return self.clf

    def knn_classifier(self, **kwargs):
        if kwargs is not None and len(kwargs) > 0:
            self.clf = KNeighborsClassifier(**kwargs)
        else:
            self.clf = KNeighborsClassifier(n_neighbors=1)
        return self.clf

    def logistic_regression(self, **kwargs):
        if kwargs is not None and len(kwargs) > 0:
            self.reg = linear_model.LogisticRegression(**kwargs)
        else:
            self.reg = linear_model.LogisticRegression(C=1.0, penalty='l1', tol=1e-6)
        return self.reg

    def polynomial_regression(self, degree=2, **kwargs):
        return make_pipeline(PolynomialFeatures(degree), LinearRegression(**kwargs))
