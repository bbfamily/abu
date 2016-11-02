# -*- encoding:utf-8 -*-
"""

封装grid

由于现在在写一本关于股票量化方面的书，会涉及到相关文章里的一些内容，
出版社方面不希望我现在开源全部代码，但是一定会开源，最晚等书出版发行
以后，现在只能开源文章中涉及的部分代码，整个系统的开源会稍后，请谅解
我其实也觉着有人能看你的代码就已经很给面子了，但是。。。再次抱歉！！

"""
from __future__ import division
from __future__ import print_function

import matplotlib.pyplot as plt
import numpy as np
from sklearn.grid_search import GridSearchCV
from sklearn.neighbors import KNeighborsClassifier

__author__ = 'BBFamily'


def grid_search_common_clf(estimator, x, y, cv=5, n_jobs=1, **kwargs):
    """
        普遍适应
        :param estimator:
        :param x:
        :param y:
        :param cv:
        :param n_jobs:
        :param kwargs:
        :return:
    """
    param_grid = kwargs
    grid = GridSearchCV(estimator, param_grid, cv=cv, scoring='accuracy', n_jobs=n_jobs)
    grid.fit(x, y)
    return grid


def grid_search_estimators_clf(estimator, x, y, cv=5, n_estimators=range(10, 500, 10), n_jobs=-1, show=True):
    """
        可以对n_estimators多个
        :param estimator:
        :param x:
        :param y:
        :param cv:
        :param n_estimators:
        :param n_jobs:
        :param show:
        :return:
    """
    param_grid = dict(n_estimators=n_estimators)
    grid = GridSearchCV(estimator, param_grid, cv=cv, scoring='accuracy', n_jobs=n_jobs)
    grid.fit(x, y)

    if show:
        grid_mean_scores = [result[1] for result in grid.grid_scores_]
        plt.plot(n_estimators, grid_mean_scores)
        plt.plot(grid.best_params_['n_estimators'], grid.best_score_, 'ro', markersize=12, markeredgewidth=1.5,
                 markerfacecolor='None', markeredgecolor='r')
        plt.title('search estimators')
        plt.show()
    return grid.best_score_, grid.best_params_


def grid_search_neighbors_clf(x, y, cv=5, neighbors_range=None, n_jobs=1, show=True):
    if neighbors_range is None:
        neighbors_range = range(1, np.minimum(20, x.shape[0] // 2))
    param_grid = dict(n_neighbors=neighbors_range)
    grid = GridSearchCV(KNeighborsClassifier(), param_grid, cv=cv, scoring='accuracy', n_jobs=n_jobs)
    grid.fit(x, y)

    if show:
        grid_mean_scores = [result[1] for result in grid.grid_scores_]
        plt.plot(neighbors_range, grid_mean_scores)
        plt.plot(grid.best_params_['n_neighbors'], grid.best_score_, 'ro', markersize=12, markeredgewidth=1.5,
                 markerfacecolor='None', markeredgecolor='r')
        plt.title('search neighbors')
        plt.show()

    return grid.best_score_, grid.best_params_
