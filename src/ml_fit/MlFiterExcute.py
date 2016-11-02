# -*- encoding:utf-8 -*-
"""

封装常用的分析方式及流程模块

由于现在在写一本关于股票量化方面的书，会涉及到相关文章里的一些内容，
出版社方面不希望我现在开源全部代码，但是一定会开源，最晚等书出版发行
以后，现在只能开源文章中涉及的部分代码，整个系统的开源会稍后，请谅解
我其实也觉着有人能看你的代码就已经很给面子了，但是。。。再次抱歉！！

"""
from __future__ import division
from __future__ import print_function

import os

import matplotlib.cbook as cbook
import matplotlib.pyplot as plt
import numpy as np
from scipy import interp
from sklearn import metrics
from sklearn import tree
from sklearn.cross_validation import KFold
from sklearn.learning_curve import learning_curve
from sklearn.metrics import roc_curve, auc

import ZEnv
import ZLog

__author__ = 'BBFamily'

g_enable_kf_cache = False
g_kf_cache = None


def kf_folds_proxy(y, n_folds=10):
    global g_kf_cache, g_enable_kf_cache
    if g_enable_kf_cache and g_kf_cache is not None and g_kf_cache.n == len(y):
        return g_kf_cache
    kf = KFold(len(y), n_folds=n_folds, shuffle=True)
    if g_enable_kf_cache:
        g_kf_cache = kf
    return kf


def run_prob_cv_estimator(estimator, x, y, n_folds=10):
    kf = kf_folds_proxy(y, n_folds)
    y_prob = np.zeros((len(y), 2))

    for train_index, test_index in kf:
        x_train, x_test = x[train_index], x[test_index]
        y_train = y[train_index]

        estimator.fit(x_train, y_train)
        y_prob[test_index] = estimator.predict_proba(x_test)
    return y_prob[:, 1]


def run_prob_cv_class(x, y, clf_class, n_folds=10, **kwargs):
    kf = kf_folds_proxy(y, n_folds)
    y_prob = np.zeros((len(y), 2))
    for train_index, test_index in kf:
        x_train, x_test = x[train_index], x[test_index]
        y_train = y[train_index]
        clf = clf_class(**kwargs)
        clf.fit(x_train, y_train)
        y_prob[test_index] = clf.predict_proba(x_test)
    return y_prob[:, 1]


def run_cv_estimator(estimator, x, y, n_folds=10):
    kf = kf_folds_proxy(y, n_folds)
    y_pred = y.copy()

    for train_index, test_index in kf:
        x_train, x_test = x[train_index], x[test_index]
        y_train = y[train_index]
        estimator.fit(x_train, y_train)
        y_pred[test_index] = estimator.predict(x_test)
    return y_pred


def run_cv_class(x, y, clf_class, n_folds=10, **kwargs):
    kf = kf_folds_proxy(y, n_folds)
    y_pred = y.copy()

    for train_index, test_index in kf:
        x_train, x_test = x[train_index], x[test_index]
        y_train = y[train_index]

        clf = clf_class(**kwargs)
        clf.fit(x_train, y_train)
        y_pred[test_index] = clf.predict(x_test)
    return y_pred


def plot_learning_curve_error(estimator, x, y, cv=5,
                              train_sizes=np.linspace(.05, 1., 20)):
    def rms_error(model, x, y):
        y_pred = model.predict(x)
        return np.sqrt(np.mean((y - y_pred) ** 2))

    def plot_with_err(x, data, **kwargs):
        mu, std = data.mean(1), data.std(1)
        lines = plt.plot(x, mu, '-', **kwargs)
        plt.fill_between(x, mu - std, mu + std, edgecolor='none',
                         facecolor=lines[0].get_color(), alpha=0.2)

    n_train, val_train, val_test = learning_curve(estimator,
                                                  x, y, train_sizes, cv=cv,
                                                  scoring=rms_error)
    plot_with_err(n_train, val_train, label='training scores')
    plot_with_err(n_train, val_test, label='validation scores')
    plt.xlabel('Training Set Size')
    plt.ylabel('rms error')
    plt.legend()


def plot_learning_curve(estimator, x, y, ylim=None, cv=5, n_jobs=1,
                        train_sizes=np.linspace(.05, 1., 20), verbose=0, show=True):
    train_sizes, train_scores, test_scores = learning_curve(
        estimator, x, y, cv=cv, n_jobs=n_jobs, train_sizes=train_sizes, verbose=verbose)

    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)

    if show:
        plt.figure()
        plt.title('learning curve')
        if ylim is not None:
            plt.ylim(*ylim)
        plt.xlabel("train sizes")
        plt.ylabel("scores")
        plt.gca().invert_yaxis()
        plt.grid()

        plt.fill_between(train_sizes, train_scores_mean - train_scores_std, train_scores_mean + train_scores_std,
                         alpha=0.1, color="b")
        plt.fill_between(train_sizes, test_scores_mean - test_scores_std, test_scores_mean + test_scores_std,
                         alpha=0.1, color="r")
        plt.plot(train_sizes, train_scores_mean, 'o-', color="b", label="train scores")
        plt.plot(train_sizes, test_scores_mean, 'o-', color="r", label="test scores")

        plt.legend(loc="best")

        plt.draw()
        plt.gca().invert_yaxis()
        plt.show()

    midpoint = ((train_scores_mean[-1] + train_scores_std[-1]) + (test_scores_mean[-1] - test_scores_std[-1])) / 2
    diff = (train_scores_mean[-1] + train_scores_std[-1]) - (test_scores_mean[-1] - test_scores_std[-1])
    return midpoint, diff


def plot_decision_boundary(pred_func, x, y):
    x_min, x_max = x[:, 0].min() - .5, x[:, 0].max() + .5
    y_min, y_max = x[:, 1].min() - .5, x[:, 1].max() + .5
    # h = (x_max - x_min) / 200
    # import pdb
    # pdb.set_trace()
    h = 0.01
    while True:
        if len(np.arange(x_min, x_max, h)) > 400:
            h *= 5
        else:
            break

    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    z = pred_func(np.c_[xx.ravel(), yy.ravel()])
    z = z.reshape(xx.shape)
    plt.contourf(xx, yy, z, cmap=plt.cm.Spectral)
    plt.scatter(x[:, 0], x[:, 1], c=y, cmap=plt.cm.Spectral)
    plt.show()


def plot_decision_function(estimator, ax=None):
    if ax is None:
        ax = plt.gca()
    x = np.linspace(plt.xlim()[0], plt.xlim()[1], 30)
    y = np.linspace(plt.ylim()[0], plt.ylim()[1], 30)
    y_mes, x_mes = np.meshgrid(y, x)
    p = np.zeros_like(x_mes)
    for i, xi in enumerate(x):
        for j, yj in enumerate(y):
            w = np.array([xi, yj])
            if hasattr(estimator, "decision_function"):
                p[i, j] = estimator.decision_function(w.reshape(1, -1))
            else:
                p[i, j] = estimator.predict_proba(w.reshape(1, -1))[:, 1]
    # plot the margins
    ax.contour(x_mes, y_mes, p, colors='k',
               levels=[-1, 0, 1], alpha=0.5,
               linestyles=['--', '-', '--'])
    plt.show()


def plot_confusion_matrices(estimator, x, y, n_folds=10):
    y_pred = run_cv_estimator(estimator, x, y, n_folds=n_folds)

    class_names = np.unique(y).tolist()

    confusion_matrix = metrics.confusion_matrix(y, y_pred)

    ZLog.info(confusion_matrix)

    fig = plt.figure()
    ax = fig.add_subplot(111)
    cax = ax.matshow(confusion_matrix)
    plt.title('Confusion matrix for %s' % estimator.__class__.__name__)
    fig.colorbar(cax)
    ax.set_xticklabels([''] + class_names)
    ax.set_yticklabels([''] + class_names)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.show()


def plot_roc_estimator(estimator, x, y):
    kf = KFold(len(y), n_folds=10, shuffle=True)
    y_prob = np.zeros((len(y), 2))
    mean_tpr = 0.0
    mean_fpr = np.linspace(0, 1, 100)
    for i, (train_index, test_index) in enumerate(kf):
        x_train, x_test = x[train_index], x[test_index]
        y_train = y[train_index]

        estimator.fit(x_train, y_train)
        y_prob[test_index] = estimator.predict_proba(x_test)
        fpr, tpr, thresholds = roc_curve(y[test_index], y_prob[test_index, 1])
        mean_tpr += interp(mean_fpr, fpr, tpr)
        mean_tpr[0] = 0.0
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, lw=1, label='ROC fold %d (area = %0.2f)' % (i, roc_auc))
    mean_tpr /= len(kf)
    mean_tpr[-1] = 1.0
    mean_auc = auc(mean_fpr, mean_tpr)
    plt.plot(mean_fpr, mean_tpr, 'k--', label='Mean ROC (area = %0.2f)' % mean_auc, lw=2)

    plt.plot([0, 1], [0, 1], '--', color=(0.6, 0.6, 0.6), label='Random')
    plt.xlim([-0.05, 1.05])
    plt.ylim([-0.05, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic')
    plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
    plt.show()


def plot_roc_class(x, y, fit_class, **kwargs):
    kf = KFold(len(y), n_folds=10, shuffle=True)
    y_prob = np.zeros((len(y), 2))
    mean_tpr = 0.0
    mean_fpr = np.linspace(0, 1, 100)
    for i, (train_index, test_index) in enumerate(kf):
        x_train, x_test = x[train_index], x[test_index]
        y_train = y[train_index]
        clf = fit_class(**kwargs)
        clf.fit(x_train, y_train)
        # Predict probabilities, not classes
        y_prob[test_index] = clf.predict_proba(x_test)
        fpr, tpr, thresholds = roc_curve(y[test_index], y_prob[test_index, 1])
        mean_tpr += interp(mean_fpr, fpr, tpr)
        mean_tpr[0] = 0.0
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, lw=1, label='ROC fold %d (area = %0.2f)' % (i, roc_auc))
    mean_tpr /= len(kf)
    mean_tpr[-1] = 1.0
    mean_auc = auc(mean_fpr, mean_tpr)
    plt.plot(mean_fpr, mean_tpr, 'k--', label='Mean ROC (area = %0.2f)' % mean_auc, lw=2)

    plt.plot([0, 1], [0, 1], '--', color=(0.6, 0.6, 0.6), label='Random')
    plt.xlim([-0.05, 1.05])
    plt.ylim([-0.05, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic')
    plt.legend(loc="lower right")
    plt.show()


def graphviz_tree(estimator, features, x, y):
    if not hasattr(estimator, 'tree_'):
        ZLog.info('only tree can graphviz!')
        return

    estimator.fit(x, y)

    tree.export_graphviz(estimator.tree_, out_file='graphviz.dot', feature_names=features)
    os.system("dot -T png graphviz.dot -o graphviz.png")

    '''
        !open $path
        要是方便用notebook直接open其实显示效果好，plt，show的大小不好调整
    '''
    # path = ZEnv.shell_cmd_result('pwd') + '/graphviz.png'
    # !open $path
    image_file = cbook.get_sample_data(ZEnv.shell_cmd_result('pwd') + '/graphviz.png')
    image = plt.imread(image_file)
    plt.imshow(image)
    plt.axis('off')  # clear x- and y-axes
    plt.show()


def visualize_tree(estimator, x, y, boundaries=True,
                   xlim=None, ylim=None):
    estimator.fit(x, y)

    if xlim is None:
        xlim = (x[:, 0].min() - 0.1, x[:, 0].max() + 0.1)
    if ylim is None:
        ylim = (x[:, 1].min() - 0.1, x[:, 1].max() + 0.1)

    x_min, x_max = xlim
    y_min, y_max = ylim
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 100),
                         np.linspace(y_min, y_max, 100))
    z = estimator.predict(np.c_[xx.ravel(), yy.ravel()])

    # Put the result into a color plot
    z = z.reshape(xx.shape)
    plt.figure()
    plt.pcolormesh(xx, yy, z, alpha=0.2, cmap='rainbow')
    plt.clim(y.min(), y.max())

    # Plot also the training points
    plt.scatter(x[:, 0], x[:, 1], c=y, s=50, cmap='rainbow')
    plt.axis('off')

    plt.xlim(x_min, x_max)
    plt.ylim(y_min, y_max)
    plt.clim(y.min(), y.max())

    # Plot the decision boundaries
    def plot_boundaries(i, xlim, ylim):
        if i < 0:
            return

        tree_inner = estimator.tree_

        if tree_inner.feature[i] == 0:
            plt.plot([tree_inner.threshold[i], tree_inner.threshold[i]], ylim, '-k')
            plot_boundaries(tree_inner.children_left[i],
                            [xlim[0], tree_inner.threshold[i]], ylim)
            plot_boundaries(tree_inner.children_right[i],
                            [tree_inner.threshold[i], xlim[1]], ylim)

        elif tree_inner.feature[i] == 1:
            plt.plot(xlim, [tree_inner.threshold[i], tree_inner.threshold[i]], '-k')
            plot_boundaries(tree_inner.children_left[i], xlim,
                            [ylim[0], tree_inner.threshold[i]])
            plot_boundaries(tree_inner.children_right[i], xlim,
                            [tree_inner.threshold[i], ylim[1]])

    if boundaries and hasattr(estimator, 'tree_'):
        """
            简单决策树才去画决策边界
        """
        plot_boundaries(0, plt.xlim(), plt.ylim())
