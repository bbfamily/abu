# -*- encoding:utf-8 -*-
"""封装常用的分析方式及流程模块"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import logging
import os

import matplotlib.cbook as cbook
import matplotlib.pyplot as plt
import numpy as np
from scipy import interp
from sklearn import metrics
from sklearn import tree
from sklearn.base import ClusterMixin, clone
from sklearn.metrics import roc_curve, auc

from ..CoreBu.ABuFixes import KFold, learning_curve
from ..UtilBu.ABuDTUtil import warnings_filter
# noinspection PyUnresolvedReferences
from ..CoreBu.ABuFixes import range
from ..UtilBu.ABuFileUtil import file_exist

__author__ = '阿布'
__weixin__ = 'abu_quant'

__all__ = [
           'run_silhouette_cv_estimator',
           'run_prob_cv_estimator',
           'run_cv_estimator',
           'plot_learning_curve',
           'plot_decision_boundary',
           'plot_confusion_matrices',
           'plot_roc_estimator',
           'graphviz_tree',
           'visualize_tree'
           ]


# noinspection PyUnresolvedReferences
def run_silhouette_cv_estimator(estimator, x, n_folds=10):
    """
    只针对kmean的cv验证，使用silhouette_score对聚类后的结果labels_
    进行度量使用silhouette_score，kmean的cv验证只是简单的通过np.random.choice
    进行随机筛选x数据进行聚类的silhouette_score度量，并不涉及训练集测试集
    :param estimator: keman或者支持estimator.labels_, 只通过if not isinstance(estimator, ClusterMixin)进行过滤
    :param x: x特征矩阵
    :param n_folds: int，透传KFold参数，切割训练集测试集参数，默认10
    :return: eg: array([ 0.693 ,  0.652 ,  0.6845,  0.6696,  0.6732,  0.6874,  0.668 ,
                         0.6743,  0.6748,  0.671 ])
    """

    if not isinstance(estimator, ClusterMixin):
        print('estimator must be ClusterMixin')
        return

    silhouette_list = list()
    # eg: n_folds = 10, len(x) = 150 -> 150 * 0.9 = 135
    choice_cnt = int(len(x) * ((n_folds - 1) / n_folds))
    choice_source = np.arange(0, x.shape[0])

    # 所有执行fit的操作使用clone一个新的
    estimator = clone(estimator)
    for _ in np.arange(0, n_folds):
        # 只是简单的通过np.random.choice进行随机筛选x数据
        choice_index = np.random.choice(choice_source, choice_cnt)
        x_choice = x[choice_index]
        estimator.fit(x_choice)
        # 进行聚类的silhouette_score度量
        silhouette_score = metrics.silhouette_score(x_choice, estimator.labels_, metric='euclidean')
        silhouette_list.append(silhouette_score)
    return silhouette_list


def run_prob_cv_estimator(estimator, x, y, n_folds=10):
    """
    通过KFold和参数n_folds拆分训练集和测试集，使用
    np.zeros((len(y), len(np.unique(y))))初始化prob矩阵，
    通过训练estimator.fit(x_train, y_train)后的分类器使用
    predict_proba将y_prob中的对应填数据

    :param estimator: 支持predict_proba的有监督学习, 只通过hasattr(estimator, 'predict_proba')进行过滤
    :param x: 训练集x矩阵，numpy矩阵
    :param y: 训练集y序列，numpy序列
    :param n_folds: int，透传KFold参数，切割训练集测试集参数，默认10
    :return: eg: y_prob
                array([[ 0.8726,  0.1274],
                       [ 0.0925,  0.9075],
                       [ 0.2485,  0.7515],
                       ...,
                       [ 0.3881,  0.6119],
                       [ 0.7472,  0.2528],
                       [ 0.8555,  0.1445]])

    """
    if not hasattr(estimator, 'predict_proba'):
        print('estimator must has predict_proba')
        return

    # 所有执行fit的操作使用clone一个新的
    estimator = clone(estimator)
    kf = KFold(len(y), n_folds=n_folds, shuffle=True)
    y_prob = np.zeros((len(y), len(np.unique(y))))
    """
        根据y序列的数量以及y的label数量构造全是0的矩阵
        eg: y_prob
        array([[ 0.,  0.,  0.],
               [ 0.,  0.,  0.],
               [ 0.,  0.,  0.],
                ..............
               [ 0.,  0.,  0.],
               [ 0.,  0.,  0.],
               [ 0.,  0.,  0.],
    """

    for train_index, test_index in kf:
        x_train, x_test = x[train_index], x[test_index]
        y_train = y[train_index]

        # clf = clone(estimator)
        estimator.fit(x_train, y_train)
        # 使用predict_proba将y_prob中的对应填数据
        y_prob[test_index] = estimator.predict_proba(x_test)

    return y_prob


def run_cv_estimator(estimator, x, y, n_folds=10):
    """
    通过KFold和参数n_folds拆分训练集和测试集，使用
    y.copy()初始化y_pred矩阵，迭代切割好的训练集与测试集，
    不断通过 estimator.predict(x_test)将y_pred中的值逐步替换

    :param estimator: 有监督学习器对象
    :param x: 训练集x矩阵，numpy矩阵
    :param y: 训练集y序列，numpy序列
    :param n_folds: int，透传KFold参数，切割训练集测试集参数，默认10
    :return: y_pred序列
    """
    if not hasattr(estimator, 'predict'):
        print('estimator must has predict')
        return

    # 所有执行fit的操作使用clone一个新的
    estimator = clone(estimator)
    kf = KFold(len(y), n_folds=n_folds, shuffle=True)
    # 首先copy一个一摸一样的y
    y_pred = y.copy()
    """
        eg: y_pred
        array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
               0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
               0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
               1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
               1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2,
               2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2,
               2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2])
    """

    for train_index, test_index in kf:
        x_train, x_test = x[train_index], x[test_index]
        y_train = y[train_index]
        estimator.fit(x_train, y_train)
        # 通过 estimator.predict(x_test)将y_pred中的值逐步替换
        y_pred[test_index] = estimator.predict(x_test)
    return y_pred


# warnings_filter针对多标签使用OneVsRestClassifier出现的版本警告
@warnings_filter
def plot_learning_curve(estimator, x, y, cv=5, n_jobs=1,
                        train_sizes=np.linspace(.05, 1., 20)):
    """
    绘制学习曲线，train_sizes使用np.linspace(.05, 1., 20)即训练集从5%－100%递进

            np.linspace(.05, 1., 20)
            array([ 0.05,  0.1 ,  0.15,  0.2 ,  0.25,  0.3 ,  0.35,  0.4 ,  0.45,
                    0.5 ,  0.55,  0.6 ,  0.65,  0.7 ,  0.75,  0.8 ,  0.85,  0.9 ,
                    0.95,  1.  ])

    套接sklern中learning_curve函数，传递estimator，cv等参数

    :param estimator: 学习器对象，透传learning_curve
    :param x: 训练集x矩阵，numpy矩阵
    :param y: 训练集y序列，numpy序列
    :param cv: 透传learning_curve，cv参数，默认5，int
    :param n_jobs: 透传learning_curve，并行进程数，默认1，即使用单进程执行
    :param train_sizes: train_sizes使用np.linspace(.05, 1., 20)即训练集从5%－100%递进
    """

    # 套接learning_curve，返回训练集和测试集的score和对应的size
    train_sizes, train_scores, test_scores = learning_curve(
        estimator, x, y, cv=cv, n_jobs=n_jobs, train_sizes=train_sizes)
    """
        eg:  train_scores shape = (20, 5)
        array([[ 0.8571,  0.9143,  0.9143,  0.9143,  0.9143],
               [ 0.8169,  0.8732,  0.8732,  0.8732,  0.8732],
               [ 0.8208,  0.8396,  0.8396,  0.8396,  0.8396],
               [ 0.8028,  0.8099,  0.8099,  0.8099,  0.8099],
               [ 0.8146,  0.8202,  0.8146,  0.8146,  0.8146],
               [ 0.8263,  0.8263,  0.8216,  0.8216,  0.8216],
               [ 0.8153,  0.8273,  0.8112,  0.8112,  0.8112],
               [ 0.8063,  0.8169,  0.7993,  0.7993,  0.7993],
               [ 0.8156,  0.8281,  0.8063,  0.8063,  0.8063],
               [ 0.8169,  0.8254,  0.8254,  0.8254,  0.8254],
               [ 0.8184,  0.8235,  0.8261,  0.8312,  0.8312],
               [ 0.815 ,  0.822 ,  0.8197,  0.822 ,  0.822 ],
               [ 0.816 ,  0.8203,  0.8203,  0.8182,  0.8182],
               [ 0.8133,  0.8173,  0.8173,  0.8253,  0.8253],
               [ 0.8109,  0.8127,  0.8146,  0.8202,  0.8221],
               [ 0.8155,  0.819 ,  0.8172,  0.8207,  0.8225],
               [ 0.8149,  0.8248,  0.8231,  0.8248,  0.8198],
               [ 0.8187,  0.8281,  0.825 ,  0.8328,  0.8219],
               [ 0.8254,  0.8299,  0.8284,  0.8343,  0.8166],
               [ 0.8272,  0.8315,  0.8301,  0.8343,  0.8174]])
    """
    train_scores_mean = np.mean(train_scores, axis=1)
    """
        eg: train_scores_mean
            array([ 0.9029,  0.862 ,  0.8358,  0.8085,  0.8157,  0.8235,  0.8153,
            0.8042,  0.8125,  0.8237,  0.8261,  0.8201,  0.8186,  0.8197,
            0.8161,  0.819 ,  0.8215,  0.8253,  0.8269,  0.8281])
    """
    train_scores_std = np.std(train_scores, axis=1)

    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)
    """
         eg: test_scores_std
            array([ 0.0751,  0.0607,  0.0314,  0.0059,  0.0047,  0.0066,  0.0074,
                    0.0051,  0.0107,  0.0115,  0.0107,  0.012 ,  0.0142,  0.018 ,
                    0.0134,  0.0167,  0.0167,  0.0127,  0.0128,  0.0113])
    """
    # 开始可视化学习曲线
    plt.figure()
    plt.title('learning curve')
    plt.xlabel("train sizes")
    plt.ylabel("scores")
    plt.gca().invert_yaxis()
    plt.grid()
    # 对train_scores的均值和方差区域进行填充
    plt.fill_between(train_sizes, train_scores_mean - train_scores_std, train_scores_mean + train_scores_std,
                     alpha=0.1, color="g")
    # 对test_scores的均值和方差区域进行填充
    plt.fill_between(train_sizes, test_scores_mean - test_scores_std, test_scores_mean + test_scores_std,
                     alpha=0.1, color="r")
    # 把train_scores_mean标注圆圈
    plt.plot(train_sizes, train_scores_mean, 'o-', color="g", label="train scores")
    # 把ttest_scores_mean标注圆圈
    plt.plot(train_sizes, test_scores_mean, 'o-', color="r", label="test scores")
    plt.legend(loc="best")
    plt.draw()
    plt.gca().invert_yaxis()
    plt.show()


def graphviz_tree(estimator, features, x, y):
    """
    绘制决策树或者core基于树的分类回归算法的决策示意图绘制，查看
    学习器本身hasattr(fiter, 'tree_')是否有tree_属性，内部clone(estimator)学习器
    后再进行训练操作，完成训练后使用sklearn中tree.export_graphvizd导出graphviz.dot文件
    需要使用第三方dot工具将graphviz.dot进行转换graphviz.png，即内部实行使用
    运行命令行：
                os.system("dot -T png graphviz.dot -o graphviz.png")
    最后读取决策示意图显示

    :param estimator: 学习器对象，透传learning_curve
    :param x: 训练集x矩阵，numpy矩阵
    :param y: 训练集y序列，numpy序列
    :param features: 训练集x矩阵列特征所队员的名称，可迭代序列对象
    """
    if not hasattr(estimator, 'tree_'):
        logging.info('only tree can graphviz!')
        return

    # 所有执行fit的操作使用clone一个新的
    estimator = clone(estimator)
    estimator.fit(x, y)
    # TODO out_file path放倒cache中
    tree.export_graphviz(estimator.tree_, out_file='graphviz.dot', feature_names=features)
    os.system("dot -T png graphviz.dot -o graphviz.png")

    '''
        !open $path
        要是方便用notebook直接open其实显示效果好，plt，show的大小不好调整
    '''
    graphviz = os.path.join(os.path.abspath('.'), 'graphviz.png')

    # path = graphviz
    # !open $path
    if not file_exist(graphviz):
        logging.info('{} not exist! please install dot util!'.format(graphviz))
        return

    image_file = cbook.get_sample_data(graphviz)
    image = plt.imread(image_file)
    image_file.close()
    plt.imshow(image)
    plt.axis('off')  # clear x- and y-axes
    plt.show()


def visualize_tree(estimator, x, y, boundaries=True):
    """
    需要x矩阵特征列只有两个维度，根据x，y，通过meshgrid构造训练集平面特征
    通过z = estimator.predict(np.c_[xx.ravel(), yy.ravel()])对特征平面
    进行predict生成z轴，可视化meshgrid构造训练集平面特征使用生成的z生成
    pcolormesh进行可视化

    :param estimator: 学习器对象，内部clone(estimator)
    :param x: 训练集x矩阵，numpy矩阵，需要特征列只有两个维度
    :param y: 训练集y序列，numpy序列
    :param boundaries: 是否绘制决策边界

    """
    if x.shape[1] != 2:
        logging.info('be sure x shape[1] == 2!')
        return

    # 所有执行fit的操作使用clone一个新的
    estimator = clone(estimator)
    estimator.fit(x, y)

    xlim = (x[:, 0].min() - 0.1, x[:, 0].max() + 0.1)
    ylim = (x[:, 1].min() - 0.1, x[:, 1].max() + 0.1)
    x_min, x_max = xlim
    y_min, y_max = ylim
    # 通过训练集中x的min和max，y的min，max构成meshgrid
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 100),
                         np.linspace(y_min, y_max, 100))
    # 摊平xx，yy进行z轴的predict
    z = estimator.predict(np.c_[xx.ravel(), yy.ravel()])
    # z的shape跟随xx
    z = z.reshape(xx.shape)
    plt.figure()
    plt.pcolormesh(xx, yy, z, alpha=0.2, cmap='rainbow')
    plt.clim(y.min(), y.max())

    # 将之前的训练集中的两个特征进行scatter绘制，颜色使用y做区分
    plt.scatter(x[:, 0], x[:, 1], c=y, s=50, cmap='rainbow')
    plt.axis('off')

    plt.xlim(x_min, x_max)
    plt.ylim(y_min, y_max)
    plt.clim(y.min(), y.max())

    def plot_boundaries(i, p_xlim, p_ylim):
        """
        针对有tree_属性的学习器绘制决策边界
        :param i: 内部递归调用使用ree_inner.children_left[i]和tree_inner.children_right[i]
        :param p_xlim: 原始参数使用plt.xlim()
        :param p_ylim: 原始参数使用plt.ylim()
        """
        if i < 0:
            return
        # 拿到tree_使用plot_boundaries继续递归绘制
        tree_inner = estimator.tree_

        if tree_inner.feature[i] == 0:
            # 绘制0的边界
            plt.plot([tree_inner.threshold[i], tree_inner.threshold[i]], p_ylim, '-k')
            # 即x轴固定p_ylim，xlim=[p_xlim[0], tree_inner.threshold[i]], [tree_inner.threshold[i], p_xlim[1]]
            plot_boundaries(tree_inner.children_left[i],
                            [p_xlim[0], tree_inner.threshold[i]], p_ylim)
            plot_boundaries(tree_inner.children_right[i],
                            [tree_inner.threshold[i], p_xlim[1]], p_ylim)
        elif tree_inner.feature[i] == 1:
            # 绘制1的边界
            plt.plot(p_xlim, [tree_inner.threshold[i], tree_inner.threshold[i]], '-k')
            # 即y轴固定p_xlim，ylim=[p_ylim[0], tree_inner.threshold[i]], [tree_inner.threshold[i], p_ylim[1]]
            plot_boundaries(tree_inner.children_left[i], p_xlim,
                            [p_ylim[0], tree_inner.threshold[i]])
            plot_boundaries(tree_inner.children_right[i], p_xlim,
                            [tree_inner.threshold[i], p_ylim[1]])

    if boundaries and hasattr(estimator, 'tree_'):
        # 简单决策树才去画决策边界
        plot_boundaries(0, plt.xlim(), plt.ylim())


def plot_decision_boundary(pred_func, x, y):
    """
    通过x，y以构建meshgrid平面区域，要x矩阵特征列只有两个维度，在区域中使用外部传递的
    pred_func函数进行z轴的predict，通过contourf绘制特征平面区域，最后使用
    plt.scatter(x[:, 0], x[:, 1], c=y, cmap=plt.cm.Spectral)在平面区域上填充原始特征
    点

    :param pred_func: callable函数，eg：pred_func: lambda p_x: fiter.predict(p_x), x, y
    :param x: 训练集x矩阵，numpy矩阵，需要特征列只有两个维度
    :param y: 训练集y序列，numpy序列
    """
    xlim = (x[:, 0].min() - 0.1, x[:, 0].max() + 0.1)
    ylim = (x[:, 1].min() - 0.1, x[:, 1].max() + 0.1)
    x_min, x_max = xlim
    y_min, y_max = ylim
    # 通过训练集中x的min和max，y的min，max构成meshgrid
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 200),
                         np.linspace(y_min, y_max, 200))
    # 摊平xx，yy进行z轴的predict, pred_func: lambda p_x: fiter.predict(p_x), x, y
    z = pred_func(np.c_[xx.ravel(), yy.ravel()])
    # z的shape跟随xx
    z = z.reshape(xx.shape)
    # 使用contourf绘制xx, yy, z，即特征平面区域以及z的颜色区别
    # noinspection PyUnresolvedReferences
    plt.contourf(xx, yy, z, cmap=plt.cm.Spectral)
    # noinspection PyUnresolvedReferences
    # 在特征区域的基础上将原始，两个维度使用scatter绘制以y为颜色的点
    plt.scatter(x[:, 0], x[:, 1], c=y, cmap=plt.cm.Spectral)
    plt.show()


def plot_roc_estimator(estimator, x, y, pos_label=None):
    """
    固定n_folds=10通过kf = KFold(len(y), n_folds=10, shuffle=True)拆分
    训练测试集，使用estimator.predict_proba对测试集数据进行概率统计，直接使用
    sklearn中的roc_curve分别对多组测试集计算fpr, tpr, thresholds，并计算roc_auc
    最后绘制roc_auc曲线进行可视化操作

    :param estimator: 分类器对象，内部clone(estimator)
    :param x: 训练集x矩阵，numpy矩阵
    :param y: 训练集y序列，numpy序列
    :param pos_label: 对y大于2个label的数据，roc_curve需要指定pos_label，如果不指定，默认使用y的第一个label值
    """

    if not hasattr(estimator, 'predict_proba'):
        # 分类器必须要有predict_proba方法
        logging.info('estimator must has predict_proba!')
        return

    # 所有执行fit的操作使用clone一个新的
    estimator = clone(estimator)
    estimator.fit(x, y)

    # eg: y_unique = [0, 1]
    y_unique = np.unique(y)
    kf = KFold(len(y), n_folds=10, shuffle=True)
    y_prob = np.zeros((len(y), len(y_unique)))
    """
        eg:  y_prob
            array([[ 0.,  0.],
                   [ 0.,  0.],
                   [ 0.,  0.],
                   ...,
                   [ 0.,  0.],
                   [ 0.,  0.],
                   [ 0.,  0.]])
    """
    mean_tpr = 0.0
    # 0-1分布100个
    mean_fpr = np.linspace(0, 1, 100)
    for i, (train_index, test_index) in enumerate(kf):
        x_train, x_test = x[train_index], x[test_index]
        y_train = y[train_index]
        estimator.fit(x_train, y_train)
        y_prob[test_index] = estimator.predict_proba(x_test)
        """
            eg: y_prob[test_index]
            array([[ 0.8358,  0.1642],
                   [ 0.4442,  0.5558],
                   [ 0.1351,  0.8649],
                   [ 0.8567,  0.1433],
                   [ 0.6953,  0.3047],
                   ..................
                   [ 0.1877,  0.8123],
                   [ 0.8465,  0.1535],
                   [ 0.1916,  0.8084],
                   [ 0.8421,  0.1579]])

        """
        if len(y_unique) != 2 and pos_label is None:
            # 对y大于2个label的数据，roc_curve需要指定pos_label，如果不指定，默认使用y的第一个label值
            pos_label = y_unique[0]
            logging.info('y label count > 2 and param pos_label is None, so choice y_unique[0]={} for pos_label!'.
                         format(pos_label))

        fpr, tpr, thresholds = roc_curve(y[test_index], y_prob[test_index, 1], pos_label=pos_label)
        """
            eg:
                fpr
                array([ 0.    ,  0.0169,  0.0169,  0.0339,  0.0339,  0.0508,  0.0508,
                        0.0847,  0.0847,  0.1017,  0.1017,  0.1186,  0.1186,  0.2034,
                        0.2034,  0.2542,  0.2542,  0.5254,  0.5254,  0.5763,  0.5763,  1.    ])
                tpr
                array([ 0.0323,  0.0323,  0.4839,  0.4839,  0.5484,  0.5484,  0.6452,
                        0.6452,  0.7419,  0.7419,  0.7742,  0.7742,  0.8387,  0.8387,
                        0.9032,  0.9032,  0.9355,  0.9355,  0.9677,  0.9677,  1.    ,  1.    ])

                thresholds
                array([ 0.9442,  0.9288,  0.8266,  0.8257,  0.8123,  0.8122,  0.8032,
                        0.7647,  0.7039,  0.5696,  0.5558,  0.4854,  0.4538,  0.2632,
                        0.2153,  0.2012,  0.1902,  0.1616,  0.1605,  0.1579,  0.1561,
                        0.1301])
        """

        # interp线性插值计算
        mean_tpr += interp(mean_fpr, fpr, tpr)
        # 把第一个值固定0，最后会使用mean_tpr[-1] = 1.0把最后一个固定1.0
        mean_tpr[0] = 0.0
        # 直接使用 sklearn中的metrics.auc计算
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, lw=1, label='ROC fold %d (area = %0.2f)' % (i, roc_auc))

    mean_tpr /= len(kf)
    # 最后一个固定1.0
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


def plot_confusion_matrices(estimator, x, y, n_folds=10):
    """
    套接run_cv_estimator进行通过参数n_folds进行训练集测试集拆封
    使用y_pred和y做为参数，透传给metrics.confusion_matrix函数
    进行混淆矩阵的计算，通过ax.matshow可视化混淆矩阵

    :param estimator: 分类器对象，内部clone(estimator)
    :param x: 训练集x矩阵，numpy矩阵
    :param y: 训练集y序列，numpy序列
    :param n_folds: 透传KFold参数，切割训练集测试集参数
    """
    y_pred = run_cv_estimator(estimator, x, y, n_folds=n_folds)
    """
        eg: y_pred
        array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
           0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
           0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
           1, 2, 1, 1, 1, 1, 1, 1, 2, 1, 1, 1, 1, 1, 2, 1, 1, 1, 1, 1, 1, 1, 1,
           1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 1, 2, 2, 2, 2, 2, 2, 2, 2,
           2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2,
           2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2])
    """
    y_unique = np.unique(y)

    confusion_matrix = metrics.confusion_matrix(y, y_pred)
    """
        eg: confusion_matrix
         [[50  0  0]
         [ 0 47  3]
         [ 0  1 49]]
    """
    logging.info(confusion_matrix)

    fig = plt.figure()
    # 颜色条的颜色数量设置使用len(y_unique) * len(y_unique)，即如果y是3个label->9颜色。2->4
    cmap = plt.get_cmap('jet', len(y_unique) * len(y_unique))
    cmap.set_under('gray')
    ax = fig.add_subplot(111)
    # ax.matshow可视化化混淆矩阵
    cax = ax.matshow(confusion_matrix, cmap=cmap,
                     vmin=confusion_matrix.min(),
                     vmax=confusion_matrix.max())
    plt.title('Confusion matrix for %s' % estimator.__class__.__name__)
    # 辅助颜色边bar显示
    fig.colorbar(cax)
    # noinspection PyTypeChecker
    ax.set_xticklabels('x: '.format(y_unique))
    # noinspection PyTypeChecker
    ax.set_yticklabels('y: '.format(y_unique))
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.show()
