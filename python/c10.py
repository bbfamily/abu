# -*- encoding:utf-8 -*-
from __future__ import print_function
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
# import warnings

# noinspection PyUnresolvedReferences
import abu_local_env
import abupy
from abupy import abu
from abupy import ABuSymbolPd

import sklearn.preprocessing as preprocessing

# warnings.filterwarnings('ignore')
sns.set_context(rc={'figure.figsize': (14, 7)})
# 使用沙盒数据，目的是和书中一样的数据环境
abupy.env.enable_example_env_ipython()

"""
    第10章 量化系统——机器学习•猪老三

    abu量化系统github地址：https://github.com/bbfamily/abu (您的star是我的动力！)
    abu量化文档教程ipython notebook：https://github.com/bbfamily/abu/tree/master/abupy_lecture
"""

"""
    10.2 猪老三世界中的量化环境
"""

"""
    是否开启date_week噪音, 开启这个的目的是让分类结果正确率降低，接近真实
"""
g_with_date_week_noise = False


def _gen_another_word_price(kl_another_word):
    """
    生成股票在另一个世界中的价格
    :param kl_another_word:
    :return:
    """
    for ind in np.arange(2, kl_another_word.shape[0]):
        # 前天数据
        bf_yesterday = kl_another_word.iloc[ind - 2]
        # 昨天
        yesterday = kl_another_word.iloc[ind - 1]
        # 今天
        today = kl_another_word.iloc[ind]
        # 生成今天的收盘价格
        kl_another_word.close[ind] = _gen_another_word_price_rule(
            yesterday.close, yesterday.volume,
            bf_yesterday.close, bf_yesterday.volume,
            today.volume, today.date_week)


def _gen_another_word_price_rule(yesterday_close, yesterday_volume,
                                 bf_yesterday_close,
                                 bf_yesterday_volume,
                                 today_volume, date_week):
    """
        通过前天收盘量价，昨天收盘量价，今天的量，构建另一个世界中的价格模型
    """
    # 昨天收盘价格与前天收盘价格的价格差
    price_change = yesterday_close - bf_yesterday_close
    # 昨天成交量与前天成交量的量差
    volume_change = yesterday_volume - bf_yesterday_volume

    # 如果量和价变动一致，今天价格涨，否则跌
    # 即量价齐涨－>涨, 量价齐跌－>涨，量价不一致－>跌
    sign = 1.0 if price_change * volume_change > 0 else -1.0

    # 通过date_week生成噪音，否则之后分类100%分对
    if g_with_date_week_noise:
        # 针对sign生成噪音，噪音的生效的先决条件是今天的量是这三天最大的
        gen_noise = today_volume > np.max(
            [yesterday_volume, bf_yesterday_volume])
        # 如果量是这三天最大 且是周五，下跌
        if gen_noise and date_week == 4:
            sign = -1.0
        # 如果量是这三天最大，如果是周一，上涨
        elif gen_noise and date_week == 0:
            sign = 1.0

    # 今天的涨跌幅度基础是price_change（昨天前天的价格变动）
    price_base = abs(price_change)
    # 今天的涨跌幅度变动因素：量比，
    # 今天的成交量/昨天的成交量 和 今天的成交量/前天的成交量 的均值
    price_factor = np.mean([today_volume / yesterday_volume,
                            today_volume / bf_yesterday_volume])

    if abs(price_base * price_factor) < yesterday_close * 0.10:
        # 如果 量比 * price_base 没超过10%，今天价格计算
        today_price = yesterday_close + \
                      sign * price_base * price_factor
    else:
        # 如果涨跌幅度超过10%，限制上限，下限为10%
        today_price = yesterday_close + sign * yesterday_close * 0.10
    return today_price


def change_real_to_another_word(symbol):
    """
    将原始真正的股票数据价格列只保留前两天数据，成交量，周几列完全保留
    价格列其他数据使用_gen_another_word_price变成另一个世界价格
    :param symbol:
    :return:
    """
    kl_pd = ABuSymbolPd.make_kl_df(symbol)
    if kl_pd is not None:
        # 原始股票数据也只保留价格，周几，成交量
        kl_pig_three = kl_pd.filter(['close', 'date_week', 'volume'])
        # 只保留原始头两天的交易收盘价格，其他的的都赋予nan
        kl_pig_three['close'][2:] = np.nan
        # 将其他nan价格变成猪老三世界中价格使用_gen_another_word_price
        _gen_another_word_price(kl_pig_three)
        return kl_pig_three


def sample_102(show=True):
    """
    10.2 生成猪老三的世界中的映射股票数据
    :return:
    """
    choice_symbols = ['usNOAH', 'usSFUN', 'usBIDU', 'usAAPL', 'usGOOG',
                      'usTSLA', 'usWUBA', 'usVIPS']
    another_word_dict = {}
    real_dict = {}
    for symbol in choice_symbols:
        # 猪老三世界的股票走势字典
        another_word_dict[symbol] = change_real_to_another_word(symbol)
        # 真实世界的股票走势字典，这里不考虑运行效率问题
        real_dict[symbol] = ABuSymbolPd.make_kl_df(symbol)
    if show:
        # 表10-1所示
        print('another_word_dict[usNOAH].head():\n', another_word_dict['usNOAH'].head())

        print('real_dict[usNOAH].head():\n', real_dict['usNOAH'].head().filter(['close', 'date_week', 'volume']))

        import itertools
        # 4 ＊ 2
        _, axs = plt.subplots(nrows=4, ncols=2, figsize=(20, 15))
        # 将画布序列拉平
        axs_list = list(itertools.chain.from_iterable(axs))

        for symbol, ax in zip(choice_symbols, axs_list):
            # 绘制猪老三世界的股价走势
            another_word_dict[symbol].close.plot(ax=ax)
            # 同样的股票在真实世界的股价走势
            real_dict[symbol].close.plot(ax=ax)
            ax.set_title(symbol)
        plt.show()
    return another_word_dict


"""
    10.3 有监督机器学习
"""


def gen_pig_three_feature(kl_another_word):
    """
    猪老三构建特征模型函数
    :param kl_another_word: 即上一节使用_gen_another_word_price
    生成的dataframe有收盘价，周几，成交量列
    :return:
    """
    # y值使用close.pct_change即涨跌幅度
    kl_another_word['regress_y'] = kl_another_word.close.pct_change()
    # 前天收盘价格
    kl_another_word['bf_yesterday_close'] = 0
    # 昨天收盘价格
    kl_another_word['yesterday_close'] = 0
    # 昨天收盘成交量
    kl_another_word['yesterday_volume'] = 0
    # 前天收盘成交量
    kl_another_word['bf_yesterday_volume'] = 0

    # 对齐特征，前天收盘价格即与今天的收盘错2个时间单位，[2:] = [:-2]
    kl_another_word['bf_yesterday_close'][2:] = \
        kl_another_word['close'][:-2]
    # 对齐特征，前天成交量
    kl_another_word['bf_yesterday_volume'][2:] = \
        kl_another_word['volume'][:-2]
    # 对齐特征，昨天收盘价与今天的收盘错1个时间单位，[1:] = [:-1]
    kl_another_word['yesterday_close'][1:] = \
        kl_another_word['close'][:-1]
    # 对齐特征，昨天成交量
    kl_another_word['yesterday_volume'][1:] = \
        kl_another_word['volume'][:-1]

    # 特征1: 价格差
    kl_another_word['feature_price_change'] = \
        kl_another_word['yesterday_close'] - \
        kl_another_word['bf_yesterday_close']

    # 特征2: 成交量差
    kl_another_word['feature_volume_Change'] = \
        kl_another_word['yesterday_volume'] - \
        kl_another_word['bf_yesterday_volume']

    # 特征3: 涨跌sign
    kl_another_word['feature_sign'] = np.sign(
        kl_another_word['feature_price_change'] * kl_another_word[
            'feature_volume_Change'])

    # 特征4: 周几
    kl_another_word['feature_date_week'] = kl_another_word[
        'date_week']

    """
        构建噪音特征, 因为猪老三也不可能全部分析正确真实的特征因素
        这里引入一些噪音特征
    """
    # 成交量乘积
    kl_another_word['feature_volume_noise'] = \
        kl_another_word['yesterday_volume'] * \
        kl_another_word['bf_yesterday_volume']

    # 价格乘积
    kl_another_word['feature_price_noise'] = \
        kl_another_word['yesterday_close'] * \
        kl_another_word['bf_yesterday_close']

    # 将数据标准化
    scaler = preprocessing.StandardScaler()
    kl_another_word['feature_price_change'] = scaler.fit_transform(
        kl_another_word['feature_price_change'].values.reshape(-1, 1))
    kl_another_word['feature_volume_Change'] = scaler.fit_transform(
        kl_another_word['feature_volume_Change'].values.reshape(-1, 1))
    kl_another_word['feature_volume_noise'] = scaler.fit_transform(
        kl_another_word['feature_volume_noise'].values.reshape(-1, 1))
    kl_another_word['feature_price_noise'] = scaler.fit_transform(
        kl_another_word['feature_price_noise'].values.reshape(-1, 1))

    # 只筛选feature_开头的特征和regress_y，抛弃前两天数据，即[2:]
    kl_pig_three_feature = kl_another_word.filter(
        regex='regress_y|feature_*')[2:]
    return kl_pig_three_feature


def sample_103_0(show=True):
    """
    10.3 生成猪老三的训练集特征示例
    :return:
    """
    another_word_dict = sample_102(show=False)
    pig_three_feature = None
    for symbol in another_word_dict:
        # 首先拿出对应的走势数据
        kl_another_word = another_word_dict[symbol]
        # 通过走势数据生成训练集特征通过gen_pig_three_feature
        kl_feature = gen_pig_three_feature(kl_another_word)
        # 将每个股票的特征数据都拼接起来，形成训练集
        pig_three_feature = kl_feature if pig_three_feature is None \
            else pig_three_feature.append(kl_feature)

    # Dataframe -> matrix
    feature_np = pig_three_feature.as_matrix()
    # x特征矩阵
    train_x = feature_np[:, 1:]
    # 回归训练的连续值y
    train_y_regress = feature_np[:, 0]
    # 分类训练的离散值y，之后分类技术使用
    # noinspection PyTypeChecker
    train_y_classification = np.where(train_y_regress > 0, 1, 0)

    if show:
        print('pig_three_feature.shape:', pig_three_feature.shape)
        print('pig_three_feature.tail():\n', pig_three_feature.tail())
        print('train_x[:5], train_y_regress[:5], train_y_classification[:5]:\n', train_x[:5], train_y_regress[:5],
              train_y_classification[:5])

    return train_x, train_y_regress, train_y_classification, pig_three_feature


"""
    猪老三使用回归预测股价
"""


def sample_1031_1():
    """
    10.3.1_1 猪老三使用回归预测股价：生成训练集数据和测试集数据
    :return:
    """

    # noinspection PyShadowingNames
    def gen_feature_from_symbol(symbol):
        """
        封装由一个symbol转换为特征矩阵序列函数
        :param symbol:
        :return:
        """
        # 真实世界走势数据转换到老三的世界
        kl_another_word = change_real_to_another_word(symbol)
        # 由走势转换为特征dataframe通过gen_pig_three_feature
        kl_another_word_feature_test = gen_pig_three_feature(kl_another_word)
        # 转换为matrix
        feature_np_test = kl_another_word_feature_test.as_matrix()
        # 从matrix抽取y回归
        test_y_regress = feature_np_test[:, 0]
        # y回归 －> y分类
        # noinspection PyTypeChecker
        test_y_classification = np.where(test_y_regress > 0, 1, 0)
        # 从matrix抽取x特征矩阵
        test_x = feature_np_test[:, 1:]
        return test_x, test_y_regress, test_y_classification, kl_another_word_feature_test

    # 生成训练集数据
    train_x, train_y_regress, train_y_classification, pig_three_feature = sample_103_0(show=False)
    # 生成测试集数据
    test_x, test_y_regress, test_y_classification, kl_another_word_feature_test = gen_feature_from_symbol('usFB')

    print('训练集：{}, 测试集：{}'.format(pig_three_feature.shape[0], kl_another_word_feature_test.shape[0]))

    return train_x, train_y_regress, train_y_classification, pig_three_feature, \
           test_x, test_y_regress, test_y_classification, kl_another_word_feature_test


def regress_process(estimator, train_x, train_y_regress, test_x,
                    test_y_regress):
    # 训练训练集数据
    estimator.fit(train_x, train_y_regress)
    # 使用训练好的模型预测测试集对应的y，即根据usFB的走势特征预测股价涨跌幅度
    test_y_prdict_regress = estimator.predict(test_x)

    # 绘制usFB实际股价涨跌幅度
    plt.plot(test_y_regress.cumsum())
    # 绘制通过模型预测的usFB股价涨跌幅度
    plt.plot(test_y_prdict_regress.cumsum())

    # 针对训练集数据做交叉验证
    from abupy import cross_val_score
    from abupy.CoreBu.ABuFixes import mean_squared_error_scorer
    scores = cross_val_score(estimator, train_x,
                             train_y_regress, cv=10,
                             scoring=mean_squared_error_scorer)
    # mse开方 -> rmse
    mean_sc = -np.mean(np.sqrt(-scores))
    print('{} RMSE: {}'.format(estimator.__class__.__name__, mean_sc))


def sample_1031_2():
    """
    10.3.1_2 猪老三使用回归预测股价：LinearRegressio
    :return:
    """
    train_x, train_y_regress, train_y_classification, pig_three_feature, \
    test_x, test_y_regress, test_y_classification, kl_another_word_feature_test = sample_1031_1()

    # 实例化线性回归对象estimator
    from sklearn.linear_model import LinearRegression
    estimator = LinearRegression()
    # 将回归模型对象，训练集x，训练集连续y值，测试集x，测试集连续y传入
    regress_process(estimator, train_x, train_y_regress, test_x,
                    test_y_regress)
    plt.show()

    from abupy import ABuMLExecute
    ABuMLExecute.plot_learning_curve(estimator, train_x, train_y_regress, cv=10)


def sample_1031_3():
    """
    10.3.1_3 猪老三使用回归预测股价：PolynomialFeatures
    :return:
    """
    train_x, train_y_regress, train_y_classification, pig_three_feature, \
    test_x, test_y_regress, test_y_classification, kl_another_word_feature_test = sample_1031_1()

    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import PolynomialFeatures
    from sklearn.linear_model import LinearRegression

    # pipeline套上 degree=3 ＋ LinearRegression
    estimator = make_pipeline(PolynomialFeatures(degree=3),
                              LinearRegression())
    # 继续使用regress_process，区别是estimator变了
    regress_process(estimator, train_x, train_y_regress, test_x,
                    test_y_regress)
    plt.show()


def sample_1031_4():
    """
    10.3.1_4 猪老三使用回归预测股价：使用集成学习算法预测股价AdaBoost与RandomForest
    :return:
    """
    train_x, train_y_regress, train_y_classification, pig_three_feature, \
    test_x, test_y_regress, test_y_classification, kl_another_word_feature_test = sample_1031_1()

    # AdaBoost
    from sklearn.ensemble import AdaBoostRegressor

    estimator = AdaBoostRegressor(n_estimators=100)
    regress_process(estimator, train_x, train_y_regress, test_x,
                    test_y_regress)
    plt.show()
    # RandomForest
    from sklearn.ensemble import RandomForestRegressor

    estimator = RandomForestRegressor(n_estimators=100)
    regress_process(estimator, train_x, train_y_regress, test_x, test_y_regress)
    plt.show()


"""
    10.3.2 猪老三使用分类预测股票涨跌
"""


def classification_process(estimator, train_x, train_y_classification,
                           test_x, test_y_classification):
    from sklearn import metrics
    # 训练数据，这里分类要所以要使用y_classification
    estimator.fit(train_x, train_y_classification)
    # 使用训练好的分类模型预测测试集对应的y，即根据usFB的走势特征预测涨跌
    test_y_prdict_classification = estimator.predict(test_x)
    # 通过metrics.accuracy_score度量预测涨跌的准确率
    print("{} accuracy = {:.2f}".format(
        estimator.__class__.__name__,
        metrics.accuracy_score(test_y_classification,
                               test_y_prdict_classification)))

    from abupy import cross_val_score
    # 针对训练集数据做交叉验证scoring='accuracy'，cv＝10
    scores = cross_val_score(estimator, train_x,
                             train_y_classification,
                             cv=10,
                             scoring='accuracy')
    # 所有交叉验证的分数取平均值
    mean_sc = np.mean(scores)
    print('cross validation accuracy mean: {:.2f}'.format(mean_sc))


def sample_1032_1():
    """
    10.3.2_1 猪老三使用分类预测股票涨跌：LogisticRegression
    :return:
    """
    train_x, train_y_regress, train_y_classification, pig_three_feature, \
    test_x, test_y_regress, test_y_classification, kl_another_word_feature_test = sample_1031_1()

    # 无噪音分类正确100%
    from sklearn.linear_model import LogisticRegression
    estimator = LogisticRegression(C=1.0, penalty='l1', tol=1e-6)
    # 将分类器，训练集x，训练集y分类，测试集，测试集y分别传入函数
    classification_process(estimator, train_x, train_y_classification,
                           test_x, test_y_classification)

    # 开启噪音，再来一遍，有噪音正确率93%, 之后的都开启g_with_date_week_noise
    global g_with_date_week_noise
    g_with_date_week_noise = True
    train_x, train_y_regress, train_y_classification, pig_three_feature, \
    test_x, test_y_regress, test_y_classification, kl_another_word_feature_test = sample_1031_1()
    classification_process(estimator, train_x, train_y_classification,
                           test_x, test_y_classification)


def sample_1032_2():
    """
    10.3.2_2 猪老三使用分类预测股票涨跌：svm
    :return:
    """
    global g_with_date_week_noise
    g_with_date_week_noise = True

    train_x, train_y_regress, train_y_classification, pig_three_feature, \
    test_x, test_y_regress, test_y_classification, kl_another_word_feature_test = sample_1031_1()

    from sklearn.svm import SVC

    estimator = SVC(kernel='rbf')
    classification_process(estimator, train_x, train_y_classification,
                           test_x, test_y_classification)


def sample_1032_3():
    """
    10.3.2_3 猪老三使用分类预测股票涨跌：RandomForestClassifier
    :return:
    """
    global g_with_date_week_noise
    g_with_date_week_noise = True

    train_x, train_y_regress, train_y_classification, pig_three_feature, \
    test_x, test_y_regress, test_y_classification, kl_another_word_feature_test = sample_1031_1()

    from sklearn.ensemble import RandomForestClassifier

    estimator = RandomForestClassifier(n_estimators=100)
    classification_process(estimator, train_x, train_y_classification,
                           test_x, test_y_classification)


def sample_1032_4(show=True):
    """
    10.3.2_4 猪老三使用分类预测股票涨跌：train_test_split
    :return:
    """
    from sklearn import metrics
    from abupy import train_test_split

    # noinspection PyShadowingNames
    def train_test_split_xy(estimator, x, y, test_size=0.5,
                            random_state=0):
        # 通过train_test_split将原始训练集随机切割为新训练集与测试集
        train_x, test_x, train_y, test_y = \
            train_test_split(x, y, test_size=test_size,
                             random_state=random_state)

        if show:
            print(x.shape, y.shape)
            print(train_x.shape, train_y.shape)
            print(test_x.shape, test_y.shape)

        clf = estimator.fit(train_x, train_y)
        predictions = clf.predict(test_x)

        if show:
            # 度量准确率
            print("accuracy = %.2f" %
                  (metrics.accuracy_score(test_y, predictions)))

            # 度量查准率
            print("precision_score = %.2f" %
                  (metrics.precision_score(test_y, predictions)))

            # 度量回收率
            print("recall_score = %.2f" %
                  (metrics.recall_score(test_y, predictions)))

        return test_y, predictions

    global g_with_date_week_noise
    g_with_date_week_noise = True
    train_x, train_y_regress, train_y_classification, pig_three_feature, \
    test_x, test_y_regress, test_y_classification, kl_another_word_feature_test = sample_1031_1()

    from sklearn.ensemble import RandomForestClassifier
    estimator = RandomForestClassifier(n_estimators=100)

    test_y, predictions = train_test_split_xy(estimator, train_x, train_y_classification)
    return estimator, train_x, train_y_classification, test_y, predictions


def sample_1032_5():
    """
    10.3.2_5 猪老三使用分类预测股票涨跌：混淆矩阵和roc曲线
    :return:
    """

    from sklearn import metrics

    # noinspection PyShadowingNames
    def confusion_matrix_with_report(test_y, predictions):
        confusion_matrix = metrics.confusion_matrix(test_y, predictions)
        # print("Confusion Matrix ", confusion_matrix)
        print("          Predicted")
        print("         |  0  |  1  |")
        print("         |-----|-----|")
        print("       0 | %3d | %3d |" % (confusion_matrix[0, 0],
                                          confusion_matrix[0, 1]))
        print("Actual   |-----|-----|")
        print("       1 | %3d | %3d |" % (confusion_matrix[1, 0],
                                          confusion_matrix[1, 1]))
        print("         |-----|-----|")

        print(metrics.classification_report(test_y, predictions))

    estimator, train_x, train_y_classification, test_y, predictions = sample_1032_4(show=False)
    confusion_matrix_with_report(test_y, predictions)
    from abupy import ABuMLExecute
    ABuMLExecute.plot_roc_estimator(estimator, train_x, train_y_classification)


def sample_1033_1():
    """
    10.3.3 通过决策树分类，绘制出决策图
    这里需要安装dot graphviz，才能通过os.system("dot -T png graphviz.dot -o graphviz.png")生成png
    :return:
    """
    from sklearn.tree import DecisionTreeClassifier
    from sklearn import tree
    import os

    estimator = DecisionTreeClassifier(max_depth=2, random_state=1)

    # noinspection PyShadowingNames
    def graphviz_tree(estimator, features, x, y):
        if not hasattr(estimator, 'tree_'):
            print('only tree can graphviz!')
            return

        estimator.fit(x, y)
        # 将决策模型导出graphviz.dot文件
        tree.export_graphviz(estimator.tree_, out_file='graphviz.dot',
                             feature_names=features)
        # 通过dot将模型绘制决策图，保存png
        os.system("dot -T png graphviz.dot -o graphviz.png")

    global g_with_date_week_noise
    g_with_date_week_noise = True
    train_x, train_y_regress, train_y_classification, pig_three_feature, \
    test_x, test_y_regress, test_y_classification, kl_another_word_feature_test = sample_1031_1()

    # 这里会使用到特征的名称列pig_three_feature.columns[1:]
    graphviz_tree(estimator, pig_three_feature.columns[1:], train_x,
                  train_y_classification)

    import PIL.Image
    PIL.Image.open('graphviz.png').show()


def sample_1033_2():
    """
    10.3.3 特征的重要性排序及支持度评级
    :return:
    """
    global g_with_date_week_noise
    g_with_date_week_noise = True
    train_x, train_y_regress, train_y_classification, pig_three_feature, \
    test_x, test_y_regress, test_y_classification, kl_another_word_feature_test = sample_1031_1()

    # noinspection PyShadowingNames
    def importances_coef_pd(estimator):
        """
            特征的重要性
        """
        if hasattr(estimator, 'feature_importances_'):
            # 有feature_importances_的通过sort_values排序
            return pd.DataFrame(
                {'feature': list(pig_three_feature.columns[1:]),
                 'importance': estimator.feature_importances_}).sort_values('importance')

        elif hasattr(estimator, 'coef_'):
            # 有coef_的通过coef排序
            return pd.DataFrame(
                {"columns": list(pig_three_feature.columns)[1:], "coef": list(estimator.coef_.T)}).sort_values('coef')
        else:
            print('estimator not hasattr feature_importances_ or coef_!')

    # 使用随机森林分类器
    from sklearn.ensemble import RandomForestClassifier
    estimator = RandomForestClassifier(n_estimators=100)
    # 训练数据模型
    estimator.fit(train_x, train_y_classification)
    # 对训练后的模型特征的重要度进行判定，重要程度由小到大，表10-4所示
    print('importances_coef_pd(estimator):\n', importances_coef_pd(estimator))

    from sklearn.feature_selection import RFE

    # noinspection PyShadowingNames
    def feature_selection(estimator, x, y):
        """
            支持度评级
        """
        selector = RFE(estimator)
        selector.fit(x, y)
        print('RFE selection')
        print(pd.DataFrame(
            {'support': selector.support_, 'ranking': selector.ranking_},
            index=pig_three_feature.columns[1:]))

    print('feature_selection(estimator, train_x, train_y_classification):\n',
          feature_selection(estimator, train_x, train_y_classification))


"""
    10.4 无监督机器学习
"""


def sample_1041():
    """
    10.4.1 使用降维可视化数据
    :return:
    """
    train_x, train_y_regress, train_y_classification, pig_three_feature, \
    test_x, test_y_regress, test_y_classification, kl_another_word_feature_test = sample_1031_1()

    from sklearn.decomposition import PCA
    from abupy import ABuMLExecute

    # noinspection PyShadowingNames
    def plot_decision_function(estimator, x, y):
        # pca进行降维，只保留2个特征序列
        pca_2n = PCA(n_components=2)
        x = pca_2n.fit_transform(x)

        # 进行训练
        estimator.fit(x, y)
        plt.scatter(x[:, 0], x[:, 1], c=y, s=50, cmap='spring')
        ABuMLExecute.plot_decision_boundary(
            lambda p_x: estimator.predict(p_x), x, y)

    from sklearn.ensemble import RandomForestClassifier
    estimator = RandomForestClassifier(n_estimators=100)
    plot_decision_function(estimator, train_x, train_y_classification)


# noinspection PyTypeChecker
def sample_1042():
    """
    10.4.2 猪老三使用聚类算法提高正确率
    :return:
    """
    global g_with_date_week_noise
    g_with_date_week_noise = True
    train_x, train_y_regress, train_y_classification, pig_three_feature, \
    test_x, test_y_regress, test_y_classification, kl_another_word_feature_test = sample_1031_1()

    # 使用随机森林作为分类器
    from sklearn.ensemble import RandomForestClassifier
    estimator = RandomForestClassifier(n_estimators=100)
    estimator.fit(train_x, train_y_classification)
    test_y_prdict_classification = estimator.predict(test_x)

    from sklearn import metrics
    print("accuracy = %.2f" % (
        metrics.accuracy_score(test_y_classification,
                               test_y_prdict_classification)))

    # 测试集feature即usFB的kl feature
    pig_three_kmean_feature = kl_another_word_feature_test
    # 测试集真实的涨跌结果test_y_classification
    pig_three_kmean_feature['y'] = test_y_classification
    # 使用刚刚的随机森林作为分类器的预测涨跌结果test_y_prdict_classification
    pig_three_kmean_feature['y_prdict'] = test_y_prdict_classification
    # 即生成一列新数据记录预测是否正确
    pig_three_kmean_feature['y_same'] = np.where(
        pig_three_kmean_feature['y'] ==
        pig_three_kmean_feature['y_prdict'], 1, 0)
    # 将feature中只保留刚刚得到的y_same
    pig_three_kmean_feature = pig_three_kmean_feature.filter(['y_same'])

    from sklearn.cluster import KMeans

    # 使用刚刚得到的只有y_same列的数据赋值x_kmean
    x_kmean = pig_three_kmean_feature.values
    # n_clusters=2, 即只聚两类数据
    kmean = KMeans(n_clusters=2)
    kmean.fit(x_kmean)
    # 将聚类标签赋予新的一列cluster
    pig_three_kmean_feature['cluster'] = kmean.predict(x_kmean)
    # 将周几这个特征合并过来
    pig_three_kmean_feature['feature_date_week'] = \
        kl_another_word_feature_test['feature_date_week']
    # 表10-5所示
    print('pig_three_kmean_feature.tail():\n', pig_three_kmean_feature.tail())

    # 表10-6所示
    print('pd.crosstab(pig_three_kmean_feature.feature_date_week, pig_three_kmean_feature.cluster):\n',
          pd.crosstab(pig_three_kmean_feature.feature_date_week, pig_three_kmean_feature.cluster))


"""
    10.5 梦醒时分
"""


def sample_105_0():
    """
    10.5 AbuML
    :return:
    """
    global g_with_date_week_noise
    g_with_date_week_noise = True
    train_x, train_y_regress, train_y_classification, pig_three_feature, \
    test_x, test_y_regress, test_y_classification, kl_another_word_feature_test = sample_1031_1()

    from abupy import AbuML
    # 通过x, y矩阵和特征的DataFrame对象组成AbuML
    ml = AbuML(train_x, train_y_classification, pig_three_feature)
    # 使用随机森林作为分类器
    _ = ml.estimator.random_forest_classifier()

    # 交织验证结果的正确率
    print('ml.cross_val_accuracy_score():\n', ml.cross_val_accuracy_score())
    # 特征的选择
    print('ml.feature_selection():\n', ml.feature_selection())


"""
    如下内容不能使用沙盒环境, 建议对照阅读：
        abu量化文档－第十九节 数据源
        第20节 美股交易UMP决策
"""


def sample_1051_0():
    """
    10.5.1 回测中生成特征，切分训练测试集，成交买单快照: 数据准备

    如果没有运行过abu量化文档－第十九节 数据源：中使用腾讯数据源进行数据更新，需要运行
    如果运行过就不要重复运行了：
    """
    from abupy import EMarketTargetType, EMarketSourceType, EDataCacheType
    # 关闭沙盒数据环境
    abupy.env.disable_example_env_ipython()
    abupy.env.g_market_source = EMarketSourceType.E_MARKET_SOURCE_tx
    abupy.env.g_data_cache_type = EDataCacheType.E_DATA_CACHE_CSV
    # 首选这里预下载市场中所有股票的6年数据(做5年回测，需要预先下载6年数据)
    abu.run_kl_update(start='2011-08-08', end='2017-08-08', market=EMarketTargetType.E_MARKET_TARGET_US)


def sample_1051_1(from_cache=False, show=True):
    """
    10.5.1 回测中生成特征，切分训练测试集，成交买单快照: 数据准备
    :return:
    """
    from abupy import AbuMetricsBase
    from abupy import AbuFactorBuyBreak
    from abupy import AbuFactorAtrNStop
    from abupy import AbuFactorPreAtrNStop
    from abupy import AbuFactorCloseAtrNStop

    # 关闭沙盒数据环境
    abupy.env.disable_example_env_ipython()
    from abupy import EMarketDataFetchMode
    # 因为sample_94_1下载了预先数据，使用缓存，设置E_DATA_FETCH_FORCE_LOCAL，实际上run_kl_update最后会把设置set到FORCE_LOCAL
    abupy.env.g_data_fetch_mode = EMarketDataFetchMode.E_DATA_FETCH_FORCE_LOCAL

    # 设置选股因子，None为不使用选股因子
    stock_pickers = None
    # 买入因子依然延用向上突破因子
    buy_factors = [{'xd': 60, 'class': AbuFactorBuyBreak},
                   {'xd': 42, 'class': AbuFactorBuyBreak}]

    # 卖出因子继续使用上一章使用的因子
    sell_factors = [
        {'stop_loss_n': 1.0, 'stop_win_n': 3.0,
         'class': AbuFactorAtrNStop},
        {'class': AbuFactorPreAtrNStop, 'pre_atr_n': 1.5},
        {'class': AbuFactorCloseAtrNStop, 'close_atr_n': 1.5}
    ]

    # 回测生成买入时刻特征
    abupy.env.g_enable_ml_feature = True
    # 回测将symbols切割分为训练集数据和测试集数据
    abupy.env.g_enable_train_test_split = True
    # 下面设置回测时切割训练集，测试集使用的切割比例参数，默认为10，即切割为10份，9份做为训练，1份做为测试，
    # 由于美股股票数量多，所以切割分为4份，3份做为训练集，1份做为测试集
    abupy.env.g_split_tt_n_folds = 4

    from abupy import EStoreAbu
    if from_cache:
        abu_result_tuple = \
            abu.load_abu_result_tuple(n_folds=5, store_type=EStoreAbu.E_STORE_CUSTOM_NAME,
                                      custom_name='train_us')
    else:
        # 初始化资金500万，资金管理依然使用默认atr
        read_cash = 5000000
        # 每笔交易的买入基数资金设置为万分之15
        abupy.beta.atr.g_atr_pos_base = 0.0015
        # 使用run_loop_back运行策略，因子使用和之前一样，
        # choice_symbols=None为全市场回测，5年历史数据回测
        abu_result_tuple, _ = abu.run_loop_back(read_cash,
                                                buy_factors, sell_factors,
                                                stock_pickers,
                                                choice_symbols=None,
                                                start='2012-08-08', end='2017-08-08')
        # 把运行的结果保存在本地，以便之后分析回测使用，保存回测结果数据代码如下所示
        abu.store_abu_result_tuple(abu_result_tuple, n_folds=5, store_type=EStoreAbu.E_STORE_CUSTOM_NAME,
                                   custom_name='train_us')

    if show:
        metrics = AbuMetricsBase(*abu_result_tuple)
        metrics.fit_metrics()
        metrics.plot_returns_cmp(only_show_returns=True)

    "＊＊＊＊＊＊＊＊＊＊＊＊＊＊＊＊＊＊＊＊＊＊＊＊＊＊＊＊＊＊＊＊＊＊＊＊＊＊＊＊＊＊＊＊＊＊＊＊＊＊＊＊＊＊＊＊＊＊＊＊＊＊＊＊＊"
    abupy.env.g_enable_train_test_split = False
    # 使用切割好的测试数据
    abupy.env.g_enable_last_split_test = True

    from abupy import EStoreAbu
    if from_cache:
        abu_result_tuple_test = \
            abu.load_abu_result_tuple(n_folds=5, store_type=EStoreAbu.E_STORE_CUSTOM_NAME,
                                      custom_name='test_us')
    else:
        read_cash = 5000000
        abupy.beta.atr.g_atr_pos_base = 0.007
        choice_symbols = None
        abu_result_tuple_test, kl_pd_manager_test = abu.run_loop_back(read_cash,
                                                                      buy_factors, sell_factors, stock_pickers,
                                                                      choice_symbols=choice_symbols, start='2012-08-08',
                                                                      end='2017-08-08')
        abu.store_abu_result_tuple(abu_result_tuple_test, n_folds=5, store_type=EStoreAbu.E_STORE_CUSTOM_NAME,
                                   custom_name='test_us')

    if show:
        metrics = AbuMetricsBase(*abu_result_tuple_test)
        metrics.fit_metrics()
        metrics.plot_returns_cmp(only_show_returns=True)
        print(abu_result_tuple.orders_pd[abu_result_tuple.orders_pd.result != 0].head())

    return abu_result_tuple, abu_result_tuple_test


# noinspection PyUnresolvedReferences
def sample_1052():
    """
    10.5.2 基于特征的交易预测
    :return:
    """
    # 需要在有缓存的情况下运行
    abu_result_tuple, _ = sample_1051_1(from_cache=True, show=False)

    from abupy.UmpBu.ABuUmpMainMul import UmpMulFiter
    mul = UmpMulFiter(orders_pd=abu_result_tuple.orders_pd, scaler=False)
    print('mul.df.head():\n', mul.df.head())

    # 默认使用svm作为分类器
    print('decision_tree_classifier cv please wait...')
    mul.estimator.decision_tree_classifier()
    mul.cross_val_accuracy_score()

    # 默认使用svm作为分类器
    print('knn_classifier cv please wait...')
    # 默认使用svm作为分类器, 改分类器knn
    mul.estimator.knn_classifier()
    mul.cross_val_accuracy_score()

    from abupy.UmpBu.ABuUmpMainBase import UmpDegFiter
    deg = UmpDegFiter(orders_pd=abu_result_tuple.orders_pd)
    print('deg.df.head():\n', deg.df.head())

    print('xgb_classifier cv please wait...')
    # 分类器使用GradientBoosting
    deg.estimator.xgb_classifier()
    deg.cross_val_accuracy_score()

    print('adaboost_classifier cv please wait...')
    # 分类器使用adaboost
    deg.estimator.adaboost_classifier(base_estimator=None)
    deg.cross_val_accuracy_score()

    print('train_test_split_xy please wait...')
    deg.train_test_split_xy()


if __name__ == "__main__":
    sample_102()
    # sample_103_0()
    # sample_1031_1()
    # sample_1031_2()
    # sample_1031_3()
    # sample_1031_4()
    # sample_1032_1()
    # sample_1032_2()
    # sample_1032_3()
    # sample_1032_4()
    # sample_1032_5()
    # sample_1033_1()
    # sample_1033_2()
    # sample_1041()
    # sample_1042()
    # sample_105_0()
    # sample_1051_0()
    # sample_1051_1(from_cache=True)
    # sample_1051_1(from_cache=False)
    # sample_1052()
