# -*- encoding:utf-8 -*-
from __future__ import print_function

import warnings
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

warnings.filterwarnings('ignore')
sns.set_context(rc={'figure.figsize': (14, 7)})


"""
    第三章 量化工具——NumPy
    abu量化系统github地址：https://github.com/bbfamily/abu (您的star是我的动力！)
    abu量化文档教程ipython notebook：https://github.com/bbfamily/abu/tree/master/abupy_lecture
"""


def sample_311():
    """
    3.1.1 并行化思想
    :return:
    """
    # 注意 * 3的操作被运行在每一个元素上
    np_list = np.ones(5) * 3
    print('np_list:', np_list)
    # 普通的列表把*3操作认为是整体性操作
    normal_list = [1, 1, 1, 1, 1] * 3
    print('normal_list:', normal_list, len(normal_list))


# 200支股票
stock_cnt = 200
# 504个交易日
view_days = 504
# 生成服从正态分布：均值期望＝0，标准差＝1的序列
stock_day_change = np.random.standard_normal((stock_cnt, view_days))
try:
    # 使用沙盒数据，目的是和书中一样的数据环境，不需要注视掉
    stock_day_change = np.load('../gen/stock_day_change.npy')
except Exception as e:
    print('../gen/stock_day_change.npy load error:{}'.format(e))


def sample_312():
    """
    3.1.2 初始化操作
    :return:
    """
    np_list = np.arange(10000)

    # 100个0
    print('np.zeros(100):\n', np.zeros(100))
    # shape：3行2列 全是0
    print('np.zeros((3, 2):\n', np.zeros((3, 2)))

    # shape： 3行2列 全是1
    print('np.ones((3, 2):\n', np.ones((3, 2)))
    # shape：x=2, y=3, z=3 值随机
    print('np.empty((2, 3, 3):\n', np.empty((2, 3, 3)))

    # 初始化序列与np_list一样的shape，值全为1
    print('np.ones_like(np_list):\n', np.ones_like(np_list))
    # 初始化序列与np_list一样的shape，值全为0
    print('np.zeros_like(np_list):\n', np.zeros_like(np_list))
    # eye得到对角线全为1的单位矩阵
    print('np.eye(3):\n', np.eye(3))

    # 打印shape (200, 504) 200行504列
    print('stock_day_change.shape:', stock_day_change.shape)
    # 打印出第一支只股票，头五个交易日的涨跌幅情况
    print('stock_day_change[0:1, :5]:\n', stock_day_change[0:1, :5])


"""
    3.1.3 索引选取和切片选
"""

# tmp = a
tmp = stock_day_change[0:2, 0:5].copy()
# a = b
stock_day_change[0:2, 0:5] = stock_day_change[-2:, -5:]
# b = tmp
stock_day_change[-2:, -5:] = tmp


def sample_313():
    """
    3.1.3 索引选取和切片选
    :return:
    """
    # 0:2第一，第二支股票，0:5头五个交易日的涨跌幅数据
    print('stock_day_change[0:2, 0:5]:\n', stock_day_change[0:2, 0:5])

    # -2:倒数一，第二支股票，-5:最后五个交易日的涨跌幅数据
    print('stock_day_change[-2:, -5:]:\n', stock_day_change[-2:, -5:])

    # view result
    print('[0:2, 0:5], [-2:, -5:]:\n', stock_day_change[0:2, 0:5], stock_day_change[-2:, -5:])


def sample_314():
    """
    3.1.4 数据转换与规整
    :return:
    """
    print('stock_day_change[0:2, 0:5]:\n', stock_day_change[0:2, 0:5])
    print('[0:2, 0:5].astype(int):\n', stock_day_change[0:2, 0:5].astype(int))
    # 2代表保留两位小数
    print('around 2:\n', np.around(stock_day_change[0:2, 0:5], 2))
    # 使用copy目的是不修改原始序列
    tmp_test = stock_day_change[0:2, 0:5].copy()
    # 将第一个元素改成nan
    tmp_test[0][0] = np.nan
    print('tmp_test:\n', tmp_test)


def sample_315():
    """
    3.1.5 逻辑条件进行数据筛选
    :return:
    """
    # 找出上述切片内涨幅超过0.5的股票时段, 通过输出结果你可以看到返回的是boolean的数组
    mask = stock_day_change[0:2, 0:5] > 0.5
    print('mask:\n', mask)
    tmp_test = stock_day_change[0:2, 0:5].copy()
    # 使用上述的mask数组筛选出符合条件的数组, 即中筛选mask中对应index值为True的
    print('tmp_test[mask]:\n', tmp_test[mask])

    tmp_test[tmp_test > 0.5] = 1
    print('tmp_test:\n', tmp_test)

    tmp_test = stock_day_change[-2:, -5:]
    print('tmp_test2:\n', tmp_test)
    print('tmp_test[(tmp_test > 1) | (tmp_test < -1)]:\n', tmp_test[(tmp_test > 1) | (tmp_test < -1)])


# noinspection PyTypeChecker
def sample_316():
    """
    3.1.6 通用序列函数
    :return:
    """
    # np.all判断序列中的所有元素是否全部是true, 即对bool序列进行与操作
    # 本例实际判断stock_day_change[0:2, 0:5]中是否全是上涨的
    print('np.all(stock_day_change[0:2, 0:5] > 0):\n', np.all(stock_day_change[0:2, 0:5] > 0))

    # np.any判断序列中是否有元素为true, 即对bool序列进行或操作
    # 本例实际判断stock_day_change[0:2, 0:5]中是至少有一个是上涨的
    print('np.any(stock_day_change[0:2, 0:5] > 0):\n', np.any(stock_day_change[0:2, 0:5] > 0))

    # 对两个序列对应的元素两两比较，maximum结果集取大,相对使用minimum为取小的结果集
    print('np.maximum(stock_day_change[0:2, 0:5], stock_day_change[-2:, -5:]):\n',
          np.maximum(stock_day_change[0:2, 0:5], stock_day_change[-2:, -5:]))

    change_int = stock_day_change[0:2, 0:5].astype(int)
    print('change_int:\n', change_int)
    # 序列中数值值唯一且不重复的值组成新的序列
    print('np.unique(change_int):\n', np.unique(change_int))

    # axis＝1
    print('np.diff(stock_day_change[0:2, 0:5]):\n', np.diff(stock_day_change[0:2, 0:5]))

    # 唯一区别 axis=0
    print('np.diff(stock_day_change[0:2, 0:5], axis=0):\n', np.diff(stock_day_change[0:2, 0:5], axis=0))

    tmp_test = stock_day_change[-2:, -5:]
    print('np.where(tmp_test > 0.5, 1, 0):\n', np.where(tmp_test > 0.5, 1, 0))
    print('np.where(tmp_test > 0.5, 1, tmp_test):\n', np.where(tmp_test > 0.5, 1, tmp_test))

    # 序列中的值大于0.5并且小于1的赋值为1，否则赋值为0
    print('np.where(np.logical_and(tmp_test > 0.5, tmp_test < 1), 1, 0):\n',
          np.where(np.logical_and(tmp_test > 0.5, tmp_test < 1), 1, 0))

    # 序列中的值大于0.5或者小于－0.5的赋值为1，否则赋值为0
    print('np.where(np.logical_or(tmp_test > 0.5, tmp_test < -0.5), 1, 0):\n',
          np.where(np.logical_or(tmp_test > 0.5, tmp_test < -0.5), 1, 0))


"""
    3.1.7 数据本地序列化操作
"""
stock_day_change = np.load('../gen/stock_day_change.npy')
np.save('../gen/stock_day_change', stock_day_change)

"""
    3.2 统计概念与函数使用
"""

stock_day_change_four = stock_day_change[:4, :4]


def sample_320():
    """
    3.2.0 统计概念与函数使用
    :return:
    """
    print('stock_day_change_four:\n', stock_day_change_four)


def sample_321():
    """
    3.2.1 统计基础函数使用
    :return:
    """
    print('最大涨幅 {}'.format(np.max(stock_day_change_four, axis=1)))

    print('最大跌幅 {}'.format(np.min(stock_day_change_four, axis=1)))
    print('振幅幅度 {}'.format(np.std(stock_day_change_four, axis=1)))
    print('平均涨跌 {}'.format(np.mean(stock_day_change_four, axis=1)))

    print('最大涨幅 {}'.format(np.max(stock_day_change_four, axis=0)))

    print('最大涨幅股票{}'.format(np.argmax(stock_day_change_four, axis=0)))
    print('最大跌幅股票{}'.format(np.argmin(stock_day_change_four, axis=0)))

    print('最大跌幅 {}'.format(np.min(stock_day_change_four, axis=0)))
    print('振幅幅度 {}'.format(np.std(stock_day_change_four, axis=0)))
    print('平均涨跌 {}'.format(np.mean(stock_day_change_four, axis=0)))


def sample_322():
    """
    3.2.2 统计基础概念
    :return:
    """
    a_investor = np.random.normal(loc=100, scale=50, size=(100, 1))
    b_investor = np.random.normal(loc=100, scale=20, size=(100, 1))

    # a交易者
    print('a交易者期望{0:.2f}元, 标准差{1:.2f}, 方差{2:.2f}'.format(
        a_investor.mean(), a_investor.std(), a_investor.var()))

    # b交易者
    print('b交易者期望{0:.2f}元, 标准差{1:.2f}, 方差{2:.2f}'.format(
        b_investor.mean(), b_investor.std(), b_investor.var()))

    # a交易者期望
    a_mean = a_investor.mean()
    # a交易者标注差
    a_std = a_investor.std()
    # 收益绘制曲线
    plt.plot(a_investor)
    # 水平直线 上线
    plt.axhline(a_mean + a_std, color='r')
    # 水平直线 均值期望线
    plt.axhline(a_mean, color='y')
    # 水平直线 下线
    plt.axhline(a_mean - a_std, color='g')
    plt.show()

    b_mean = b_investor.mean()
    b_std = b_investor.std()
    # b交易者收益绘制曲线
    plt.plot(b_investor)
    # 水平直线 上线
    plt.axhline(b_mean + b_std, color='r')
    # 水平直线 均值期望线
    plt.axhline(b_mean, color='y')
    # 水平直线 下线
    plt.axhline(b_mean - b_std, color='g')
    plt.show()


def sample_331():
    """
    3.3.1 正态分布基础概念
    :return:
    """
    import scipy.stats as scs

    # 均值期望
    stock_mean = stock_day_change[0].mean()
    # 标准差
    stock_std = stock_day_change[0].std()
    print('股票0 mean均值期望:{:.3f}'.format(stock_mean))
    print('股票0 std振幅标准差:{:.3f}'.format(stock_std))

    # 绘制股票0的直方图
    plt.hist(stock_day_change[0], bins=50, normed=True)

    # linspace从股票0 最小值－> 最大值生成数据
    fit_linspace = np.linspace(stock_day_change[0].min(),
                               stock_day_change[0].max())

    # 概率密度函数(PDF，probability density function)
    # 由均值，方差，来描述曲线，使用scipy.stats.norm.pdf生成拟合曲线
    pdf = scs.norm(stock_mean, stock_std).pdf(fit_linspace)
    print(pdf)
    # plot x, y
    plt.plot(fit_linspace, pdf, lw=2, c='r')
    plt.show()


def sample_332():
    """
    3.3.2 实例1：正态分布买入策略
    :return:
    """
    # 保留后50天的随机数据作为策略验证数据
    keep_days = 50
    # 统计前454, 切片切出0-454day，view_days = 504
    stock_day_change_test = stock_day_change[:stock_cnt, 0:view_days - keep_days]
    # 打印出前454跌幅最大的三支，总跌幅通过np.sum计算，np.sort对结果排序
    print('np.sort(np.sum(stock_day_change_test, axis=1))[:3]:', np.sort(np.sum(stock_day_change_test, axis=1))[:3])
    # 使用np.argsort针对股票跌幅进行排序，返回序号，即符合买入条件的股票序号
    stock_lower_array = np.argsort(np.sum(stock_day_change_test, axis=1))[:3]
    # 输符合买入条件的股票序号
    print('stock_lower_array:', stock_lower_array)

    def show_buy_lower(p_stock_ind):
        """
        :param p_stock_ind: 股票序号,即在stock_day_change中的位置
        :return:
        """
        # 设置一个一行两列的可视化图表
        _, axs = plt.subplots(nrows=1, ncols=2, figsize=(16, 5))
        # view_days504 - keep_days50 = 454
        # 绘制前454天股票走势图，np.cumsum()：序列连续求和
        axs[0].plot(np.arange(0, view_days - keep_days),
                    stock_day_change_test[p_stock_ind].cumsum())

        # [view_days504 - keep_days50 = 454 : view_days504]
        # 从第454天开始到504天的股票走势
        cs_buy = stock_day_change[p_stock_ind][
                 view_days - keep_days:view_days].cumsum()

        # 绘制从第454天到504天股票走势图
        axs[1].plot(np.arange(view_days - keep_days, view_days), cs_buy)
        # 返回从第454天开始到第504天计算盈亏的盈亏序列的最后一个值
        return cs_buy[-1]

    # 最后输出的盈亏比例
    profit = 0
    # 跌幅最大的三支遍历序号
    for stock_ind in stock_lower_array:
        # profit即三支股票从第454天买入开始计算，直到最后一天的盈亏比例
        profit += show_buy_lower(stock_ind)
    plt.show()

    # str.format 支持{:.2f}形式保留两位小数
    print('买入第 {} 支股票，从第454个交易日开始持有盈亏:{:.2f}%'.format(
        stock_lower_array, profit))


def sample_342():
    """
    3.4.2 实例2：如何在交易中获取优势
    :return:
    """

    # 设置100个赌徒
    gamblers = 100

    def casino(win_rate, win_once=1, loss_once=1, commission=0.01):
        """
            赌场：简单设定每个赌徒一共有1000000一共想在赌场玩10000000次，
            但是你要是没钱了也别想玩了
            win_rate:   输赢的概率
            win_once:   每次赢的钱数
            loss_once:  每次输的钱数
            commission: 手续费这里简单的设置了0.01 1%
        """
        my_money = 1000000
        play_cnt = 10000000
        commission = commission
        for _ in np.arange(0, play_cnt):
            # 使用伯努利分布根据win_rate来获取输赢
            w = np.random.binomial(1, win_rate)
            if w:
                # 赢了 +win_once
                my_money += win_once
            else:
                # 输了 -loss_once
                my_money -= loss_once
            # 手续费
            my_money -= commission
            if my_money <= 0:
                # 没钱就别玩了，不赊账
                break
        return my_money

    """
        如果有numba使用numba进行加速, 这个加速效果非常明显，不使用numba非常非常非常慢
    """
    import numba as nb
    casino = nb.jit(casino)

    print('heaven_moneys....')
    # 100个赌徒进场天堂赌场，胜率0.5，赔率1，还没手续费
    heaven_moneys = [casino(0.5, commission=0) for _ in
                     np.arange(0, gamblers)]

    print('cheat_moneys....')
    # 100个赌徒进场开始，胜率0.4，赔率1，没手续费
    cheat_moneys = [casino(0.4, commission=0) for _ in
                    np.arange(0, gamblers)]

    print('commission_moneys....')
    # 100个赌徒进场开始，胜率0.5，赔率1，手续费0.01
    commission_moneys = [casino(0.5, commission=0.01) for _ in
                         np.arange(0, gamblers)]

    print('casino(0.5, commission=0.01, win_once=1.02, loss_once=0.98.....')
    # 100个赌徒进场开始，胜率0.5，赔率1.04，手续费0.01
    f1_moneys = [casino(0.5, commission=0.01, win_once=1.02, loss_once=0.98)
                 for _ in np.arange(0, gamblers)]

    print('casino(0.45, commission=0.01, win_once=1.02, loss_once=0.98.....')
    # 100个赌徒进场开始，胜率0.45，赔率1.04，手续费0.01
    f2_moneys = [casino(0.45, commission=0.01, win_once=1.02, loss_once=0.98)
                 for _ in np.arange(0, gamblers)]

    _ = plt.hist(heaven_moneys, bins=30)
    plt.show()
    _ = plt.hist(cheat_moneys, bins=30)
    plt.show()
    _ = plt.hist(commission_moneys, bins=30)
    plt.show()
    _ = plt.hist(f1_moneys, bins=30)
    plt.show()
    _ = plt.hist(f2_moneys, bins=30)
    plt.show()


if __name__ == "__main__":
    sample_311()
    # sample_312()
    # sample_313()
    # sample_314()
    # sample_315()
    # sample_316()
    # sample_320()
    # sample_321()
    # sample_322()
    # sample_331()
    # sample_332()
    # sample_342()
