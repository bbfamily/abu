# -*- encoding:utf-8 -*-
from __future__ import print_function
from __future__ import division

import warnings
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
# noinspection PyUnresolvedReferences
import abu_local_env
import abupy
from abupy import ABuSymbolPd
from abupy import six, xrange

from abc import ABCMeta, abstractmethod


warnings.filterwarnings('ignore')
sns.set_context(rc={'figure.figsize': (14, 7)})
# 使用沙盒数据，目的是和书中一样的数据环境
abupy.env.enable_example_env_ipython()

tsla_close = ABuSymbolPd.make_kl_df('usTSLA').close
# x序列: 0，1，2, ...len(tsla_close)
x = np.arange(0, tsla_close.shape[0])
# 收盘价格序列
y = tsla_close.values


"""
    第六章 量化工具——数学：你一生的追求到底能带来多少幸福

    abu量化系统github地址：https://github.com/bbfamily/abu (您的star是我的动力！)
    abu量化文档教程ipython notebook：https://github.com/bbfamily/abu/tree/master/abupy_lecture
"""


def sample_611_1(show=True):
    """
    6.1.1 线性回归
    :return:
    """
    import statsmodels.api as sm
    from statsmodels import regression

    def regress_y(_y):
        _y = _y
        # x序列: 0，1，2, ...len(y)
        _x = np.arange(0, len(_y))
        _x = sm.add_constant(_x)
        # 使用OLS做拟合
        _model = regression.linear_model.OLS(_y, _x).fit()
        return _model

    model = regress_y(y)
    b = model.params[0]
    k = model.params[1]
    # y = kx + b
    y_fit = k * x + b
    if show:
        plt.plot(x, y)
        plt.plot(x, y_fit, 'r')
        plt.show()
        # summary模型拟合概述，表6-1所示
        print(model.summary())
    return y_fit


# noinspection PyPep8Naming
def sample_611_2():
    """
    6.1.1 线性回归
    :return:
    """
    y_fit = sample_611_1(show=False)

    MAE = sum(np.abs(y - y_fit)) / len(y)
    print('偏差绝对值之和(MAE)={}'.format(MAE))
    MSE = sum(np.square(y - y_fit)) / len(y)
    print('偏差绝对值之和(MSE)={}'.format(MSE))
    RMSE = np.sqrt(sum(np.square(y - y_fit)) / len(y))
    print('偏差绝对值之和(RMSE)={}'.format(RMSE))

    from sklearn import metrics
    print('sklearn偏差绝对值之和(MAE)={}'.format(metrics.mean_absolute_error(y, y_fit)))
    print('sklearn偏差平方(MSE)={}'.format(metrics.mean_squared_error(y, y_fit)))
    print('sklearn偏差平方和开平方(RMSE)={}'.format(np.sqrt(metrics.mean_squared_error(y, y_fit))))


# noinspection PyCallingNonCallable
def sample_612():
    """
    6.1.2 多项式回归
    :return:
    """
    import itertools

    # 生成9个subplots 3*3
    _, axs = plt.subplots(nrows=3, ncols=3, figsize=(15, 15))

    # 将 3 * 3转换成一个线性list
    axs_list = list(itertools.chain.from_iterable(axs))
    # 1-9次多项式回归
    poly = np.arange(1, 10, 1)
    for p_cnt, ax in zip(poly, axs_list):
        # 使用polynomial.Chebyshev.fit进行多项式拟合
        p = np.polynomial.Chebyshev.fit(x, y, p_cnt)
        # 使用p直接对x序列代人即得到拟合结果序列
        y_fit = p(x)
        # 度量mse值
        from sklearn import metrics
        mse = metrics.mean_squared_error(y, y_fit)
        # 使用拟合次数和mse误差大小设置标题
        ax.set_title('{} poly MSE={}'.format(p_cnt, mse))
        ax.plot(x, y, '', x, y_fit, 'r.')
    plt.show()


def sample_613():
    """
    6.1.3 插值
    :return:
    """
    from scipy.interpolate import interp1d, splrep, splev

    # 示例两种插值计算方式
    _, axs = plt.subplots(nrows=1, ncols=2, figsize=(14, 5))

    # 线性插值
    linear_interp = interp1d(x, y)
    # axs[0]左边的
    axs[0].set_title('interp1d')
    # 在相同坐标系下，同样的x，插值的y值使r.绘制（红色点）
    axs[0].plot(x, y, '', x, linear_interp(x), 'r.')

    # B-spline插值
    splrep_interp = splrep(x, y)
    # axs[1]右边的
    axs[1].set_title('splrep')
    # #在相同坐标系下，同样的x，插值的y值使g.绘制（绿色点）
    axs[1].plot(x, y, '', x, splev(x, splrep_interp), 'g.')
    plt.show()


"""
    6.2 蒙特卡洛方法与凸优化
    6.2.1 你一生的追求到底能带来多少幸福
"""

# 每个人平均寿命期望是75年，约75*365=27375天
K_INIT_LIVING_DAYS = 27375


class Person(object):
    """
        人类
    """

    def __init__(self):
        # 初始化人平均能活的寿命
        self.living = K_INIT_LIVING_DAYS
        # 初始化幸福指数
        self.happiness = 0
        # 初始化财富值
        self.wealth = 0
        # 初始化名望权利
        self.fame = 0
        # 活着的第几天
        self.living_day = 0

    def live_one_day(self, seek):
        """
        每天只能进行一个seek，这个seek决定了你今天追求的是什么，得到了什么
        seek的类型属于下面将编写的BaseSeekDay
        :param seek:
        :return:
        """
        # 调用每个独特的BaseSeekDay类都会实现的do_seek_day，得到今天的收获
        consume_living, happiness, wealth, fame = seek.do_seek_day()
        # 每天要减去生命消耗，有些seek前面还会增加生命
        self.living -= consume_living
        # seek得到的幸福指数积累
        self.happiness += happiness
        # seek得到的财富积累
        self.wealth += wealth
        # seek得到的名望权力积累
        self.fame += fame
        # 活完这一天了
        self.living_day += 1


class BaseSeekDay(six.with_metaclass(ABCMeta, object)):
    def __init__(self):
        # 每个追求每天消耗生命的常数
        self.living_consume = 0

        # 每个追求每天幸福指数常数
        self.happiness_base = 0

        # 每个追求每天财富积累常数
        self.wealth_base = 0
        # 每个追求每天名望权利积累常数
        self.fame_base = 0

        # 每个追求每天消耗生命的可变因素序列
        self.living_factor = [0]

        # 每个追求每天幸福指数的可变因素序列
        self.happiness_factor = [0]

        # 每个追求每天财富积累的可变因素序列
        self.wealth_factor = [0]
        # 每个追求每天名望权利的可变因素序列
        self.fame_factor = [0]

        # 追求了多少天了这一生
        self.do_seek_day_cnt = 0
        # 子类进行常数及可变因素序列设置
        self._init_self()

    @abstractmethod
    def _init_self(self, *args, **kwargs):
        # 子类必须实现，设置自己的生命消耗的常数，幸福指数常数等常数设置
        pass

    @abstractmethod
    def _gen_living_days(self, *args, **kwargs):
        # 子类必须实现，设置自己的可变因素序列
        pass

    def do_seek_day(self):
        """
        每一天的追求具体seek
        :return:
        """
        # 生命消耗=living_consume:消耗常数 * happiness_factor:可变序列
        if self.do_seek_day_cnt >= len(self.living_factor):
            # 超出len(self.living_factor), 就取最后一个living_factor[-1]
            consume_living = \
                self.living_factor[-1] * self.living_consume
        else:
            # 每个类自定义这个追求的消耗生命常数，以及living_factor，比如
            # HealthSeekDay追求健康，living_factor序列的值即由负值->正值
            # 每个子类living_factor会有自己特点的变化速度及序列长度，导致每个
            # 追求对生命的消耗随着追求的次数变化不一
            consume_living = self.living_factor[self.do_seek_day_cnt] \
                             * self.living_consume
        # 幸福指数=happiness_base:幸福常数 * happiness_factor:可变序列
        if self.do_seek_day_cnt >= len(self.happiness_factor):
            # 超出len(self.happiness_factor), 就取最后一个
            # 由于happiness_factor值由:n—>0 所以happiness_factor[-1]=0
            # 即随着追求一个事物的次数过多后会变的没有幸福感
            happiness = self.happiness_factor[
                            -1] * self.happiness_base
        else:
            # 每个类自定义这个追求的幸福指数常数，以及happiness_factor
            # happiness_factor子类的定义一般是从高－>低变化
            happiness = self.happiness_factor[
                            self.do_seek_day_cnt] * self.happiness_base
        # 财富积累=wealth_base:积累常数 * wealth_factor:可变序列
        if self.do_seek_day_cnt >= len(self.wealth_factor):
            # 超出len(self.wealth_factor), 就取最后一个
            wealth = self.wealth_factor[-1] * self.wealth_base
        else:
            # 每个类自定义这个追求的财富指数常数，以及wealth_factor
            wealth = self.wealth_factor[
                         self.do_seek_day_cnt] * self.wealth_base
        # 权利积累=fame_base:积累常数 * fame_factor:可变序列
        if self.do_seek_day_cnt >= len(self.fame_factor):
            # 超出len(self.fame_factor), 就取最后一个
            fame = self.fame_factor[-1] * self.fame_base
        else:
            # 每个类自定义这个追求的名望权利指数常数，以及fame_factor
            fame = self.fame_factor[
                       self.do_seek_day_cnt] * self.fame_base
        # 追求了多少天了这一生 + 1
        self.do_seek_day_cnt += 1
        # 返回这个追求这一天对生命的消耗，得到的幸福，财富，名望权利
        return consume_living, happiness, wealth, fame


def regular_mm(group):
    # 最小-最大规范化
    return (group - group.min()) / (group.max() - group.min())


"""
    HealthSeekDay
"""


class HealthSeekDay(BaseSeekDay):
    """
        HealthSeekDay追求健康长寿的一天:
        形象：健身，旅游，娱乐，做感兴趣的事情。
        抽象：追求健康长寿。
    """

    def _init_self(self):
        # 每天对生命消耗的常数＝1，即代表1天
        self.living_consume = 1
        # 每天幸福指数常数＝1
        self.happiness_base = 1
        # 设定可变因素序列
        self._gen_living_days()

    def _gen_living_days(self):
        # 只生成12000个序列，因为下面的happiness_factor序列值由1－>0
        # 所以大于12000次的追求都将只是单纯消耗生命，并不增加幸福指数
        # 即随着做一件事情的次数越来越多，幸福感越来越低，直到完全体会不到幸福
        days = np.arange(1, 12000)
        # 基础函数选用sqrt, 影响序列变化速度
        living_days = np.sqrt(days)

        """
            对生命消耗可变因素序列值由-1->1, 也就是这个追求一开始的时候对生命
            的消耗为负增长，延长了生命，随着追求的次数不断增多对生命的消耗转为正
            数因为即使一个人天天锻炼身体，天天吃营养品，也还是会有自然死亡的那
            一天
        """
        # *2-1的目的:regular_mm在0-1之间,HealthSeekDay要结果在－1，1之间
        self.living_factor = regular_mm(living_days) * 2 - 1
        # 结果在1-0之间 [::-1]: 将0->1转换到1->0
        self.happiness_factor = regular_mm(days)[::-1]


def sample_621_1():
    """
    6.2.1_1 你一生的故事：HealthSeekDay
    :return:
    """
    # 初始化我
    me = Person()
    # 初始化追求健康长寿快乐
    seek_health = HealthSeekDay()
    while me.living > 0:
        # 只要还活着，就追求健康长寿快乐
        me.live_one_day(seek_health)

    print('只追求健康长寿快乐活了{}年，幸福指数{},积累财富{},名望权力{}'.format
          (round(me.living_day / 365, 2), round(me.happiness, 2),
           me.wealth, me.fame))

    plt.plot(seek_health.living_factor * seek_health.living_consume)
    plt.plot(seek_health.happiness_factor * seek_health.happiness_base)
    plt.legend(['living_factor', 'happiness_factor'], loc='best')
    plt.show()


"""
    StockSeekDay
"""


class StockSeekDay(BaseSeekDay):
    """
        StockSeekDay追求财富金钱的一天:
        形象：做股票投资赚钱的事情。
        抽象：追求财富金钱
    """

    def _init_self(self, show=False):
        # 每天对生命消耗的常数＝2，即代表2天
        self.living_consume = 2
        # 每天幸福指数常数＝0.5
        self.happiness_base = 0.5
        # 财富积累常数＝10，默认＝0
        self.wealth_base = 10
        # 设定可变因素序列
        self._gen_living_days()

    def _gen_living_days(self):
        # 只生成10000个序列
        days = np.arange(1, 10000)
        # 针对生命消耗living_factor的基础函数还是sqrt
        living_days = np.sqrt(days)
        # 由于不需要像HealthSeekDay从负数开始，所以直接regular_mm 即:0->1
        self.living_factor = regular_mm(living_days)

        # 针对幸福感可变序列使用了np.power4，即变化速度比sqrt快
        happiness_days = np.power(days, 4)
        # 幸福指数可变因素会快速递减由1->0
        self.happiness_factor = regular_mm(happiness_days)[::-1]

        """
            这里简单设定wealth_factor=living_factor
            living_factor(0-1), 导致wealth_factor(0-1), 即财富积累越到
            后面越有效率，速度越快，头一个100万最难赚
        """
        self.wealth_factor = self.living_factor


def sample_621_2():
    """
    6.2.1_2 你一生的故事：StockSeekDay
    :return:
    """
    # 初始化我
    me = Person()
    # 初始化追求财富金钱
    seek_stock = StockSeekDay()
    while me.living > 0:
        # 只要还活着，就追求财富金钱
        me.live_one_day(seek_stock)

    print('只追求财富金钱活了{}年，幸福指数{}, 积累财富{}, 名望权力{}'.format
          (round(me.living_day / 365, 2), round(me.happiness, 2),
           round(me.wealth, 2), me.fame))
    plt.plot(seek_stock.living_factor * seek_stock.living_consume)
    plt.plot(seek_stock.happiness_factor * seek_stock.happiness_base)
    plt.legend(['living_factor', 'happiness_factor'], loc='best')
    plt.show()


"""
    FameSeekDay
"""


class FameSeekDay(BaseSeekDay):
    """
        FameTask追求名望权力的一天:
        追求名望权力
    """

    def _init_self(self):
        # 每天对生命消耗的常数＝3，即代表3天
        self.living_consume = 3
        # 每天幸福指数常数＝0.6
        self.happiness_base = 0.6
        # 名望权利积累常数＝10，默认＝0
        self.fame_base = 10
        # 设定可变因素序列
        self._gen_living_days()

    def _gen_living_days(self):
        # 只生成12000个序列
        days = np.arange(1, 12000)
        # 针对生命消耗living_factor的基础函数还是sqrt
        living_days = np.sqrt(days)
        # 由于不需要像HealthSeekDay从负数开始，所以直接regular_mm 即:0->1
        self.living_factor = regular_mm(living_days)

        # 针对幸福感可变序列使用了np.power2
        # 即变化速度比StockSeekDay慢但比HealthSeekDay快
        happiness_days = np.power(days, 2)
        # 幸福指数可变因素递减由1->0
        self.happiness_factor = regular_mm(happiness_days)[::-1]

        # 这里简单设定fame_factor=living_factor
        self.fame_factor = self.living_factor


def sample_621_3():
    """
    6.2.1_3 你一生的故事：FameSeekDay
    :return:
    """
    # 初始化我
    me = Person()
    # 初始化追求名望权力
    seek_fame = FameSeekDay()
    while me.living > 0:
        # 只要还活着，就追求名望权力
        me.live_one_day(seek_fame)

    print('只追求名望权力活了{}年，幸福指数{}, 积累财富{}, 名望权力{}'.format
          (round(me.living_day / 365, 2), round(me.happiness, 2),
           round(me.wealth, 2), round(me.fame, 2)))

    plt.plot(seek_fame.living_factor * seek_fame.living_consume)
    plt.plot(seek_fame.happiness_factor * seek_fame.happiness_base)
    plt.legend(['living_factor', 'happiness_factor'], loc='best')
    plt.show()


"""
    6.2.2 使用蒙特卡洛方法计算怎样度过一生最幸福
"""


def my_life(weights):
    """
        追求健康长寿快乐的权重:weights[0]
        追求财富金钱的权重:weights[1]
        追求名望权力的权重:weights[2]
    """
    # 追求健康长寿快乐
    seek_health = HealthSeekDay()
    # 追求财富金钱
    seek_stock = StockSeekDay()
    # 追求名望权力
    seek_fame = FameSeekDay()

    # 放在一个list中对对应下面np.random.choice中的index[0, 1, 2]
    seek_list = [seek_health, seek_stock, seek_fame]

    # 初始化我
    me = Person()
    # 加权随机抽取序列。80000天肯定够了, 80000天快220年了。。。
    seek_choice = np.random.choice([0, 1, 2], 80000, p=weights)

    while me.living > 0:
        # 追求从加权随机抽取序列已经初始化好的
        seek_ind = seek_choice[me.living_day]
        seek = seek_list[seek_ind]
        # 只要还活着，就追求
        me.live_one_day(seek)
    return round(me.living_day / 365, 2), round(me.happiness, 2), round(me.wealth, 2), round(me.fame, 2)


def sample_622():
    """
    6.2.2 使用蒙特卡洛方法计算怎样度过一生最幸福
    :return:
    """
    living_day, happiness, wealth, fame = my_life([0.4, 0.3, 0.3])
    print('活了{}年，幸福指数{}, 积累财富{}, 名望权力{}'.format(
        living_day, happiness, wealth, fame))

    from abupy import AbuProgress
    progress = AbuProgress(2000, 0, label='my_life...')

    result = []
    for pos, _ in enumerate(xrange(2000)):
        # 2000次随机权重分配
        weights = np.random.random(3)
        weights /= np.sum(weights)
        # result中：tuple[0]权重weights,，tuple[1]my_life返回的结果
        result.append((weights, my_life(weights)))
        progress.show(a_progress=pos + 1)

    # result中tuple[1]=my_life返回的结果, my_life[1]=幸福指数，so->x[1][1]
    sorted_scores = sorted(result, key=lambda p_x: p_x[1][1], reverse=True)
    # 将最优权重sorted_scores[0][0]代入my_life得到结果
    living_day, happiness, wealth, fame = my_life(sorted_scores[0][0])

    print('活了{}年，幸福指数{}, 积累财富{}, 名望权力{}'.format
          (living_day, happiness, wealth, fame))

    print('人生最优权重：追求健康{:.3f},追求财富{:.3f},追求名望{:.3f}'.format(
        sorted_scores[0][0][0], sorted_scores[0][0][1],
        sorted_scores[0][0][2]))

    # noinspection PyUnresolvedReferences
    from mpl_toolkits.mplot3d import Axes3D
    """
        result中: tuple[0]权重weights, tuple[1]my_life返回的结果
        r[0][0]: 追求健康长寿快乐的权重
        r[0][1]: 追求财富金钱的权重
        r[0][2]: 追求名望权力的权重
        r[1][1]: my_life[1]=幸福指数
    """
    result_show = np.array(
        [[r[0][0], r[0][1], r[0][2], r[1][1]] for r in result])

    fig = plt.figure(figsize=(9, 6))
    ax = fig.gca(projection='3d')
    ax.view_init(30, 60)
    """
        x:追求健康长寿快乐的权重, y:追求财富金钱的权重
        z:追求名望权力的权重, c:color 幸福指数, 颜色越深越幸福
    """
    ax.scatter3D(result_show[:, 0], result_show[:, 1], result_show[:, 2],
                 c=result_show[:, 3], cmap='spring')
    ax.set_xlabel('health')
    ax.set_ylabel('stock')
    ax.set_zlabel('fame')
    plt.show()

    # 幸福指数
    happiness_result = result_show[:, 3]
    # 使用qcut分10份
    print('pd.qcut(happiness_result, 10).value_counts():\n', pd.qcut(happiness_result, 10).value_counts())


"""
    6.2.3 凸优化基础概念
"""


# noinspection PyTypeChecker
def sample_623():
    """
    6.2.3 趋势骨架图
    :return:
    """
    import scipy.optimize as sco
    from scipy.interpolate import interp1d

    # 继续使用TSLA收盘价格序列
    # interp1d线性插值函数
    linear_interp = interp1d(x, y)
    # 绘制插值
    plt.plot(linear_interp(x))

    # fminbound寻找给定范围内的最小值：在linear_inter中寻找全局最优范围1－504
    global_min_pos = sco.fminbound(linear_interp, 1, 504)
    # 绘制全局最优点，全局最小值点，r<：红色三角
    plt.plot(global_min_pos, linear_interp(global_min_pos), 'r<')

    # 每个单位都先画一个点，由两个点连成一条直线形成股价骨架图
    last_postion = None
    # 步长50，每50个单位求一次局部最小
    for find_min_pos in np.arange(50, len(x), 50):
        # fmin_bfgs寻找给定值的局部最小值
        local_min_pos = sco.fmin_bfgs(linear_interp, find_min_pos, disp=0)
        # 形成最小点位置信息(x, y)
        draw_postion = (local_min_pos, linear_interp(local_min_pos))
        # 第一个50单位last_postion＝none, 之后都有值
        if last_postion is not None:
            # 将两两临近局部最小值相连，两个点连成一条直线
            plt.plot([last_postion[0][0], draw_postion[0][0]],
                     [last_postion[1][0], draw_postion[1][0]], 'o-')
        # 将这个步长单位内的最小值点赋予last_postion
        last_postion = draw_postion
    plt.show()


def sample_624():
    """
    6.2.4 全局最优求解怎样度过一生最幸福
    :return:
    """
    import scipy.optimize as sco

    def minimize_happiness_global(weights):
        if np.sum(weights) != 1:
            # 过滤权重和不等于1的权重组合
            return 0
        # 最优都是寻找最小值，所以要得到幸福指数最大的权重，
        # 返回-my_life，这样最小的结果其实是幸福指数最大的权重配比
        return -my_life(weights)[1]

    opt_global = sco.brute(minimize_happiness_global,
                           ((0, 1.1, 0.1), (0, 1.1, 0.1), (0, 1.1, 0.1)))
    print(opt_global)

    living_day, happiness, wealth, fame = my_life(opt_global)
    print('活了{}年，幸福指数{}, 积累财富{}, 名望权力{}'.format
          (living_day, happiness, wealth, fame))


# noinspection PyTypeChecker
def sample_625():
    """
    6.2.5 非凸函数计算怎样度过一生最幸福
    :return:
    """
    import scipy.optimize as sco

    method = 'SLSQP'
    # 提供一个函数来规范参数,np.sum(weights) = 1 -> np.sum(weights) - 1 = 0
    constraints = ({'type': 'eq', 'fun': lambda p_x: np.sum(p_x) - 1})
    # 参数的范围选定
    bounds = tuple((0, 0.9) for _ in xrange(3))
    print('bounds:', bounds)

    def minimize_happiness_local(weights):
        # print(weights)
        return -my_life(weights)[1]

    # 初始化猜测最优参数，这里使用brute计算出的全局最优参数作为guess
    guess = [0.5, 0.2, 0.3]
    opt_local = sco.minimize(minimize_happiness_local, guess,
                             method=method, bounds=bounds,
                             constraints=constraints)
    print('opt_local:', opt_local)


# noinspection PyShadowingNames
def sample_626():
    """
    6.2.6 标准凸函数求最优
    :return:
    """
    import scipy.optimize as sco

    fig = plt.figure()
    from mpl_toolkits.mplot3d import Axes3D
    ax = Axes3D(fig)
    x = np.arange(-10, 10, 0.5)
    y = np.arange(-10, 10, 0.5)
    x_grid, y_grid = np.meshgrid(x, y)
    # z^2 = x^2 + y^2
    z_grid = x_grid ** 2 + y_grid ** 2

    ax.plot_surface(x_grid, y_grid, z_grid, rstride=1, cstride=1,
                    cmap='hot')
    plt.show()

    def convex_func(xy):
        return xy[0] ** 2 + xy[1] ** 2

    bounds = ((-10, 10), (-10, 10))
    guess = [5, 5]
    for method in ['SLSQP', 'TNC', 'L-BFGS-B']:
        # 打印start
        print(method + ' start')
        # noinspection PyTypeChecker
        ret = sco.minimize(convex_func, guess, method=method,
                           bounds=bounds)
        print(ret)
        # 这里通过np.allclose判定结果是不是（0， 0）
        print('result is (0, 0): {}'.format(
            np.allclose(ret['x'], [0., 0.], atol=0.001)))
        # 打印end
        print(method + ' end')


"""
    6.3 线性代数
"""

# 获取多支股票数据组成panel
my_stock_df = ABuSymbolPd.make_kl_df(
    ['usBIDU', 'usGOOG', 'usFB', 'usAAPL', 'us.IXIC'], n_folds=2)
# 变换轴向，形成新的切面
my_stock_df = my_stock_df.swapaxes('items', 'minor')
my_stock_df_close = my_stock_df['close'].dropna(axis=0)


def regular_std(group):
    # z-score规范化也称零-均值规范化
    return (group - group.mean()) / group.std()


def sample_630():
    """
    获取多支股票数据组成panel
    :return:
    """
    print('my_stock_df_close.tail():\n', my_stock_df_close.tail())

    my_stock_df_close_std = regular_std(my_stock_df_close)
    my_stock_df_close_std.plot()
    plt.show()


def sample_631():
    """
    6.3.1 矩阵基础知识
    :return:
    """
    from scipy import linalg

    # dataframe转换matrix通过as_matrix
    cs_matrix = my_stock_df_close.as_matrix()
    # cs_matrix本身有5列数据(5支股票)，要变成方阵即保留5行数据0:5
    cs_matrix = cs_matrix[0:5, :]
    print('cs_matrix.shape:', cs_matrix.shape)
    print('cs_matrix:\n', cs_matrix)

    eye5 = np.eye(5)
    print(eye5)

    cs_matrix_inv = linalg.inv(cs_matrix)
    print('逆矩阵: cs_matrix_inv')
    print(cs_matrix_inv)
    # 上面打印cs_matrix_inv输出上并非绝对标准单位矩阵，是对角线值元素接近与1，非对
    # 角线元素接近与0的矩阵，需要使用np.allclose来确认结果
    print('相乘后的结果是单位矩阵：{}'.format(
        np.allclose(np.dot(cs_matrix, cs_matrix_inv), eye5)))


def sample_632():
    """
    6.3.2 特征值和特征向量
    :return:
    """
    from scipy import mat, linalg

    a = mat('[1.5 -0.5; -0.5 1.5]')
    u, d = linalg.eig(a)
    print('特征值向量：{}'.format(u))
    print('特征向量（列向量）矩阵：{}'.format(d))


def sample_634():
    """
    6.3.4 PCA和SVD使用实例
    :return:
    """
    from sklearn.decomposition import PCA

    my_stock_df_close_std = regular_std(my_stock_df_close)
    # n_components=1只保留一个维度
    pca = PCA(n_components=1)
    # 稍后会有展示fit_transform的实现，以及关键核心代码抽取
    my_stock_df_trans_pca = \
        pca.fit_transform(my_stock_df_close_std.as_matrix())

    plt.plot(my_stock_df_trans_pca)
    plt.show()

    # 可视化维度和主成分关系，参数空
    pca = PCA()
    # 直接使用fit，不用fit_transform
    pca.fit(my_stock_df_close_std)

    # x:保留的维度 y:保留的维度下的方差比总和即保留了多少主成分
    plt.plot(np.arange(1, len(pca.explained_variance_ratio_) + 1),
             np.cumsum(pca.explained_variance_ratio_))
    plt.xlabel('component')
    plt.ylabel('explained variance')
    plt.show()

    # 0.95即保留95%主成分
    pca = PCA(0.95)
    # 稍后会有展示fit_transform的实现，以及关键核心代码抽取
    my_stock_df_trans_pca = \
        pca.fit_transform(my_stock_df_close_std.as_matrix())
    plt.plot(my_stock_df_trans_pca)
    plt.show()

    # noinspection PyPep8Naming
    def my_pca(n_components=1):
        from scipy import linalg

        # svd奇异值分解
        U, S, V = linalg.svd(my_stock_df_close_std.as_matrix(),
                             full_matrices=False)
        # 通过n_components进行降维
        U = U[:, :n_components]
        U *= S[:n_components]
        # 绘制降维后的矩阵
        plt.plot(U)

    # 输出如图6－19所示
    my_pca(n_components=3)
    plt.show()


if __name__ == "__main__":
    sample_611_1()
    # sample_611_2()
    # sample_612()
    # sample_613()
    # sample_621_1()
    # sample_621_2()
    # sample_621_3()
    # sample_622()
    # sample_623()
    # sample_624()
    # sample_625()
    # sample_626()
    # sample_630()
    # sample_631()
    # sample_632()
    # sample_634()
