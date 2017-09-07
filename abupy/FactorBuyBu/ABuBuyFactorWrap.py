# -*- encoding:utf-8 -*-
"""买入因子类装饰器模块"""

from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

from ..CoreBu.ABuFixes import six
from ..TLineBu.ABuTL import AbuTLine

__author__ = '阿布'
__weixin__ = 'abu_quant'


class AbuLeastPolyWrap(object):
    """示例做为买入因子策略装饰器封装show_least_valid_poly对大盘震荡大的情况下封锁交易"""

    def __call__(self, cls):
        """只做为买入因子策略类的装饰器"""

        if isinstance(cls, six.class_types):
            # 只做为类装饰器使用

            init_self = cls._init_self
            org_fit_day = cls.fit_day

            # fit_month不是必须实现的
            org_fit_month = getattr(cls, 'fit_month', None)

            def init_self_wrapped(*args, **kwargs):
                # 拿出被装饰的self对象
                warp_self = args[0]
                # 外部可以设置poly阀值，self.poly在fit_month中和每一个月大盘计算的poly比较，
                # 若是大盘的poly大于poly认为走势震荡
                warp_self.poly = kwargs.pop('poly', 2)
                # 是否封锁买入策略进行择时交易
                warp_self.lock = False
                # 调用原始的_init_self
                init_self(*args, **kwargs)

            def fit_day_wrapped(*args, **kwargs):
                # 拿出被装饰的self对象
                warp_self = args[0]
                if warp_self.lock:
                    # 如果封锁策略进行交易的情况下，策略不进行择时
                    return None
                return org_fit_day(*args, **kwargs)

            def fit_month_wrapped(*args, **kwargs):
                warp_self = args[0]
                today = args[1]
                # fit_month即在回测策略中每一个月执行一次的方法
                # 策略中拥有self.benchmark，即交易基准对象，AbuBenchmark实例对象，benchmark.kl_pd即对应的市场大盘走势
                benchmark_df = warp_self.benchmark.kl_pd
                # 拿出大盘的今天
                benchmark_today = benchmark_df[benchmark_df.date == today.date]
                if benchmark_today.empty:
                    return 0
                # 要拿大盘最近一个月的走势，准备切片的start，end
                end_key = int(benchmark_today.iloc[0].key)
                start_key = end_key - 20
                if start_key < 0:
                    return 0

                # 使用切片切出从今天开始向前20天的数据
                benchmark_month = benchmark_df[start_key:end_key + 1]
                # 通过大盘最近一个月的收盘价格做为参数构造AbuTLine对象
                benchmark_month_line = AbuTLine(benchmark_month.close, 'benchmark month line')
                # 计算这个月最少需要几次拟合才能代表走势曲线
                least = benchmark_month_line.show_least_valid_poly(show=False)

                if least >= warp_self.poly:
                    # 如果最少的拟合次数大于阀值self.poly，说明走势成立，大盘非震荡走势，解锁交易
                    warp_self.lock = False
                else:
                    # 如果最少的拟合次数小于阀值self.poly，说明大盘处于震荡走势，封锁策略进行交易
                    warp_self.lock = True

                if org_fit_month is not None:
                    return org_fit_month(*args, **kwargs)

            cls._init_self = init_self_wrapped
            init_self_wrapped.__name__ = '_init_self'

            cls.fit_day = fit_day_wrapped
            fit_day_wrapped.__name__ = 'fit_day'

            cls.fit_month = fit_month_wrapped
            fit_month_wrapped.__name__ = 'fit_month'

            return cls
        else:
            raise TypeError('AbuLeastPolyWrap just for class warp')
