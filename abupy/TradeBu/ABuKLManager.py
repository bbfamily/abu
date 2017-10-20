# -*- encoding:utf-8 -*-
"""
    金融时间序列管理模块
"""

from __future__ import print_function
from __future__ import absolute_import
from __future__ import division

import logging

from ..TradeBu import AbuBenchmark
from ..UtilBu import ABuDateUtil
from ..CoreBu.ABuEnv import EMarketDataSplitMode, EMarketDataFetchMode
from ..MarketBu import ABuSymbolPd
from ..MarketBu.ABuMarket import split_k_market
from ..CoreBu.ABuEnvProcess import add_process_env_sig, AbuEnvProcess
from ..CoreBu.ABuParallel import delayed, Parallel
from ..CoreBu import ABuEnv
from ..CoreBu.ABuEnv import EDataCacheType
from ..UtilBu.ABuProgress import AbuMulPidProgress
from ..UtilBu.ABuFileUtil import batch_h5s
# noinspection PyUnresolvedReferences
from ..CoreBu.ABuFixes import filter

__author__ = '阿布'
__weixin__ = 'abu_quant'


# noinspection PyUnusedLocal
@add_process_env_sig
def gen_dict_pick_time_kl_pd(target_symbols, capital, benchmark, show_progress=True):
    """
    在AbuKLManager中batch_get_pick_time_kl_pd批量获取择时时间序列中使用做为并行多进程委托方法
    :param target_symbols: 请求的symbol
    :param capital: 资金类AbuCapital实例化对象 （实现中暂时不使用其中信息）
    :param benchmark: 交易基准对象，AbuBenchmark实例对象
    :param show_progress: 是否显示ui进度条
    """

    # 构建的返回时间序列交易数据组成的字典
    pick_kl_pd_dict = dict()

    # 为batch_h5s装饰器准备参数，详见batch_h5装饰器实现
    h5s_fn = None
    if ABuEnv.g_data_cache_type == EDataCacheType.E_DATA_CACHE_HDF5 and ABuEnv.g_data_fetch_mode == \
            EMarketDataFetchMode.E_DATA_FETCH_FORCE_LOCAL:
        # 存储使用hdf5且使用本地数据模式才赋予h5s_fn路径
        # noinspection PyProtectedMember
        h5s_fn = ABuEnv.g_project_kl_df_data

    @batch_h5s(h5s_fn)
    def _batch_gen_dict_pick_time_kl_pd():
        # 启动多进程进度条
        with AbuMulPidProgress(len(target_symbols), 'gen kl_pd complete', show_progress=show_progress) as progress:
            for epoch, target_symbol in enumerate(target_symbols):
                progress.show(epoch + 1)
                # 迭代target_symbols，获取对应时间交易序列
                kl_pd = ABuSymbolPd.make_kl_df(target_symbol, data_mode=EMarketDataSplitMode.E_DATA_SPLIT_UNDO,
                                               benchmark=benchmark, n_folds=benchmark.n_folds)
                # 以target_symbol为key将时间金融序列kl_pd添加到返回字典中
                pick_kl_pd_dict[target_symbol] = kl_pd
    _batch_gen_dict_pick_time_kl_pd()
    return pick_kl_pd_dict


class AbuKLManager(object):
    """金融时间序列管理类"""

    def __init__(self, benchmark, capital):
        """
        :param benchmark: 交易基准对象，AbuBenchmark实例对象
        :param capital: 资金类AbuCapital实例化对象
        """
        self.benchmark = benchmark
        self.capital = capital
        # 选股时间交易序列字典
        pick_stock_kl_pd_dict = dict()
        # 择时时间交易序列字典
        pick_time_kl_pd_dict = dict()
        # 类字典pick_kl_pd_dict将选股和择时字典包起来
        self.pick_kl_pd_dict = {'pick_stock': pick_stock_kl_pd_dict, 'pick_time': pick_time_kl_pd_dict}

    def __str__(self):
        """打印对象显示：pick_stock + pick_time keys, 即所有symbol信息"""
        keys = set(self.pick_kl_pd_dict['pick_stock'].keys()) | set(self.pick_kl_pd_dict['pick_time'].keys())
        return 'pick_stock + pick_time keys :{}'.format(keys)

    __repr__ = __str__

    def __len__(self):
        """对象长度：选股字典长度 + 择时字典长度"""
        return len(self.pick_kl_pd_dict['pick_stock']) + len(self.pick_kl_pd_dict['pick_time'])

    def __contains__(self, item):
        """成员测试：在择时字典中或者在选股字典中"""
        return item in self.pick_kl_pd_dict['pick_stock'] or item in self.pick_kl_pd_dict['pick_time']

    def __missing__(self, key):
        """对象缺失：需要根据key使用code_to_symbol进行fetch数据，暂未实现"""
        # TODO 需要根据key使用code_to_symbol进行fetch数据
        raise NotImplementedError('TODO AbuKLManager __missing__')

    def __getitem__(self, key):
        """索引获取：尝试分别从选股字典，择时字典中查询，返回两个字典的查询结果"""
        pick_stock_item = None
        if key in self.pick_kl_pd_dict['pick_stock']:
            pick_stock_item = self.pick_kl_pd_dict['pick_stock'][key]
        pick_time_item = None
        if key in self.pick_kl_pd_dict['pick_time']:
            pick_time_item = self.pick_kl_pd_dict['pick_time'][key]
        return pick_stock_item, pick_time_item

    def __setitem__(self, key, value):
        """索引设置：抛错误，即不准许外部设置"""
        raise AttributeError("AbuKLManager set value!!!")

    def _fetch_pick_stock_kl_pd(self, xd, target_symbol):
        """
        根据选股周期和symbol获取选股时段金融时间序列，相对择时金融时间序列获取要复杂，
        因为要根据条件构造选股时段benchmark，且在类变量中存储选股时段benchmark
        :param xd: 选股周期（默认一年的交易日长度）
        :param target_symbol: 选股symbol
        :return: 选股时段金融时间序列
        """

        # 从设置的择时benchmark中取第一个日期即为选股时段最后一个日期
        end = ABuDateUtil.timestamp_to_str(self.benchmark.kl_pd.index[0])

        if xd == ABuEnv.g_market_trade_year:
            # 一般都是默认的1年，不需要使用begin_date提高效率
            n_folds = 1
            pre_bc_key = 'pre_benchmark_{}'.format(n_folds)
            start = None
        else:
            # 1年除1年交易日数量，浮点数n_folds eg: 0.88
            n_folds = float(xd / ABuEnv.g_market_trade_year)
            # 为了计算start，xd的单位是交易日，换算为自然日
            delay_day = 365 * n_folds
            start = ABuDateUtil.begin_date(delay_day, date_str=end, fix=False)
            # 根据选股start，end拼接选股类变量key，eg：pre_benchmark_2011-09-09_2016-07-26
            pre_bc_key = 'pre_benchmark_{}-{}'.format(start, end)
        if hasattr(self, pre_bc_key):
            # 从类变量中直接获取选股benchmark，eg: self.pre_benchmark_2011-09-09_2016-07-26
            pre_benchmark = getattr(self, pre_bc_key)
        else:
            # 类变量中没有，实例一个AbuBenchmark，根据n_folds和end获取benchmark选股时段
            pre_benchmark = AbuBenchmark(n_folds=n_folds, start=start, end=end)
            # 类变量设置选股时段benchmark
            setattr(self, pre_bc_key, pre_benchmark)
        # 以选股时段benchmark做为参数，获取选股时段对应symbol的金融时间序列
        return ABuSymbolPd.make_kl_df(target_symbol, data_mode=EMarketDataSplitMode.E_DATA_SPLIT_UNDO,
                                      benchmark=pre_benchmark, n_folds=pre_benchmark.n_folds, start=start, end=end)

    def _fetch_pick_time_kl_pd(self, target_symbol):
        """获取择时时段金融时间序列"""
        return ABuSymbolPd.make_kl_df(target_symbol, data_mode=EMarketDataSplitMode.E_DATA_SPLIT_UNDO,
                                      benchmark=self.benchmark, n_folds=self.benchmark.n_folds)

    def get_pick_time_kl_pd(self, target_symbol):
        """对外获取择时时段金融时间序列，首先在内部择时字典中寻找，没找到使用_fetch_pick_time_kl_pd获取，且保存择时字典"""
        if target_symbol in self.pick_kl_pd_dict['pick_time']:
            kl_pd = self.pick_kl_pd_dict['pick_time'][target_symbol]
            if kl_pd is not None:
                # 因为在多进程的时候拷贝会丢失name信息
                kl_pd.name = target_symbol
            return kl_pd
        # 字典中每找到，进行fetch，获取后保存在择时字典中
        kl_pd = self._fetch_pick_time_kl_pd(target_symbol)
        self.pick_kl_pd_dict['pick_time'][target_symbol] = kl_pd
        return kl_pd

    def filter_pick_time_choice_symbols(self, choice_symbols):
        """
        使用filter筛选出choice_symbols中的symbol对应的择时时间序列不在内部择时字典中的symbol序列
        :param choice_symbols: 支持迭代的symbol序列
        :return: 不在内部择时字典中的symbol序列
        """
        return list(filter(lambda target_symbol: target_symbol not in self.pick_kl_pd_dict['pick_time'],
                           choice_symbols))

    def batch_get_pick_time_kl_pd(self, choice_symbols, n_process=ABuEnv.g_cpu_cnt, show_progress=True):
        """
        统一批量获取择时金融时间序列获保存在内部的择时字典中，以多进程并行方式运行
        :param choice_symbols: 支持迭代的symbol序列
        :param n_process: 择时金融时间序列获取并行启动的进程数，默认16个，属于io操作多，所以没有考虑cpu数量
        :param show_progress: 是否显示ui进度条
        """
        if len(choice_symbols) == 0:
            return

        if n_process <= 0:
            # 因为下面要n_process > 1做判断而且要根据n_process来split_k_market
            n_process = ABuEnv.g_cpu_cnt

        # TODO 需要区分hdf5和csv不同存贮情况，csv存贮模式下可以并行读写
        # 只有E_DATA_FETCH_FORCE_LOCAL才进行多任务模式，否则回滚到单进程模式n_process = 1
        if n_process > 1 and ABuEnv.g_data_fetch_mode != EMarketDataFetchMode.E_DATA_FETCH_FORCE_LOCAL:
            # 1. hdf5多进程还容易写坏数据
            # 2. MAC OS 10.9 之后并行联网＋numpy 系统bug crash，卡死等问题
            logging.info('batch get only support E_DATA_FETCH_FORCE_LOCAL for Parallel!')
            n_process = 1

        # 根据输入的choice_symbols和要并行的进程数，分配symbol到n_process个进程中
        process_symbols = split_k_market(n_process, market_symbols=choice_symbols)

        # 因为切割会有余数，所以将原始设置的进程数切换为分割好的个数, 即32 -> 33 16 -> 17
        if n_process > 1:
            n_process = len(process_symbols)

        parallel = Parallel(
            n_jobs=n_process, verbose=0, pre_dispatch='2*n_jobs')

        # gen_dict_pick_time_kl_pd被装饰器add_process_env_sig装饰，需要进程间内存拷贝对象AbuEnvProcess，详ABuEnvProcess.py
        p_nev = AbuEnvProcess()
        # 开始并行任务执行
        out_pick_kl_pd_dict = parallel(delayed(gen_dict_pick_time_kl_pd)(target_symbols, self.capital, self.benchmark,
                                                                         show_progress=show_progress,
                                                                         env=p_nev)
                                       for target_symbols in process_symbols)

        for pick_kl_pd_dict in out_pick_kl_pd_dict:
            # 迭代多任务组成的out_pick_kl_pd_dict，分别更新保存在内部的择时字典中
            self.pick_kl_pd_dict['pick_time'].update(pick_kl_pd_dict)

    def get_pick_stock_kl_pd(self, target_symbol, xd=ABuEnv.g_market_trade_year,
                             min_xd=int(ABuEnv.g_market_trade_year / 2)):
        """
        对外获取选股时段金融时间序列，首先在内部择时字典中寻找，没找到使用_fetch_pick_stock_kl_pd获取，且保存选股字典
        :param target_symbol: 选股symbol
        :param xd: 选股周期（默认一年的交易日长度）
        :param min_xd: 对fetch的选股金融序列进行过滤参数，即最小金融序列长度
        :return:
        """

        if target_symbol in self.pick_kl_pd_dict['pick_stock']:
            xd_dict = self.pick_kl_pd_dict['pick_stock'][target_symbol]
            if xd in xd_dict:
                # 缓存中找到形如：self.pick_kl_pd_dict['pick_stock']['usTSLA']['252']
                # noinspection PyTypeChecker
                kl_pd = xd_dict[xd]
                if kl_pd is not None:
                    # 因为在多进程的时候深拷贝会丢失name
                    kl_pd.name = target_symbol
                return kl_pd

        # 字典中每找到，进行fetch
        kl_pd = self._fetch_pick_stock_kl_pd(xd, target_symbol)
        """选股字典是三层字典结构，比择时字典多一层，因为有选股周期做为第三层字典的key"""
        if kl_pd is None or kl_pd.shape[0] == 0:
            self.pick_kl_pd_dict['pick_stock'][target_symbol] = {xd: None}
            return None

        """由于_fetch_pick_stock_kl_pd中获取kl_pd使用了标尺模式，所以这里的min_xd要设置大于标尺才有实际意义"""
        if kl_pd.shape[0] < min_xd:
            # 如果时间序列有数据但是 < min_xd, 抛弃数据直接{xd: None}
            self.pick_kl_pd_dict['pick_stock'][target_symbol] = {xd: None}
            return None
        # 第三层字典{xd: kl_pd}
        self.pick_kl_pd_dict['pick_stock'][target_symbol] = {xd: kl_pd}
        return kl_pd
