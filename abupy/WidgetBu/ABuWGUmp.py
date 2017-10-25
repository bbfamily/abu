# -*- encoding:utf-8 -*-
"""ump回测裁判训练以及交易预测拦截图形可视化"""

from __future__ import print_function
from __future__ import absolute_import
from __future__ import division

import os
import datetime
from contextlib import contextmanager

import ipywidgets as widgets

from ..WidgetBu.ABuWGBase import WidgetBase, show_msg_toast_func, permission_denied
# noinspection PyUnresolvedReferences
from ..CoreBu.ABuFixes import filter
from ..CoreBu.ABuStore import load_custom_abu_index, load_custom_ump_index, delete_abu_result_tuple
from ..CoreBu import ABuEnv
from ..CoreBu.ABuStore import dump_custom_ump_index_csv, del_custom_ump_index
from ..CoreBu.ABu import load_abu_result_tuple, store_abu_result_tuple
from ..CoreBu.ABuStore import EStoreAbu, dump_custom_abu_index_csv
from ..TradeBu import ABuMLFeature
from ..TradeBu.ABuMLFeature import AbuFeatureDegExtend
from ..UtilBu.ABuStrUtil import to_unicode
from ..UtilBu.ABuFileUtil import del_file

from ..UmpBu.ABuUmpMainDeg import AbuUmpMainDeg, AbuUmpMainDegExtend
from ..UmpBu.ABuUmpMainPrice import AbuUmpMainPrice
from ..UmpBu.ABuUmpMainMul import AbuUmpMainMul

from ..UmpBu.ABuUmpEdgeDeg import AbuUmpEdgeDeg, AbuUmpEegeDegExtend
from ..UmpBu.ABuUmpEdgePrice import AbuUmpEdgePrice
from ..UmpBu.ABuUmpEdgeMul import AbuUmpEdgeMul
from ..UmpBu import ABuUmpManager

__author__ = '阿布'
__weixin__ = 'abu_quant'


# noinspection PyProtectedMember
class WidgetUmp(WidgetBase):
    """回测裁判训练以及交易预测拦截ui类"""

    # noinspection PyProtectedMember
    def __init__(self):
        """构建回测需要的各个组件形成tab"""
        sub_widget_tab = widgets.Tab()
        # 裁判特征采集界面初始化
        feature_tab = self._init_feature_ui()
        # 裁判特征训练界面初始化
        train_tab = self._init_train_ui()
        # 裁判预测拦截界面初始化
        predict_tab = self._init_predict_ui()
        # 裁判数据管理界面初始化
        manager_tab = self._init_manager_ui()

        sub_widget_tab.children = [feature_tab, train_tab, predict_tab, manager_tab]
        for ind, name in enumerate([u'裁判特征采集', u'裁判特征训练', u'裁判预测拦截', u'裁判数据管理']):
            sub_widget_tab.set_title(ind, name)
        self.widget = widgets.VBox([sub_widget_tab])

    def load_abu_result(self):
        """读取回测结果的索引描述csv文件"""
        index_csv_df = load_custom_abu_index()
        train_options = list()
        if index_csv_df is not None:
            train_options = [u'{}. {}:{}'.format(ind + 1, index_csv_df.index[ind], to_unicode(description)) for
                             ind, description
                             in enumerate(index_csv_df.description)]
        self.abu_result.options = train_options

    def load_train_ump(self, ump_select_ui):
        """读取已经完成训练的ump裁判的本地索引描述csv文件"""
        index_csv_df = load_custom_ump_index()
        ump_options = list()
        if index_csv_df is not None:
            ump_options = [u'{}. {}:{}:{}'.format(
                ind + 1, u'主裁' if ump_type_key == 'main' else u'边裁', index_csv_df.index[ind],
                to_unicode(description)) for
                           ind, (ump_type_key, description) in
                           enumerate(zip(index_csv_df.is_main_ump, index_csv_df.description))]
        ump_select_ui.options = ump_options

    def run_before(self):
        """在回测模块开始回测前调用，根据回测中是否开启特征记录，以及是否使用裁判进行预测交易拦截对回测进行设置"""
        # 先clear一下ABuMLFeature和ABuUmpManager
        ABuMLFeature.clear_user_feature()
        ABuUmpManager.clear_user_ump()

        if self.choice_umps.options is not None and len(self.choice_umps.options) > 0:
            # 有选择使用裁判对交易结果进行人工拦截干预, 打开生成回测交易特征开关
            self.enable_ml_feature.value = 1

            # 打开使用用户自定义裁判开关
            ABuUmpManager.g_enable_user_ump = True

            ump_class_dict = {AbuUmpMainDeg.class_unique_id(): AbuUmpMainDeg,
                              AbuUmpMainPrice.class_unique_id(): AbuUmpMainPrice,
                              AbuUmpMainMul.class_unique_id(): AbuUmpMainMul,
                              AbuUmpMainDegExtend.class_unique_id(): AbuUmpMainDegExtend,

                              AbuUmpEdgeDeg.class_unique_id(): AbuUmpEdgeDeg,
                              AbuUmpEdgePrice.class_unique_id(): AbuUmpEdgePrice,
                              AbuUmpEdgeMul.class_unique_id(): AbuUmpEdgeMul,
                              AbuUmpEegeDegExtend.class_unique_id(): AbuUmpEegeDegExtend}
            for choice_ump in self.choice_umps.options:
                unique_class_key = choice_ump.split(':')[1]
                ump_custom_fn = choice_ump.split(':')[2]
                ump_class = ump_class_dict[unique_class_key]
                ump_object = ump_class(predict=True, market_name=ump_custom_fn)
                # 把读取的裁判都做为自定义裁判加入到ABuUmpManager中，即可在回测中使用裁判进行交易拦截
                ABuUmpManager.append_user_ump(ump_object)

        ABuEnv.g_enable_ml_feature = self.enable_ml_feature.value
        if self.enable_ml_feature.value:
            # 如果开启回测记录特征需要加入AbuFeatureDegExtend到ABuMLFeature
            ABuMLFeature.append_user_feature(AbuFeatureDegExtend)

    def run_end(self, abu_result_tuple, choice_symbols, buy_desc_list, sell_desc_list, ps_desc_list):
        """保存回测结果以及回测结果索引文件存贮"""

        if self.enable_ml_feature.value:
            # 只有启动特征采集的情况下才进行保存回测
            custom_name = datetime.datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
            store_abu_result_tuple(abu_result_tuple, store_type=EStoreAbu.E_STORE_CUSTOM_NAME,
                                   custom_name=custom_name)
            symbol_desc = u'{}全市场'.format(
                ABuEnv.g_market_target.value) if choice_symbols is None else u'{}个symbol'.format(
                len(choice_symbols))
            ump_desc = u'有裁判预测拦截交易' \
                if self.choice_umps.options is not None and len(self.choice_umps.options) > 0 else u'无裁判'
            factor_desc = u'|'.join(buy_desc_list + sell_desc_list + ps_desc_list)
            custom_desc = u'{}|{}|{}'.format(symbol_desc, ump_desc, factor_desc)
            # 对刚刚保存的store_abu_result_tuple的缓存进行cache描述以及索引保存
            dump_custom_abu_index_csv(custom_name, custom_desc)
            # 通知训练界面进行更新
            self.load_abu_result()

    def _init_manager_ui(self):
        """裁判数据管理界面初始化"""
        description = widgets.Textarea(
            value=u'删除选择的裁判本地数据：\n'
                  u'删除所选择的已训练好的本地裁判数据，谨慎操作！\n'
                  u'分享选择的裁判：\n'
                  u'将训练好的裁判数据分享到交易社区，供其他交易者使用\n'
                  u'下载更多的裁判：\n'
                  u'从交易社区，下载更多训练好的裁判数据\n',

            disabled=False,
            layout=widgets.Layout(height='150px')
        )

        self.manager_umps = widgets.Select(
            options=[],
            description=u'本地裁判:',
            disabled=False,
            layout=widgets.Layout(width='100%', align_items='stretch')
        )
        self.load_train_ump(self.manager_umps)
        delete_bt = widgets.Button(description=u'删除选择的裁判本地数据', layout=widgets.Layout(width='98%'),
                                   button_style='warning')
        delete_bt.on_click(self._do_delete_ump)

        share_bt = widgets.Button(description=u'分享选择的裁判', layout=widgets.Layout(width='98%'),
                                  button_style='info')
        share_bt.on_click(permission_denied)
        down_bt = widgets.Button(description=u'下载更多的裁判', layout=widgets.Layout(width='98%'),
                                 button_style='info')
        down_bt.on_click(permission_denied)

        return widgets.VBox([description, self.manager_umps, delete_bt, share_bt, down_bt])

    def _init_predict_ui(self):
        """裁判预测拦截界面初始化"""
        description = widgets.Textarea(
            value=u'裁判预测拦截：\n'
                  u'通过在\'裁判特征训练\'选中\'指定的裁判，选中的裁判将在对应的\n'
                  u'回测中生效，即开始在回测中对交易进行预测拦截等智能交易干涉行为',

            disabled=False,
            layout=widgets.Layout(height='150px')
        )
        # ump已选框
        self.choice_umps = widgets.SelectMultiple(
            description=u'已选裁判:',
            disabled=False,
            layout=widgets.Layout(width='100%', align_items='stretch')
        )
        self.choice_umps.observe(self.remove_ump_select, names='value')

        self.umps = widgets.SelectMultiple(
            description=u'备选裁判:',
            disabled=False,
            layout=widgets.Layout(width='100%', align_items='stretch')
        )
        self.umps.observe(self.on_ump_select, names='value')
        self.load_train_ump(self.umps)

        return widgets.VBox([description, self.choice_umps, self.umps])

    def remove_ump_select(self, select):
        """ump已选中点击删除股票池中对应的symbol"""
        # FIXME BUG 低版本ipywidgets下删除的不对
        self.choice_umps.options = list(set(self.choice_umps.options) - set(select['new']))

    def on_ump_select(self, select):
        """从备选ump中选择放入到已选ump中"""
        st_ump = [ump for ump in list(select['new'])]
        self.choice_umps.options = list(set(st_ump + list(self.choice_umps.options)))

    def _init_train_ui(self):
        """裁判特征训练面初始化"""
        description = widgets.Textarea(
            value=u'裁判特征训练：\n'
                  u'通过在\'裁判特征采集\'选中\'回测过程生成交易特征\'可在回测完成后保存当此回测结果\n'
                  u'所有回测的结果将显示在下面的\'备选回测:\'框中\n'
                  u'通过\'开始训练裁判\'进行指定的回测裁判训练，训练后的裁判在\'裁判预测拦截\'下可进行选择，选中的裁判将在对应的'
                  u'回测中生效，即开始在回测中对交易进行预测拦截等智能交易干涉行为',

            disabled=False,
            layout=widgets.Layout(height='150px')
        )

        self.abu_result = widgets.Select(
            options=[],
            description=u'备选回测:',
            disabled=False,
            layout=widgets.Layout(width='100%', align_items='stretch')
        )
        self.load_abu_result()

        train_bt = widgets.Button(description=u'开始训练裁判', layout=widgets.Layout(width='98%'),
                                  button_style='info')
        train_bt.on_click(self._do_train)
        delete_bt = widgets.Button(description=u'删除选择的备选回测本地数据', layout=widgets.Layout(width='98%'),
                                   button_style='warning')
        delete_bt.on_click(self._do_delete_abu_result)

        return widgets.VBox([description, self.abu_result, train_bt, delete_bt])

    def _init_feature_ui(self):
        """裁判特征采集界面初始化"""
        ml_feature_description = widgets.Textarea(
            value=u'裁判特征采集\n'
                  u'裁判是建立在机器学习技术基础上的，所以必然会涉及到特征，abu量化系统支持在回测过程中生成特征数据，切分训练测试集，'
                  u'甚至成交买单快照图片，通过打开下面的开关即可在生成最终的输出结果数据订单信息上加上买入时刻的很多信息，'
                  u'比如价格位置、趋势走向、波动情况等等特征, 注意需要生成特征后回测速度效率会降低\n'
                  u'如在下拉选择中选中\'回测过程生成交易特征\'在回测完成后将保存回测结果，通过在\'裁判特征训练\'可进行查看并进行'
                  u'裁判训练',
            disabled=False,
            layout=widgets.Layout(height='150px')
        )

        self.enable_ml_feature = widgets.Dropdown(
            options={u'回测过程不生成交易特征': 0,
                     u'回测过程生成交易特征': 1},
            value=0,
            description=u'特征生成:',
        )
        return widgets.VBox([ml_feature_description, self.enable_ml_feature])

    @contextmanager
    def _parse_custom(self):
        """从记录描述ui文字描述中解析abu_custom_name和abu_custom_desc"""
        if self.abu_result.value is None:
            show_msg_toast_func(u'未选择任何特征回测结果！')
            return

        s_pos = self.abu_result.value.find('.')
        e_pos = self.abu_result.value.find(':')
        if s_pos > 0 and e_pos > 0:
            abu_custom_name = self.abu_result.value[s_pos + 1:e_pos].strip()
            # 截取回测的文字描述内容之后传递给ump_custom_desc
            abu_custom_desc = self.abu_result.value[e_pos:]

            yield abu_custom_name, abu_custom_desc

            # 下文进行索引文件的重新加载，ui刷新
            self.load_train_ump(self.umps)
            self.load_train_ump(self.manager_umps)
            self.load_abu_result()

    # noinspection PyUnusedLocal
    def _do_delete_ump(self, bt):
        """执行删除已训练好的ump数据文件以及对应的索引描述行"""
        choice_ump = self.manager_umps.value
        if choice_ump is None:
            show_msg_toast_func(u'未选择任何本地裁判数据！')
            return

        is_main_ump = choice_ump.split(':')[0].find(u'主裁') > 0
        unique_class_key = choice_ump.split(':')[1]
        ump_custom_fn = choice_ump.split(':')[2]

        # 通过是否主裁，ump唯一id，以及custom name来唯一确定要删除的裁判具体名称
        del_fn = 'ump_main_{}_{}'.format(
            ump_custom_fn, unique_class_key) if is_main_ump else 'ump_edge_{}_{}'.format(
            ump_custom_fn, unique_class_key)

        ump_fn = os.path.join(ABuEnv.g_project_data_dir, 'ump', del_fn)
        # 删除ump数据文件
        del_file(ump_fn)
        # 删除索引描述行
        del_custom_ump_index('{}:{}'.format(unique_class_key, ump_custom_fn))
        show_msg_toast_func(u'删除{}成功！'.format(ump_fn))

        # ui刷新
        self.load_train_ump(self.umps)
        self.load_train_ump(self.manager_umps)

    # noinspection PyUnusedLocal
    def _do_delete_abu_result(self, bt):
        """内部通过上下文_parse_custom读取abu_custom_name删除对应的回测以及索引描述行"""
        with self._parse_custom() as (abu_custom_name, _):
            delete_abu_result_tuple(store_type=EStoreAbu.E_STORE_CUSTOM_NAME, custom_name=abu_custom_name,
                                    del_index=True)
            show_msg_toast_func(u'删除{}成功！'.format(abu_custom_name))

    # noinspection PyUnusedLocal
    def _do_train(self, bt):
        """
            内部通过上下文_parse_custom读取abu_custom_name，abu_custom_desc
            读取对应的回测单子，依次开始训练ump主裁：

            1. 角度主裁
            2. 价格主裁
            3. mul单混主裁
            4. 扩展角度主裁

            依次开始训练ump边裁：

            1. 角度边裁
            2. 价格边裁
            3. mul单混边裁
            4. 扩展角度边裁
        """

        # 通过上下文_parse_custom读取abu_custom_name，abu_custom_desc
        with self._parse_custom() as (abu_custom_name, abu_custom_desc):
            # 读取对应的回测单子
            abu_result_tuple_train = load_abu_result_tuple(store_type=EStoreAbu.E_STORE_CUSTOM_NAME,
                                                           custom_name=abu_custom_name)
            orders_pd_train = abu_result_tuple_train.orders_pd
            if orders_pd_train is None:
                show_msg_toast_func(u'特征交易文件读取失败！')
                return

            if orders_pd_train.shape[0] < 50:
                show_msg_toast_func(u'生成交易订单数量小于50，不能训练裁判！')
                return

            # gmm训练默认沙盒数据少分类，其它的内部自行计算
            p_ncs = slice(20, 40, 1) if ABuEnv._g_enable_example_env_ipython else None

            def train_main_ump(ump_class, ump_name):
                # 训练好的ump custom_name, 在dump_file_fn内部实际会拼接class_unique_id
                ump_custom_name = datetime.datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
                _ = ump_class.ump_main_clf_dump(orders_pd_train, p_ncs=p_ncs, market_name=ump_custom_name,
                                                save_order=False, show_order=False)

                ump_unique = ump_class.class_unique_id()
                ump_key = 'main'
                ump_custom_desc = u'{}基于{}的训练结果'.format(ump_name, abu_custom_desc)
                dump_custom_ump_index_csv(ump_custom_name, ump_unique, ump_key, ump_custom_desc)

            # 依次开始训练ump主裁
            train_main_ump(AbuUmpMainDeg, u'角度主裁')
            train_main_ump(AbuUmpMainPrice, u'价格主裁')
            train_main_ump(AbuUmpMainMul, u'mul单混主裁')
            train_main_ump(AbuUmpMainDegExtend, u'扩展角度主裁')

            def train_edge_ump(ump_class, ump_name):
                # 训练好的ump custom_name, 在dump_file_fn内部实际会拼接class_unique_id
                ump_custom_name = datetime.datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
                _ = ump_class.ump_edge_clf_dump(orders_pd_train, market_name=ump_custom_name)

                ump_unique = ump_class.class_unique_id()
                ump_key = 'edge'
                ump_custom_desc = u'{}基于{}的训练结果'.format(ump_name, abu_custom_desc)
                dump_custom_ump_index_csv(ump_custom_name, ump_unique, ump_key, ump_custom_desc)
                print(u'边裁训练：{} 完成！'.format(ump_custom_desc))

            # 依次开始训练ump边裁
            train_edge_ump(AbuUmpEdgeDeg, u'角度边裁')
            train_edge_ump(AbuUmpEdgePrice, u'价格边裁')
            train_edge_ump(AbuUmpEdgeMul, u'mul单混边裁')
            train_edge_ump(AbuUmpEegeDegExtend, u'扩展角度边裁')
