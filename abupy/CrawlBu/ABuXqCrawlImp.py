# -*- encoding:utf-8 -*-
from __future__ import print_function
from __future__ import absolute_import
from __future__ import division

from abc import ABCMeta, abstractmethod

import logging
import time

from . import ABuXqFile

from ..CoreBu import env
# noinspection PyUnresolvedReferences
from ..CoreBu.ABuFixes import map, reduce, filter
from .ABuXqApi import BASE_XQ_HQ_URL
from .ABuXqApi import BASE_XQ_STOCK_INFO
from ..ExtBu import six

__author__ = '小青蛙'
__weixin__ = 'abu_quant'


def _bs4_html(content):
    """
    使用BeautifulSoup解析html
    :param content: html
    :return: BeautifulSoup
    """
    from bs4 import BeautifulSoup
    return BeautifulSoup(content, 'lxml')


def _xpath(content):
    """
    使用xpath解析html 
    :param content:
    :return:
    """
    from lxml import etree
    selector = etree.HTML(content)
    return selector


class BaseXQCrawlBrower(six.with_metaclass(ABCMeta, object)):
    """
    使用chrome浏览器的自动化测试驱动接口，获取网页数据
    """

    def __init__(self, base_url):
        self._base_url = base_url
        if env.g_crawl_chrome_driver is not None:
            self.driver_path = env.g_crawl_chrome_driver
        else:
            raise RuntimeError('driver_path error!!!, abupy.CoreBu.ABuEnv.g_crawl_chrome_driver must be right')

        # noinspection PyUnresolvedReferences
        from selenium.webdriver.support import ui
        # noinspection PyUnresolvedReferences
        from selenium import webdriver
        self.driver = webdriver.Chrome(self.driver_path)
        self.wait = ui.WebDriverWait(self.driver, 10)

    @abstractmethod
    def _crawl_imp(self, *args, **kwargs):
        pass

    def get(self, url):
        self.driver.get(url)

    @property
    def content(self):
        return self.driver.page_source

    def crawl(self, *args, **kwargs):
        """
        执行完任务是自动退出，避免占用资源，在多进程爬时会启动多个chrome实例
        :param args:
        :param kwargs:
        :return: crawl_imp
        """
        ret = None
        try:
            self.driver.get(self._base_url)
            self.driver.maximize_window()
            ret = self._crawl_imp(*args, **kwargs)
        except Exception as e:
            logging.exception(e)
        return ret

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.driver.quit()

    def _scroll_to_bottom(self, element):
        loc = element.location
        self.driver.execute_script('window.scrollBy({},{})'.format(loc['x'], loc['y']))


class BaseHQCrawlBrower(BaseXQCrawlBrower):
    def _crawl_imp(self, *args, **kwargs):
        pass

    def __init__(self, url):
        super(BaseHQCrawlBrower, self).__init__(BASE_XQ_HQ_URL)
        self._base_url = self._base_url + url


class NavHQCrawlBrower(BaseHQCrawlBrower):
    def _crawl_imp(self, *args, **kwargs):
        return _parse_nav(self.driver.page_source)

    def __init__(self):
        super(NavHQCrawlBrower, self).__init__('')


class StockListCrawlBrower(BaseHQCrawlBrower):
    def _ensure_max_page_size(self):
        """
        使每页展示的stock数最多，总页数变少，使网络请求数变少
        """
        max_page_tag = self.driver.find_element_by_xpath('//*[@id="stockList-header"]/div[2]/a[3]')
        max_page_tag.click()
        time_out = 30
        while time_out:
            time.sleep(1)
            time_out -= 1
            _, total = self._curr_total_page()
            #  直到 最大size生效
            if total == 1 or self._curr_page_counts() == int(max_page_tag.text):
                break

    def _curr_page_counts(self):
        selector = _xpath(self.content)
        items = selector.xpath('//*[@id="stockList"]/div[1]/table/tbody/tr')
        return len(items)

    def _curr_total_page(self):

        selector = _xpath(self.content)
        pages = selector.xpath('//*[@id="pageList"]/div/ul/li/a/text()')
        cur_page = selector.xpath('//*[@id="pageList"]/div/ul/li[@class="active"]/a/text()')

        # 存在pages的最后一个值，否则cur和total都是1
        if len(pages):
            return int(cur_page[0]), int(pages[-1])
        else:
            return 1, 1

    def _curr_page_items(self):
        selector = _xpath(self.content)
        # code = selector.xpath('//*[@id="stockList"]/div[1]/table/tbody/tr/td[1]/a/text()')
        # name = selector.xpath('//*[@id="stockList"]/div[1]/table/tbody/tr/td[2]/a/text()')
        # a标签下的text() 可能不存在，而xpath会把不存在的过滤掉，导致code，和name的长度不一致，产生错位，故先找到a，a。text为kong也占位，就能一一对应
        code = selector.xpath('//*[@id="stockList"]/div[1]/table/tbody/tr/td[1]/a')
        name = selector.xpath('//*[@id="stockList"]/div[1]/table/tbody/tr/td[2]/a')
        code = list(map(lambda a: a.text, code))
        name = list(map(lambda a: a.text, name))
        return name, code

    def _goto_next_page(self):
        next_page = self.driver.find_element_by_xpath('//*[@id="pageList"]/div/ul/li[@class="next"]/a')
        if next_page is not None:
            # 滚动到next_page 标签显示出来，否则click可能会报错
            self.wait.until(lambda dr: next_page.is_enabled())
            self.driver.execute_script('arguments[0].click()', next_page)
            time.sleep(1)

    def _crawl_imp(self, *args, **kwargs):
        self._ensure_max_page_size()

        cur_page, total_page = self._curr_total_page()
        names = []
        symbols = []
        # page index start 1
        for page in range(1, total_page + 1):
            self.wait.until(lambda dr: dr.find_element_by_xpath('//*[@id="stockList"]/div[1]/table').is_displayed())
            cur_page, _ = self._curr_total_page()
            temp_names, temp_symbols = self._curr_page_items()
            names += temp_names
            symbols += temp_symbols
            if page < total_page:
                self._goto_next_page()
            else:
                break

        return names, symbols

    def __init__(self, url):
        super(StockListCrawlBrower, self).__init__(url)


def _parse_nav(content):
    soup = _bs4_html(content)
    nav_tags = soup.select('.industry-nav > div')

    def parse_nav_tag(tag):
        nav = {}
        first_nav = tag.select('.first-nav > span')
        if len(first_nav) > 0:
            second_nav = tag.select('.second-nav > li')
            nav[first_nav[0].string] = {}
            for nav_2 in second_nav:
                a_tag = nav_2.select('a')
                if len(a_tag) <= 0:
                    continue
                third_nav = nav_2.select('.third-nav')
                if len(third_nav) > 0:
                    second_nav_name = str(nav_2).replace('<li><i class="list-style"></i>', '')
                    second_nav_name = second_nav_name[: second_nav_name.index('<i')]
                    nav[first_nav[0].string][second_nav_name] = list(map(lambda a: {a.get('title'): a.get('href')},
                                                                         a_tag))
                else:
                    nav[first_nav[0].string][a_tag[0].get('title')] = a_tag[0].get('href')
        return nav

    def merge(dict1, dict2):
        return dict(dict1, **dict2)

    return reduce(lambda d1, d2: merge(d1, d2), map(lambda tag: parse_nav_tag(tag), nav_tags))


class StockInfoListBrower(BaseXQCrawlBrower):
    def __init__(self, market, symbols):
        super(StockInfoListBrower, self).__init__(BASE_XQ_STOCK_INFO)

        self._market = market
        self._symbols = symbols

    def __exit__(self, exc_type, exc_val, exc_tb):
        self._market = None
        self._symbols = None
        super(StockInfoListBrower, self).__exit__(exc_type, exc_val, exc_tb)

    def _parse_stock_info(self):
        selector = _xpath(self.content)
        # 特斯拉(NASDAQ:TSLA)
        stock_name = selector.xpath('//*[@id="center"]/div[2]/div[2]/div[1]/div[1]/span[1]/strong/text()')
        company_info_p = selector.xpath('//*[@id="center"]/div[3]/div/div[2]/div/p')
        company_industry = selector.xpath('//*[@id="relatedIndustry"]/h2/a/text()')

        quate_items = selector.xpath('//*[@id="center"]/div[2]/div[2]/div[2]/table/tbody/tr/td')

        info = {}
        if len(stock_name):
            stock_name_info = stock_name[0]
            st = stock_name_info.rfind('(')
            en = stock_name_info.rfind(')')
            market_symbol = stock_name_info[st + 1:en]
            sp_result = market_symbol.split(':')
            if len(sp_result) == 2:
                company_name = stock_name_info[:st]
                exchange_name = sp_result[0]
                company_symbol = sp_result[1]
                info['name'] = company_name
                info['exchange'] = exchange_name
                info['symbol'] = company_symbol

        if len(company_info_p):
            last_key = None
            for p in company_info_p:
                for child in p.getchildren():
                    if child.tag == 'strong':
                        info[child.text] = child.tail
                        last_key = child.text
                    if child.tag == 'a':
                        info[last_key] = child.get('href')
        if len(quate_items):
            for item in quate_items:
                if len(item.getchildren()) == 1:
                    info[item.text] = item.getchildren()[0].text

        if len(company_industry):
            info['industry'] = company_industry[0]

        return info

    def _crawl_imp(self, *args, **kwargs):
        for index, symbol in enumerate(self._symbols):
            try:
                if not ABuXqFile.exist_stock_info(self._market, symbol) or ('replace' in kwargs and kwargs['replace']):
                    self.get(self._base_url + symbol)
                    stock_info = self._parse_stock_info()
                    ABuXqFile.save_cache_stock_info(stock_info, self._market, symbol)
                if (index + 1) % 200 == 0:
                    print(
                        '{}: {} {}  {}/{}'.format(kwargs['process'], self._market, symbol, index + 1,
                                                  len(self._symbols)))
            except Exception as e:
                # 记录失败的symbol
                ABuXqFile.error_stock_info(self._market, symbol, e)
                logging.exception(e)
        return 'Done'
