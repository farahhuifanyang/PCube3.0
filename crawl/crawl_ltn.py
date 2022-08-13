'''
Author: Li Zekun
Date: 2020-12-29 11:09:24
LastEditTime: 2021-02-22 16:54:47
LastEditors: Please set LastEditors
Description: Crawl articles from www.chinatimes.com
FilePath: /PCube3/crawl/crawl_chinatimes.py
'''
import os
import sys
sys.path.append(r'F:\New_Life\项目组\人立方\PCube3')
import json
import time
import hashlib
import requests
from lxml import etree
from os.path import isdir
from openccpy.opencc import Opencc
from configure import globalFLAGS
from classes.Article import Article
import random


def main(simple=True):
    """
    定时启动的爬虫程序

    Args:
        simple (bool, optional): 选择是否以简体存储. Defaults to True.
    """
    os.system("curl 'http://10.3.8.211/login' --data 'user=2019140570&pass=103010&line='")
    topics = json.load(open(os.path.join(globalFLAGS.static_dir, "crawl/chinatimes_topics.json")))  # 读取感兴趣的主题列表
    for key, topic in topics.items():
        save_dir = os.path.join(globalFLAGS.news_data_dir, f"{time.strftime('%Y')}-{time.strftime('%m')}")
        if not isdir(save_dir):
            os.mkdir(save_dir)
        save_dir = os.path.join(save_dir, f"{key}")
        if not isdir(save_dir):
            os.mkdir(save_dir)
        article_urls, hot_urls = get_article_urls(topic, simple=simple)
        for url in article_urls:
            article = parse_article(url, topic)
            save_path = os.path.join(save_dir, f"{article.id}.html")
            with open(save_path, 'w', encoding="utf-8") as f:
                f.write(article.to_file())
                print('[{}] INFO: {}: {} saved.'.format(time.strftime('%Y-%m-%d %H:%M:%S'), article.theme, article.title))


def get_article_urls(topic, simple=True):
    """
    从实时新闻的列表中获取文章页面url

    Args:
        topic (string): 主题
        simple (bool, optional): 选择是否以简体存储. Defaults to True.

    Returns:
        article_urls (string list): 过去24h内文章的URL
        hot_urls (string list): 右上角热门文章的URL
    """

    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/74.0.3729.131 Safari/537.36'
    }
    title_xpath = "/html/body/div[2]/div/div[2]/div/section/ul/li/div/div/div[2]/h3/a"
    date_xpath = "/html/body/div[2]/div/div[2]/div/section/ul/li/div/div/div[2]/div/time/span[2]"
    time_xpath = "/html/body/div[2]/div/div[2]/div/section/ul/li/div/div/div[2]/div/time/span[1]"
    hot_title_xpath = "/html/body/div[2]/div/div[2]/aside/section/ol/li/h4/a"

    article_urls = []
    hot_urls = []

    hot_topic_url = f"https://www.chinatimes.com/cn/{topic}/total?chdtv" if simple else f"https://www.chinatimes.com/{topic}/total?chdtv"
    try:
        main_response = requests.get(hot_topic_url, headers=headers, timeout=(3, 10))
    except requests.exceptions.RequestException:
        print('[{}] ERROR: {} connection time out.'.format(time.strftime('%Y-%m-%d %H:%M:%S'), hot_topic_url))
        return [], []
    main_html = etree.HTML(main_response.text+"?chdtv")
    title_items = main_html.xpath(hot_title_xpath)
    for title in title_items:
        hot_urls.append(title.attrib['href'])

    for i in range(1, 11):
        topic_url = f"https://www.chinatimes.com/cn/{topic}/total?page={i}&chdtv" if simple else f"https://www.chinatimes.com/{topic}/total?page={i}&chdtv"
        try:
            main_response = requests.get(topic_url, headers=headers, timeout=(3, 10))
        except requests.exceptions.RequestException:
            print('[{}] ERROR: {} connection time out.'.format(time.strftime('%Y-%m-%d %H:%M:%S'), topic_url))
            continue

        main_html = etree.HTML(main_response.text)
        date_items = main_html.xpath(date_xpath)
        title_items = main_html.xpath(title_xpath)

        for date, title in zip(date_items, title_items):
            date_ = date.text.split('/')
            if date_[2] == time.strftime('%d') and date_[1] == time.strftime('%m'):
                article_urls.append("https://www.chinatimes.com" + title.attrib['href']+"?chdtv")

    return article_urls, hot_urls


def parse_article(url, theme):
    """
    根据URL获取并处理单篇文章

    Args:
        url (string): 文章页面url
        theme (theme): 文章主题

    Returns:
        (Article): Article对象，None表示文章无效
    """
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/74.0.3729.131 Safari/537.36'
    }

    title_xpath = "/html/body/div[2]/div/div[2]/div/div/article/div/header/h1"
    time_xpath = "/html/body/div[2]/div/div[2]/div/div/article/div/header/div/div[1]/div/div/time/span[1]"
    date_xpath = "/html/body/div[2]/div/div[2]/div/div/article/div/header/div/div[1]/div/div/time/span[2]"
    theme_xpath = "/html/body/div[2]/div/div[1]/nav/ol/li[2]/a/span"
    content_xpath = "/html/body/div[2]/div/div[2]/div/div/article/div/div[1]/div[2]/div[2]/div[2]/p"
    keyword_xpath = "/html/body/div[2]/div/div[2]/div/div/article/div/div[1]/div[2]/div[2]/div[2]/div[5]/span/a"

    try:
        page = requests.get(url, headers=headers, timeout=(3, 5))
    except requests.exceptions.RequestException:
        print('[{}] ERROR: {} connection time out.'.format(time.strftime('%Y-%m-%d %H:%M:%S'), url))
        return None
    page.encoding = page.apparent_encoding  # 设置合适的解析编码
    page = page.text

    html = etree.HTML(page)
    time_item = html.xpath(time_xpath)
    date_item = html.xpath(date_xpath)
    # theme_item = html.xpath(theme_xpath)
    title_item = html.xpath(title_xpath)
    content_item = html.xpath(content_xpath)
    keyword_item = html.xpath(keyword_xpath)

    time_ = date_item[0].text.replace("/", "-") + " " + time_item[0].text + ":00"
    # theme = theme_item[0].text if len(theme_item) > 0 else theme
    title = title_item[0].text
    keywords = [item.text for item in keyword_item]

    content = ""
    for item in content_item:
        if item.text:
            content += item.text + "\\n"

    file_md5 = hashlib.md5(content.encode())
    id_ = file_md5.hexdigest()

    return Article(id_, url, time_, title, theme, keywords, content)


def test():
    xpath = '/html/body/div[10]/section/div[4]/h1'
    topic_url = r"https://news.ltn.com.tw/news/politics/breakingnews/3427727?chdtv"
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/88.0.4324.190 Safari/537.36'
    }
    main_response = requests.get(topic_url, headers=headers)
    # main_response.encoding = main_response.apparent_encoding  # 设置合适的解析编码
    # print(main_response.encoding)
    # with open('main_html', 'w', encoding='utf-8') as f:
    #     f.write(main_response.text)
    print(main_response.text)
    main_html = etree.HTML(main_response.text)
    # t = etree.tostring(main_html, encoding="utf-8", pretty_print=True)
    # print(t.decode("utf-8"))
    title_items = main_html.xpath(xpath)
    print(title_items)


def fix():
    root_dir = "/home/disk2/nuclear/news_data/PCube/2021-01"
    for topic_dir in os.listdir(root_dir):
        for file in os.listdir(os.path.join(root_dir, topic_dir)):
            file = os.path.join(root_dir, topic_dir, file)
            article = Article.init_from_file(file)
            if not article.keywords:
                article = parse_article(article.url, topic_dir)
            else:
                article.theme = topic_dir
            with open(file, 'w', encoding="utf-8") as f:
                f.write(article.to_file())


def history_news():
    """
    用以爬取中时电子报历史数据的算法
    抽取结果存入文件

    Returns:
        [type]: [description]
    """
    kw_save_path = r"F:\New_Life\项目组\人立方\PCube3\crawl\keywords.txt"
    url_save_path = r"F:\New_Life\项目组\人立方\PCube3\crawl\history_url.txt"
    done_kw_save_path = r"F:\New_Life\项目组\人立方\PCube3\crawl\done_keywords.txt"
    log_save_path = r"F:\New_Life\项目组\人立方\PCube3\crawl\log.txt"
    topics = json.load(open(os.path.join(globalFLAGS.static_dir, "crawl/ltn_topics.json"), encoding='utf-8'))
    my_headers=["Mozilla/5.0 (Windows NT 6.3; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/39.0.2171.95 Safari/537.36",
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_9_2) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/35.0.1916.153 Safari/537.36",
        "Mozilla/5.0 (Windows NT 6.1; WOW64; rv:30.0) Gecko/20100101 Firefox/30.0",
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_9_2) AppleWebKit/537.75.14 (KHTML, like Gecko) Version/7.0.3 Safari/537.75.14",
        "Mozilla/5.0 (compatible; MSIE 10.0; Windows NT 6.2; Win64; x64; Trident/6.0)"
    ]

    def reload_kw():
        """
        用于断点恢复，从文件中读取现有的所有关键词

        Returns:
            set: 无重复字符串集合
        """
        if os.path.exists(kw_save_path):
            keywords = []
            with open(kw_save_path, "r", encoding='utf-8') as kwf:
                for word in kwf:
                    keywords.append(word.strip())
            keywords = set(keywords)
        else:
            keywords = set()
        if os.path.exists(done_kw_save_path):
            done_keywords = []
            with open(done_kw_save_path, "r", encoding='utf-8') as kwf:
                for word in kwf:
                    done_keywords.append(word.strip())
            done_keywords = set(done_keywords)
        else:
            done_keywords = set()
        keywords = keywords - done_keywords
        return keywords

    def reload_url():
        """
        用于断点恢复，从文件中读取现有的所有URL

        Returns:
            list: 无重复URL集合
        """
        if os.path.exists(url_save_path):
            urls = []
            with open(url_save_path, "r") as urlf:
                for url in urlf:
                    urls.append(url.strip().split("\t"))
        else:
            urls = []
        return urls

    def getKeywords():
        """
        从一月抽取到的新闻中提取所有的关键词，以此作为搜索的关键词
        这一步运行很快，得到的结果存入文件
        """
        keywords = reload_kw()
        root_dir = "/home/disk2/nuclear/news_data/PCube/2021-01"
        for topic_dir in os.listdir(root_dir):
            for file in os.listdir(os.path.join(root_dir, topic_dir)):
                file = os.path.join(root_dir, topic_dir, file)
                article = Article.init_from_file(file)
                for word in article.keywords:
                    try:
                        word = Opencc.to_traditional(word)
                        keywords.add(word)
                    except Exception:
                        pass

        root_dir = "/home/disk2/nuclear/news_data/PCube/2020-12"
        for topic_dir in os.listdir(root_dir):
            for file in os.listdir(os.path.join(root_dir, topic_dir)):
                file = os.path.join(root_dir, topic_dir, file)
                article = Article.init_from_file(file)
                for word in article.keywords:
                    try:
                        word = Opencc.to_traditional(word)
                        keywords.add(word)
                    except Exception:
                        pass

        with open(kw_save_path, "w") as wf:
            for word in keywords:
                wf.write(word+"\n")

    def getUrls():
        """
        根据关键词搜索历史新闻，提取这些新闻的URL
        这一步可能需要数小时，每得到一个结果就保存到文件最后一行
        """
        keywords = reload_kw()
        # urls = reload_url()
        start_time = '20170101'
        end_time = '20210220'
        title_xpath = "/html/body/section/div[6]/ul/li/div[@class='cont']/a[@class='tit']"
        date_xpath = "/html/body/section/div[6]/ul/li/div/span[@class='time']"
        theme_xpath = "/html/body/section/div[6]/ul/li/div[@class='cont']/i"

	
        for word in keywords:
            print(f"[INFO] {word} begin.")
            i = 1
            while True:
                context_url = f"https://search.ltn.com.tw/list?keyword={word}&start_time={start_time}&end_time={end_time}&sort=date&type=all&page={i}"  # 这里是搜索用的URL，需要根据网址修改
                # print(f"[INFO] {word} page {i}. url={context_url}")
                try:
                    headers = {
                        'User-Agent': random.choice(my_headers)
                    }
                    main_response = requests.get(context_url, headers=headers, timeout=(3, 10))
                except requests.exceptions.RequestException:
                    print('[{}] ERROR: {} connection time out.'.format(time.strftime('%Y-%m-%d %H:%M:%S'), context_url))
                    with open(log_save_path, 'a') as f:
                        f.write('[{}] ERROR: {} connection time out.\n'.format(time.strftime('%Y-%m-%d %H:%M:%S'), context_url))
                    continue
                try:
                    main_html = etree.HTML(main_response.text)
                    date_items = main_html.xpath(date_xpath)
                    title_items = main_html.xpath(title_xpath)
                    theme_items = main_html.xpath(theme_xpath)
                except AttributeError:
                    print('[{}] ERROR: {} AttributeError.'.format(time.strftime('%Y-%m-%d %H:%M:%S'), context_url))
                    with open(log_save_path, 'a') as f:
                        f.write('[{}] ERROR: {} AttributeError.\n'.format(time.strftime('%Y-%m-%d %H:%M:%S'), context_url))
                    continue

                if not title_items:
                    break

                date_ = date_items[0].text.split('/')
                j = 1
                for date, theme, title in zip(date_items, theme_items, title_items):
                    # print(f"\t [INFO] {word} page {i}, number {j}.")
                    try:
                        date_ = date.text.split('/')
                        theme_ = [Opencc.to_simple(char) for char in theme.text]
                        theme_ = "".join(theme_)
                        url_ = title.attrib['href']
                        if int(date_[0]) >= 2019 and theme_ in topics:
                            new_url = [theme_, url_]
                            # urls.append(new_url)
                            with open(url_save_path, "a", encoding='utf-8') as wf:
                                wf.write("\t".join(new_url)+"\n")
                        elif int(date_[0]) < 2019:
                            break
                    except Exception:
                        print(f"[ERRO] {word} page {i}, number {j} failed.")
                        with open(log_save_path, 'a') as f:
                            f.write(f"[ERRO] {word} page {i}, number {j} failed.\n")
                        continue
                    j += 1

                i += 1
            print(f"[INFO] {word} complete.")
            with open(done_kw_save_path, 'a', encoding='utf-8') as f:
                f.write(word+'\n')

    # getKeywords()     # 从头开始的话释放这两行代码
    getUrls()

    """
    根据URL获取所有的历史信息并存入相应的目录下，这一步运行非常慢，且容易出错，每完成一篇文章就保存一篇
    """
    # urls = reload_url()
    # for topic, url in urls:
    #     try:
    #         article = parse_article(url, topic)
    #     except Exception:
    #         continue
    #     if not article:
    #         os.system("curl 'http://10.3.8.211/login' --data 'user=2019140570&pass=103010&line='")  # 目前发现的问题均为没有在网关登陆的问题
    #         continue
    #     year, month, _ = article.time.split(" ")[0].split("-")
    #     save_dir = f"/home/disk2/nuclear/news_data/PCube/{year}-{month}"  # 保存到相应的位置
    #     if not isdir(save_dir):
    #         os.mkdir(save_dir)
    #     save_dir = os.path.join(save_dir, f"{topic}")
    #     if not isdir(save_dir):
    #         os.mkdir(save_dir)
    #     save_path = os.path.join(save_dir, f"{article.id}.html")

    #     with open(save_path, 'w', encoding="utf-8") as f:
    #         f.write(article.to_file())
    #         print('[{}] INFO: {}: {} saved.'.format(time.strftime('%Y-%m-%d %H:%M:%S'), article.theme, article.url))


if __name__ == "__main__":
    # main(simple=True)
    # parse_article("https://www.chinatimes.com/cn/realtimenews/20201230002663-260407?chdtv")
    # test()
    history_news()
