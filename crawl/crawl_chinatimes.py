'''
Author: Li Zekun
Date: 2020-12-29 11:09:24
LastEditTime: 2021-07-22 16:20:52
LastEditors: Please set LastEditors
Description: Crawl articles from www.chinatimes.com
FilePath: /PCube3/crawl/crawl_chinatimes.py
'''
import os
import json
import time
import hashlib
import requests
from lxml import etree
from os.path import isdir
from openccpy.opencc import Opencc
from configure import globalFLAGS
from classes.Article import Article
from DAO.ElasticSearchFile import ElasticSearchTemplate


def main(simple=True):
    """
    定时启动的爬虫程序

    Args:
        simple (bool, optional): 选择是否以简体存储. Defaults to True.
    """
    os.system("curl 'http://10.3.8.211/login' --data 'user=2019140570&pass=103010&line='")
    topics = json.load(open(os.path.join(globalFLAGS.static_dir, "crawl/chinatimes_topics.json")))  # 读取感兴趣的主题列表
    logf = open(globalFLAGS.log_dir+f"/{time.strftime('%Y-%m')}-chinatimes.log", "a", encoding="utf-8")
    for key, topic in topics.items():
        save_dir = os.path.join(globalFLAGS.news_data_dir, f"CT/{time.strftime('%Y')}-{time.strftime('%m')}")
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
                logf.write('[{}] INFO: {}: {} saved.\n'.format(time.strftime('%Y-%m-%d %H:%M:%S'), article.theme, article.title))
    logf.close()


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

    return Article(id_, url, time_, title, theme, keywords, content, "CT")


def test():
    xpath = "/html/body/div[2]/div/div[1]/nav/ol/li[2]/a/span"
    topic_url = "https://www.chinatimes.com/realtimenews/20201228005805-260407?chdtv"
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/74.0.3729.131 Safari/537.36'
    }
    main_response = requests.get(topic_url, headers=headers, timeout=(3, 10))
    main_html = etree.HTML(main_response.text)
    title_items = main_html.xpath(xpath)
    print(title_items)


def fix():
    root_dir = "/home/disk2/nuclear/news_data/PCube/CT/2021-01"
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
    kw_save_path = "/home/disk2/nuclear/news_data/PCube/CT/keywords.txt"
    url_save_path = "/home/disk2/nuclear/news_data/PCube/CT/history_url.txt"
    topics = json.load(open(os.path.join(globalFLAGS.static_dir, "crawl/chinatimes_topics.json")))
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/74.0.3729.131 Safari/537.36'
    }

    def reload_kw():
        """
        用于断点恢复，从文件中读取现有的所有关键词

        Returns:
            set: 无重复字符串集合
        """
        if os.path.exists(kw_save_path):
            keywords = []
            with open(kw_save_path, "r") as kwf:
                for word in kwf:
                    keywords.append(word.strip())
            keywords = set(keywords)
        else:
            keywords = set()
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
        root_dir = "/home/disk2/nuclear/news_data/PCube/CT/2021-01"
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

        root_dir = "/home/disk2/nuclear/news_data/PCube/CT/2020-12"
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
        urls = reload_url()

        title_xpath = "/html/body/div/div/div/div/section/div/ul/li/div/div/div/h3/a"
        # time_xpath = "/html/body/div/div/div/div/section/div/ul/li/div/div/div/div/time/span[1]"
        date_xpath = "/html/body/div/div/div/div/section/div/ul/li/div/div/div/div/time/span[2]"
        theme_xpath = "/html/body/div/div/div/div/section/div/ul/li/div/div/div/div/div[@class='category']/a"
        for word in keywords:
            i = 1
            while True:
                context_url = f"https://www.chinatimes.com/Search/{word}?page={i}&chdtv"  # 这里是搜索用的URL，需要根据网址修改
                try:
                    main_response = requests.get(context_url, headers=headers, timeout=(3, 10))
                except requests.exceptions.RequestException:
                    print('[{}] ERROR: {} connection time out.'.format(time.strftime('%Y-%m-%d %H:%M:%S'), context_url))
                    continue

                main_html = etree.HTML(main_response.text)
                date_items = main_html.xpath(date_xpath)
                title_items = main_html.xpath(title_xpath)
                theme_items = main_html.xpath(theme_xpath)

                if not title_items:
                    break

                date_ = date_items[0].text.split('/')
                if int(date_[0]) < 2019:
                    break

                for date, theme, title in zip(date_items, theme_items, title_items):
                    try:
                        date_ = date.text.split('/')
                        theme_ = [Opencc.to_simple(char) for char in theme.text]
                        theme_ = "".join(theme_)
                        if int(date_[0]) >= 2019 and theme_ in topics:
                            new_url = [theme_, title.attrib['href'].replace("com", "com/cn")+"?chdtv"]
                            if new_url not in urls:
                                urls.append(new_url)
                                with open(url_save_path, "a") as wf:
                                    wf.write("\t".join(new_url)+"\n")
                        elif int(date_[0]) < 2019:
                            break
                    except Exception:
                        continue

                i += 1
            print(f"[INFO] {word} complete.")

    # getKeywords()     # 从头开始的话释放这两行代码
    # getUrls()

    def change_type(byte):
        """
        用于将含有字节流的Article对象转为Json, 目前采用__dict__操作
        """
        if isinstance(byte, bytes):
            return str(byte, encoding = "utf-8")
        return json.JSONEncoder.default(byte)

    """
    根据URL获取所有的历史信息并存入相应的目录下，这一步运行非常慢，且容易出错，每完成一篇文章就保存一篇
    """
    urls = reload_url()
    for topic, url in urls:
        try:
            article = parse_article(url, topic)
        except Exception:
            continue
        if not article:
            os.system("curl 'http://10.3.8.211/login' --data 'user=2019140570&pass=103010&line='")  # 目前发现的问题均为没有在网关登陆的问题
            continue
        year, month, _ = article.time.split(" ")[0].split("-")
        # # 保存到文件
        # save_dir = f"/home/disk2/nuclear/news_data/PCube/CT/{year}-{month}"  # 保存到相应的位置
        # if not isdir(save_dir):
        #     os.mkdir(save_dir)
        # save_dir = os.path.join(save_dir, f"{topic}")
        # if not isdir(save_dir):
        #     os.mkdir(save_dir)
        # save_path = os.path.join(save_dir, f"{article.id}.html")

        # with open(save_path, 'w', encoding="utf-8") as f:
        #     f.write(article.to_file())
        #     print('[{}] INFO: {}: {} saved.'.format(time.strftime('%Y-%m-%d %H:%M:%S'), article.theme, article.url))

        # 保存到ES
        es = ElasticSearchTemplate()
        es.write(index = "article", doc_type = "doc", id = article.id, data = article.__dict__)


if __name__ == "__main__":
    # main(simple=True)
    # parse_article("https://www.chinatimes.com/cn/realtimenews/20201230002663-260407?chdtv")
    # test()
    
    # 执行 export PYTHONPATH=./ 在PCube3.0下
    history_news()
