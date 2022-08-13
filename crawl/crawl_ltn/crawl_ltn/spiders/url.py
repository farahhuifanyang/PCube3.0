import scrapy
import hashlib
from zhconv import convert
from crawl_ltn.items import CrawlLtnItem


url_save_path = r"F:\New_Life\项目组\人立方\PCube3\crawl\history_url_dedup.txt"
done_url_save_path = r"F:\New_Life\项目组\人立方\PCube3\crawl_ltn\done_url.txt"


class UrlSpider(scrapy.Spider):
    name = 'url'
    allowed_domains = ['https://news.ltn.com.tw/news/']
    title_xpath = "/html/body//div[@class='content']//div[@class='whitecon']/h1/text()"
    time_xpath = "/html/body//div[@class='content']//div[@class='whitecon']//span[@class='time']/text()"
    # theme_xpath 直接用url解析
    content_xpath = "/html/body//div[@class='content']//div[@class='whitecon']//div[@class='text boxTitle boxText']/p[not(@class)]/text()"

    def start_requests(self):
        urls, done_urls = set(), set()
        with open(url_save_path, 'r', encoding='utf-8') as f:
            for line in f.readlines():
                url = line.strip().split()[1]
                urls.add(url)
        
        with open(done_url_save_path, 'r', encoding='utf-8') as f:
            for line in f.readlines():
                url = line.strip()
                done_urls.add(url)

        self.logger.debug(f'总共需要解析 {len(urls)} 条新闻')
        urls = urls - done_urls
        self.logger.debug(f'已解析 {len(done_urls)} 条新闻')
        
        # proxy = {'proxy': 'http://127.0.0.1:8001'}
        for url in urls:
            # self.logger.info('发送请求 %s', url)
            yield scrapy.Request(url, callback=self.parse, dont_filter=True)

    def parse(self, response):

        # 转换繁体到简体
        def get_topic(url):
            topic_eg = url.split('/')[4]
            dic = {
                "政治": "politics",
                "社会": "society",
                "生活": "life",
                "国际": "world"
            }
            dic = dict(zip(dic.values(), dic.keys()))
            return dic[topic_eg]

        # self.logger.info('收到请求 %s', response.url)
        url_ = response.url
        title_ = response.xpath(self.title_xpath).extract()[0]
        title_ = title_.strip()  # 去除标题末尾的换行符
        time_ = response.xpath(self.time_xpath).extract()[0].split()
        content_list_ = response.xpath(self.content_xpath).extract()
        content_list_ = [content.strip() for content in content_list_]
        time_ = time_[0].replace("/", "-") + ' ' + time_[1] + ':00'
        content_ = "\\n".join(content_list_)
        keywords_ = []
        title_ = convert(title_, 'zh-cn')
        content_ = convert(content_, 'zh-cn')
        file_md5 = hashlib.md5(content_.encode())
        id_ = file_md5.hexdigest()
        theme_ = get_topic(url_)
        # self.logger.info(f'解析完毕 {url_}, id={id_}, theme={theme_}, title={title_}, time={time_}, content={content_}')

        item = CrawlLtnItem()
        item['url'] = url_
        item['title'] = title_
        item['time'] = time_
        item['keywords'] = keywords_
        item['content'] = content_
        item['id_'] = id_
        item['theme'] = theme_
        yield item




