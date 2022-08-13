# Define your item pipelines here
#
# Don't forget to add your pipeline to the ITEM_PIPELINES setting
# See: https://docs.scrapy.org/en/latest/topics/item-pipeline.html


# useful for handling different item types with a single interface
import time
from configure import globalFLAGS
import json
from os.path import isdir
import os
from classes.Article import Article
from itemadapter import ItemAdapter
import sys
sys.path.append(r'F:\New_Life\项目组\人立方\PCube3')


class CrawlLtnPipeline:
    def process_item(self, item, spider):
        id_ = item['id_']
        url = item['url']
        title = item['title']
        time_ = item['time']
        theme = item['theme']
        content = item['content']
        keywords = item['keywords']

        article = Article(id_, url, time_, title, theme, keywords, content)
        year, month, _ = article.time.split(" ")[0].split("-")

        save_dir = f"F:/New_Life/项目组/人立方/news_data/{year}-{month}"    # 保存到相应的位置
        done_url_sv_path = r'F:/New_Life/项目组/人立方/PCube3/crawl_ltn/done_url.txt'

        if not isdir(save_dir):
            os.mkdir(save_dir)
        save_dir = os.path.join(save_dir, f"{theme}")
        if not isdir(save_dir):
            os.mkdir(save_dir)
        save_path = os.path.join(save_dir, f"{article.id}.html")
        # print(save_dir)

        with open(save_path, 'w', encoding="utf-8") as f:
            f.write(article.to_file())
            # print(article.to_file())
            # self.logger.info("保存成功 %s", url)
            print('[{}] INFO: {}: {} saved.'.format(time.strftime('%Y-%m-%d %H:%M:%S'), article.theme, article.url))

        with open(done_url_sv_path, 'a', encoding='utf-8') as f:
            f.write(url + '\n')
