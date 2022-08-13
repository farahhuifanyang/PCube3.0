'''
Author: your name
Date: 2020-12-30 14:28:47
LastEditTime: 2020-12-30 17:12:11
LastEditors: Please set LastEditors
Description: In User Settings Edit
FilePath: /PCube3/move.py
'''
import os
import re
import json
import hashlib
from os.path import isdir
from configure import globalFLAGS
from classes.Article import Article


def move():
    topics = {"两岸": "两岸", "产业": "财经", "全球": "国际", "军事": "军事",
              "国际": "国际", "政治": "政治", "政治要闻": "政治", "生活": "生活",
              "社会": "社会", "财经": "财经", "金融": "财经", "证券": "财经"}
    root_dir = "/home/disk2/nuclear/java/data"
    for topic_dir in os.listdir(root_dir):
        for file in os.listdir(os.path.join(root_dir, topic_dir)):
            file = os.path.join(root_dir, topic_dir, file)
            with open(file, 'r') as rf:
                lines = rf.readlines()
                url = lines[0].strip().split("\t")[1]
                time = lines[1].strip().split("\t")[1]
                title = lines[2].strip().split("\t")[1]
                theme = lines[3].strip().split("\t")[1]
                content = lines[4].strip().split("\t")[1]

                file_md5 = hashlib.md5(content.encode())
                id_ = file_md5.hexdigest()

                # re中字符串需要是正则式，其本身不能被python当作特殊字符处理，故用r使得re接收到完整的正则式字符串
                time = re.sub(r"(\d{4}-\d{2}-\d{2})T(\d{2}:\d{2}:\d{2}).*", r"\1 \2", time)
                year = re.sub(r"(\d{4})-\d{2}-\d{2}.*", r"\1", time)
                month = re.sub(r"\d{4}-(\d{2})-\d{2}.*", r"\1", time)

                article = Article(id_, url, time, title, theme, content)

            if theme in topics:
                save_dir = os.path.join(globalFLAGS.news_data_dir, f"{year}-{month}")
                if not isdir(save_dir):
                    os.mkdir(save_dir)
                save_dir = os.path.join(save_dir, f"{topic_dir}")
                if not isdir(save_dir):
                    os.mkdir(save_dir)
                    save_path = os.path.join(save_dir, f"{article.id}.html")
                    with open(save_path, 'w', encoding="utf-8") as f:
                        f.write(article.to_file())
                        print('INFO: {}: {} saved.'.format(topic_dir, article.title))


def old():
    topics = json.load(open(os.path.join(globalFLAGS.static_dir, "crawl/chinatimes_topics.json")))
    root_dir = "/home/disk2/nuclear/news_data/PCube/old"
    for file in os.listdir(root_dir):
        file = os.path.join(root_dir, file)
        with open(file, 'r') as rf:
            lines = rf.readlines()
            i = 0
            while i < len(lines):
                url = lines[i+1].strip().split("\t")[1]
                time = lines[i+2].strip().split("\t")[1]
                title = lines[i+3].strip().split("\t")[1]

                try:
                    theme = lines[i+4].strip().split("\t")[1]
                    content = lines[i+5].strip().split("\t")[1]
                except Exception:
                    title += lines[i+4].strip()
                    theme = lines[i+5].strip().split("\t")[1]
                    content = lines[i+6].strip().split("\t")[1]
                    i += 1
                i += 6

                file_md5 = hashlib.md5(content.encode())
                id_ = file_md5.hexdigest()

                # re中字符串需要是正则式，其本身不能被python当作特殊字符处理，故用r使得re接收到完整的正则式字符串
                time = re.sub(r"(\d{4}-\d{2}-\d{2})T(\d{2}:\d{2}:\d{2}).*", r"\1 \2", time)
                year = re.sub(r"(\d{4})-\d{2}-\d{2}.*", r"\1", time)
                month = re.sub(r"\d{4}-(\d{2})-\d{2}.*", r"\1", time)

                article = Article(id_, url, time, title, theme, content)

                valid = False
                for topic, _ in topics.items():
                    if theme in topic or topic in theme:
                        valid = True
                        break

                if valid:
                    save_dir = os.path.join(globalFLAGS.news_data_dir, f"{year}-{month}")
                    if not isdir(save_dir):
                        os.mkdir(save_dir)
                    save_dir = os.path.join(save_dir, f"{theme}")
                    if not isdir(save_dir):
                        os.mkdir(save_dir)
                        save_path = os.path.join(save_dir, f"{article.id}.html")
                        with open(save_path, 'w', encoding="utf-8") as f:
                            f.write(article.to_file())
                            print('INFO: {}: {} saved.'.format(theme, article.title))


if __name__ == "__main__":
    old()
