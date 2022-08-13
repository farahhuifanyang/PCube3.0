'''
Author: your name
Date: 2020-12-30 09:08:42
LastEditTime: 2021-06-26 16:13:38
LastEditors: Please set LastEditors
Description: In User Settings Edit
FilePath: /PCube3/classes/Article.py
'''


class Article(object):
    def __init__(self, id, url, time, title, theme, keywords, content, source):
        """
        从爬虫的结果中构造
        :param id:  文章id
        :param url: 原文网址
        :param time:    原文发布时间
        :param title:   文章标题
        :param theme:   文章主题topic
        :param keywords:    文章关键词，包括实体、事件
        :param content: 文章内容
        :param source:  文章来源
        """
        self.id = id
        self.url = url
        self.time = time
        self.title = title
        self.theme = theme
        self.keywords = keywords
        self.content = content
        self.source = source

    def to_dict(self):
        """
        转化为dict类型数据
        :return: 所有属性均以key-value表示的dict
        """
        class_dict = {  # 组合成文章数据
            'id': self.id,
            'url': self.url,
            'time': self.time,
            'title': self.title,
            'theme': self.theme,
            'keywords': "\t".join(self.keywords),
            'content': self.content,
            'source': self.source
        }

        return class_dict

    def to_file(self):
        file_content = ""
        file_content += 'id:\t' + self.id + '\n'
        file_content += 'url:\t' + self.url + '\n'
        file_content += 'time:\t' + self.time + '\n'
        file_content += 'title:\t' + self.title + '\n'
        file_content += 'theme:\t' + self.theme + '\n'
        file_content += 'keywords:\t' + "\t".join(self.keywords) + '\n'
        file_content += 'content:\t' + self.content + '\n'
        file_content += 'source:\t' + self.source + '\n'
        return file_content

    def __str__(self):
        """
        重写tostring方法
        :return: 转化为dict再字符串化
        """
        str_dict = str(self.to_dict())
        return str_dict

    def __lt__(self, other):
        """
        less than函数，排序时使用，先按照time顺序排列，相同时按照id排列
        :param other: 另一个Article对象
        :return:
        """
        if self.time < other.time:
            return True
        elif self.time == other.time:
            return self.id < other.id
        else:
            return False

    @classmethod
    def init_from_db(cls, dbresult):
        """
        从数据库查询结果构造, 未完成
        :param dbresult:
        """
        id = dbresult["id"]

    @classmethod
    def init_from_file(cls, file):
        """
        从数据库查询结果构造, 未完成
        :param file: 文件路径
        """
        with open(file, 'r', encoding='utf-8') as rf:
            lines = rf.readlines()
            while "\n" in lines:
                lines.remove("\n")
            id = lines[0].strip().split("\t")[1]
            url = lines[1].strip().split("\t")[1]
            time = lines[2].strip().split("\t")[1]
            title = lines[3].strip().split("\t")[1]
            theme = lines[4].strip().split("\t")[1]
            if len(lines) >= 7:
                keywords = lines[5].strip().split("\t")[1:]
                content = "".join(lines[6].strip().split("\t")[1:] + lines[7:])
            else:
                keywords = []
                content = lines[5].strip().split("\t")[1]
        return Article(id, url, time, title, theme, keywords, content, "")
