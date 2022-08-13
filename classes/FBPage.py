'''
Author: Liuhezi
Date: 2021-04-23 15:13:37
LastEditTime: 2021-04-28 11:26:36
LastEditors: Liuhezi
Description: Facebook page class
FilePath: /PCube3.0/classes/FBPage.py
'''


class FBPage(object):
    def __init__(self,id,name,post):
        """从爬虫结果中构造facebook主页类

        Args:
            id   (str): wikidata对应的人物ID
            name (str): 用户名
            post (str): 用户所有帖子
        """
        self.id = id
        self.name = name
        self.post = post
    
    def to_dict(self):
        """
        转化为dict类型数据
        """
        class_dict = {
            'id':self.id,
            'name':self.name,
            'post':self.post
        }

        return class_dict
    
    @classmethod
    def init_from_db(cls,dbresult):
        """
        从数据库查询结果构造, 未完成
        :param dbresult:
        """
        id = dbresult["id"]

    @classmethod
    def init_from_file(cls,root,files):
        """从本地爬虫文件中构造facebook主页类
        
        Args:
            root (str): 用户目录
            files (str): 该用户下帖子文件目录
        """
        post = ''
        if files:
            for file in files:
                f = open(root+'/'+file)
                iter_f = iter(f)
                s = ''
                for line in iter_f:
                    if line != '':
                        s+=line.strip()
                if s:
                    post+=s+'<sep>'
            name = root.split('/')[-1].split('_')[0]
            id = root.split('/')[-1].split('_')[1]
        return FBPage(id,name,post)

        
    