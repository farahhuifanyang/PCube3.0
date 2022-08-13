'''
Author: your name
Date: 2021-06-09 10:09:19
LastEditTime: 2021-06-09 10:14:46
LastEditors: Cao Chenyu
Description: In User Settings Edit
FilePath: /PCube3.0/DAO/Elastic/Entity.py
'''
from DAO.ElasticSearchTemplate import ElasticSearchTemplate


class EntityFile(ElasticSearchTemplate):
    def __init__(self, url, timeout=1000, max_retries=10, retry=True) -> None:
        super().__init__(url, timeout=timeout, max_retries=max_retries, retry=retry)
        self.index = "entity"
        self.doc_type = "entity"

    def searchByKey(self, key):
        """
        按照es文件的主键进行查找

        Args:
            key (str): 主键

        Returns:
            dict: 返回唯一的文章内容或None
        """
        try:
            res = self.es.get(index=self.index, id=key)
        except Exception:
            # 数据库没有该数据
            return None
        res = res["_source"]
        return res

    def searchByValue(self, filter):
        """
        按照属性值查找es文件

        Args:
            filter (dict): 参与筛选的键值对

        Returns:
            list[dict]: 所有比中的文件数据
        """
        body = {
            "query": {
                "match": filter
            }
        }
        res = self.es.search(body=body)
        if res["hits"]["total"]["value"] <= 0:
            return None  # 数据库没有该数据
        res = [hit["_source"] for hit in res["hits"]["hits"]]
        return res

    def write(self, data, id):
        """
        向ES写入一条文件数据

        Args:
            data (dict): 文件的主要内容
            id (str): 文件的主键
        """
        self.es.index(index=self.index, doc_type=self.doc_type, id=id, body=data)

    def deleteByID(self, ids):
        """
        删除特定id的文件

        Args:
            index (str): 索引
            doc_type (str): 文件类型，以后可能会弃用
            id (str): 文件的主键
        """
        for eid in ids:
            self.es.delete(index=self.index, doc_type=self.doc_type, id=eid)
