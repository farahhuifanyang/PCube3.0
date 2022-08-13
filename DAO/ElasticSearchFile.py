'''
Author: your name
Date: 2021-05-14 10:57:46
LastEditTime: 2021-07-14 20:13:19
LastEditors: Please set LastEditors
Description: 定义保存在ES中的文件的基类
FilePath: /PCube3/DAO/ElasticSearchFile.py
'''
from elasticsearch import Elasticsearch


class ElasticSearchTemplate(object):
    def __init__(self, url = ['10.105.242.74:9200'], timeout=1000, max_retries=10, retry=True) -> None:
        self.es = Elasticsearch(url, timeout=timeout, max_retries=max_retries, retry_on_timeout=retry)

    def searchByKey(self, key, index=None):
        """
        按照es文件的主键进行查找

        Args:
            key (str): 主键
            index (str, optional): es的索引，给定即可加速查找. Defaults to None.

        Returns:
            dict: 返回唯一的文章内容或None
        """
        if index:
            try:
                res = self.es.get(index=index, id=key)
            except Exception:
                # 数据库没有该数据
                return None
            res = res["_source"]
        else:
            body = {
                "query": {
                    "ids": {
                        "values": [
                            key
                        ]
                    }
                }
            }

            res = self.es.search(body=body)
            if res["hits"]["total"]["value"] <= 0:
                return None  # 数据库没有该数据
            res = res["hits"]["hits"][0]["_source"]
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

    def write(self,  index, doc_type, id, data):
        """
        向ES写入一条文件数据

        Args:
            index (str): 索引
            doc_type (str): 文件类型，以后可能会弃用
            id (str): 文件的主键
            data (dict): 文件的主要内容
        """
        self.es.index(index=index, doc_type=doc_type, id=id, body=data)

    def deleteByID(self, index, doc_type, ids):
        """
        删除特定id的文件

        Args:
            index (str): 索引
            doc_type (str): 文件类型，以后可能会弃用
            id (str): 文件的主键
        """
        for eid in ids:
            self.es.delete(index=index, doc_type=doc_type, id=eid)


if __name__ == "__main__":
    # 执行后可以将20个典型实体的信息存入ES
    from configure import globalFLAGS
    import json
    f = "static/processOnBaseData/criticalEntity.json"
    es = ElasticFile(globalFLAGS.ES_url)

    # with open(f, 'r') as cef:
    #     datas = json.load(cef)
    #     for i, data in enumerate(datas):
    #         es.searchByValue({"eid": "Q8349465"})
            # res = es.es.delete(index="entity", doc_type="entity", id=data["eid"])
            # es.write(data, "entity", "entity", data["eid"])
            # print(data["id"])
    # res = es.es.indices.delete('entity') 
    es.searchByValue()
