'''
Author: your name
Date: 2021-06-10 08:54:29
LastEditTime: 2021-06-22 17:32:25
LastEditors: Please set LastEditors
Description: In User Settings Edit
FilePath: /PCube3.0/DAO/Elastic/EventFile.py
'''
from DAO.ElasticSearchTemplate import ElasticSearchTemplate


class ElasticSearchTemplate(ElasticSearchTemplate):
    def __init__(self, url, timeout=1000, max_retries=10, retry=True) -> None:
        super().__init__(url, timeout=timeout, max_retries=max_retries, retry=retry)
        self.index = "event_summary"
        self.doc_type = "doc"

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

    def searchByValue(self, filter, precise=False):
        """
        按照属性值查找es文件

        Args:
            filter (dict): 参与筛选的键值对
            precise (bool): 选择是否精确匹配

        Returns:
            list[dict]: 所有比中的文件数据
        """
        if not precise:
            body = {
                "query": {
                    "match": filter
                }
            }
        else:
            body = {
                "query": {
                    "match_phrase": filter
                }
            }
        res = self.es.search(body=body)
        if res["hits"]["total"]["value"] <= 0:
            return None  # 数据库没有该数据
        res = [hit["_source"] for hit in res["hits"]["hits"]]
        return res

    def searchByTime(self, year, month=None, day=None):
        """
        按照时间搜索事件

        Args:
            year ([type]): 年
            month ([type], optional): 月. Defaults to None.
            day ([type], optional): 日. Defaults to None.

        Returns:
            list[dict]: 所有比中的文件数据
        """
        prefix = str(year)
        if month:
            prefix += "-"+str(month).zfill(2)
            if day:
                prefix += "-"+str(day).zfill(2)
        body = {
            "query": {
                "match_phrase_prefix": {
                    "eid": prefix
                }
            },
            "size": 10000
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


if __name__ == "__main__":
    # 执行后可以将20个典型实体的信息存入ES
    from configure import globalFLAGS
    import json
    f = "static/processOnBaseData/criticalEntity.json"
    es = ElasticSearchTemplate(globalFLAGS.ES_url)

    with open(f, 'r') as cef:
        datas = json.load(cef)
        for i, data in enumerate(datas):
            es.searchByTime("2021", "03")
            # es.searchByKey("2020-11-230000007502")
            # es.searchByValue({"eid": "2020-11-230000007502"}, precise=True)
            # res = es.es.delete(index="entity", doc_type="entity", id=data["eid"])
            # es.write(data, "entity", "entity", data["eid"])
            # print(data["id"])
    # res = es.es.indices.delete('entity')
