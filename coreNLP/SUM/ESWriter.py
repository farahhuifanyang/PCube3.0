'''
Author: Guowenying
Date: 2021-06-03 11:14:27
LastEditTime: 2021-06-10 10:01:56
LastEditors: Guowenying
Description: In User Settings Edit
FilePath: /PCube3/coreNLP/SUM/ESWriter.py
'''
from elasticsearch5 import Elasticsearch
import datetime
from DAO.ElasticFile import ElasticFile
from configure import globalFLAGS
# elasticsearch集群服务器的地址
ES = [
    '10.105.242.74:9200'
]

# 创建elasticsearch客户端
es = Elasticsearch(
    ES,
    # 启动前嗅探es集群服务器
    sniff_on_start=True,
    # es集群服务器结点连接异常时是否刷新es节点信息
    sniff_on_connection_fail=True,
    # 每60秒刷新节点信息
    sniffer_timeout=60
)


# 或者
body = {
    "query":{
        "match_all":{}
    },
    # "size":50000
    "size": "10000"

}
res = es.search(index="event_summary",doc_type="doc",body=body)
# print(type(res["hits"]["hits"]))
for value in res["hits"]["hits"]:
    # print(value["_source"]['timestamp'])
    if value["_source"]['timestamp'].startswith('2021-06-03'):
        print(value["_source"])
# print(res['hits']['hits'])
# es.indices.create(index='event_summary')

# es.index(index="event_summary",doc_type="doc",id='2020-06-030000026943',body={"eid":'2020-06-030000026943', "abstract":'新冠肺炎疫情期间，花卉销量大减，为了帮农民行销花卉，彰化县政府向农粮署争取经费在校园开办花艺美学课程，让8万名中小学生学习花艺，彰化县长王惠美今天到湖南国小和学生们一起上插花课表示，希望这些花材能充分运用也促进花卉买气，更将美感教育结合产业，让美感教育向下扎根。', "name":"为振兴花卉产业，台中举办校园花艺课程","timestamp":datetime.datetime.now()}) 
# search = ElasticFile(globalFLAGS.ES_url)
# print(search.searchByKey(key = "2020-06-030000026943", index = 'event_summary'))