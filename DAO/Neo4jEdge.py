'''
Author: your name
Date: 2021-05-14 10:56:55
LastEditTime: 2021-06-09 09:20:08
LastEditors: Please set LastEditors
Description: 定义保存在Neo4j中的边的基类
FilePath: /PCube3.0/DAO/Neo4jEdge.py
'''
import json
from py2neo import Graph, Node, Relationship
from py2neo.matching import RelationshipMatcher


class Neo4jEdge(object):
    def __init__(self, url, user, passwd) -> None:
        self.graph = Graph(url, auth=(user, passwd), secure=False)
        self.matcher = RelationshipMatcher(self.graph)

    def searchByKey(self, eid1=None, eid2=None):
        """
        按照两个实体的id进行搜索

        Args:
            eid1 (str, optional): 源节点的实体id. Defaults to None.
            eid2 (str, optional): 被指向节点的实体id. Defaults to None.

        Returns:
            list[py2neo.Relation]: 无返回时为None或者空list，有返回时为py2neo的关系对象，用法与dict几乎无异，["name"]字段保存关系值
        """
        if not eid1 and not eid2:   # 不允许二者皆为空
            return
        if eid1:
            e1 = self.graph.nodes.match(eid=eid1).first()
            if not e1:
                print("E1 not exist, please create first")
                return None
            e1 = [e1]
        else:
            e1 = []
        if eid2:
            e2 = self.graph.nodes.match(eid=eid2).first()
            if not e2:
                print("E2 not exist, please create first")
                return None
            e2 = [e2]
        else:
            e2 = []
        nodes = e1+e2
        res = [rel for rel in self.matcher.match(nodes, r_type="OpenRel")]
        return res

    def write(self, eid1, eid2, relation):
        """
        写入一条关系到neo4j中

        Args:
            eid1 (str): 源节点的实体id
            eid2 (str): 被指向节点的实体id
            relation (str): 关系名称(指示词)

        Returns:
            str: 若有一个节点还没有创建，返回该节点的eid，要求创建，执行成功则返回None
        """
        e1 = self.graph.nodes.match(eid=eid1).first()
        if not e1:
            print("E1 not exist, please create first")
            return eid1
        e2 = self.graph.nodes.match(eid=eid2).first()
        if not e2:
            print("E2 not exist, please create first")
            return eid2
        edge = Relationship(e1, "OpenRel", e2, name=relation)
        self.graph.merge(edge)
        return None


if __name__ == "__main__":
    # 执行后将江启臣和蔡英文的样例关系存入neo4j
    from configure import globalFLAGS
    from DAO.Neo4jNode import Neo4jNode
    from DAO.Hbase.Entity import Entity
    from coreNLP.processOnBaseNewsData import getWikibase
    neo4jedge = Neo4jEdge(globalFLAGS.neo4j_url, globalFLAGS.neo4j_usr, globalFLAGS.neo4j_passwd)
    test_datas = [['Q8279603', '主席', 'Q31113', '中国国民党'],
                  ['Q8279603', '家乡', 'Q569604', '丰原区'],
                  ['Q8279603', '博士', 'Q1024426', '美国南卡罗来纳大学'],
                  ['Q8279603', '任教于', 'Q1024426', '美国南卡罗来纳大学'],
                  ['Q8279603', '硕士', 'Q235034', '匹兹堡大学'],
                  ['Q8279603', '前任', 'Q709307', '郝龙斌'],
                  ['Q8279603', '同事', 'Q16193401', '曾铭宗'],
                  ['Q8279603', '会见', 'Q9455382', '黄之锋'],
                  ['Q8279603', '击败', 'Q8350409', '翁美春'],
                  ['Q8279603', '对手', 'Q9095925', '赵少康'],
                  ['Q8279603', '结盟', 'Q9095925', '赵少康'],
                  ['Q8279603', '不认同', 'Q15031', '习近平'],
                  ['Q233984', '主席', 'Q903822', '民主进步党'],
                  ['Q233984', '出生地', 'Q271167', '中山区'],
                  ['Q233984', '接触', 'Q7825', '世界贸易组织'],
                  ['Q233984', '委员', 'Q715869', '立法院'],
                  ['Q233984', '下属', 'Q315528', '李登辉'],
                  ['Q233984', '下属', 'Q22368', '陈水扁'],
                  ['Q233984', '毕业于', 'Q32746', '国立台湾大学'],
                  ['Q233984', '硕士', 'Q49115', '康奈尔大学'],
                  ['Q233984', '博士', 'Q174570', '伦敦政治经济学院'],
                  ['Q233984', '会见', 'Q9455382', '黄之锋'],
                  ['Q233984', '主委', 'Q705141', '陆委会'],
                  ['Q233984', '搭配', 'Q551904', '苏贞昌'],
                  ['Q233984', '击败', 'Q8274095', '韩国瑜'],
                  ['Q233984', '继任', 'Q19216', '马英九']]
    neo4j = Neo4jNode(globalFLAGS.neo4j_url, globalFLAGS.neo4j_usr, globalFLAGS.neo4j_passwd)
    hbase = Entity(globalFLAGS.Hbase_ip, globalFLAGS.Hbase_prefix)
    for data in test_datas:
        res = neo4jedge.searchByKey(data[0], data[2])
        res = neo4jedge.write(data[0], data[2], data[1])
        while res:
            for_neo4j, for_hbase = getWikibase(data[2], data[3])
            # 保存至neo4j和hbase
            if for_neo4j:
                neo4j.write(**for_neo4j)
            if for_hbase:
                hbase.write(for_hbase["eid"], for_hbase)
            res = neo4jedge.write(data[0], data[2], data[1])
        print(res)
