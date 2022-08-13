'''
Author: your name
Date: 2021-05-14 10:56:32
LastEditTime: 2021-06-09 10:20:39
LastEditors: Please set LastEditors
Description: 定义保存在Neo4j中的节点的基类
FilePath: /PCube3.0/DAO/Neo4jNode.py
'''
from py2neo import Graph, Node
from py2neo.matching import NodeMatcher


class Neo4jNode(object):
    def __init__(self, url, user, passwd) -> None:
        self.graph = Graph(url, auth=(user, passwd), secure=False)
        self.matcher = NodeMatcher(self.graph)

    def _getNextID(self):
        """
        私有函数，用于获取无法消歧的实体当前标注的id

        Returns:
            str: 无法消歧的实体当前标注的id
        """
        nid = self.matcher.match(name="NextID").first()
        eid = "N"+str(nid["val"]).zfill(8)
        nid["val"] += 1
        self.graph.push(nid)
        return eid

    def searchByKey(self, eid, etype=None):
        """按实体id查找

        Args:
            eid (str): 实体id

        Returns:
            [type]: [description]
        """
        # if etype:
        #     result = self.matcher.match(eid=eid)
        try:
            result = self.matcher.match(eid=eid)
            if len(result) > 0:
                return result.first()
            else:
                return None
        except Exception as e:
            return None

    def searchByValue(self, filter):
        """
        以列的值进行筛选性搜索
        返回匹配成功的行

        Args:
            filter (dict): 列-值的对应表，格式为{"列名": "列值"}

        """
        result = self.matcher.match(**filter)
        if len(result) > 0:
            return result
        else:
            return None

    def searchAll(self):
        """
        查询已有的所有实体数据

        Returns:
            generator: 依次返回查询结果
        """
        all_cypher = "match (n) where any(label in labels(n) WHERE label in ['PER','LOC','ORG','REG','OTH']) return n"
        res = self.graph.run(all_cypher)
        for node in res:
            yield node[0]

    def write(self, name, etype, eid=None, name_cht=None, summary=None, party=None, position=None):
        """
        写入之前不存在的新的节点

        Args:
            name (str): 实体名
            etype (str): 实体类型
            eid (str, optional): 实体ID，消歧成功时有. Defaults to None.
            name_cht (str, optional): 繁体中文名. Defaults to None.
            summary (str, optional): 简介，消歧成功时有. Defaults to None.
            party (str, optional): 少数特别人物实体有. Defaults to None.
            position (str, optional): 少数特别人物实体有. Defaults to None.
        Returns:
            str: 返回创建的实体的id，为没有提供id的实体使用
        """
        if not eid:
            eid = self._getNextID()
        node = Node(etype, name=name, eid=eid, name_cht=name_cht, summary=summary, party=party, position=position)
        self.graph.create(node)
        return eid

    def update(self, eid, properties):
        """
        在现有节点上进行修改

        Args:
            eid (str): 节点id
            properties (dict): 需要修改的属性
        """
        node = self.searchByKey(eid)
        for k, v in properties.items():
            node[k] = v
        self.graph.push(node)

    def deleteAll(self):
        """
        极度危险 ！！！！！！！
        删除人立方所有保存在neo4j的数据
        """
        entity_types = [
            'LOC', 'ORG', 'PER', 'REG', 'OTH'
        ]
        for t in entity_types:
            delete_cypher = "match (e:{})-[r]-() delete r".format(t)
            self.graph.run(delete_cypher)
            delete_cypher = "match (e:{}) delete e".format(t)
            self.graph.run(delete_cypher)

        id_reset = "match (n:counter) set n.val=2"
        self.graph.run(id_reset)


if __name__ == "__main__":
    # 执行后可以删除neo4j中所有的节点
    from configure import globalFLAGS
    neo4jnode = Neo4jNode(globalFLAGS.neo4j_url, globalFLAGS.neo4j_usr, globalFLAGS.neo4j_passwd)
    # neo4jnode.deleteAll()  # 若要删除所有节点，释放该行
