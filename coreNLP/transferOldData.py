'''
Author: your name
Date: 2021-06-08 08:59:44
LastEditTime: 2021-06-09 11:38:11
LastEditors: Please set LastEditors
Description: In User Settings Edit
FilePath: /PCube3.0/coreNLP/transferOldData.py
'''
from py2neo import Graph
from configure import globalFLAGS
from DAO.Neo4jNode import Neo4jNode
from DAO.Neo4jEdge import Neo4jEdge
from DAO.Elastic.EntityFile import EntityFile


def getWikibase(form):
    """
    根据旧数据库中的实体名查询wikiID

    Args:
        form (str): 旧数据库实体名

    Returns:
        dict, dict: 需要存入neo4j和存入ES的数据
    """
    import wptools
    from openccpy.opencc import Opencc
    formal = form

    page = wptools.page(form, lang='zh')
    try:
        wiki = page.get(proxy=f'http://127.0.0.1:{globalFLAGS.EL_proxy_port}', timeout=20)
        introduction = wiki.data["exrest"]   # !!!!
    except Exception:
        return None, None

    if "label" in wiki.data:
        formal = wiki.data["label"]
    name = [Opencc.to_simple(char) for char in formal]
    name = "".join(name)
    name_cht = Opencc.to_traditional(formal)
    summary = wiki.data['description']
    if "claims" in wiki.data and "P856" in wiki.data["claims"]:
        web_site = wiki.data["claims"]["P856"][0]
    else:
        web_site = None
    if "wikibase" not in wiki.data or not wiki.data["wikibase"]:
        for_neo4j = {"name": name, "name_cht": name_cht, "summary": summary}
    else:
        for_neo4j = {"eid": wiki.data['wikibase'], "name": name, "name_cht": name_cht, "summary": summary}
    for_es = {"webSite": {"官网": web_site}, "introduction": introduction}
    for_es = dict(for_neo4j, **for_es)

    return for_neo4j, for_es


def getOldRels():
    """
    从旧的数据库里提取关系

    Yields:
        dict, str, dict: (实体1信息, 关系, 实体2信息)
    """
    cypher = "match (n)-[r]->(m) where any(label in labels(m) WHERE label in ['Location','Organization','Person']) return n,r,m"
    old_graph = Graph("bolt://10.112.235.173:7687", auth=("neo4j", "neo4j"))
    rels = old_graph.run(cypher)
    for rel in rels:
        e1 = dict(rel[0])
        r = rel[1]["relationship"]
        e2 = dict(rel[2])
        yield (e1, r, e2)


def save2new(old_rels):
    """
    将旧有的关系存入新数据库

    Args:
        old_rels (generator[tuple]): getOldRels返回的生成器

    Returns:
        [type]: [description]
    """
    neo4jnode = Neo4jNode(globalFLAGS.neo4j_url, globalFLAGS.neo4j_usr, globalFLAGS.neo4j_passwd)
    neo4jedge = Neo4jEdge(globalFLAGS.neo4j_url, globalFLAGS.neo4j_usr, globalFLAGS.neo4j_passwd)
    es = EntityFile(globalFLAGS.ES_url)
    for e1, rel, e2 in old_rels:
        if not rel:
            continue

        def saveNode(entity):
            node = neo4jnode.searchByValue({"name": entity["name"]})
            if node:    # 新数据库里有
                node = node.first()
            if not node:  # 新数据库里没有
                node, for_es = getWikibase(entity["name"])
                if node:    # 从网上查到了，写入新数据库
                    if "party" in entity:
                        node["party"] = entity["party"]
                    if entity["type"] == "人物":
                        node["etype"] = "PER"
                    elif entity["type"] == "地名":
                        node["etype"] = "LOC"
                    elif entity["type"] == "组织":
                        node["etype"] = "ORG"
                    else:
                        print(entity)
                    eid = neo4jnode.write(**node)
                    node = neo4jnode.searchByKey(eid=eid)
                else:   # 从网上没查到，自己解析
                    node = {"name": entity["name"]}
                    if entity["type"] == "人物":
                        node["etype"] = "PER"
                    elif entity["type"] == "地名":
                        node["etype"] = "LOC"
                    elif entity["type"] == "组织":
                        node["etype"] = "ORG"
                    else:
                        print(entity)

                    eid = neo4jnode.write(**node)  # 得到新eid
                    node = neo4jnode.searchByKey(eid=eid)

                if for_es:
                    if "eid" not in for_es:
                        for_es["eid"] = node["eid"]
                    es.write(for_es, for_es["eid"])
            elif "party" in entity:  # 更新新数据的党派字段
                neo4jnode.update(node["eid"], {"party": entity["party"]})
            return node

        e1node = saveNode(e1)
        e2node = saveNode(e2)

        neo4jedge.write(e1node["eid"], e2node["eid"], rel)


if __name__ == "__main__":
    # 执行后即可将旧数据库(xtyno2)中的关系数据迁移到新数据库(918-02)
    old_rels = getOldRels()
    save2new(old_rels)
