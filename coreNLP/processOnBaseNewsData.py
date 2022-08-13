'''
Author: your name
Date: 2021-05-11 08:58:52
LastEditTime: 2021-06-10 10:52:23
LastEditors: Please set LastEditors
Description: process performed on base data, the result of every step must be stored and could be restart in check point
FilePath: /PCube3.0/coreNLP/processOnBaseData.py
'''
import json
import os

from configure import globalFLAGS
from classes.Article import Article

from preprocess.forRealTime import processOneArticle
from coreNLP.NER.main import EntityRecognizer

from coreNLP.EL.WikiLinker import WikiLinker
from coreNLP.EL.main import EntityLinker

from DAO.Neo4jNode import Neo4jNode
from DAO.Hbase.Entity import Entity
from DAO.Hbase.ArticleEntity import ArticleEntity
from DAO.Elastic.EntityFile import EntityFile


def getWikibase(wiki_id, form):
    """
    实体数据写入数据库的部分使用的函数，用于从wiki获取基本信息

    Args:
        wiki_id (str): 实体的wikiID
        form (str): 实体在句子里的名称

    Returns:
        [type]: [description]
    """
    import wptools
    from openccpy.opencc import Opencc
    page = wptools.page(wikibase=wiki_id, lang='zh')
    try:
        wiki = page.get_wikidata(proxy=f'http://127.0.0.1:{globalFLAGS.EL_proxy_port}')
    except Exception:
        return None, None
    if "label" in wiki.data and wiki.data["label"]:
        formal = wiki.data["label"]
    else:
        formal = form
    name = [Opencc.to_simple(char) for char in formal]
    name = "".join(name)
    name_cht = Opencc.to_traditional(formal)
    summary = wiki.data['description']
    if "P856" in wiki.data["claims"]:
        web_site = wiki.data["claims"]["P856"][0]
    else:
        web_site = None

    def isLOC(instance):
        LOCinstance = ["Q5284036", "Q82794", "Q515", "Q486972", "Q23442", "Q1496967",
                       "Q5107", "Q2418896", "Q705296", "Q3622002", "Q35657"]
        for ins in LOCinstance:
            if ins in instance:
                return True
        return False
    if "P31" in wiki.data["claims"]:
        if "Q5" in wiki.data["claims"]["P31"] or "Q4164871" in wiki.data["claims"]["P31"]:
            etype = "PER"
        elif "Q6256" in wiki.data["claims"]["P31"] or "Q3624078" in wiki.data["claims"]["P31"]:
            etype = "REG"
        elif isLOC(wiki.data["claims"]["P31"]):
            etype = "LOC"
        elif "Q4167410" in wiki.data["claims"]["P31"]:
            etype = None
        else:
            etype = "ORG"
    else:
        etype = None

    for_neo4j = {"eid": wiki_id, "name": name, "name_cht": name_cht, "summary": summary, "etype": etype}
    for_es = for_neo4j.copy()
    for_es["webSite"] = {"官网": web_site}

    page = wptools.page(formal, lang='zh')
    try:
        wiki = page.get(proxy=f'http://127.0.0.1:{globalFLAGS.EL_proxy_port}', timeout=20)
        for_es["introduction"] = wiki.data["exrest"]   # !!!!
    except Exception:
        for_es["introduction"] = None

    return for_neo4j, for_es


def NERonBaseData():
    """
    处理目前为止已经爬取的全部数据，以每100篇文章为单位保存在临时位置
    """
    i = 0
    # outs = []
    log_parsed = 0  # 记录已经爬取过的文件名，用于断点重启
    inputs = {}  # 暂存100条文章的输入
    ner = EntityRecognizer()    # 初始化NER模块

    for root, sites, files in os.walk(globalFLAGS.news_data_dir):
        for file in files:
            file = os.path.join(root, file)
            if not file.endswith("html"):  # 不处理非文章文件
                continue
            i += 1
            if i < log_parsed:  # 不处理之前处理过的文件
                continue

            try:    # 解析文件
                aritcle = Article.init_from_file(os.path.join(globalFLAGS.news_data_dir, file))
                k, v = processOneArticle(aritcle)
            except Exception:
                print(f"{file} failed")
                continue
            inputs[k] = v   # 组装成NER输入格式
            print(f"{file} parsed")

            if i % 100 == 0:    # 每一百条启动一次批量NER
                out = ner.run(inputs)
                filename = str(i // 100)    # 每一百条保存成一个文件
                filename = filename.zfill(4)
                with open(os.path.join(globalFLAGS.tmp_result_dir, "NER", filename), "w") as wf:
                    json.dump(out, wf, ensure_ascii=False)
                inputs = {}


def ELOnBaseData():
    """
    从实体识别的结果进行实体链接和消歧
    自动进行断点重启，若想重新开始需要清除linded_dir下的所有文件
    必须在翻墙环境下运行 ！！！！！
    每个文件输入：
    {“文章ID”: [{“sentence”: “原文句子”, “entities”:
    [{“form”: “实体在文中的表述”, “type”: “实体类型”,
    “idx”: [实体开始位置, 实体结束位置]}]}]}
    """
    ner_dir = os.path.join(globalFLAGS.tmp_result_dir, "NER")
    linked_dir = os.path.join(globalFLAGS.tmp_result_dir, "EL_Linked")
    cannot = os.path.join(globalFLAGS.tmp_result_dir, "CannotEL")

    entity_linker = EntityLinker()
    wiki_linker = WikiLinker()
    for file in os.listdir(ner_dir):
        fullpath = os.path.join(ner_dir, file)
        outpath = os.path.join(linked_dir, file)
        if os.path.exists(outpath):  # 若这篇文章已经处理完成并保存过，直接跳过
            continue
        files = json.load(open(fullpath, "r", encoding="utf-8"))
        for aid, sents in files.items():
            form_map = {}   # 记录本文中已有的实体form为可能出现的简写准备
            for sidx, sent in enumerate(sents):
                for eidx, entity in enumerate(sent["entities"]):
                    if len(entity["form"]) > 1 and entity["form"] not in form_map:
                        form_map[entity["form"]] = entity["type"]
                        form = entity["form"]
                    else:
                        form = entity["form"]
                        for f, t in form_map.items():   # 文字有重合，类型相同，应该认为是共指
                            if entity["form"] in f and t == entity["type"]:
                                form = f
                                break
                    candidates = wiki_linker.queryFromDB(form)    # 先查数据库
                    if len(candidates) > 1:   # 对数据库里的消歧
                        inputs = {"sentence": sent["sentence"],
                                  "form": form, "type": entity["type"],
                                  "candidates": candidates}
                        best_cand = entity_linker.run(inputs)   # 得到得分最高的实体
                        if best_cand:
                            files[aid][sidx]["entities"][eidx]["wikiID"] = best_cand["eid"]
                            files[aid][sidx]["entities"][eidx]["name"] = best_cand["formal"]
                            break
                    elif candidates and candidates[0]["eid"]:
                        best_cand = candidates[0]   # 得到得分最高的实体
                        files[aid][sidx]["entities"][eidx]["wikiID"] = best_cand["eid"]
                        files[aid][sidx]["entities"][eidx]["name"] = best_cand["formal"]
                        break
                    elif candidates:
                        einfo = {"aid": aid, "sidx": sidx, "eidx": eidx}
                        einfo = json.dumps(einfo, ensure_ascii=False) + "\n"
                        with open(os.path.join(cannot, file), 'w') as wf:
                            wf.write(einfo)
                        break
                    candidates = wiki_linker.queryFromWiki(form)
                    if len(candidates) > 1:   # 对数据库里的消歧
                        inputs = {"sentence": sent["sentence"],
                                  "form": form, "type": entity["type"],
                                  "candidates": candidates}
                        best_cand = entity_linker.run(inputs)   # 得到得分最高的实体
                        if not best_cand:
                            einfo = {"aid": aid, "sidx": sidx, "eidx": eidx}
                            einfo = json.dumps(einfo, ensure_ascii=False) + "\n"
                            with open(os.path.join(cannot, file), 'w') as wf:
                                wf.write(einfo)
                            break
                        files[aid][sidx]["entities"][eidx]["wikiID"] = best_cand["eid"]
                        files[aid][sidx]["entities"][eidx]["name"] = best_cand["formal"]
                        wiki_linker.writeNewAlias(form, best_cand)
                    elif candidates:
                        best_cand = candidates[0]
                        files[aid][sidx]["entities"][eidx]["wikiID"] = best_cand["eid"]
                        files[aid][sidx]["entities"][eidx]["name"] = best_cand["formal"]
                        wiki_linker.writeNewAlias(form, best_cand)
                    else:
                        einfo = {"aid": aid, "sidx": sidx, "eidx": eidx}
                        einfo = json.dumps(einfo, ensure_ascii=False) + "\n"
                        with open(os.path.join(cannot, file), 'w') as wf:
                            wf.write(einfo)
                        wiki_linker.writeNewAlias(form, {"eid": ""})

        print(f"{file} finish")
        with open(outpath, 'w') as wf:    # 每个NER文件保存一次
            json.dump(files, wf, ensure_ascii=False)


def writeEntity():
    """
    从网络上收集实体信息保存入数据库
    自动进行断点重启，若想重新开始需删除static/processOnBaseData/EL_saved.txt的内容
    """
    with open("static/processOnBaseData/EL_saved.txt", "r") as ellog:
        line = ellog.readlines()
    if line:
        isParsed = True
        line = line[0].strip()
    else:
        isParsed = False
    linked_dir = os.path.join(globalFLAGS.tmp_result_dir, "EL_Linked")
    neo4j = Neo4jNode(globalFLAGS.neo4j_url, globalFLAGS.neo4j_usr, globalFLAGS.neo4j_passwd)
    es = EntityFile(globalFLAGS.ES_url)
    for file in os.listdir(linked_dir):
        fullpath = os.path.join(linked_dir, file)
        files = json.load(open(fullpath, "r", encoding="utf-8"))
        for aid, sents in files.items():
            if isParsed:
                if aid == line:
                    isParsed = False
                continue
            for sidx, sent in enumerate(sents):
                for eidx, entity in enumerate(sent["entities"]):
                    # 查neo4j
                    if "wikiID" in entity and not neo4j.searchByKey(entity["wikiID"]):
                        # neo4j没有，查wiki
                        for_neo4j, for_es = getWikibase(entity["wikiID"], entity["form"])
                        # 保存至neo4j和hbase
                        if for_neo4j:
                            if entity["type"] != "PER" or not for_neo4j["etype"]:
                                for_neo4j["etype"] = entity["type"]
                            neo4j.write(**for_neo4j)
                        if for_es:
                            es.write(for_es, for_es["eid"])
            with open("static/processOnBaseData/EL_saved.txt", "w") as ellog:
                ellog.write(aid+"\n")


def linkArticleAndEntity():
    linked_dir = os.path.join(globalFLAGS.tmp_result_dir, "EL_Linked")
    neo4j = Neo4jNode(globalFLAGS.neo4j_url, globalFLAGS.neo4j_usr, globalFLAGS.neo4j_passwd)
    hbase = ArticleEntity(globalFLAGS.Hbase_ip, globalFLAGS.Hbase_prefix)
    for file in os.listdir(linked_dir):
        fullpath = os.path.join(linked_dir, file)
        files = json.load(open(fullpath, "r", encoding="utf-8"))
        for aid, sents in files.items():
            for sidx, sent in enumerate(sents):
                for eidx, entity in enumerate(sent["entities"]):
                    # 查neo4j
                    if "wikiID" in entity and neo4j.searchByKey(entity["wikiID"]):
                        hbase.write(aid, entity["wikiID"])


def moveFromHbaseToES():
    """
    用于将暂存在hbase中的数据转存到ES中，以后将弃用
    """
    neo4j = Neo4jNode(globalFLAGS.neo4j_url, globalFLAGS.neo4j_usr, globalFLAGS.neo4j_passwd)
    hbase = Entity(globalFLAGS.Hbase_ip, globalFLAGS.Hbase_prefix)
    es = EntityFile(globalFLAGS.ES_url)
    entities = neo4j.searchAll()
    for neo4j_entity in entities:
        eid = neo4j_entity["eid"]
        neo4j_entity = dict(neo4j_entity)

        if es.searchByKey(eid):
            continue
        hbase_entity = hbase.searchByKey(eid)
        if hbase_entity:
            hbase_entity = hbase_entity[0]
            es_file = dict(neo4j_entity, **hbase_entity)
            if not es_file["webSite"]:
                es_file.pop("webSite")
            else:
                es_file["webSite"] = {"官网": es_file["webSite"]}
        else:
            es_file = neo4j_entity
        es.write(es_file, eid)


if __name__ == "__main__":
    ELOnBaseData()
    writeEntity()
    # linkArticleAndEntity()
