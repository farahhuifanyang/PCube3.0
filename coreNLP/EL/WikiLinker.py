'''
Author: your name
Date: 2021-05-14 10:42:40
LastEditTime: 2021-05-27 18:13:38
LastEditors: Please set LastEditors
Description: In User Settings Edit
FilePath: /PCube3.0/coreNLP/EL/WikiLinker.py
'''
import wptools

from configure import globalFLAGS
from DAO.Hbase.EntityAlias import EntityAlias


class WikiLinker:
    def __init__(self) -> None:
        # Hbase别名表
        self.aliasBase = EntityAlias(globalFLAGS.Hbase_ip, globalFLAGS.Hbase_prefix)
        self.port = globalFLAGS.EL_proxy_port

    def queryFromDB(self, form):
        # 从别名表查询
        res = self.aliasBase.searchByKey(form)
        return res if res else []

    def queryFromWiki(self, form, skip_eids=[], rec=True):
        """
        从wikidata查询

        Args:
            form (str): 查询的关键词
            skip_eids (list, optional): 在递归调用时，需要跳过的实体id. Defaults to [].
            rec (bool, optional): 是否需要进行递归调用，在初次查找为真，查找消歧页面实体列表为假. Defaults to True.

        Returns:
            list: [{"eid": wiki实体id, "formal": 正式名称, "exrest": 简介}]
        """
        page = wptools.page(form, lang='zh')
        try:
            wiki = page.get(proxy=f'http://127.0.0.1:{self.port}', timeout=20)
        except Exception as e:
            return []

        def isDisambiguation(wiki):
            """
            从wiki的查询返回中判断是否为消歧页面

            Args:
                wiki ([type]): wiki的返回，wiki.data中包含主要数据

            Returns:
                bool: 是否为消歧
            """
            if "disambiguation" in wiki.data:
                return True
            elif "description" in wiki.data and wiki.data["description"] and "消歧" in wiki.data["description"]:
                return True
            else:
                return False
        if "wikibase" not in wiki.data or not wiki.data["wikibase"]:
            return []
        eid = wiki.data["wikibase"]
        if rec and isDisambiguation(wiki) and eid not in skip_eids:  # 是消歧页面
            links = wiki.data["links"]
            candidates = []
            skip_eids.append(wiki.data["wikibase"])
            for link in links:
                cand = self.queryFromWiki(link, skip_eids, rec=False)  # 单进程递归查找消歧列表的实体
                candidates += cand
            return candidates
        elif eid not in skip_eids:  # 不是消歧页面直接消歧完成
            formal = wiki.data["label"] if "label" in wiki.data else wiki.data["title"]
            exrest = wiki.data["exrest"]
            return [{"eid": eid, "formal": formal, "exrest": exrest}]
        else:
            return []   # 无法找到对应实体，消歧失败

    def writeNewAlias(self, form, detail):
        """
        在别名表中写入一条歧义数据

        Args:
            form (str): 歧义名称
            detail (dict): {"detail:eid": wiki实体id, "detail:name": 正式名称, "detail:desc": 简介}
        """
        exists = self.queryFromDB(form)
        if exists:
            form = form + str(len(exists))
        self.aliasBase.write(form, detail)
