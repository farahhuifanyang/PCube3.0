import os
import json
import random
import wptools


def getWikibase(wiki_id):
    """
    从wikidata中查询特定实体id对应的实体数据

    Args:
        wiki_id (str): wikidata的实体id，必定以Q开头

    Returns:
        str, str: 实体正式名称，实体简介
    """
    page = wptools.page(wikibase=wiki_id, lang='zh')
    try:
        wiki = page.get_wikidata(proxy='http://127.0.0.1:7890')
    except Exception:
        return None, None
    formal = wiki.data["label"]
    wiki_desc = wiki.data['description']
    page = wptools.page(formal, lang='zh')
    try:
        wiki = page.get(proxy='http://127.0.0.1:7890', timeout=20)
    except Exception:
        return formal, wiki_desc
    return formal, wiki.data["exrest"]


def getCandidate(form, eids, rec=True):
    """
    根据标注的实体form从wikidata中搜索负样本候选项

    Args:
        form (str): 实体的form
        eids (list[str]): 不予处理的实体wikiID
        rec (bool, optional): 是否要从消歧页面递归进行查询. Defaults to True.

    Returns:
        list[[str, str]]: 实体正式名称，实体简介
    """
    page = wptools.page(form, lang='zh')
    try:    # 此处的proxy需要根据情况修改
        wiki = page.get(proxy='http://127.0.0.1:7890', timeout=20)
    except Exception as e:
        return []

    def isDisambiguation(wiki):
        # 判断wikidata的返回是不是消歧页面
        if "disambiguation" in wiki.data:
            return True
        elif "description" in wiki.data and wiki.data["description"] and "消歧" in wiki.data["description"]:
            return True
        else:
            return False
    if "wikibase" not in wiki.data or not wiki.data["wikibase"]:
        return []
    if rec and isDisambiguation(wiki) and wiki.data["wikibase"] not in eids:
        links = wiki.data["links"]
        candidates = []
        eids.append(wiki.data["wikibase"])
        for link in links:
            cand = getCandidate(link, eids, rec=False)
            candidates += cand
        return candidates
    elif wiki.data["wikibase"] not in eids:
        formal = wiki.data["label"]
        exrest = wiki.data["exrest"]
        return [[formal, exrest]]
    else:
        return []


def NER2EL(ner_dir, el_dir):
    """
    根据实体识别的标注数据，得到用于实体链接实验的正负样本
    注意： 本函数必须在翻墙条件下进行

    Args:
        ner_dir (str): NER数据所在的目录
        el_dir (str): 链接数据保存的目录
    """
    filelist = []
    with open("ellog.txt", "r", encoding="utf-8") as logf:
        for line in logf:
            dpath = line.split(" ")[0]
            filelist.append(dpath)
    logf = open("ellog.txt", "a", encoding="utf-8")
    for ner_path in os.listdir(ner_dir):
        ner_path = os.path.join(ner_dir, ner_path)
        if ner_path in filelist:
            continue
        el_datas = []
        id_dict = {}
        with open(ner_path, 'r', encoding='utf-8') as nerf:
            lines = json.load(nerf)
            for line in lines:
                entities = line["wikiLink"]
                for entity in entities:
                    form = entity["form"]
                    wiki_id = entity["wiki_id"]
                    if wiki_id == "*":
                        continue
                    if wiki_id not in id_dict:
                        formal, exrest = getWikibase(wiki_id)
                        if not formal:
                            continue
                        candidates = getCandidate(form, [wiki_id])
                        id_dict[wiki_id] = [formal, exrest, candidates]
                    else:
                        formal, exrest, candidates = id_dict[wiki_id]
                    one_data = {"token": line["token"], "form": form, "idx": entity["idx"], "type": entity["type"],
                                "formal": formal, "exrest": exrest, "tag": 1}
                    el_datas.append(one_data)
                    for cand in candidates:
                        formal, exrest = cand
                        one_data = {"token": line["token"], "form": form, "idx": entity["idx"], "type": entity["type"],
                                    "formal": formal, "exrest": exrest, "tag": 0}
                        el_datas.append(one_data)
        for data in el_datas:
            randnum = random.randint(0, 4)
            data = json.dumps(data, ensure_ascii=False) + "\n"
            if randnum == 0:
                with open(os.path.join(el_dir, "test.txt"), 'a', encoding='utf-8') as ief:
                    ief.write(data)
            else:
                with open(os.path.join(el_dir, "train.txt"), 'a', encoding='utf-8') as ief:
                    ief.write(data)
            with open(os.path.join(el_dir, "all.txt"), 'a', encoding='utf-8') as ief:
                ief.write(data)
        logf.write(f"{ner_path} parse finished.\n")
        logf.flush()
    logf.close()


def addNegativeSample(el_dir):
    """
    在训练集中增加负样本，即增加wikidata消歧页面的其他候选实体为负样本

    Args:
        el_dir (str): 实体链接数据所在的目录
    """
    allf = open(os.path.join(el_dir, "all_with_pos.txt"), 'r', encoding='utf-8')
    all_data = allf.readlines()
    split = len(all_data) // 4 * 3
    status = 0
    last_form = ""
    last_idx = []
    last_pos = {}
    train_datas = []
    test_datas = []

    for i, line in enumerate(all_data):
        line = json.loads(line)
        if status == 0:  # 起始状态
            last_form = line["form"]
            last_idx = line["idx"]
            last_pos = line.copy()
            status = 1
        elif status == 1:   # 一次正样本出现状态
            cur_form = line["form"]
            cur_idx = line["idx"]
            if cur_form == last_form and cur_idx == last_idx:
                # 第二次是负样本需要重采样
                status = 2
            else:
                # 单独正样本，不需要训练
                last_form = cur_form
                last_idx = cur_idx
                last_pos = line.copy()
                if i < split:
                    continue
        elif status == 2:   # 负样本持续出现状态
            cur_form = line["form"]
            cur_idx = line["idx"]
            if i < split:
                train_datas.append(last_pos)
            if cur_form != last_form or cur_idx != last_idx:
                # 样本切换
                last_form = cur_form
                last_idx = cur_idx
                last_pos = line.copy()
                status = 1
        if i < split:
            train_datas.append(line)
        else:
            test_datas.append(line)
    allf.close()

    for data in train_datas:
        data = json.dumps(data, ensure_ascii=False) + "\n"
        with open(os.path.join(el_dir, "train.txt"), 'a', encoding='utf-8') as ief:
            ief.write(data)
    for data in test_datas:
        data = json.dumps(data, ensure_ascii=False) + "\n"
        with open(os.path.join(el_dir, "test.txt"), 'a', encoding='utf-8') as ief:
            ief.write(data)


if __name__ == "__main__":
    # getWikibase("Q132")
    NER2EL("ner_data/", "el_data/")
    addNegativeSample("el_data/")
