'''
Author: your name
Date: 2021-03-05 11:04:39
LastEditTime: 2021-06-06 12:59:04
LastEditors: Please set LastEditors
Description: In User Settings Edit
FilePath: /PCube3/preprocess/ann2train.py
'''
import os
import re
import json
import itertools


def parseNERData():
    """
    将inception的输出变成序列标注算法模型易读取的格式
    """
    ann_dir = "/home/disk2/nuclear/PCubeAnn/AnnExport/NER"
    train_dir = "/home/disk2/nuclear/PCubeAnn/Traindata/NER"
    for number in os.listdir(ann_dir):
        file = f"{ann_dir}/{number}/CURATION_USER.tsv"
        start = False
        multi_entity_list = []
        sentences = []
        sentence = {"token": [], "wikiLink": [], "ner": []}
        last_length = 0

        for line in open(file, encoding="utf-8"):
            if line.startswith("#Text"):    # 原句行，一句的开始
                start = True
                sentence = {"token": [], "wikiLink": [], "ner": []}
            elif start and line == "\n":    # 终结行，当前结果加入总数据
                sentences.append(sentence)
                start = False
            elif start:     # 词语行
                line = line.strip().split("\t")
                if "." not in line[0]:  # 非词语内部实体行
                    sentence["token"] += list(line[2])
                    last_length = len(line[2])

                    # 删除WikiURL的前缀
                    wiki_id = line[3].replace("http://www.wikidata.org/entity/", "")
                    wiki_id = re.sub(r"\[\d+\]", "", wiki_id)

                    if line[4] != "_":  # 本词语是实体
                        tag = re.sub(r"\[\d+\]", "", line[4])
                        tag = tag.split("|")[0]     # 避免词语属于多个实体，仅第一个为最长实体
                        if tag == "RE\_TAG":    # 关系指示词
                            tag = "REL"
                        if tag == "*":      # debug操作，避免出现未标注实体
                            print(file)
                        if "[" in line[4]:  # 当前实体多于一个词语
                            if line[4] in multi_entity_list:   # 当前词语不是第一个词语
                                tag = ["I-" + tag]*len(line[2])
                            else:   # 当前词语是第一个词语
                                tag = ["B-" + tag] + ["I-" + tag]*(len(line[2])-1)
                                multi_entity_list.append((line[4]))
                        else:   # 单词语实体
                            tag = ["B-" + tag] + ["I-" + tag]*(len(line[2])-1)

                    else:
                        tag = ["O"]*len(line[2])
                    sentence["ner"] += tag

                    sentence["wikiLink"] += [wiki_id]*len(line[2])

                elif "." in line[0]:    # 当前字为前一个词语的内部实体
                    wiki_id = line[3].replace("http://www.wikidata.org/entity/", "")
                    wiki_id = re.sub(r"\[\d+\]", "", wiki_id)
                    if line[0].split(".")[1] == "1":    # 将前一个词语的实体ID全部删去
                        for i in range(last_length-1, -1, -1):
                            sentence["wikiLink"][i+len(sentence["token"])-last_length] = "_"

                    token = list(line[2])   # 重新按字符赋予id
                    last_word = sentence["token"][len(sentence["token"])-last_length:]
                    for i in range(last_length-1, -1, -1):
                        if last_word[i] != token[-1]:
                            continue
                        sentence["wikiLink"][i+len(sentence["token"])-last_length] = wiki_id

        cur_id = ""  # 当前实体ID
        cur_form = ""   # 当前实体名
        cur_type = ""   # 当前实体类型
        cur_idxs = []   # 当前实体所在的句子指针
        entities = json.load(open("static/preproces/CannotLink.json", encoding="utf-8"))

        for i in range(len(sentences)):
            sentence = sentences[i]
            wikiLink = []
            for j, data in enumerate(zip(sentence["token"], sentence["wikiLink"])):
                char, wiki_id = data
                if wiki_id != cur_id:   # 切换到下一实体
                    if cur_id != "" and cur_id != "_":  # 若当前的字符是实体
                        if cur_id == "*" and len(wikiLink) > 0:  # ID 为*
                            k = -1  # 判断本字符知否与上一个规约的实体是连接的，取最长实体
                            while cur_form not in entities and wikiLink[k]["idx"][-1] + 1 == cur_idxs[0]:
                                cur_form = wikiLink[k]["form"] + cur_form
                                cur_idxs = wikiLink[k]["idx"] + cur_idxs
                            if cur_form in entities:    # 若该实体名是之前出现过的，将之前的ID赋予当前实体
                                cur_id = entities[cur_form]
                                wikiLink = wikiLink[:k]
                        if cur_id != "*":   # 当前实体有直接标注的ID，判断本字符知否与上一个规约的实体是连接的，取最长实体
                            if cur_form not in entities:
                                entities[cur_form] = cur_id
                                wikiLink.append({"form": cur_form, "type": cur_type, "wiki_id": cur_id, "idx": [cur_idxs[0], cur_idxs[-1]]})
                            else:
                                wikiLink.append({"form": cur_form, "type": cur_type, "wiki_id": entities[cur_form], "idx": [cur_idxs[0], cur_idxs[-1]]})
                    cur_id = ""
                    cur_form = ""
                    cur_type = ""
                    cur_idxs = []

                cur_form += char    # 继续增加现在的实体名
                cur_idxs.append(j)  # 继续加入实体ID
                cur_type = sentence["ner"][j].split("-")[-1]
                cur_id = wiki_id
                if wiki_id == "*":
                    if cur_form in entities:
                        cur_id = entities[cur_form]
            # 解析包含着多个实体的词语，分析每个词语及其ID
            share_id = {}  # 分解之后的每个实体
            delLink = []    # 需要删除的包含着多个实体的词语
            for j in range(len(wikiLink)):
                if "|" in wikiLink[j]["wiki_id"]:
                    for id in wikiLink[j]["wiki_id"].split("|")[:-1]:
                        if id not in share_id:
                            share_id[id] = dict(wikiLink[j])
                            share_id[id]["form"] = wikiLink[j]["form"]
                            share_id[id]["type"] = wikiLink[j]["type"]
                            share_id[id]["idx"] = wikiLink[j]["idx"].copy()
                            share_id[id]["wiki_id"] = id
                        else:
                            share_id[id]["form"] += wikiLink[j]["form"]
                            share_id[id]["idx"][-1] = wikiLink[j]["idx"][-1]
                    wikiLink[j]["wiki_id"] = wikiLink[j]["wiki_id"].split("|")[-1]
                elif wikiLink[j]["wiki_id"] in share_id:
                    id = wikiLink[j]["wiki_id"]
                    if wikiLink[j]["idx"][0] - 1 == share_id[id]["idx"][-1]:
                        share_id[id]["form"] += wikiLink[j]["form"]
                        share_id[id]["idx"][-1] = wikiLink[j]["idx"][-1]
                        share_id[id]["type"] = wikiLink[j]["type"]
                        delLink.append(j)

            for offset, j in enumerate(delLink):
                wikiLink.pop(j-offset)

            for _, link in share_id.items():
                j = 0
                while j < len(wikiLink) and wikiLink[j]["idx"][-1] < link["idx"][-1]:
                    j += 1
                wikiLink.insert(j, link)

            sentence["wikiLink"] = wikiLink
            sentence["token"] = "".join(sentence["token"])
            # tag_res = tagger.tag(sentence["token"], ["chs", "seg", "pos"])
            # poss = []
            # for seg, pos in zip(tag_res["seg"][0], tag_res["pos"][0]):
            #     poss += [pos] * len(seg)
            # sentence["pos"] = poss

        with open(os.path.join(train_dir, number.replace("txt", "json")), "w", encoding="utf-8") as wf:
            json.dump(sentences, wf, ensure_ascii=False)
        print(f"Finish parsing article {number}")


def parseOREQAData():
    """
    将inception的ORE输出变成QA算法模型易读取的格式
    """
    ann_dir = "/home/disk2/nuclear/PCubeAnn/AnnExport/ORE"
    train_dir = "/home/disk2/nuclear/PCubeAnn/Traindata/ORE"

    for number in os.listdir(ann_dir):
        file = f"{ann_dir}/{number}/CURATION_USER.tsv"
        # file level paras
        eid_map = json.load(open("static/preproces/CannotLink.json", encoding="utf-8"))
        start = False
        sentences = []
        cur_entities = {}
        # sentence level paras
        cur_text = ""
        cur_edges = []
        # if number == "0006.txt":
        #     print("ssdsd")
        for line in open(file, encoding="utf-8"):
            if line =='\n':
                print(line)
            if line.startswith("#Text"):    # 原句行，一句的开始
                start = True
                cur_text = ""
                cur_edges = []
            elif start and line == "\n":    # 终结行，当前结果加入总数据
                # for edge in cur_edges:
                sentences.append({"text": cur_text, "edge": cur_edges})
                start = False
            elif start:     # 词语行
                line = line.strip().split("\t")
                if len(line) <= 5:
                    break
                if "." not in line[0]:  # 非词语内部实体行
                    cur_text += line[2]

                    # 删除WikiURL的前缀
                    wiki_id = line[3].replace("http://www.wikidata.org/entity/", "")
                    wiki_id = re.sub(r"\[\d+\]", "", wiki_id)

                    if line[4] != "_":  # 本词语是实体
                        tag = line[4].split("|")[0]  # 避免词语属于多个实体，仅第一个为最长实体
                        tag = re.sub(r"\[\d+\]", "", tag)
                        if tag == "RE\_TAG" or line[5] == "*|*":    # 关系指示词
                            tag = "REL"
                            eid1, eid2 = line[6].split("|")
                            if "[" in line[4]:  # 关系词为多个词语
                                rid = re.sub(r".*\[(.+)\]", r"\1", line[4].split("|")[0])   # 取关系词的文章全局id
                                if rid not in line[6]:  # 关系词双标签，取有关系的
                                    rid = re.sub(r".*\[(.+)\]", r"\1", line[4].split("|")[-1])
                                # 取关系的前后两个实体的id
                                f1 = eid1.split("_")[0][-1]
                                if eid1.split("_")[0][-2].isdigit():
                                    f1 = eid1.split("_")[0][-2] + f1
                                t1 = eid1.split("_")[-1][0:-1]
                                f2 = eid2.split("_")[0][-1]
                                if eid2.split("_")[0][-2].isdigit():
                                    f2 = eid2.split("_")[0][-2] + f2
                                t2 = eid2.split("_")[-1][0:-1]
                                if f1 == "0" or t1 == "0":  # 0 指的是没有全局id的实体，以词语id为实体id
                                    eid1 = re.sub(r"(.*)\[.+\]", r"\1", eid1)
                                else:
                                    eid1 = re.sub(r".*(\[.+\])", r"\1", eid1)
                                    eid1 = eid1.replace(rid, "0")
                                if f2 == "0" or t2 == "0":
                                    eid2 = re.sub(r"(.*)\[.+\]", r"\1", eid2)
                                else:
                                    eid2 = re.sub(r".*(\[.+\])", r"\1", eid2)
                                    eid2 = eid2.replace(rid, "0")
                                rid = "[" + rid + "_0]"
                            else:
                                rid = line[0]
                                if "[" in eid1:
                                    eid1 = re.sub(r".*(\[.+\])", r"\1", eid1)
                                if "[" in eid2:
                                    eid2 = re.sub(r".*(\[.+\])", r"\1", eid2)
                            cur_edges.append([eid1, rid, eid2])
                        elif tag == "*":      # debug操作，避免出现未标注实体
                            print(file)
                        if "[" in line[4]:  # 当前实体多于一个词语
                            eid = re.sub(r".*\[(\d+)\]", r"[\1_0]", line[4].split("|")[0])
                            if eid in cur_entities:   # 当前词语不是第一个词语
                                cur_entities[eid][0] += line[2]
                            else:   # 当前词语是第一个词语
                                cur_entities[eid] = [line[2], tag, wiki_id]
                            if wiki_id == "*" and cur_entities[eid][0] in eid_map:
                                wiki_id = eid_map[cur_entities[eid][0]]
                                cur_entities[eid][-1] = wiki_id
                            elif wiki_id != "*":
                                eid_map[cur_entities[eid][0]] = wiki_id
                        else:   # 单词语实体
                            if wiki_id == "*" and line[2] in eid_map:
                                wiki_id = eid_map[line[2]]
                            elif wiki_id != "*":
                                eid_map[line[2]] = wiki_id
                            cur_entities[line[0]] = [line[2], tag, wiki_id]

                elif "." in line[0]:    # 当前字为前一个词语的内部实体
                    wiki_id = line[3].replace("http://www.wikidata.org/entity/", "")
                    tag = re.sub(r"\[\d+\]", "", line[4])
                    tag = tag.split("|")[0]

                    wiki_id = re.sub(r"\[\d+\]", "", wiki_id)
                    eid = line[0].split(".")[0]
                    if eid in cur_entities:
                        otag = cur_entities[eid][1]
                        if otag == tag:
                            cur_entities.pop(eid)
                    cur_entities[line[0]] = [line[2], tag, wiki_id]

        res_sentences = []
        for sid, sent in enumerate(sentences):
            edges = sent["edge"]
            text = sent["text"]
            for edge in edges:
                query = cur_entities[edge[0]][0] + "?" + cur_entities[edge[-1]][0]  # 实体组合成query
                ans = cur_entities[edge[1]][0]
                aid = text.find(ans)
                if aid != -1:
                    ans = [aid, aid+len(ans)]
                else:
                    print("sss")
                res_sentences.append({"text": text, "query": query, "answer": ans,
                                    "etype1": cur_entities[edge[0]][1], "etype2": cur_entities[edge[-1]][1],
                                    "eid1": cur_entities[edge[0]][-1], "eid2": cur_entities[edge[-1]][-1]})
            entity_list = []
            for k, v in cur_entities.items():
                if v[1] == "REL":
                    continue
                if k.startswith(str(sid+1)):
                    entity_list.append(v)
                elif k.startswith("["):
                    if v[0] in text:
                        entity_list.append(v)
            entity_pairs = list(itertools.combinations(entity_list, 2))
            for pair in entity_pairs[:len(entity_list)]:
                positive = False
                for edge in edges:
                    if pair[0] == edge[0] and pair[-1] == edge[-1]:
                        positive = True
                        break
                if positive:
                    continue
                query = pair[0][0] + "?" + pair[-1][0]  # 实体组合成query
                ans = [0,0]
                res_sentences.append({"text": text, "query": query, "answer": ans,
                                    "etype1": pair[0][1], "etype2": pair[-1][1],
                                    "eid1": pair[0][-1], "eid2": pair[-1][-1]})
        with open(os.path.join(train_dir, number.replace("txt", "json")), "w", encoding="utf-8") as wf:
            json.dump(res_sentences, wf, ensure_ascii=False)
        print(f"Finish parsing article {number}")


def merge(train_dir):
    """
    将每个文章的解析合并成大的训练测试集
    """
    datas = []
    for file in os.listdir(train_dir):
        if not file[0].isdigit():
            continue
        file = os.path.join(train_dir, file)
        with open(file) as rf:
            lines = json.load(rf)
        for line in lines:
            datas.append(line)
    train_path = os.path.join(train_dir, "train.txt")
    test_path = os.path.join(train_dir, "dev.txt")
    all_path = os.path.join(train_dir, "all.txt")
    split = len(datas) // 2
    with open(train_path, "w") as wf:
        for data in datas[: split]:
            wf.write(json.dumps(data, ensure_ascii=False)+"\n")
    with open(test_path, "w") as wf:
        for data in datas[split:]:
            wf.write(json.dumps(data, ensure_ascii=False)+"\n")
    with open(all_path, "w") as wf:
        for data in datas:
            wf.write(json.dumps(data, ensure_ascii=False)+"\n")


if __name__ == "__main__":
    parseOREQAData()
    merge("/home/disk2/nuclear/PCubeAnn/Traindata/ORE")
