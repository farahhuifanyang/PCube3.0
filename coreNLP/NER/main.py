'''
Author: your name
Date: 2021-04-06 17:04:45
LastEditTime: 2021-07-23 16:49:19
LastEditors: Please set LastEditors
Description: In User Settings Edit
FilePath: /PCube3.0/coreNLP/NER/main.py
'''
from coreNLP.Algorithm import Algorithm
from coreNLP.NER.model import NERModel
from coreNLP.NER.dataParser import NERDataset
from coreNLP.NER.config import FLAGS
from configure import globalFLAGS

from transformers import BertTokenizer, BertConfig
import torch


class EntityRecognizer(Algorithm):
    def __init__(self) -> None:
        # 加载各种配置选项和参数
        if globalFLAGS.NER_cuda_visible_devices != -1:
            self.device = globalFLAGS.NER_cuda_visible_devices
        else:
            self.device = None
        self.max_length = FLAGS.max_length
        self.batch_size = globalFLAGS.NER_batch_size
        self.id2label = FLAGS.id2label

        # 构建预测模型
        self.bertconfig = BertConfig.from_pretrained(FLAGS.pretrained)
        self.tokenizer = BertTokenizer.from_pretrained(FLAGS.pretrained)
        self.model = NERModel(FLAGS, self.bertconfig)

        # 加载模型
        if self.device is not None:
            self.model.cuda(self.device)
        self.model.load_state_dict(torch.load(globalFLAGS.NER_model_path))
        self.model.eval()

    def run(self, inputs=None):
        """
        使用模型对一篇文章进行预测

        Args:
            inputs (dict, optional): 输入的文章，由预处理模块产生.
                    {“文章ID”: [{“sentence”: [“原文词语1”, “原文词语2”, …],
                    “pos”:[“词类1”, “词类2”, …], “dp”: [“依存关系1”, “依存关系2, …”],
                    “head”:[“依存中心1”, “依存中心2”]} ]}
                    Defaults to None.

        Returns:
            dict: 实体识别的输出
                    {“文章ID”: [{“sentence”: “原文句子”, “entities”:
                    [{“form”: “实体在文中的表述”, “type”: “实体类型”,
                    “idx”: [实体开始位置, 实体结束位置]}]}]}
        """
        if not inputs:
            return
        batch_dataset = NERDataset(inputs, self.tokenizer, self.max_length, self.device)
        batch_dataloader = torch.utils.data.DataLoader(batch_dataset, self.batch_size, drop_last=False)
        batch_tags = []
        for i, (token, pos, mask) in enumerate(batch_dataloader):
            self.model.zero_grad()
            tag = self.model.decode(token, pos, mask)
            batch_tags += tag   # 注意此时tag里有BERT的标志字符

        sent_map = batch_dataset.getSentMap()
        sent_map = self.postProcess(sent_map, batch_tags)
        return sent_map

    def postProcess(self, sent_map, batch_tags):
        """
        从模型输出的标签到算法输出的后处理阶段，有部分修正操作

        Args:
            sent_map (dict): 文章与句子原句的对应关系，用以得到实体formation {"文章id": ["句子1", "句子2"]}
            batch_tags (list[list[int]]): 每一句的模型预测值 [[tag1, tag2, ...]]
        """
        def getEntityIdx(tag):
            """
            从一句话的预测标签序列里找到实体的位置

            Args:
                tag (list[int]): 一句的模型预测值 [tag1, tag2, ...]

            Returns:
                list[list[int]]: 每个实体的起始/结束index [[start, end], ...]
                list[str]: 每个实体的类型 [type1, type2, ..]
            """
            cur_tag = "O"   # 记录当前实体类型
            cur_idx = []    # 记录当前实体的起始结束
            idxs = []
            types = []
            for j, t in enumerate(tag):
                t = self.id2label[t]
                if t == "O":
                    if cur_tag != t:  # 有旧实体结束
                        cur_idx = [cur_idx[0], j]
                        idxs.append(cur_idx)
                        types.append(cur_tag)
                        cur_idx = []
                        cur_tag = "O"
                    continue

                t = t.split("-")[-1]    # 得到当前字符的实体分类
                if t != cur_tag:
                    if cur_tag != "O":  # 有旧实体结束
                        cur_idx = [cur_idx[0], j]
                        idxs.append(cur_idx)
                        types.append(cur_tag)
                    cur_idx = [j]   # 新实体的开始
                    cur_tag = t
            return idxs, types

        i = 0
        for k, v in sent_map.items():
            sent_of_one = []
            form_dict = set()
            for s in v:
                idxs, types = getEntityIdx(batch_tags[i][1:-1])
                forms = [s[idx[0]:idx[1]] for idx in idxs]

                def filterForm(s, forms, idxs, types):
                    new_forms, new_idxs, new_types = [], [], []
                    for form, idx, tp in zip(forms, idxs, types):
                        if len(form) >= 3:
                            new_forms.append(form)
                            new_idxs.append(idx)
                            new_types.append(tp)
                            form_dict.add(form[0])
                        elif len(form) == 2:
                            if form.encode().isalpha() or form[1] in ["姓", "男", "女", "某", "嫌", "犯"]:
                                continue    # 直接判断字母对中文无效
                            new_forms.append(form)
                            new_idxs.append(idx)
                            new_types.append(tp)
                        elif form in form_dict:
                            new_forms.append(form)
                            new_idxs.append(idx)
                            new_types.append(tp)
                    return new_forms, new_idxs, new_types

                forms, idxs, types = filterForm(s, forms, idxs, types)
                entities = [{"form": form, "type": tp, "idx": idx} for form, idx, tp in zip(forms, idxs, types)]
                sent_of_one.append({"sentence": s, "entities": entities})
                i += 1
            sent_map[k] = sent_of_one

        return sent_map


if __name__ == "__main__":
    import json
    # with open("coreNLP/NER/cur_entities.json", "r") as outf:
    #     entities = json.load(outf)
    #     num = 0
    #     for k, v in entities.items():
    #         num += v
    from preprocess.forRealTime import processOneArticle
    from classes.Article import Article
    import os
    pcube_data_dir = "/home/disk2/nuclear/news_data/PCube"
    i = 0
    inputs = {}
    entities = {}
    ner = EntityRecognizer()
    for root, sites, files in os.walk(pcube_data_dir):
        for file in files:
            if not file.endswith("html"):
                continue
            file = os.path.join(root, file)
            try:
                aritcle = Article.init_from_file(os.path.join(pcube_data_dir, file))
                k, v = processOneArticle(aritcle)
            except Exception:
                continue
            inputs[k] = v

            def numberOfEntities(inputs, i):
                if i % 10 == 0:
                    out = ner.run(inputs)
                    for aid, res in out.items():
                        for sent in res:
                            for entity in sent["entities"]:
                                if entity["form"] not in entities:
                                    entities[entity["form"]] = 1
                                else:
                                    entities[entity["form"]] += 1
                    inputs = {}
                if i >= 100:
                    with open("coreNLP/NER/cur_entities.json", "w") as outf:
                        json.dump(entities, outf, ensure_ascii=False)
                    # break
                i += 1
            numberOfEntities(inputs, i)
