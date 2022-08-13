'''
Author: Liuhezi
Date: 2021-04-22 09:57:14
LastEditTime: 2021-05-11 11:27:13
LastEditors: Please set LastEditors
Description: In User Settings Edit
FilePath: /PCube3.0/coreNLP/PER/main.py
'''

from coreNLP.Algorithm import Algorithm
from coreNLP.PER.model import PERModel
from coreNLP.PER.dataParser import PERDataset
from coreNLP.PER.config import FLAGS
from configure import globalFLAGS

import json
import numpy as np
import torch

import os

os.environ['CUDA_VISIBLE_DEVICES'] = '2, 3'


class PersonalityAnalyzer(Algorithm):
    def __init__(self) -> None:
        if globalFLAGS.PER_cuda_visible_devices != -1:
            self.device = globalFLAGS.PER_cuda_visible_devices
        else:
            self.device = None
        self.batch_size = globalFLAGS.PER_batch_size
        # 载入模型所需的word层、pern层节点邻接矩阵

        self.word_pern_adj = np.load(globalFLAGS.PER_word_pern_adj_path)
        self.pern_adj = np.load(globalFLAGS.PER_pern_adj_path)
        # 构建对称邻接矩阵
        self.word_pern_adj += np.identity(self.word_pern_adj.shape[1])
        self.word_pern_adj[self.word_pern_adj != 0] = 1.0
        self.pern_adj += np.identity(self.pern_adj.shape[1])
        self.pern_adj[self.pern_adj != 0] = 1.0
        # 载入节点特征
        self.pern_features = np.load(globalFLAGS.PER_pern_features_path)
        pern_dim = self.pern_features.shape[-1]
        np.random.seed(42)
        self.word_features = np.random.rand(89, pern_dim)

        # tensor化
        self.pern_features = torch.tensor(self.pern_features)
        self.word_features = torch.tensor(self.word_features)
        self.word_pern_adj = torch.tensor(self.word_pern_adj)
        self.pern_adj = torch.tensor(self.pern_adj)

        self.pern_features -= self.pern_features.mean(0, keepdim=True)
        self.pern_features /= self.pern_features.std(0, unbiased=False, keepdim=True)
        self.word_features -= self.word_features.mean(0, keepdim=True)
        self.word_features /= self.word_features.std(0, unbiased=False, keepdim=True)
        self.pern_features = self.pern_features.cuda()
        self.word_features = self.word_features.cuda()
        self.pern_adj = self.pern_adj.cuda()
        self.word_pern_adj = self.word_pern_adj.cuda()
        # 构建模型
        self.model = PERModel(self.pern_features, self.word_features, self.pern_adj, self.word_pern_adj)
        # 加载模型
        if self.device is not None:
            self.model.cuda()
        self.model.load_state_dict(torch.load(globalFLAGS.PER_model_path))
        self.model.eval()

    def run(self, input=None):
        """使用模型对一个人物进行五人格预测

        Args:
            input (dict, optional): 输入的人物，由预处理模块产生. 
            {“人物ID”: “与图中词类节点的特征邻接矩阵”}


        Returns:
            dict: 五人格预测的输出
            {“人物ID”: [“openness”: 经验开放性分值, “conscientious”: 尽责性分值,
                        “extraversion”: 外向性分值, “agreeableness”: 宜人性分值,
                        “neuroticism”: 神经质分值]}
        """
        if not input:
            return
        batch_dataset = PERDataset(input)
        batch_dataloader = torch.utils.data.DataLoader(batch_dataset, self.batch_size)
        total_scores = []
        for idx, batch in enumerate(batch_dataloader):
            self.model.zero_grad()
            if self.device is not None:
                batch = batch.cuda()
            scores = self.model(batch)
            scores = scores.cpu().detach().numpy().tolist()
            total_scores += scores
        total_scores = self.postProcess(total_scores, input.keys())
        return total_scores

    def postProcess(self, total_scores, ids):
        """模型输出人格分值的后处理阶段，将分值与人物id对应

        Args:
            total_scores (array): 一个batch的五人格输出[[ope1,con1,ext1,agr1,neu1],
                                                       [ope2,con2,ext2,agr2,neu2]..]
            ids (list): 一个batch数据对应的人物id [Q000001,Q000002...]

        Returns:
            output (list(dict)) : 每个人物对应的五人格分数
            [{id:人物ID，ope:开放性分值,con:尽责性分值,ext:外向性分值,agr:宜人性分值,neu:神经质分值},..]
        """
        idx = 0
        output = []
        labels = ['openness', 'conscientious', 'extraversion', 'agreeableness', 'neuroticism']
        for id in ids:
            scores = {}
            scores['id'] = id
            for i in range(5):
                scores[labels[i]] = round(total_scores[idx][i], 2)
            idx += 1
            output.append(scores)
        return output


if __name__ == "__main__":
    from preprocess.forPersonality import processOneFBPage
    from classes.FBPage import FBPage
    import os
    fb_post_dir = "/home/liuhezi-19/projects/crawler/facebook_page_spider/data/"
    cnt = 0
    input = {}
    output = []
    per = PersonalityAnalyzer()
    for root, _, files in os.walk(fb_post_dir):
        if files:
            page = FBPage.init_from_file(root, files)
            id, adj = processOneFBPage(page)
            input[id] = adj
            cnt += 1
            if cnt % 10 == 0:
                out = per.run(input)
                output += out
                input = {}
    with open('coreNLP/PER/per_scores.json', 'w') as f:
        json.dump(output, f, ensure_ascii=False, indent=4)
