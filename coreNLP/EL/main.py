'''
Author: your name
Date: 2021-05-10 10:43:20
LastEditTime: 2021-06-09 09:59:55
LastEditors: Please set LastEditors
Description: In User Settings Edit
FilePath: /PCube3.0/coreNLP/EL/test.py
'''
import torch
import numpy as np

from coreNLP.Algorithm import Algorithm
from coreNLP.EL.model import ELModel
from coreNLP.EL.dataParser import ELDataset
from coreNLP.EL.config import FLAGS
from configure import globalFLAGS
from transformers import BertTokenizer, BertConfig


class EntityLinker(Algorithm):
    def __init__(self) -> None:
        if globalFLAGS.EL_cuda_visible_devices != -1:
            self.device = globalFLAGS.EL_cuda_visible_devices
        else:
            self.device = None
        self.threshold = globalFLAGS.threshold
        self.max_sen_length = FLAGS.max_sen_length
        self.max_ent_length = FLAGS.max_ent_length
        self.candi_limit = globalFLAGS.EL_candi_limit

        # 构建模型
        self.bertconfig = BertConfig.from_pretrained(FLAGS.pretrained)
        self.tokenizer = BertTokenizer.from_pretrained(FLAGS.pretrained)
        self.model = ELModel(FLAGS, self.bertconfig)
        if self.device is not None:
            self.model.cuda(self.device)
        self.model.load_state_dict(torch.load(globalFLAGS.EL_model_path))
        self.model.eval()

    def run(self, inputs=None):
        """
        使用模型对一个存在歧义的实体进行消歧

        Args:
            inputs (dict, optional): 输入的文章，由预处理模块产生.
                    {"sentence": "原文句子", "form": "实体在文中的表述", "type": "实体类型",
                    "candidates":[{"formal": "实体正式名称", "desc": "实体描述", "eid": "实体ID"}, ...]}}
                    Defaults to None.

        Returns:
            dict: 实体链接的输出
                    ["匹配度最高的实体id", 匹配得分(float 0-1)]
        """
        if not inputs:
            return
        batch_dataset = ELDataset(inputs, self.tokenizer, self.max_sen_length, self.max_ent_length, self.device)
        batch_dataloader = torch.utils.data.DataLoader(batch_dataset, self.candi_limit, drop_last=False)
        token, desc, tok_mask, desc_mask, form, formal, ment_mask, form_mask = next(iter(batch_dataloader))
        self.model.zero_grad()
        score = self.model.predict(token, desc, tok_mask, desc_mask, form, formal, ment_mask, form_mask)
        score = score.cpu().detach().numpy()
        idx = np.argmax(score)
        score = score.tolist()
        if score[idx] > self.threshold:
            return inputs["candidates"][idx]
        else:
            return None


if __name__ == "__main__":
    entityLinker = EntityLinker()
    entityLinker.run()
