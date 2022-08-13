'''
Author: Liuhezi
Date: 2021-04-19 15:54:49
LastEditTime: 2021-04-28 11:21:57
LastEditors: Liuhezi
Description: In User Settings Edit
FilePath: /PCube3.0/preprocess/forPersonality.py
'''
import json
import re
import pickle

import jieba
import numpy as np

from classes import FBPage
from configure import globalFLAGS

def processOneFBPage(page:FBPage):
    """处理单个用户facebook主页

    Args:
        page (FBPage): 用户主页对象

    Returns:
        str,array: 用户ID，主页对象预处理后的形成的用户-词邻接矩阵
        一维矩阵横行代表单个用户，各列代表统计的词类特征
    """
    # 清洗post文本内容并分词
    pattern = re.compile(r'[^\u4e00-\u9fa5]')
    cln_post = ''
    for p in page.post.split('<sep>'):
        if p:
            p = re.sub(pattern,'',p).strip()
            p = ' '.join(jieba.cut(p))
            cln_post+=p+'<sep>'
    page.post = cln_post
    # 统计特征并生成 user_word_adj
    word_list = np.load(globalFLAGS.PER_word_list_path).tolist()
    liwc_mean = np.load(globalFLAGS.PER_liwc_mean_path,allow_pickle=True).item()
    with open(globalFLAGS.PER_entity_path, 'rb') as f:
        entity = pickle.load(f)
    with open(globalFLAGS.PER_tw_liwc_dict_path,'r') as f:
        tw_liwc = json.load(f)
    # 初始化user_word_adj
    user_word_adj = np.zeros((len(word_list)))
    # 开始统计
    posts = page.post.replace('<sep>',' ')
    word_sum = len(posts.split())
    k = 0
    for key,values in tw_liwc.items():
        cnt = 0
        for value in values:
            if '*' in value:
                cnt += len(re.compile(r'{}\w*'.format(value)).findall(posts))
            else:
                cnt += posts.count(value)
        if round(cnt/word_sum,2) > liwc_mean[key]:
            if key+'+' in entity['word']:
                user_word_adj[word_list.index(key+'+')] = 1
        else:
            if key+'-' in entity['word']:
                user_word_adj[word_list.index(key+'-')] = 1
    return page.id, user_word_adj

    