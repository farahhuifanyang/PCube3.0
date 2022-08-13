'''
Author: Guowenying
Date: 2021-06-08 17:17:29
LastEditTime: 2021-06-20 14:26:32
LastEditors: Please set LastEditors
Description: In User Settings Edit
FilePath: /PCube3/coreNLP/CLUSTER/for_test.py
'''
'''
Author: Guowenying
Date: 2021-02-05 16:19:16
LastEditTime: 2021-06-02 22:04:52
LastEditors: Guowenying
Description: In User Settings Edit
FilePath: /PCube3/coreNLP/CLUSTER/forCluster.py
'''

#该脚本修改涉及到的读写后可替换/coreNLP/CLUSTER/forCluster.py 在计算相似度时使用文章的标题和文章内容的第一段计算文档相似度 计算效率提高 
import os
import numpy as np
from coreNLP.Algorithm import Algorithm
from coreNLP.CLUSTER.dataParser import CLUDataset

from coreNLP.CLUSTER.config import args
from gensim import corpora, models, similarities, matutils
from smart_open import smart_open
from tkinter import _flatten
from pyltp import Segmentor, Postagger
from openccpy.opencc import Opencc
import datetime
import json
import re

dictionary = None
class SinglePassCluster(Algorithm):

    def __init__(self):

        self.dir_name = args.dir_name
        self.stop_words_file = args.stop_words_file
        self.theta = args.theta
        self.ltp_data_dir = args.ltp_data_dir
        self.cws_model_path = os.path.join(self.ltp_data_dir, 'cws.model')
        self.pos_model_path = os.path.join(self.ltp_data_dir, 'pos.model')
        self.segmentor = Segmentor()  # 初始化实例
        self.segmentor.load(self.cws_model_path)  
        self.postagger = Postagger()  # 初始化实例
        self.postagger.load(self.pos_model_path)  # 加载模型

    def clean(self, s):
        """
            清理文档中无用的字符
        Parameters
        ----------
        s : string
            某篇文档

        Returns
        -------
        string
            经过清理后的文档
        """
        return re.sub(r'[0-9a-zA-Z.， ,\-。%《*》/•、&＆(—)-@（+）：？!！“”·:」_\t\\//\r\n]', '', s)

    def wordSegment(self, sentence):
        """
        处理文档，对文档进行去停用词、分词等操作

        Parameters
        ----------
        sentence : list
            语料库中包含的所有文档

        Returns
        -------
        list
            处理之后的全部文档
        """
        stopwords = [line.strip() for line in open(self.stop_words_file, encoding='utf-8').readlines()]
        post_list = ['n', 'nh', 'ni', 'nl', 'ns', 'nz', 'j', 'ws', 'a', 'z', 'b', 'v']
        sentence = sentence.strip().replace('\\n', '')
        sentence = self.clean(sentence)
        words = self.segmentor.segment(sentence.replace('\n', ''))  # 分词
        postags = self.postagger.postag(words)  # 词性标注
        dict_data = dict(zip(words, postags))
        table = {k: v for k, v in dict_data.items() if v in post_list}
        words = list(table.keys())
        word_segmentation = []
        for word in words:
            if word == ' ':
                continue
            if word not in stopwords:
                word_segmentation.append(word)
        return word_segmentation

    def getTfidfVectorRepresentation(self, word_segmentation, pivot=10, slope=0.1):
        """
        利用TF-IDF方法得到文本数据的空间向量表示

        Parameters
        ----------
        word_segmentation : list
            所有的文档集合
        pivot : int, optional
            gensim中tfidf模型的参数
        slope : float, optional
            gensim中tfidf模型的参数
        Returns
        -------
        list
            语料库中文档的TF-IDF向量表示
        """

        dictionary = corpora.Dictionary(word_segmentation)

        #加载之前处理得到的字典并合并
        # if os.path.exists(os.path.join(args.temp_file,'doc2bow.dict')):
        #     pre_dictionary = corpora.Dictionary.load(os.path.join(args.temp_file,'doc2bow.dict'))
        #     pre_dictionary.merge_with(dictionary)

        #加载之前处理过的语料，合并当前处理的文本语料
        corpus = []
        if os.path.exists(os.path.join(args.temp_file,'corpus')):
            with open(os.path.join(args.temp_file,'corpus'),'r') as f:
                corpus = json.load(f)['0']     
        corpus += [dictionary.doc2bow(text) for text in word_segmentation]

        tfidf = models.TfidfModel(corpus, pivot=pivot, slope=slope) # 训练TF-IDF模型   
        corpus_tfidf = tfidf[corpus]    #获得文章的TF-IDF表示
        # dictionary.save(os.path.join(args.temp_file,'doc2bow.dict'))    #保存词典
        # with open(os.path.join(args.temp_file,'corpus'),'w') as f:
        #         json.dump({'0':corpus},f) 
        return corpus_tfidf

    def getMaxSimilarity(self, dictTopic, vector, Inverted_index):
        """
        获得当前文档所属的事件
        Parameters
        ----------
        dictTopic : dict
            已经处理过的文档生成的事件集
        vector : list
            当前文档所有的单词
        Inverted_index : dict
            倒排索引

        Returns
        -------
        int, int 
            返回和该文件最相似的事件索引，相似度
        """
        max_value = 0
        max_index = -1
        index = self.getIndex(vector,Inverted_index)    
        for k in index:
            one_similarity = np.mean([matutils.cossim(vector, v) for v in dictTopic[k]]) 
            if one_similarity > max_value:
                max_value = one_similarity
                max_index = k
        return max_index, max_value

    def getIndex(self, words, cluster):
        """
        通过建立的倒排索引，获得包含该文章单词的所有事件集索引

        Parameters
        ----------
        words : list [(word,tf-idf)]
            当前处理的文章中包含的单词
        cluster : dict {word:{包含该单词的所有事件索引}}
            建立的倒排索引

        Returns
        -------
        list
            返回包含该文章单词的所有事件索引
        """
        can_cluster_ids = []
        for word, tf in words:
            can_cluster_ids.extend(cluster.get(word, []))
        return list(set(can_cluster_ids))

    def singlePass(self, corpus, texts, theta):
        """
        simglePass聚类的具体实现

        Parameters
        ----------
        corpus : 所有语料
            经过处理得到的所有文档内容及其TFIDF表示
        texts : dict
            处理得到的文档（id, content, time）
        theta : int
            划分事件集的阈值，超过该值文本划分到相应事件集否则划分新的事件

        Returns
        -------
        dict, dict
            事件集中包含的TF值， 事件集中包含的文件id
        """

        def writeData():
            with open(os.path.join(args.temp_file,'dictTopic.vector'),'w') as f:
                json.dump(id_dict,f)
            
            # 对话题中的文档排序获得事件的id:
            if not os.path.exists(os.path.join(args.temp_file,'numTopic.txt')):
                for key, value in file_name.items():
                    value.sort(key = lambda x:x[1])
                    id2num[value[0][1].split()[0]+str(key).zfill(10)] = key
                    num2id[key] = value[0][1].split()[0]+str(key).zfill(10)
                    cluster_files[value[0][1].split()[0]+str(key).zfill(10)] = [name[0] for name in value]
            else:
                for key, value in file_name.items():
                    new_key = key
                    if not key in num2id:
                        new_key = value[0][1].split()[0]+str(key).zfill(10)
                    else:
                        new_key = num2id[key]
                    id2num[new_key] = key 
                    num2id[key] = new_key
                    cluster_files[new_key] = [name[0] for name in value]

            with open(os.path.join(args.temp_file,'numTopic.txt'),'w') as f:    #保存事件集包含的文档的映射，但事件id是以自增int的形式命名的
                json.dump(file_name,f)
            
            with open(os.path.join(args.temp_file,'clusterTopic.txt'),'w') as f:    #保存事件集包含的文档的映射，事件的ID包含时间和10为自增id组成
                json.dump(cluster_files,f)

            with open(os.path.join(args.temp_file,'num2id.txt'),'w') as f:  #自增int和事件ID的映射
                json.dump(num2id,f)

            with open(os.path.join(args.temp_file,'id2num.txt'),'w') as f:  #事件ID和自增int的映射
                json.dump(id2num,f)

            for key, value in inverted_index.items():
                inverted_index[key] = list(value)

            with open(os.path.join(args.temp_file,'Inverted_index'),'w') as f:  #保存倒排索引
                json.dump(inverted_index,f)
        dict_topic = {}
        file_name = {}
        id_dict = {}
        inverted_index = {}
        num2id = {}
        id2num = {}
        cnt = 0
        num_topic = 0
        # if os.path.exists(os.path.join(args.temp_file,'Inverted_index')): #取出存在的倒排索引
        #     with open(os.path.join(args.temp_file,'Inverted_index'),'r') as f:
        #         inverted_temp = json.load(f)
        #         for key,value in inverted_temp.items():
        #             inverted_index[int(key)] = set(value)

        # if os.path.exists(os.path.join(args.temp_file,'num2id.txt')): #取出存在的倒排索引
        #     with open(os.path.join(args.temp_file,'num2id.txt'),'r') as f:
        #         pre = json.load(f)
        #         for key,value in pre.items():
        #             num2id[int(key)] = value

        # if os.path.exists(os.path.join(args.temp_file,'id2num.txt')): #取出存在的倒排索引
        #     with open(os.path.join(args.temp_file,'id2num.txt'),'r') as f:
        #         id2num = json.load(f)

        # if os.path.exists(os.path.join(args.temp_file,'numTopic.txt')):   #取出已经存在的文档所在簇集
        #     with open(os.path.join(args.temp_file,'numTopic.txt'),'r') as f:
        #             file_names = json.load(f)
        #             for key, value in file_names.items():
        #                 file_name[int(key)] = value

        # if os.path.exists(os.path.join(args.temp_file,'dictTopic.vector')):   #取出存在的文档向量
        #     with open(os.path.join(args.temp_file,'dictTopic.vector'),'r') as f:
        #         id_dicts = json.load(f)
        #         for key,value in id_dicts.items():
        #             id_dict[int(key)] = value
        #         for key,value in id_dict.items():
        #             if key not in dict_topic:
        #                 dict_topic[key] = []
        #             for i in value:
        #                 dict_topic[key].append(corpus[i])
        #     num_topic = len(dict_topic)       #主题的数量
        #     cnt = len(corpus)-len(texts)    #已经处理过cnt个文档

        for vector, text in zip(corpus[-len(texts):], texts):
            if num_topic == 0:
                dict_topic[num_topic] = []
                dict_topic[num_topic].append(vector)
                id_dict[num_topic] = []
                id_dict[num_topic].append(cnt)
                file_name[num_topic] = []
                file_name[num_topic].append(text['content'])#[text['id'],text['time']]
                for word, tf in vector:
                    if word not in inverted_index.keys(): 
                        inverted_index[word] = set()
                    inverted_index[word].add(num_topic)
                num_topic += 1
            else:
                max_index, max_value = self.getMaxSimilarity(dict_topic, vector, inverted_index)
                # 将给定语句分配到现有的、最相似的主题中
                if max_value >= theta:
                    dict_topic[max_index].append(vector)
                    id_dict[max_index].append(cnt)
                    for word, tf in vector:
                        if word not in inverted_index: 
                            inverted_index[word] = set()
                        inverted_index[word].add(max_index)
                    file_name[max_index].append(text['content'])#[text['id'],text['time']]

                # 或者创建一个新的主题
                else:
                    dict_topic[num_topic] = []
                    dict_topic[num_topic].append(vector)
                    id_dict[num_topic] = []
                    id_dict[num_topic].append(cnt)
                    file_name[num_topic] = []
                    file_name[num_topic].append(text['content'])#[text['id'],text['time']]
                    for word, tf in vector:
                        if word not in inverted_index: 
                            inverted_index[word] = set()
                        inverted_index[word].add(num_topic)
                    num_topic += 1
            cnt += 1
            if cnt % 1000 == 0:
                print("processing {}...".format(cnt))
                print("The number of topic is {}...".format(num_topic))
        # with open(os.path.join(args.temp_file,'dictTopic.vector'),'w') as f:
        #     json.dump(id_dict,f)
        for key,value in file_name.items():
            print('this is key:')
            if len(value)>1:
                for article in value:
                    print('this is  article',article)
            
        # 对话题中的文档排序获得事件的id:
        cluster_files = {}
        # writeData()
        # if not os.path.exists(os.path.join(args.temp_file,'numTopic.txt')):
        #     for key, value in file_name.items():
        #         value.sort(key = lambda x:x[1])
        #         id2num[value[0][1].split()[0]+str(key).zfill(10)] = key
        #         num2id[key] = value[0][1].split()[0]+str(key).zfill(10)
        #         cluster_files[value[0][1].split()[0]+str(key).zfill(10)] = [name[0] for name in value]
        # else:
        #     for key, value in file_name.items():
        #         new_key = key
        #         if not key in num2id:
        #             new_key = value[0][1].split()[0]+str(key).zfill(10)
        #         else:
        #             new_key = num2id[key]
        #         id2num[new_key] = key 
        #         num2id[key] = new_key
        #         cluster_files[new_key] = [name[0] for name in value]

        # with open(os.path.join(args.temp_file,'numTopic.txt'),'w') as f:    #保存事件集包含的文档的映射，但事件id是以自增int的形式命名的
        #     json.dump(file_name,f)
        
        # with open(os.path.join(args.temp_file,'clusterTopic.txt'),'w') as f:    #保存事件集包含的文档的映射，事件的ID包含时间和10为自增id组成
        #     json.dump(cluster_files,f)

        # with open(os.path.join(args.temp_file,'num2id.txt'),'w') as f:  #自增int和事件ID的映射
        #     json.dump(num2id,f)

        # with open(os.path.join(args.temp_file,'id2num.txt'),'w') as f:  #事件ID和自增int的映射
        #     json.dump(id2num,f)

        # for key, value in inverted_index.items():
        #     inverted_index[key] = list(value)

        # with open(os.path.join(args.temp_file,'Inverted_index'),'w') as f:  #保存倒排索引
        #     json.dump(inverted_index,f)
        print('生成的主题个数为：',num_topic) 
        return cluster_files

    
    def run(self, theta=0.5, n=3000, datMat = None):
        """
        使用TF-IDF作为文章的特征值对数据集进行聚类，划分事件集

        Parameters
        ----------
        theta : float, optional
            相似度阈值, by default 0.5
        n : int, optional
            对数据集中的多少文件聚类, by default 3000

        Returns
        -------
        [dict]
            {0：[article_1, ...], ... , n: [article_n, ... ]}
            返回事件集，及文章的位置
        """
        
        word_segmentation = []
        for i in range(len(datMat)):
             word_segmentation.append(self.wordSegment(datMat[i]['content']))
            # print(self.wordSegment(datMat[i]['content']))
            # for sentence in datMat[i]['content'].split('\\n'):
            #     if len(sentence)>5:
            #         word_segmentation.append(self.wordSegment(datMat[i]['title']+'\n'+sentence))
            #         break

        # 得到文本数据的空间向量表示
        corpus_tfidf = self.getTfidfVectorRepresentation(word_segmentation)
        clusterTopic = self.singlePass(corpus_tfidf, datMat, theta)
        # 按聚类语句数量对主题进行排序，找到重要的聚类群
        # clusterTopic_list = sorted(clusterTopic.items(), key=lambda x: len(x[1]), reverse=True)
        return clusterTopic


if __name__ == '__main__':


    start = datetime.datetime.now()
    single_pass_cluster = SinglePassCluster()
    datMat = CLUDataset().read(args.dir_name, args.number)
    single_pass_cluster.run(theta = args.theta, n=100, datMat = datMat)
    end = datetime.datetime.now()
    print('Running time: %s Seconds' % (end-start))
