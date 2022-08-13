'''
Author: Guowenying
Date: 2021-02-05 16:19:16
LastEditTime: 2021-05-27 21:41:13
LastEditors: Please set LastEditors
Description: In User Settings Edit
FilePath: /PCube3/preprocess/forCluster.py
'''
import os
import numpy as np
from gensim import corpora, models, similarities, matutils
from smart_open import smart_open
from tkinter import _flatten
from pyltp import Segmentor, Postagger
from openccpy.opencc import Opencc
import datetime
import itertools
import json
from gensim.models import VocabTransform


dictionary = None


class Single_Pass_Cluster(object):

    def __init__(self,
                 filename,
                 stop_words_file='停用词汇总.txt',
                 theta=0.5,
                 LTP_DATA_DIR=r'/home/gwy/ALL/ltp-model/',  # ltp模型目录的路径
                 segmentor=Segmentor(),
                 postagger=Postagger(),

                 ):

        self.filename = filename
        self.stop_words_file = stop_words_file
        self.theta = theta
        self.LTP_DATA_DIR = LTP_DATA_DIR
        self.cws_model_path = os.path.join(self.LTP_DATA_DIR, 'cws.model')
        self.pos_model_path = os.path.join(self.LTP_DATA_DIR, 'pos.model')
        self.segmentor = segmentor  # 初始化实例
        self.segmentor.load(self.cws_model_path)  # , self.LTP_DATA_DIR + 'dictionary.txt' _with_lexicon加载模型
        self.postagger = postagger  # 初始化实例
        self.postagger.load(self.pos_model_path)  # 加载模型

    def loadData(self, filename):
        Data = []
        i = 0
        with smart_open(self.filename, encoding='utf-8') as f:
            # texts = [list(SentenceSplitter.split(i.strip().strip('\ufeff'))) for i in f.readlines()]
            texts = f.readlines()
            print('未切割前的语句总数有{}条...'.format(len(texts)))
            print("............................................................................................")
            texts = [i.strip() for i in list(_flatten(texts)) if len(i) > 5]
            print('切割后的语句总数有{}条...'.format(len(texts)))
            for line in texts:
                i += 1
                Data.append(line)
        return Data

    def word_segment(self, sentence):
        stopwords = [line.strip() for line in open(self.stop_words_file, encoding='utf-8').readlines()]
        post_list = ['n', 'nh', 'ni', 'nl', 'ns', 'nz', 'j', 'ws', 'a', 'z', 'b']
        sentence = sentence.strip().replace('。', '').replace('\\n', '').replace('」', '').replace('//', '').replace('_', '').replace('-', '').replace('\r', '').replace('\n', '').replace('\t', '').replace('@', '').replace(r'\\', '').replace("''", '')
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

    def get_Tfidf_vector_representation(self, word_segmentation, pivot=10, slope=0.1):
        # 得到文本数据的空间向量表示
        dictionary = corpora.Dictionary(word_segmentation)
        if os.path.exists('/home/gwy/ALL/nuclear_data/doc2bow.dict'):
            pre_dictionary = corpora.Dictionary.load('/home/gwy/ALL/nuclear_data/doc2bow.dict')
            # dictionary = itertools.chain(pre_dictionary, dictionary)
            pre_dictionary.merge_with(dictionary)
        corpus = []
        if os.path.exists('/home/gwy/ALL/nuclear_data/corpus'):
            with open('/home/gwy/ALL/nuclear_data/corpus', 'r') as f:
                corpus = json.load(f)['0']
        corpus += [dictionary.doc2bow(text) for text in word_segmentation]
        tfidf = models.TfidfModel(corpus, pivot=pivot, slope=slope)
        corpus_tfidf = tfidf[corpus]
        dictionary.save('/home/gwy/ALL/nuclear_data/doc2bow.dict')
        with open('/home/gwy/ALL/nuclear_data/corpus', 'w') as f:
            json.dump({'0': corpus}, f)
        return corpus_tfidf

    def get_Doc2vec_vector_representation(self, word_segmentation):
        # 得到文本数据的空间向量表示

        corpus_doc2vec = [get_avg_feature_vector(i, model, num_features=50) for i in word_segmentation]
        return corpus_doc2vec

    def getMaxSimilarity(self, dictTopic, vector):
        maxValue = 0
        maxIndex = -1
        for k, cluster in dictTopic.items():
            oneSimilarity = np.mean([matutils.cossim(vector, v) for v in cluster])
            #oneSimilarity = np.mean([cosine_similarity(vector, v) for v in cluster])
            if oneSimilarity > maxValue:
                maxValue = oneSimilarity
                maxIndex = k
        return maxIndex, maxValue

    def single_pass(self, corpus, texts, theta):
        dictTopic = {}
        clusterTopic = {}
        FileName = {}
        id_dict = {}
        cnt = 0
        numTopic = 0
        if os.path.exists('/home/gwy/ALL/nuclear_data/dictTopic.vector'):
            with open('/home/gwy/ALL/nuclear_data/dictTopic.vector', 'r') as f:
                id_dict = json.load(f)
                for key, value in id_dict.items():
                    if key not in dictTopic:
                        dictTopic[key] = []
                    for i in value:
                        dictTopic[key].append(corpus[i])
            numTopic = len(dictTopic)
            cnt = len(corpus)-len(texts)
        if os.path.exists('/home/gwy/ALL/nuclear_data/clusterTopic.txt'):
            with open('/home/gwy/ALL/nuclear_data/clusterTopic.txt', 'r') as f:
                FileName = json.load(f)

        for vector, text in zip(corpus[-len(texts):], texts):
            if numTopic == 0:
                dictTopic[numTopic] = []
                dictTopic[numTopic].append(vector)
                id_dict[numTopic] = []
                id_dict[numTopic].append(cnt)
                # clusterTopic[numTopic] = []
                # clusterTopic[numTopic].append(text)
                FileName[numTopic] = []
                FileName[numTopic].append(text['id'])
                numTopic += 1
            else:
                maxIndex, maxValue = self.getMaxSimilarity(dictTopic, vector)
                # 将给定语句分配到现有的、最相似的主题中
                if maxValue >= theta:
                    dictTopic[maxIndex].append(vector)
                    id_dict[maxIndex].append(cnt)
                    # clusterTopic[maxIndex].append(text)
                    FileName[maxIndex].append(text['id'])

                # 或者创建一个新的主题
                else:
                    dictTopic[numTopic] = []
                    dictTopic[numTopic].append(vector)
                    # clusterTopic[numTopic] = []
                    # clusterTopic[numTopic].append(text)
                    id_dict[numTopic] = []
                    id_dict[numTopic].append(cnt)
                    FileName[numTopic] = []
                    FileName[numTopic].append(text['id'])
                    numTopic += 1
            cnt += 1
            if cnt % 500 == 0:
                print("processing {}...".format(cnt))
        with open('/home/gwy/ALL/nuclear_data/dictTopic.vector', 'w') as f:
            json.dump(id_dict, f)
            # 将文档和话题的映射写入文件中
        with open('/home/gwy/ALL/nuclear_data/clusterTopic.txt', 'w') as f:
            json.dump(FileName, f)

        return dictTopic, clusterTopic

    def read(self, file, n):
        file_list = file_dir(file, n)
        data = []
        for file in file_list:
            with open(file, 'r') as f:
                lines = f.readlines()
                if len(lines) == 0:
                    continue
                time = lines[2].strip().split('\t')[-1]
                content = lines[-1].strip().split('\t')[-1]
                content = [Opencc.to_simple(char) for char in content]
                content = "".join(content)
                data.append({'id': file, 'time': time, 'content': content})
        return data

    def fit_transform(self, theta=0.5, n=3000):
        datMat = self.read(self.filename, n)
        word_segmentation = []
        for i in range(len(datMat)):
            word_segmentation.append(self.word_segment(datMat[i]['content']))
        print("............................................................................................")
        print('文本已经分词完毕 !')

        # 得到文本数据的空间向量表示
        corpus_tfidf = self.get_Tfidf_vector_representation(word_segmentation)
        #corpus_tfidf =  self.get_Doc2vec_vector_representation(word_segmentation)
        dictTopic, clusterTopic = self.single_pass(corpus_tfidf, datMat, theta)
        print("............................................................................................")
        print("得到的主题数量有: {} 个 ...".format(len(dictTopic)))
        print("............................................................................................\n")
        # 按聚类语句数量对主题进行排序，找到重要的聚类群
        clusterTopic_list = sorted(clusterTopic.items(), key=lambda x: len(x[1]), reverse=True)
        # num = 0
        # for k in clusterTopic_list[:1000]:
        #     topic_order = sorted(k[1], key=lambda x: x['time'])
        #     cluster_title = '\n'.join([i['content'].split(r'\n')[0] for i in topic_order])
        #     file_name = [i['id'] for i in topic_order]
        # with open("/home/nuclear/PCube3/preprocess/Cluster/{}_{}.txt".format(str(num), str(len(k[1]))), 'w') as f:
        #     f.write(cluster_title)
        # num += 1
        # 得到每个聚类中的的主题关键词
        # word = TextRank4Keyword()
        # word.analyze(''.join(self.word_segment(''.join(cluster_title))),window = 5,lower = True)
        # w_list = word.get_keywords(num = 10,word_min_len = 2)
        # sentence = TextRank4Sentence()
        # sentence.analyze(cluster_title ,lower = True)
        # s_list = sentence.get_key_sentences(num = 3,sentence_min_len = 5)[:30]
        # print ("【主题索引】:{} \n【主题声量】：{} \n【主题关键词】： {} \n【主题中心句】 ：\n{}".format(k[0],len(k[1]),','.join([i.word for i in w_list]),'\n'.join([i.sentence.replace('\\n','\n') for i in s_list])))
        # print("【主题索引】:{} \n【主题声量】：{} ".format(k[0],len(k[1])))
        # print("【主题文章名】", [i['id'] for i in k[1]])
        # print ("-------------------------------------------------------------------------")


def file_dir(file, n):
    file_dir = os.walk(file)
    file_list = []
    num = 0
    for root, dirs, files in file_dir:
        for file in files:
            if num >= n and n != -1:
                break
            if file.endswith('txt'):
                file_list.append(os.path.join(root, file))
                num += 1
    return file_list


if __name__ == '__main__':

    start = datetime.datetime.now()
    # 中间写代码块
    single_pass_cluster = Single_Pass_Cluster('/home/gwy/ALL/HCube/NLPCC/',
                                              stop_words_file='static/preproces/stop_word.txt')
    single_pass_cluster.fit_transform(theta=0.15, n=300)
    end = datetime.datetime.now()
    print('Running time: %s Seconds' % (end-start))
