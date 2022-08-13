'''
Author: Guowenying
Date: 2021-04-28 11:32:15
LastEditTime: 2021-06-09 21:34:25
LastEditors: Guowenying
Description: In User Settings Edit
FilePath: /PCube3/coreNLP/SUM/DataParser.py
'''
import emoji
import numpy
import os
import gc
import glob
import json
import re
import torch
from coreNLP.SUM.config import args
from multiprocessing import Pool
from transformers import BertTokenizer
from os.path import join as pjoin
import subprocess
from classes.Article import Article
from elasticsearch5 import Elasticsearch
from elasticsearch5 import exceptions
import re

#将经过chinese stanford-nlp的句子还原
REMAP = {"-lrb-": "(", "-rrb-": ")", "-lcb-": "{", "-rcb-": "}",
         "-lsb-": "[", "-rsb-": "]", "``": '"', "''": '"'}

class Batch(object):
    """
    将数据转换成Batch处理
    Parameters
    ----------
    object : class
        父类
    """
    def _pad(self, data, pad_id, width=-1):
        """
        将不同长度的文章和和不同句子数量的文章进行padding
        Parameters
        ----------
        data : list
            进入模型的一batch数据
        pad_id : int
            用于padding的数字
        width : int, optional
            将这个batch的数据统一padding成同一长度, by default -1：代表padding成最长的那篇文章的长度

        Returns
        -------
        list
            经过padding之后的数据
        """
        if (width == -1):
            width = max(len(d) for d in data)
        rtn_data = [d + [pad_id] * (width - len(d)) for d in data]
        return rtn_data

    def __init__(self, data=None, device=None):
        """
        对数据padding成能参加训练的张量
        Parameters
        ----------
        data : list, optional
            padding过的batch数据，待处理成tensor, by default None
        device : int, optional
            数据处理完成后放在哪个设备上, by default None
        """
        if data is not None:
            self.batch_size = len(data)
            pre_src = [x[0] for x in data]
            pre_labels = [x[1] for x in data]
            pre_segs = [x[2] for x in data]
            pre_clss = [x[3] for x in data]

            src = torch.tensor(self._pad(pre_src, 0))

            labels = torch.tensor(self._pad(pre_labels, 0))
            segs = torch.tensor(self._pad(pre_segs, 0))
            mask = ~ (src == 0)

            clss = torch.tensor(self._pad(pre_clss, -1))
            mask_cls = ~ (clss == -1)
            clss[clss == -1] = 0

            setattr(self, 'clss', clss.to(device))
            setattr(self, 'mask_cls', mask_cls.to(device))
            setattr(self, 'src', src.to(device))
            setattr(self, 'labels', labels.to(device))
            setattr(self, 'segs', segs.to(device))
            setattr(self, 'mask', mask.to(device))

            
            src_str = [x[-2] for x in data]
            setattr(self, 'src_str', src_str)
            tgt_str = [x[-1] for x in data]
            setattr(self, 'tgt_str', tgt_str)

    def __len__(self):
        """
        返回batch的大小
        Returns
        -------
        int
            batch_size
        """
        return self.batch_size


def batch(data, batch_size):
    """
    从batch_size块中的数据中yield元素。

    Parameters
    ----------
    data : list
        一个batch中的数据
    batch_size : int
        对一个batch中数据的字数要求

    Yields
    -------
    minibatch
        返回的batch数据是延迟加载的
    """
    minibatch, size_so_far = [], 0
    for ex in data:
        minibatch.append(ex)
        size_so_far = simpleBatchSizeFn(ex, len(minibatch))
        if size_so_far == batch_size:
            yield minibatch
            minibatch, size_so_far = [], 0
        elif size_so_far > batch_size:
            yield minibatch[:-1]
            minibatch, size_so_far = minibatch[-1:], simpleBatchSizeFn(ex, 1)
    if minibatch:
        yield minibatch


def loadDataset(args, corpus_type):
    """
    数据生成器，不要在这里做额外的操作，比如print，因为要推迟到第一次加载时

    Parameters
    ----------
    args : list
        加载时必须的参数
    corpus_type : test. must in ['train' or 'valid' or 'test']

    Returns
    -------
    list
        加载的事件文章, 该部分是延迟加载

    """
    assert corpus_type in ["train", "valid", "test"]

    def lazyDatasetLoader(pt_file):
        dataset = torch.load(pt_file)
        return dataset

    
    pts = sorted(glob.glob(args.bert_data_path + '.' + corpus_type + '.[0-9]*.pt'))
    if pts:
        for pt in pts:
            yield lazyDatasetLoader(pt)
    else:
        pt = args.bert_data_path + '.' + corpus_type + '.pt'
        yield lazyDatasetLoader(pt)


def simpleBatchSizeFn(new, count):
    """
    当前读取多长文档内容，用于确定batch_size.
    note: 每个batch中不是包含固定数量的文章，batch_size的确定通过src_elements不超过某个值
    Parameters
    ----------
    new : list
        [事件的内容， 事件中句子的label-仅在测试时有意义]
    count : int
        当前batch中包含几个事件

    Returns
    -------
    int
        该batch中篇幅最长的事件的字数*该batch中包含的事件的个数
    """
    src, labels = new[0], new[1]
    global max_n_sents, max_n_tokens, max_size
    if count == 1:
        max_size = 0
        max_n_sents=0
        max_n_tokens=0
    max_n_sents = max(max_n_sents, len(src))
    max_size = max(max_size, max_n_sents)
    src_elements = count * max_size
    return src_elements


class SUMDataset(object):
    def __init__(self, args, datasets,  batch_size,
                 device):
        self.args = args
        self.datasets = datasets
        self.batch_size = batch_size
        self.device = device
        self.cur_iter = self._nextDatasetIterator(datasets)

        assert self.cur_iter is not None

    def __iter__(self):
        """
        SUMDataset类变成可迭代类
        Yields
        -------
        [type]
            [description]
        """
        dataset_iter = (d for d in self.datasets)
        while self.cur_iter is not None:
            for batch in self.cur_iter:
                yield batch
            self.cur_iter = self._nextDatasetIterator(dataset_iter)


    def _nextDatasetIterator(self, dataset_iter):
        try:
            if hasattr(self, "cur_dataset"):
                self.cur_dataset = None
                gc.collect()
                del self.cur_dataset
                gc.collect()

            self.cur_dataset = next(dataset_iter)
        except StopIteration:
            return None

        return DataIterator(args = self.args,
            dataset=self.cur_dataset,  batch_size=self.batch_size,
            device=self.device)


class DataIterator(object):
    def __init__(self, args, dataset,  batch_size,  device=None):
        self.args = args
        self.batch_size, self.dataset = batch_size, dataset
        self.iterations = 0
        self.device = device
        
        self.sort_key = lambda x: len(x[1])

        self._iterations_this_epoch = 0

    def data(self):
        xs = self.dataset
        return xs

    def preprocess(self, ex):
        """
        读取每个事件中必要的信息
        Parameters
        ----------
        ex : dict
            事件

        Returns
        -------
            事件中包含的信息
        """
        src = ex['src']
        labels = ex['labels']
        segs = ex['segs']
        clss = ex['clss']
        src_txt = ex['src_txt']
        tgt_txt = ex['tgt_txt']

        return src, labels, segs, clss, src_txt, tgt_txt

    def batchBuffer(self, data, batch_size):
        """
        确定batch
        Parameters
        ----------
        data : dict
            事件
        batch_size : int
            batch中的事件应该遵循的字数限制

        Yields
        -------
        list
            该batch中包含的事件个数
        """
        minibatch, size_so_far = [], 0
        for ex in data:
            if(len(ex['src'])==0):
                continue
            ex = self.preprocess(ex)
            if(ex is None):
                continue
            minibatch.append(ex)
            size_so_far = simpleBatchSizeFn(ex, len(minibatch))
            if size_so_far == batch_size:
                yield minibatch
                minibatch, size_so_far = [], 0
            elif size_so_far > batch_size:
                yield minibatch[:-1]
                minibatch, size_so_far = minibatch[-1:], simpleBatchSizeFn(ex, 1)
        if minibatch:
            yield minibatch

    def createBatches(self):
        """
        创建batch前需要的操作， 首先按照max(事件长度)*事件个数 <= self.batch_size * 50，确定该batch中应该包含多少个事件
        Yields
        -------
        list
            一个batch中的事件
        """
        data = self.data()
        for buffer in self.batchBuffer(data, self.batch_size * 50):

            p_batch = sorted(buffer, key=lambda x: len(x[3]))
            p_batch = batch(p_batch, self.batch_size)

            p_batch = list(p_batch)
            for b in p_batch:
                yield b

    def __iter__(self):
        """
        返回可参与训练的batch数据
        Yields
        -------
        Batch class
            返回的batch数据
        """
        while True:
            self.batches = self.createBatches()
            for idx, minibatch in enumerate(self.batches):
                if self._iterations_this_epoch > idx:
                    continue
                self.iterations += 1
                self._iterations_this_epoch += 1
                batch = Batch(minibatch, self.device)

                yield batch
            return



# 数据的预处理阶段，生成模型需要的数据格式

def clean(x):
    """

    Parameters
    ----------
    x : string
        经过分词和分句处理后的数据

    Returns
    -------
    string
        对REMAP中的符号进行转换
    """
    return re.sub(
        r"-lrb-|-rrb-|-lcb-|-rcb-|-lsb-|-rsb-|``|''",
        lambda m: REMAP.get(m.group()), x)

def tokenize():
    """
    对每个事件进行分词和分句处理， 使用chinese stanford-nlp工具
    Raises
    ------
    Exception
        如果分词和分句处理前后的文章个数不同则抛出异常
    """
    stories_dir = os.path.abspath(args.raw_path)
    tokenized_stories_dir = os.path.abspath(args.save_path)

    stories = os.listdir(stories_dir)
    # make IO list file
    with open("mapping_for_corenlp.txt", "w") as f:
        for s in stories:
            f.write("%s\n" % (os.path.join(stories_dir, s)))
    command = ['java', 'edu.stanford.nlp.pipeline.StanfordCoreNLP', '-props', '/home/nuclear/stanford-corenlp-full-2018-10-05/chinese', '-annotators', 'tokenize,ssplit', '-ssplit.newlineIsSentenceBreak', 'always', '-filelist', 'mapping_for_corenlp.txt', '-outputFormat', 'json', '-continueOnAnnotateError','true', '-outputDirectory', tokenized_stories_dir]
    subprocess.call(command)
    os.remove("mapping_for_corenlp.txt")

    # Check that the tokenized stories directory contains the same number of files as the original directory
    num_orig = len(os.listdir(stories_dir))
    num_tokenized = len(os.listdir(tokenized_stories_dir))
    if num_orig != num_tokenized:
        raise Exception(
            "The tokenized stories directory %s contains %i files, but it should contain the same number as %s (which has %i files). Was there an error during tokenization?" % (
            tokenized_stories_dir, num_tokenized, stories_dir, num_orig))
    print("Successfully finished tokenizing %s to %s.\n" % (stories_dir, tokenized_stories_dir))


def format_to_Lines():
    """
    将单独的每篇事件转化成包含shard_size个事件的若干block
    """
    test_files = glob.glob(pjoin(args.save_path, '*'))
    a_lst = [(f, args) for f in test_files]
    pool = Pool(args.n_cpus)
    dataset = []
    p_ct = 0
    for d in pool.imap_unordered(_format_to_Lines, a_lst):
        dataset.append(d)
        if (len(dataset) > args.shard_size):
            pt_file = "{:s}.{:s}.{:d}.json".format(args.json_path, 'test', p_ct)
            with open(pt_file, 'w') as save:
                save.write(json.dumps(dataset, ensure_ascii=False))
                p_ct += 1
                dataset = []
    pool.close()
    pool.join()
    if (len(dataset) > 0):
        pt_file = "{:s}.{:s}.{:d}.json".format(args.json_path, 'test', p_ct)
        with open(pt_file, 'w') as save:
            save.write(json.dumps(dataset, ensure_ascii=False))
            p_ct += 1
            dataset = []
            
def _format_to_Lines(params):
    """
    加载每一个事件的事件内容，及其事件ID

    Parameters
    ----------
    params : list
        文件名及其他必需的参数

    Returns
    -------
    dict
        {'src': 事件内容, 'tgt': 事件的ID}
    """
    f, args = params
    source, tgt = loadJson(f)
    return {'src': source, 'tgt': tgt}

def loadJson(p):
    """
    加载分词和分句后的事件内容， 转化成简洁的形式
    Parameters
    ----------
    p : string
        文件名

    Returns
    -------
    source, eventID
        事件的内容, 事件的ID
    """
    source = []
    eventID = []
    flag = False
    for sent in json.load(open(p))['sentences']:
        tokens = [t['word'] for t in sent['tokens']]
        if (tokens[0] == 'eventID'):
            continue
        if (tokens[0] == 'content'):
            flag = True
            continue
        if flag:
            source.append(tokens)
        else:
            eventID.append(tokens)

    source = [clean(' '.join(sent)).split() for sent in source]

    assert len(eventID) == 1
    return source, ''.join(eventID[0])       

def format_to_Bert():
    """
    ALBERT对数据进行分词和转化成计算机能够处理的形式，利用多线程的方式
    """
    a_lst = []
    for json_f in glob.glob(pjoin(args.json_read_path, '*' + 'test' + '.*.json')):
        real_name = json_f.split('/')[-1]
        a_lst.append((json_f, args, pjoin(args.bert_path, real_name.replace('json', 'albert.pt'))))
    print(a_lst)
    pool = Pool(args.n_cpus)
    for d in pool.imap(_format_to_Bert, a_lst):
        pass

    pool.close()
    pool.join()

def _format_to_Bert(params):
    """
    具体用ALBERT进行转化的操作
    Parameters
    ----------
    params : list
        具体的参数，要处理的文章内容，对句子和事件的具体的要求，保存的路径
    """
    json_file, args, save_file = params
    if (os.path.exists(save_file)):
        return

    bert = BertData(args)
    jobs = json.load(open(json_file))
    datasets = []
    for d in jobs:
        source, tgt = d['src'], d['tgt']
        b_data = bert.preprocess(source, tgt)
        if (b_data is None):
            continue
        indexed_tokens, labels, segments_ids, cls_ids, src_txt, tgt_txt = b_data
        b_data_dict = {"src": indexed_tokens, "labels": labels, "segs": segments_ids, 'clss': cls_ids,
                       'src_txt': src_txt, "tgt_txt": tgt_txt}
        datasets.append(b_data_dict)
    torch.save(datasets, save_file)
    datasets = []
    gc.collect()

class BertData():
    """
    ALBERT对事件进行处理的具体操作，利用preprocess进行处理
    """
    def __init__(self, args):
        self.args = args
        self.tokenizer = BertTokenizer.from_pretrained('voidful/albert_chinese_base')

        self.sep_vid = self.tokenizer.vocab['[SEP]']
        self.cls_vid = self.tokenizer.vocab['[CLS]']
        self.pad_vid = self.tokenizer.vocab['[PAD]']

    def preprocess(self, src, tgt):
        """
        对事件进行处理， 包括用BERTokenizer对事件内容分词，加上句子的段信息，句子开始标志以及记录事件的ID信息等
        Parameters
        ----------
        src : list
            事件的内容
        tgt : string
            事件的ID

        Returns
        -------
        dict
            每一篇文章必须的信息
        """
        if (len(src) == 0):
            return None

        original_src_txt = [''.join(s) for s in src]

        labels = [0] * len(src)

        idxs = [i for i, s in enumerate(src) if (len(s) > self.args.min_src_ntokens)]

        src = [src[i][:self.args.max_src_ntokens] for i in idxs]
        labels = [labels[i] for i in idxs]
        src = src[:self.args.max_nsents]
        labels = labels[:self.args.max_nsents]

        if (len(src) < self.args.min_nsents):
            return None
        if (len(labels) == 0):
            return None

        src_txt = [''.join(sent) for sent in src]
        text = ' [SEP] [CLS] '.join(src_txt)
        src_subtokens = self.tokenizer.tokenize(text)
        src_subtokens = src_subtokens[:510]
        src_subtokens = ['[CLS]'] + src_subtokens + ['[SEP]']

        src_subtoken_idxs = self.tokenizer.convert_tokens_to_ids(src_subtokens)
        _segs = [-1] + [i for i, t in enumerate(src_subtoken_idxs) if t == self.sep_vid]
        segs = [_segs[i] - _segs[i - 1] for i in range(1, len(_segs))]
        segments_ids = []
        for i, s in enumerate(segs):
            if (i % 2 == 0):
                segments_ids += s * [0]
            else:
                segments_ids += s * [1]
        cls_ids = [i for i, t in enumerate(src_subtoken_idxs) if t == self.cls_vid]
        labels = labels[:len(cls_ids)]

        src_txt = [original_src_txt[i] for i in idxs]
        return src_subtoken_idxs, labels, segments_ids, cls_ids, src_txt, tgt

def delete_Dir(path):
    for root, dirs, files in os.walk(path, topdown=False): 
        for name in files: 
            os.remove(os.path.join(root, name)) 


def read(events):
    """
    根据eventID和对应的ArticleIDS读取相应文章内容创建事件
    Parameters
    ----------
    events : dict
        {event_ID:事件的ID , article_IDS: 事件包含的所有文章}
    """ 
    delete_Dir(args.raw_path)
    delete_Dir(args.save_path)
    delete_Dir(args.json_read_path)
    delete_Dir(args.bert_path)

    for event_ID, article_IDs in events.items():
        event_content = []
        with open(pjoin(args.raw_path, event_ID),'w') as f:
            f.write('eventID\n'+event_ID+'\n'+'content\n')
            for article in article_IDs:
                paragraphs = Article.init_from_file(pjoin(args.source_path, article+'.html')).to_dict()['content']
                event_content.append(paragraphs.split('\\n')[0])              
            f.write('\n'.join(event_content))

def read_From_ES(events):
    """
    根据eventID和对应的ArticleIDS读取相应文章内容创建事件
    Parameters
    ----------
    events : dict
        {event_ID:事件的ID , article_IDS: 事件包含的所有文章}
    """ 

    ES = [
        '10.105.242.74:9200'
    ]   #elasticsearch集群服务器的地址
    es = Elasticsearch(
        ES, 
        # 启动前嗅探es集群服务器
        sniff_on_start=True,
        # es集群服务器结点连接异常时是否刷新es节点信息
        sniff_on_connection_fail=True,
        # 每60秒刷新节点信息
        sniffer_timeout=60
    )
    
    delete_Dir(args.raw_path)
    delete_Dir(args.save_path)
    delete_Dir(args.json_read_path)
    delete_Dir(args.bert_path)
    for event_ID, article_IDs in events.items():
        event_content = []
        with open(pjoin(args.raw_path, event_ID),'w') as f:
            f.write('eventID\n'+event_ID+'\n'+'content\n')
            for article in article_IDs:
                try:
                    res = es.get(index="udn", doc_type="doc", id=article)
                except exceptions.NotFoundError:
                    try:
                        res = es.get(index="ltn", doc_type="doc", id=article)
                    except exceptions.NotFoundError:
                        res = es.get(index="ct", doc_type="doc", id=article)
                # paragraphs = Article.init_from_file(pjoin(args.source_path, article+'.html')).to_dict()['content']
                content = res['_source']['content'].split('\\n')
                for i in content:
                    if len(i)>3:
                        event_content.append(i.replace(' ','').replace('\u2028',''))
                        break                
            f.write('\n'.join(event_content))

if __name__ == '__main__':
    with open(args.map_path,'r') as f:
        events = json.load(f)
    # read_From_ES(events)    
    # read(events)
    # tokenize()
    format_to_Lines()
    format_to_Bert()
