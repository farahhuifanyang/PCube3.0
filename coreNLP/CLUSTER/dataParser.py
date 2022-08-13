'''
Author: Guowenying
Date: 2021-04-27 09:10:09
LastEditTime: 2021-06-09 08:46:10
LastEditors: Guowenying
Description: In User Settings Edit
FilePath: /PCube3/coreNLP/CLUSTER/dataParser.py
'''

import os 
from classes.Article import Article
from coreNLP.CLUSTER.config import args
from openccpy.opencc import Opencc
class CLUDataset(object):
    def file_dir(self, dir, n):
        """
        读取指定文件夹中的n个文件  

        Parameters
        ----------
        dir : string
            准备读取的文件夹
        n : int
            指定读取文件夹下文件的个数。n等于-1时读取文件夹下所有文件

        Returns
        -------
        list
            该文件夹下所有文件列表
        """
        log_parsed = []
        if os.path.exists(os.path.join(args.temp_file,'log.txt')):
            with open(os.path.join(args.temp_file,'log.txt'), 'r') as logf: # 存储已经处理过的文件名
                for line in logf:
                    log_parsed.append(line.replace('\n', ''))

        file_dir = os.walk(dir)
        file_list = []
        num = 0
        for root, dirs, files in file_dir:
            for file in files:
                if num >= n and n != -1:
                    break
                if file.endswith(args.file_suffix):# and os.path.join(root, file) not in log_parsed:
                    file_list.append(os.path.join(root, file))
                    num += 1
        print("{}'s files will be processed...".format(num))
        return file_list

    def read(self, dir, n):
        """
        读取所有文件

        Parameters
        ----------
        dir : string
            待处理的文件夹
        n : int
            对数据集中的多少文件聚类

        Returns
        -------
        list
            所有文件的内容
        """
        file_list = self.file_dir(dir, n)
        data = []
        for file in file_list:
            try:    # 解析文件
                article = Article.init_from_file(file)
                data.append(article.to_dict())
            except Exception:
                print(f"{file} failed\n")
                continue
        return data
