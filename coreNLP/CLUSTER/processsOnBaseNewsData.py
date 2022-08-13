'''
Author: Guowenying
Date: 2021-05-12 08:27:22
LastEditTime: 2021-05-12 22:15:04
LastEditors: Guowenying
Description: In User Settings Edit
FilePath: /PCube3/coreNLP/CLUSTER/processsOnBaseNewsData.py
'''
from coreNLP.CLUSTER.dataParser import CLUDataset
from coreNLP.CLUSTER.config import args
from classes.Article import Article
from coreNLP.CLUSTER.forCluster import SinglePassCluster
import os

def process():
    reader = CLUDataset()
    file_list = reader.file_dir(args.dir_name, args.number)
    data = []
    cur_parsed = []
    actuator = SinglePassCluster()
    for file in file_list:
        try:    # 解析文件
            article = Article.init_from_file(file)
            data.append(article.to_dict())
            cur_parsed.append(file)
        except Exception:
            print(f"{file} failed")
            continue
        if len(data) == 5000:
            print('已经读取五千篇文章')
            actuator.run(theta = args.theta, n = args.number, datMat= data)
            with open(os.path.join(args.temp_file,'log.txt'), 'a') as logf:
                for log in cur_parsed:
                    logf.write(log+'\n')
            data = []
            cur_parsed = []
    return data
if __name__ == '__main__':
    process()
