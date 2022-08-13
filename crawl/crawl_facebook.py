'''
Author: Liuhezi
Date: 2021-04-16 15:26:02
LastEditTime: 2021-04-28 11:10:05
LastEditors: Liuhezi
Description: In User Settings Edit
FilePath: /PCube3.0/crawl/crawl_facebook.py
'''

import os

from configure import globalFLAGS
from facebook_scraper import get_posts
from tqdm import tqdm


def get_user_urls(file):
    """
    :param file: 脸书用户与url映射文件
    :return: 读取映射文件内容保存url映射表
    """
    user_url = {}
    user_id = {}
    with open(file, 'r') as f:
        for line in f.readlines():
            line = line.strip()
            name = line.split('\t')[0]
            id = line.split('\t')[1]
            url = line.split('\t')[2]
            user_url[name] = user_url
            user_id[name] = id
    return user_url,user_id


def crawl_posts(user_url,user_id, save_path, post_cnt):
    """
    :param user_url: 脸书用户url映射文件
    :param user_id: 脸书用户对应wikidataID映射文件
    :param save_path: 保存至本地的路径
    :param post_cnt: 每个用户爬取的帖子数
    :return: 
    """
    for k, v in tqdm(user_url.items()):
        cnt = 0
        # 保存用户目录为 '用户中文名_wikidataID的形式'
        path = save_path + k+'_'+user_id[k]
        if not os.path.exists(path):
            os.makedirs(path)
        # print(v)
        for post in tqdm(get_posts(v, pages=None)):
            if cnt == post_cnt:
                break
            # user_posts.setdefault(k, []).append(post['text'][:-1])
            with open(path + '/' + str(post_cnt) + '.txt', 'w', encoding='utf-8') as f:
                f.write(post['text'][:-1])
            cnt += 1


if __name__ == '__main__':
    user_url,user_id = get_user_id_url(globalFLAGS.PER_user_id_url)
    crawl_posts(user_url,user_id, './data/', 100)
