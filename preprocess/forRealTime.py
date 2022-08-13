'''
Author: your name
Date: 2021-04-07 10:13:15
LastEditTime: 2021-06-09 08:48:02
LastEditors: Please set LastEditors
Description: In User Settings Edit
FilePath: /PCube3.0/preprocess/main.py
'''
from classes import Article
from preprocess.LTPtagger import LTPtagger


def processOneArticle(article: Article):
    """处理单篇文章

    Args:
        article (Article): 文章对象

    Returns:
        str, dict: 文章id与预处理后的内容元组， 内容：
                {“sentence”: [“原文词语1”, “原文词语2”, …],
                “pos”:[“词类1”, “词类2”, …], “dp”: [“依存关系1”, “依存关系2, …”],
                “head”:[“依存中心1”, “依存中心2”]}

    """
    tagger = LTPtagger()
    title_tagged = tagger.tag(article.title, ["chs", "seg", "pos", "dp"])
    content_tagged = tagger.tag(article.content, ["split", "chs", "seg", "pos", "dp"])
    segs = title_tagged["seg"] + content_tagged["seg"]
    poss = title_tagged["pos"] + content_tagged["pos"]
    heads = title_tagged["head"] + content_tagged["head"]
    dps = title_tagged["dp"] + content_tagged["dp"]
    parsedArticle = [{"sentence": seg, "pos": pos, "dp": dp, "head": head}
                     for seg, pos, dp, head in zip(segs, poss, dps, heads)]

    return article.id, parsedArticle


def main(articles):
    """
    预处理一批文章

    Args:
        articles (list[Article]): 文章列表，每个元素都是Article对象

    Returns:
        dict: {“文章ID”: [{“sentence”: [“原文词语1”, “原文词语2”, …],
                “pos”:[“词类1”, “词类2”, …], “dp”: [“依存关系1”, “依存关系2, …”],
                “head”:[“依存中心1”, “依存中心2”]} ]}
    """
    output = {}
    for article in articles:
        k, v = processOneArticle(article)
        output[k] = v
    return output
