'''
Author: your name
Date: 2021-02-03 10:23:51
LastEditTime: 2021-06-05 10:11:48
LastEditors: Please set LastEditors
Description: In User Settings Edit
FilePath: /PCube3/preprocess/forAnnotation.py
'''
from wiki_content_extractor import wiki_html_to_text
from LTPtagger import LTPtagger
import matplotlib.pyplot as plt
import wptools
import requests
import json
import os

tagger = LTPtagger()
plt.rc("font", family='YouYuan')


def parseOneFile(input, output):
    """
    对一篇文章原文进行LTP预处理

    Args:
        input (str): 输入文章路径
        output (str): 输出文章路径
    """
    with open(input, "r", encoding="utf-8") as rf:
        content = "".join(rf.readlines())
    content_tagged = tagger.tag(content, ["split", "chs", "seg", "pos", "dp", "ner"])
    segs = content_tagged["seg"]
    poss = content_tagged["pos"]
    heads = content_tagged["head"]
    dps = content_tagged["dp"]
    ners = content_tagged["ner"]

    def toConll2002(segs, ners):
        """以conll2002的格式进行转化，只有NER标注

        Args:
            segs (list[str]): 分词
            ners (list[str]): 分词对应NER

        Returns:
            [type]: [description]
        """
        outlines = []
        for seg, ner in zip(segs, ners):
            for word, tag in zip(seg, ner):
                word = word.replace(" ", "")
                outlines.append(word+" "+tag+"\n")
            outlines.append("\n")
        return outlines

    def toConllLike(segs, poss, heads, dps, ners):
        """以inception规定的conll的格式进行转化，但该形式可能无法正常导入，目前弃用

        Args:
            segs (list[str]): 分词
            poss (list[str]): 词性标注
            heads (list[int]): 依存中心词位置
            dps (list[str]): 依存关系
            ners (list[str]): 分词对应NER

        Returns:
            [type]: [description]
        """
        outlines = []
        for seg, pos, head, dp, ner in zip(segs, poss, heads, dps, ners):
            for i, data in enumerate(zip(seg, pos, head, dp, ner)):
                word, postag, headind, dprel, nertag = data
                word = word.replace(" ", "")
                outlines.append(f"{i}\t{word}\t_\t{postag}\t{nertag}\t{headind}\t{dprel}\n")
            outlines.append("\n")
        return outlines

    outlines = toConll2002(segs, ners)
    # outlines = toConllLike(segs, poss, heads, dps, ners)
    with open(output, "w", encoding="utf-8") as wf:
        wf.writelines(outlines)


def statistics():
    years = {}
    topics = {}
    root = "/home/disk2/nuclear/news_data/PCube/"
    for year in os.listdir(root):
        yeardir = root + year
        if not os.path.isdir(yeardir):
            continue
        for topic in os.listdir(yeardir):
            topicdir = yeardir + "/" + topic
            if not os.path.isdir(topicdir):
                continue
            for file in os.listdir(topicdir):
                if topic in topics:
                    topics[topic] += 1
                else:
                    topics[topic] = 1

                if year in years:
                    years[year] += 1
                else:
                    years[year] = 1

    x = [key[2:] for key, value in years.items()]
    y = [value for key, value in years.items()]
    plt.figure(figsize=(20, 5))
    plt.plot(x, y)
    plt.savefig("./preprocess/year.jpg")

    x = [key for key, value in topics.items()]
    y = [value for key, value in topics.items()]
    plt.clf()
    plt.bar(x, y)
    plt.savefig("./preprocess/topic.jpg")


def getWikiArticles():
    selected_pages = ["臺灣", "中華民國", "海峽兩岸關係", "泛藍", "泛綠", "中國國民黨", "民主進步黨", "台灣民眾黨_(2019年)", "中華民國總統府", "行政院", "立法院", "司法院", "考試院", "監察院", "李登輝", "陳水扁", "馬英九", "蔡英文", "韓國瑜",
                      "朱立倫", "宋楚瑜", "台灣獨立運動", "二二八事件", "九二香港會談", "台灣海峽飛彈危機", "2016年中華民國總統選舉", "2020年中華民國總統選舉", "臺灣進口美國肉類問題"]
    for title in selected_pages:
        page = wptools.page(title, lang='zh')
        result = page.get_query(proxy='http://127.0.0.1:7890')   # http://127.0.0.1:1080

        print(result.data["url"])
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/74.0.3729.131 Safari/537.36'
        }
        response = requests.get(result.data["url"], headers=headers, timeout=(3, 10))
        # with open(f"wiki_articles/{title}.html", "w", encoding="utf-8") as wf:
        #     wf.write(response.text)
        content = wiki_html_to_text(response.text)
        with open(f"wiki_articles/{title}.txt", "w", encoding="utf-8") as wf:
            wf.write(content)


def preprocessWiki():
    infdir = "wiki_articles"
    otfdir = "ann_articles"
    for file in os.listdir(infdir):
        inf = os.path.join(infdir, file)
        otf = os.path.join(otfdir, file)
        parseOneFile(inf, otf)


if __name__ == '__main__':
    # inf = "/home/disk2/nuclear/news_data/PCube/2021-01/财经/e21ae85e98e814fe0468d11da5134be3.html"
    # otf = "/home/disk2/nuclear/PCubeAnn/试行标注6.txt"
    # parseOneFile(inf, otf)
    getWikiArticles()
    preprocessWiki()
