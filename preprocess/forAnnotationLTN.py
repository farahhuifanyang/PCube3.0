'''
Author: your name
Date: 2021-02-03 10:23:51
LastEditTime: 2021-03-17 09:13:49
LastEditors: Please set LastEditors
Description: In User Settings Edit
FilePath: /PCube3/preprocess/forAnnotation.py
'''
import os
import json
import random
import matplotlib.pyplot as plt
from preprocess.LTPtagger import LTPtagger
import sys
sys.path.append("/home/nuclear/PCube3/")

tagger = LTPtagger()
plt.rc("font", family='YouYuan')


def parseOneFile(input, output):
    with open(input, "r") as rf:
        lines = rf.readlines()
        title = lines[3].strip().split('\t')[-1]
        content = lines[-1].strip().split('\t')[-1]
    title_tagged = tagger.tag(title, ["chs", "seg", "pos", "dp", "ner"])
    content_tagged = tagger.tag(content, ["split", "chs", "seg", "pos", "dp", "ner"])
    segs = title_tagged["seg"] + content_tagged["seg"]
    poss = title_tagged["pos"] + content_tagged["pos"]
    heads = title_tagged["head"] + content_tagged["head"]
    dps = title_tagged["dp"] + content_tagged["dp"]
    ners = title_tagged["ner"] + content_tagged["ner"]

    def toConll2002(segs, ners):
        outlines = []
        for seg, ner in zip(segs, ners):
            for word, tag in zip(seg, ner):
                word = word.replace(" ", "")
                outlines.append(word+" "+tag+"\n")
            outlines.append("\n")
        return outlines

    def toConllLike(segs, poss, heads, dps, ners):
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
    with open(output, "w") as wf:
        wf.writelines(outlines)


def statistics():
    years = {}
    topics = {}
    root = "/home/disk2/nuclear/news_data/PCube/CT"
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


def randomSample():
    root = "/home/disk2/nuclear/news_data/PCube/LTN/"
    inputpaths = []
    filemap = {}
    for year in os.listdir(root):
        yeardir = root + year
        if not os.path.isdir(yeardir):
            continue
        for topic in os.listdir(yeardir):
            topicdir = yeardir + "/" + topic
            if not os.path.isdir(topicdir) or topic == "生活":
                continue
            for file in os.listdir(topicdir):
                inputpath = topicdir + "/" + file
                inputpaths.append(inputpath)
    inputpaths = random.sample(inputpaths, 500)
    num = 1
    for inputpath in inputpaths:
        name = str(num).zfill(4)
        outpath = f"/home/disk2/nuclear/PCubeAnn/LTNAnnImport/{name}.txt"
        try:
            parseOneFile(inputpath, outpath)
        except Exception as e:
            print(f"No.{num} file error: {e}")
            continue
        filemap[inputpath] = outpath
        print(f"No.{num} file parase done!")
        if num >= 300:
            break
        num += 1

    with open("static/preproces/LTNAnnFileMap.json", "w") as wf:
        json.dump(filemap, wf, ensure_ascii=False)


if __name__ == '__main__':
    # inf = "/home/disk2/nuclear/news_data/PCube/news_data/2019-01/国际/5369a903d7b8180f9f1ad6a3bdca871d.html"
    # otf = "/home/disk2/nuclear/PCubeAnn/test.txt"
    # parseOneFile(inf, otf)
    randomSample()
