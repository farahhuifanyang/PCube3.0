'''
Author: your name
Date: 2021-02-03 10:24:54
LastEditTime: 2021-02-07 12:32:12
LastEditors: Please set LastEditors
Description: In User Settings Edit
FilePath: /PCube3/preprocess/LTPtagger.py
'''
from ltp import LTP
from openccpy.opencc import Opencc


class LTPtagger(object):
    def __init__(self, scale='base'):
        self.ltp = LTP(path=scale)
        self.tagmap = {"Nh": "PER", "Ni": "ORG", "Ns": "LOC"}

    def tag(self, content, args=["split", "chs", "seg", "pos", "dp", "ner"]):
        result = {}
        if "chs":
            content = [Opencc.to_simple(char) for char in content]
            content = "".join(content)

        if "split" in args:
            sents = self.ltp.sent_split(content.split(r"\n"), "zh")
            result["split"] = sents
        else:
            sents = [content]

        if "seg" in args:
            words, hiddens = self.ltp.seg(sents)
            result["seg"] = words

            if "pos" in args:
                poss = self.ltp.pos(hiddens)
                result["pos"] = poss

            if "dp" in args:
                dps = self.ltp.dep(hiddens)
                result["dp"] = [[dp[2] for dp in sent] for sent in dps]
                result["head"] = [[dp[1] for dp in sent] for sent in dps]

            if "ner" in args:
                ners = self.ltp.ner(hiddens)
                result["ner"] = []
                for segs, ner in zip(words, ners):
                    # 处理一句话
                    tags = ["O"] * len(segs)
                    for tag in ner:
                        # tags[tag[1]] = self.tagmap[tag[0]]
                        if tag[1] != 0 and tags[tag[1]-1] != "O" and tags[tag[1]-1].split("-")[-1] == self.tagmap[tag[0]]:
                            tags[tag[1]] = "I-"+self.tagmap[tag[0]]
                        else:
                            tags[tag[1]] = "B-"+self.tagmap[tag[0]]
                        if tag[1] != tag[2]:
                            for i in range(tag[1] + 1, tag[2]):
                                tags[i] = "I-"+self.tagmap[tag[0]]
                    result["ner"].append(tags)
        else:
            if "pos" in args or "dp" in args or "ner" in args:
                raise ValueError("\"seg\" must be chosen while \"pos\", \"dp\" or \"ner\" is chosen!")

        return result
