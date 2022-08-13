'''
Author: your name
Date: 2021-06-22 09:52:20
LastEditTime: 2021-06-24 10:35:08
LastEditors: Please set LastEditors
Description: In User Settings Edit
FilePath: /PCube3.0/crawl/crawlEntityPictures.py
'''
import os
import io
import json
import wptools
import requests
from PIL import Image
from configure import globalFLAGS
from DAO.Neo4jNode import Neo4jNode


def getPicture(eid, form):
    """
    从wiki获取图片URL并下载图片

    Args:
        eid ([type]): 实体id
        form ([type]): 实体名
    """
    path = os.path.join(globalFLAGS.picture_dir, eid+f".png")
    if os.path.exists(path):
        return
    page = wptools.page(form, lang='zh')
    try:
        wiki = page.get(
            proxy=f'http://127.0.0.1:{globalFLAGS.EL_proxy_port}', timeout=20)
    except Exception:
        return
    pic_url = ""
    cur_pix = 0
    if "image" not in wiki.data:
        return
    for image in wiki.data["image"]:
        if "url" in image:
            if image["height"] > cur_pix:
                pic_url = image["url"]
                cur_pix = image["height"]

    if pic_url:
        proxies = {
            "https": f'127.0.0.1:{globalFLAGS.EL_proxy_port}'
        }
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/74.0.3729.131 Safari/537.36'
        }
        r = requests.get(pic_url, headers=headers,
                         proxies=proxies, timeout=100)

        byte_stream = io.BytesIO(r.content)
        roiImg = Image.open(byte_stream)  # Image打开二进制流Byte字节流数据
        imgByteArr = io.BytesIO()   # 创建一个空的Bytes对象
        roiImg.save(imgByteArr, format='PNG')  # PNG就是图片格式，我试过换成JPG/jpg都不行
        imgByteArr = imgByteArr.getvalue()  # 这个就是保存的二进制流
        # 下面这一步只是本地测试， 可以直接把imgByteArr，当成参数上传到七牛云
        with open(path, 'wb') as f:
            f.write(imgByteArr)


def getEntityList():
    """
    从neo4j获取实体列表，返回一个生成器
    """
    neo4j = Neo4jNode(globalFLAGS.neo4j_url,
                      globalFLAGS.neo4j_usr, globalFLAGS.neo4j_passwd)
    res = neo4j.searchAll()
    for node in res:
        yield node["eid"], node["name"]


def entityListFromEL():
    """
    从实体链接暂存结果进行图片获取，数据库无法使用的临时办法

    Yields:
        eid, name: 实体id，实体正式名
    """
    with open("static/processOnBaseData/EL_saved.txt", "r") as ellog:
        line = ellog.readlines()
    if line:
        isParsed = True
        line = line[0].strip()
    else:
        isParsed = False
    linked_dir = os.path.join(globalFLAGS.tmp_result_dir, "EL_Linked")
    for file in os.listdir(linked_dir):
        fullpath = os.path.join(linked_dir, file)
        files = json.load(open(fullpath, "r", encoding="utf-8"))
        for aid, sents in files.items():
            if isParsed:
                if aid == line:
                    isParsed = False
                continue
            for sidx, sent in enumerate(sents):
                for eidx, entity in enumerate(sent["entities"]):
                    if "wikiID" in entity:
                        yield entity["wikiID"], entity["form"]


if __name__ == "__main__":
    entity_list = getEntityList()
    for eid, name in entity_list:
        getPicture(eid, name)
