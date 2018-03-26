# -*- coding: utf-8 -*-
import scrapy
import json
import time
import re

from crawl_picture.items import ImageItem


class BaiduImageSpider(scrapy.Spider):
    name = 'baidu_image'
    allowed_domains = ['image.baidu.com']

    url_format = "https://image.baidu.com/search/acjson?tn=resultjson_com&ipn=rj&ct=201326592&is=&fp=result&queryWord={word}&cl=2&lm=-1&ie=utf-8&oe=utf-8&adpicid=&st=&z=3&ic=&word={word}&s=&se=&tab=&width=0&height=0&face=&istype=&qc=&nc=&fr=&pn={pn}&rn=30&gsm={gsm}&{ts}="

    def __init__(self, query_word, crawl_count=0, *args, **kwargs):
        super(BaiduImageSpider, self).__init__(*args, **kwargs)
        self.query_word = query_word
        self.crawl_count = int(crawl_count)

    def start_requests(self):
        gsm = hex(0)[2:]
        url = self.url_format.format(word=self.query_word, pn=1, gsm=gsm, ts=int(time.time()))
        yield scrapy.Request(url=url)

    def parse(self, response):
        data = json.loads(response.body, strict=False)
        img_urls = [img.get("objURL", "") for img in data["data"] if img.get("objURL")]
        img_urls = [decode(url) for url in img_urls]
        item = ImageItem()
        item["image_urls"] = img_urls
        yield item
        pn = re.findall("pn=(\d+)", response.url)
        pn = int(pn[0]) + 30
        if self.crawl_count <= 0 or pn < self.crawl_count:
            gsm = hex(pn)[2:]
            url = self.url_format.format(word=self.query_word, pn=pn, gsm=gsm, ts=int(time.time()))
            yield scrapy.Request(url=url)


str_table = {
    '_z2C$q': ':',
    '_z&e3B': '.',
    'AzdH3F': '/'
}

char_table = {
    'w': 'a',
    'k': 'b',
    'v': 'c',
    '1': 'd',
    'j': 'e',
    'u': 'f',
    '2': 'g',
    'i': 'h',
    't': 'i',
    '3': 'j',
    'h': 'k',
    's': 'l',
    '4': 'm',
    'g': 'n',
    '5': 'o',
    'r': 'p',
    'q': 'q',
    '6': 'r',
    'f': 's',
    'p': 't',
    '7': 'u',
    'e': 'v',
    'o': 'w',
    '8': '1',
    'd': '2',
    'n': '3',
    '9': '4',
    'c': '5',
    'm': '6',
    '0': '7',
    'b': '8',
    'l': '9',
    'a': '0'
}

# str 的translate方法需要用单个字符的十进制unicode编码作为key
# value 中的数字会被当成十进制unicode编码转换成字符
# 也可以直接用字符串作为value
char_table = {ord(key): ord(value) for key, value in char_table.items()}


def decode(url):
    # 先替换字符串
    for key, value in str_table.items():
        url = url.replace(key, value)
    # 再替换剩下的字符
    return url.translate(char_table)
