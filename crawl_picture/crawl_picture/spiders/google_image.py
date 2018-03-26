# -*- coding: utf-8 -*-
import scrapy


class GoogleImageSpider(scrapy.Spider):
    name = 'google_image'
    allowed_domains = ['www.google.com/imghp']
    start_urls = ['http://www.google.com/imghp/']

    def parse(self, response):
        pass
