# Define here the models for your scraped items
#
# See documentation in:
# https://docs.scrapy.org/en/latest/topics/items.html

import scrapy


class CrawlLtnItem(scrapy.Item):
    # define the fields for your item here like:
    id_ = scrapy.Field()
    url = scrapy.Field()
    title = scrapy.Field()
    time = scrapy.Field()
    theme = scrapy.Field()
    content = scrapy.Field()
    keywords = scrapy.Field()
 
    
