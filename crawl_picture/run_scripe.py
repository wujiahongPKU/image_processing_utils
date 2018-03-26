#coding=utf-8

import os
import numpy as np
import sys


source_file=""
name_list=[]

def get_file_list(source_file):
    tempt_list=[]
    file_open=open(source_file)
    lines=file_open.readlines()
    for line in lines:
        line=line.strip().split(' ')
        # print("the line is:",line)
        for name in line:
            name=name.strip()
            # print name
            if name in tempt_list:
                continue
            else:
                tempt_list.append(name)

    return tempt_list


if __name__=="__main__":
    source_file="name.txt"
    print ("start step")
    name_list=get_file_list(source_file)
    count=0
    for name in name_list:
        count+=1
        print("start all count :",count)
        # name="asdf"
        os.environ['name']=str(name)
        dir_path="./image/"+str(name)
        print("dir path is:",dir_path)
        try:
            os.makedirs(dir_path)
            os.system("scrapy crawl baidu_image -a query_word=$name -a crawl_count=50 -s IMAGES_STORE=./image/$name/")
        except:
            continue


    print ("step have done")

