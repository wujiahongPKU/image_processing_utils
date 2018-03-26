

#coding=utf-8

import numpy as np
import os
import sys

file_source="film.rgb"

file_open=open(file_source,'r')




name=file_open.seek(1920*1080*3,0)
print ("name:",name)