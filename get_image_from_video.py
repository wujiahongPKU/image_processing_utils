#coding=utf-8


import skvideo.io
import os
import cv2
import matplotlib.pyplot as plt


video_path="/home/wujiahong/ToXiaohong_ebest/video_test/"
image_path="/home/wujiahong/ToXiaohong_ebest/image_gen/"
video_id_list=os.listdir(video_path)
first_count=0
for video_id in video_id_list:
    first_count+=1
    video_name_path = os.path.join(video_path, video_id)
    print("video name path is:",video_name_path)
    videodata=skvideo.io.vreader(video_name_path)
    print ("vide",videodata)
    second_count=0
    try:
        for frame in videodata:
            second_count+=1
            if second_count%10==0:
                cv2.imwrite(image_path+'the_'+str(first_count)+"_"+str(second_count)+".jpg",cv2.cvtColor(frame,cv2.COLOR_RGB2BGR))
    except:
        continue

print("the process have done")