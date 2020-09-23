import glob
import cv2
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import math
from tqdm import tqdm
import os
from pathlib import Path
from defisheye import Defisheye


fov = 180
pfov = 120

dtype = "equalarea" #   #"linear","stereographic","orthographic"
format_ = "fullframe" #  #"circular"

dataset_dir = "images"

images_dir = dataset_dir + "/images3/"     ##enter dataset directory of frames

'''
Data format will be like:
../images3/
    01_frame.jpg
    02_frame.jpg
    ..
    ..
    ..

'''

img_name = glob.glob(images_dir + "*.jpg")

img_list = []
for name in img_name:
    name = (Path(name).stem)+'.jpg'
    img_list.append(name)

list.sort(img_list)
print("Total images: " + str(len(img_list)))


#output_video = cv2.VideoWriter('video.mp4', cv2.VideoWriter_fourcc(*'mp4v'), 20, (1056, 1056))

image_file = os.path.join(images_dir,img_list[0])
frame0 = cv2.imread(image_file)
fourcc = cv2.VideoWriter_fourcc(*'mp4v') 
height,width,layers= frame0.shape
output_video=cv2.VideoWriter('defished.avi', fourcc, 20, (width,height))  ##saved video format, name


for i in tqdm(range(0, len(img_list))):
    image_file = os.path.join(images_dir,img_list[i])
    img = cv2.imread(image_file)
    obj = Defisheye(img, dtype=dtype, format=format_, fov=fov, pfov=pfov)
    frame_to_write = obj.convert()

    output_video.write(frame_to_write)

#cv2.destroyAllWindows()
output_video.release()
