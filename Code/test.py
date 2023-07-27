import xml.etree.ElementTree as ET
import pickle
import os 
from os import listdir, getcwd
from os.path import join
import random
from shutil import copyfile
import cv2
from tkinter import filedialog
import json

def convert(size,box):
    dw = 1./size[0]
    dh = 1./size[1]
    x = (float(box[2]) + float(box[0]))/2.0
    y = (float(box[1]) + float(box[3]))/2.0
    w = float(box[2]) - float(box[0])
    h = float(box[3]) - float(box[1])
    x = round(x*dw,6)
    w = round(w*dw,6)
    y = round(y*dh,6)
    h = round(h*dh,6)
    return (x,y,w,h)

data_list= list(sorted(os.listdir(os.path.join('Scaphoid\Images\\Normal'))))
path = os.path.join('Scaphoid\Images\\Normal')
for i in range(len(data_list)):
    data = data_list[i]

    name = data.split(".")[-2]
    data_path = os.path.join(path,data)
    print(data_path)
    img_data = cv2.imread(data_path)
    w,h,_= img_data.shape
    
    size =[h,w]
    annotation_path = os.path.join('Scaphoid\Annotations\Scaphoid_Slice',name + '.json')
    with open(annotation_path) as jsonFile:
        jsonObject = json.load(jsonFile)
        jsonFile.close()
    bboxlist = jsonObject[0]['bbox']

    a,b,c,d = convert(size,bboxlist)
    txt_path = os.path.join('label',name + '.txt')

    f= open(txt_path,'w')
    f.write('1 '+str(a)+' '+str(b)+' '+str(c)+' '+str(d))
    f.close()



