# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
import cv2
from cv2 import findChessboardCorners
import tkinter as tk
import numpy as np
import glob
import torch
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import os
from PIL import Image
from tkinter import filedialog
from PIL import  ImageTk, Image, ImageDraw
from torch import nn
from torch.optim import SGD
from pathlib import Path
from torchvision import datasets
from torch.autograd import Variable
from tensorboardX import SummaryWriter 
import json
# model = torchvision.models.resnet152()
# model.fc = nn.Linear(2048, 2)
# device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
# resnet50_model = model.to(device)
# num_workers = 0
# batch_size = 2
# LR = 0.001
# opt = SGD(model.parameters(), lr=LR)
# Path_train = "Scaphoid_for_2\\crop\\Class1\\train"
# Path_val = "Scaphoid_for_2\\crop\\Class1\\val"

# TRAIN = Path(Path_train)
# VALID = Path(Path_val)
# lossFn = nn.CrossEntropyLoss()
# train_transforms = transforms.Compose([transforms.Resize((224,224)),
#                                   transforms.ToTensor(),
#                                   transforms.Normalize([0.485, 0.456, 0.406],
#                                                         [0.229, 0.224, 0.225]),
#                                   ])
# valid_transforms = transforms.Compose([transforms.Resize((224,224)),
#                                   transforms.ToTensor(),
#                                   transforms.Normalize([0.485, 0.456, 0.406],
#                                                         [0.229, 0.224, 0.225])])

# train_data = datasets.ImageFolder(TRAIN,transform=train_transforms)
# valid_data = datasets.ImageFolder(VALID,transform=valid_transforms)
# train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, num_workers=num_workers,shuffle=True)
# valid_loader = torch.utils.data.DataLoader(valid_data, batch_size=batch_size,  num_workers=num_workers,shuffle=True)
# opt = SGD(model.parameters(), lr=LR)
# lossFn = nn.CrossEntropyLoss()
# trainSteps = len(train_loader.dataset) 
# valSteps = len(valid_loader.dataset)
# acc_num = torch.zeros((1,2))

# epochs = 20
# for e in range(0,epochs):
#     print(e)
#     model.train()
#     totalTrainLoss = 0
#     totalValLoss = 0 
#     trainCorrect = 0
#     valCorrect = 0
#     total = 0
#     correct = 0
#     test_loss =0
#     predict_num = torch.zeros((1,2))
#     target_num = torch.zeros((1,2))
#     acc_num = torch.zeros((1,2))
    
#     for idx,(x,y) in enumerate(train_loader):
#         (x,y) = (x.to(device),y.to(device))
#         pred =  model(x)
#         loss = lossFn(pred,y)
#         opt.zero_grad()
#         loss.backward()
#         opt.step()
#         totalTrainLoss += loss.item() 
#         _, pred_out = torch.max(pred,1)
#         num_correct = (pred_out == y).sum()
#         trainCorrect += num_correct.item()
        
#     with torch.no_grad():
#         model.eval()
#         for idx,(x,y) in enumerate(valid_loader):
#             (x,y) = (x.to(device),y.to(device))
#             x,y=Variable(x,volatile=True),Variable(y)
#             outputs = model(x)
#             _,predicted = torch.max(outputs.data,1)
#             total += y.size(0)
#             correct += predicted.eq(y.data).cpu().sum()
#             pre_mask = torch.zeros(outputs.size()).scatter_(1,predicted.cpu().view(-1,1),1.)
#             predict_num += pre_mask.sum(0)
#             tar_mask = torch.zeros(outputs.size()).scatter_(1,y.data.cpu().view(-1,1),1.)
#             target_num += tar_mask.sum(0)
#             acc_mask = pre_mask*tar_mask
#             acc_num += acc_mask.sum(0) 
    
#     recall = acc_num/target_num
#     recall = (recall.numpy()[0]*100).round(3)
    
#     precision = acc_num/predict_num
#     precision = (precision.numpy()[0]*100).round(3)

#     F1 = 2*recall*precision/(recall+precision)
    
#     accuracy = acc_num.sum(1)/target_num.sum(1)
#     accuracy = (accuracy.numpy()[0]*100).round(3)
    
#     print("[INFO] EPOCH: {}/{}".format(e + 1, epochs))
#     print('Recall'," ".join("%s" % id for id in recall))
#     print('Precision'," ".join('%s' % id for id in precision))
#     print('F1'," ".join('%s' % id for id in F1))
#     print("Accuracy",accuracy)
    
# classes = ('Sca','No')
# model_trained_cpu = torch.load('Save_model_ver1.pt')
# model_trained = model_trained_cpu.to(device)
# model_trained.eval()

# x_img = x.permute(1,2,0).numpy()
# x_img = cv2.resize(x_img, (224*10,224*10))
# format_img = np.zeros([1,x.shape[0],x.shape[1],x.shape[2]])
# format_img[0] = x
# format_img = torch.tensor(format_img,dtype = torch.float)
# with torch.no_grad():
#     format_img = format_img.to(device) 
#     pred = model_trained(format_img)
#     _,predicted = pred.max(1)
#     predicted = predicted.cpu().numpy().astype(np.uint8)
#     ans = str(predicted[0])
# pred = pred.cpu().numpy().astype(np.float32)

# # print([predicted[i] for i in range(10)])
# print("predict :",classes[int(ans)])
# plt.title("Class:" + classes[int(ans)])
# plt.imshow(x_img)
# plt.show()

def get_iou(bbox_ai_r,bbox_gt_r):
    bbox_ai = [bbox_ai_r[0],bbox_ai_r[1],bbox_ai_r[2]-bbox_ai_r[0],bbox_ai_r[3]-bbox_ai_r[1]]
    bbox_gt = [bbox_gt_r[0],bbox_gt_r[1],bbox_gt_r[2]-bbox_gt_r[0],bbox_gt_r[3]-bbox_gt_r[1]]
    iou_x = max(bbox_ai[0], bbox_gt[0]) # x
    iou_y = max(bbox_ai[1], bbox_gt[1]) # y
    iou_w = min(bbox_ai[2]+bbox_ai[0], bbox_gt[2]+bbox_gt[0]) - iou_x # w
    iou_w = max(iou_w, 0)
    iou_h = min(bbox_ai[3]+bbox_ai[1], bbox_gt[3]+bbox_gt[1]) - iou_y # h
    iou_h = max(iou_h, 0)
    iou_area = iou_w * iou_h
    all_area = bbox_ai[2]*bbox_ai[3] + bbox_gt[2]*bbox_gt[3] - iou_area

    return max(iou_area/all_area, 0)

# pre_bbox = []
def load():
    global file_path
    file_path = filedialog.askdirectory()  
    list_dir = list(sorted(os.listdir(os.path.join(file_path))))
    # tp = 0
    # tn = 0
    # fp = 0
    # fn = 0
    # IOU = 0
    # for i in range(len(list_dir)):
    #     class_word = []
    #     print(list_dir[i])
    #     txt_path = os.path.join('label',list_dir[i].split('.')[-2]+'.txt')
    #     with open(txt_path) as f:
    #         for line in f.readlines():
    #             s = line.split(' ')
    #             class_word.append(int(s[0]))
    #     # print(class_word[0])
    #     model = torch.hub.load('ultralytics/yolov5', 'custom', path='batch4_epoch40_v5x_f1.pt')
    #     device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    #     img_path = os.path.join(file_path,list_dir[i])
    #     model = model.to(device)
    #     img = Image.open(img_path)
    #     results = model(img, size=640)
    #     gt_class = results.pandas().xyxy[0]['class'][0]
    #     # print('pred',int(class_word[0]),'GT',gt_class)
    #     if (class_word[0] == int(gt_class)):
    #         if(int(class_word[0])==1):
    #             tp += 1
    #         elif(int(class_word[0])==0):
    #             tn +=1
    #     elif(int(class_word[0]) != int(gt_class)):
    #         if(int(class_word[0])==1):
    #             fp +=1
    #         elif(int(class_word[0])==0):
    #             fn +=1
    #     xmin = round(results.pandas().xyxy[0]['xmin'])
    #     ymin = round(results.pandas().xyxy[0]['ymin'])
    #     xmax = round(results.pandas().xyxy[0]['xmax'])
    #     ymax = round(results.pandas().xyxy[0]['ymax'])
    
    #     pre_bbox = [xmin[0],ymin[0],xmax[0],ymax[0]]
    #     img_pic = list_dir[i]
    #     ann_name = img_pic.split('.')[-2]
    #     annotation_path = os.path.join('D:\Justin\Class\影像處理\HW2\Scaphoid_for_2\Annotations\Scaphoid_Slice' ,ann_name +'.json')
    #     with open(annotation_path) as jsonFile:
    #         jsonObject = json.load(jsonFile)
    #         jsonFile.close()
    #     bboxlist = jsonObject[0]['bbox']
    #     ori_img = Image.open(img_path)
    #     result_path = os.path.join('result','image0.jpg')
    #     a_img = Image.open(result_path)
    #     d_h = a_img.height/ori_img.height 
    #     d_w = a_img.width/ori_img.width
    #     r_xmin = round(float(bboxlist[0]))
    #     r_ymin = round(float(bboxlist[1]))
    #     r_xmax = round(float(bboxlist[2]))
    #     r_ymax = round(float(bboxlist[3]))
    #     bbox_ai_r =[r_xmin,r_ymin,r_xmax,r_ymax]
    #     IOU += round(get_iou(bbox_ai_r, pre_bbox),5)
    #     print(round(get_iou(bbox_ai_r,pre_bbox),5))
        
    # precision = tp/(tp+fp)
    # recall = tp/(tp+fn)
    # F1 = 2/((1/precision)+(1/recall))
    # print('Pre',precision)
    # print('Recall',recall)
    # print('F1',F1)
    # print('IOU', IOU/len(list_dir))
    
    global img_pic
    img_pic = list_dir[int(num)]
    global img_path
    img_path = os.path.join(file_path,img_pic)
    img = Image.open(img_path)
    img = img.resize( (img.width // 5, img.height // 5))   
    imgTk =  ImageTk.PhotoImage(img)                      
    lbl_2 = tk.Label(win,image=imgTk)                   
    lbl_2.image = imgTk
    lbl_2.place(x=300, y=50)
    model = torch.hub.load('ultralytics/yolov5', 'custom', path='batch4_epoch40_v5x_f1.pt')
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    model = model.to(device)
    results = model(img, size=640)
    class_conf =[results.pandas().xyxy[0]['class'],results.pandas().xyxy[0]['confidence']]
    class_i = ['Scaphoid','Normal']
    results.save('result')
    xmin = round(results.pandas().xyxy[0]['xmin'])
    ymin = round(results.pandas().xyxy[0]['ymin'])
    xmax = round(results.pandas().xyxy[0]['xmax'])
    ymax = round(results.pandas().xyxy[0]['ymax'])
    pre_bbox = [float(xmin),float(ymin),float(xmax),float(ymax)]
    ann_name = img_pic.split('.')[-2]
    annotation_path = os.path.join('D:\Justin\Class\影像處理\HW2\Scaphoid_for_2\Annotations\Scaphoid_Slice' ,ann_name +'.json')
    with open(annotation_path) as jsonFile:
        jsonObject = json.load(jsonFile)
        jsonFile.close()
    bboxlist = jsonObject[0]['bbox']
    ori_img = Image.open(img_path)
    result_path = os.path.join('result','image0.jpg')
    pred_img = cv2.imread(result_path)
    a_img = Image.open(result_path)
    d_h = a_img.height/ori_img.height 
    d_w = a_img.width/ori_img.width
    r_xmin = round(float(bboxlist[0])*d_w)
    r_ymin = round(float(bboxlist[1])*d_h)
    r_xmax = round(float(bboxlist[2])*d_w)
    r_ymax = round(float(bboxlist[3])*d_h)
    bbox_ai_r =[r_xmin,r_ymin,r_xmax,r_ymax]
    pred_img = cv2.rectangle(pred_img,(r_xmin,r_ymin),(r_xmax,r_ymax),(255,0,255),1)
    cv2.imwrite(result_path, pred_img)
    # global txt
    # txt = "IOU:",round(get_iou(bbox_ai_r, pre_bbox),2)
    global txt_class
    txt_class = 'Class:',class_i[int(class_conf[0])]
    global txt_conf
    txt_conf = 'Confidence:',round(float(class_conf[1]),3)
    crop_img = ori_img.crop((r_xmin/d_w,r_ymin/d_h,r_xmax/d_w,r_ymax/d_h))
    crop_img.save('crop\\crop.jpg')
    
def show_result():
    result_path = os.path.join('result','image0.jpg')
    f_img = Image.open(result_path)
    f_img = f_img.resize((f_img.width, f_img.height))
    imgTk =  ImageTk.PhotoImage(f_img)                      
    # lbl_1 = tk.Label(win, text=txt, bg='gray', fg='#263238', font=('Arial', 12), width=20, height=2)
    # lbl_1.place(x=1200, y=50)
    lbl_2 = tk.Label(win,image=imgTk)                   
    lbl_2.image = imgTk
    lbl_2.place(x=600, y=50)
    crop_path = os.path.join('crop','crop.jpg')
    c_img = Image.open(crop_path)
    c_img = c_img.resize((c_img.width, c_img.height))
    img_c =  ImageTk.PhotoImage(c_img)                      
    lbl_3 = tk.Label(win,image=img_c)                   
    lbl_3.image = img_c
    lbl_3.place(x=950, y=10)
    lbl_4 = tk.Label(win, text=txt_class, bg='gray', fg='#263238', font=('Arial', 12), width=20, height=2)
    lbl_4.place(x=1200, y=100)
    lbl_5 = tk.Label(win, text=txt_conf, bg='gray', fg='#263238', font=('Arial', 12), width=20, height=2)
    lbl_5.place(x=1200, y=150)
    # list_dir = list(sorted(os.listdir(os.path.join(file_path))))
    # txt_path = os.path.join('label',list_dir[int(num)].split('.')[-2]+'.txt')
    class_word = []
    # with open(txt_path) as f:
    #     for line in f.readlines():
    #         s = line.split(' ')
    #         class_word.append(int(s[0]))
    # class_i = ['Scaphoid','Normal']
    # txt_gt = 'Answer:' , class_i[class_word[0]]
    # lbl_6 = tk.Label(win, text=txt_gt, bg='gray', fg='#263238', font=('Arial', 12), width=20, height=2)
    # lbl_6.place(x=1200, y=200)

global win 
win = tk.Tk()
win.title("Hw2")
win.geometry("1400x500") 
win.resizable(0,0) 
win.config(bg = "#A0A0A0") 
win.attributes("-alpha",1) 
win.attributes("-topmost",0)  
group_1 = tk.LabelFrame(win, text ="Function",padx=5,pady=25)
group_1.pack(padx=10,pady=10)
group_1.place(x = 20,y = 50)
group_1.config(bg = "#A0A0A0")
load_b = tk.Button(group_1 , text="Load", bg = "#E0E0E0") #顏色也可以用config
load_b.config(width = 25,height = 2) # 長寬是以網格做調整
load_b.config(command = load) # 使用function
load_b.grid(row = 2,column = 0,padx=10,pady=7)
run_b = tk.Button(group_1 , text="Show result", bg = "#E0E0E0") #顏色也可以用config
run_b.config(width = 25,height = 2) # 長寬是以網格做調整
run_b.config(command = show_result) # 使用function
run_b.grid(row = 3,column = 0,padx=10,pady=7)
data_num = tk.Entry(group_1,width = 15)
data_num.grid(row = 0,column = 0,padx=0,pady=7)

def get_entry():
    global num
    num = data_num.get()  

OK = tk.Button(group_1, text="OK", bg = "#E0E0E0")
OK.config(width = 5,height = 1) # 長寬是以網格做調整
OK.config(command = get_entry) #使用function
OK.grid(row = 1,column = 0,padx=10,pady=7)

win.mainloop()

