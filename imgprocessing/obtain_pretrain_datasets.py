# -*- coding: utf-8 -*-
"""
Created on Sun Mar  6 16:38:02 2022

@author: zs
"""
import os
import cv2
import numpy as np
import SimpleITK as sitk
import random
from skimage.filters import threshold_otsu
from skimage.metrics import structural_similarity as ssim
from get_mask import get_mask

def get_regionalFD(image, bin = 20):
    from sklearn.metrics.cluster import entropy
    xs = np.floor(np.linspace(0, image.shape[0], bin))
    ys = np.floor(np.linspace(0, image.shape[1], bin))
    xx, yy = np.meshgrid(xs, ys, sparse=False, indexing='ij') #'ij'纵向为i轴，横向为j轴
    x1x2x1x2 = np.concatenate((xx[:-1, :, np.newaxis], xx[1:, :, np.newaxis]), axis=-1).reshape((-1, 2))
    y1y2y1y2 = np.concatenate((yy[:, :-1, np.newaxis], yy[:, 1:, np.newaxis]), axis=-1).reshape((-1, 2))

    DF_map = np.zeros_like(image) *1. #乘1.0变成float
    for x1x2, y1y2 in zip(x1x2x1x2, y1y2y1y2):
        patch_image = image[int(x1x2[0]):int(x1x2[1]), int(y1y2[0]):int(y1y2[1])]
        DF = entropy(patch_image)
        DF_map[int(x1x2[0]):int(x1x2[1]), int(y1y2[0]):int(y1y2[1])] = DF

    return DF_map

def get_box(image):  #裁剪超声图像，将边缘黑色框和病人信息等去掉

    DF_map_sum = np.zeros_like(image) * 1.
    #bins = [30, 40, 50, 60, 70 ,80]
    bins = [30, 50, 100, 150]
    for bin in bins:
        DF_map = get_regionalFD(image, bin=bin)
        DF_map_sum += (DF_map / np.max(DF_map)) #先除以最大值，防止溢出，再累加

    thresh = threshold_otsu(DF_map_sum)
    DF_map_sum[DF_map_sum>=thresh] = 255
    DF_map_sum[DF_map_sum<=thresh] = 0

    binary = DF_map_sum.astype(np.uint8)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5,5)) #morph_rect掩码传递信息
    binary = cv2.morphologyEx(binary, cv2.MORPH_DILATE, kernel,iterations = 1) #腐蚀膨胀

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (50,50))
    binary = cv2.morphologyEx(binary, cv2.MORPH_ERODE, kernel,iterations = 1)

    contours,hierarchy = cv2.findContours(binary,cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)#二值图像中查找轮廓
    area = [cv2.contourArea(cnt) for cnt in contours]
    max_area = max(area)
    for i,cnt in enumerate(contours):
        if area[i]/max_area < 0.2:
            cv2.fillPoly(binary, [cnt], 0)
    binary = cv2.morphologyEx(binary, cv2.MORPH_DILATE, kernel,iterations = 1)

    contours,hierarchy = cv2.findContours(binary,cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    area = [cv2.contourArea(cnt) for cnt in contours]
    x,y,w,h = cv2.boundingRect(contours[np.argmax(area)])
    return y,y+h,x,x+w

def loadfile(filename,cut = True):
    #导入视频图像
    if os.path.splitext(filename)[1].lower() in ['.avi','.mp4','.wmv'] :#返回文件扩展名,并转化为小写
        return load_avi(filename,cut)
    else:
        return load_dcm(filename,cut)

def load_dcm(filename,cut):
  #利用simpleITK读取dcm文件 ，读取医学影像文件
  path,name = os.path.split(filename)
  here = os.path.abspath('.')
  os.chdir(path)
  try:
    ds = sitk.ReadImage(name)
    img_array = sitk.GetArrayFromImage(ds) #图像变数组
  except:
    print('invalid data or file not exists')
    os.chdir(here)
    return None
  os.chdir(here)
  #读取的图片为YBR_FULL422格式，只取Y即可得到灰度图，每个像素有自己的Y亮度，但是两个像素共用相同的Blue和Red值
  if img_array.ndim>3:
    img_array=img_array[...,0] #注意数组最后一个维度是channel，只取第一个通道的值，也就是取亮度Y
  for i in range(img_array.shape[0]): #数组的每一行代表视频里面每一张图片
      if img_array[i].sum()!=0:
          #删掉文件开始时可能存在的全黑帧
          img_array=img_array[i:]
          break
  if cut:
      y_min,y_max,x_min,x_max = get_box(img_array[0])
  else:
      y_min,y_max,x_min,x_max = 0,img_array.shape[-2],0,img_array.shape[-1]
  img_array=img_array[:,y_min:y_max,x_min:x_max]
  return img_array

def load_avi(filename,cut):
    #利用opencv获取avi文件
    if not os.path.exists(filename):
        print('invalid data or file not exists')
        return None
    cap = cv2.VideoCapture(filename) #opencv读取图片的格式是BGR(H,w,channel),torchvision中的PIL则是( channel,H,w),数组格式是(H,w,channel)
    c=0
    x_min=0
    x_max=0
    y_min=0
    y_max=0
    img_array = []
    while 1:
        ret,frame = cap.read()
        if ret:
            img = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY) #RGB/BGR和灰度之间转变
            if img.sum()==0: continue
            if c==0:
                if cut:
                    y_min,y_max,x_min,x_max = get_box(img)
                    print(y_min,y_max,x_min,x_max)
                else:
                    y_max,x_max = img.shape[:2]
            img_array.append(img[y_min:y_max,x_min:x_max])
            c+=1
        else:
            break
    img_array=np.stack(img_array)
    return img_array


def find_all_video(path):  #遍历某个目录下的所有视频文件，如果某有后缀可能是DICOM文件
    for root,ds,fs in os.walk(path):
        for f in fs:
            ends = os.path.splitext(f)[1].lower()
            if ends in ['.avi','.wmv','.mp4','.dcm','']:#,
                fullname = os.path.join(root, f)
                yield fullname

name=''
ids={}

N = 0
#num=0

    
for file in find_all_video('/home/jzhang/data-med/zs-collection/ultrasound/breast/originaldata-new'):  # yield和for搭配能循环生成fullname
    print('operate {}'.format(file))
    person = os.path.split(os.path.split(os.path.split(file)[0])[0])[1]
    video = os.path.splitext(os.path.split(file)[1])[0]
#记录处理的患者名字，可能有bug，是通过文件路径提取患者名字

#    if name in ids:  #记录处理过的患者
#        num=ids[name]  #记录该患者的第num个视频
#        ids[name]+=1
#    else:
#        num = 0
#        ids[name] = 1
    
    img_array = loadfile(file,cut = True)
    if img_array is not None:
        img = np.random.randint(0,256,img_array[0].shape,dtype = np.uint8)
        for i,image in enumerate(img_array):
            if i%5==0:  #每15帧处理一次
                if ssim(image,img) < 0.45: #每当两张图像相似度小于某个阈值处理一次0.35
                    img = image
                    cls,mask = get_mask(img)  #使用深度模型判断有无肿瘤并分割
              #if cls>0.5:  #有肿瘤和无肿瘤分开保存
              #    cv2.imencode('.png', img)[1].tofile('tumor/'+name+'{}_{:0>3d}.png'.format(num,i))
              #else:
              #    cv2.imencode('.png', img)[1].tofile('notumor/'+name+'{}_{:0>3d}.png'.format(num,i))
            #cv2.imencode('.png', img)[1].tofile('tumor/'+name+'/{}_{:0>3d}-{:d}.png'.format(num,i,(cls>0.5))) #将同一个人的视频保存在一个目录下，以1.png结尾为有肿瘤的，以0.png结尾是无肿瘤的
                    
                    if cls>0.5:
                        if person not in ids:
                            N += 1
                            ids[person]={}
                            if video not in ids[person]:
                                ids[person][video] = {'编号':len(ids[person].keys())+1,'num':1}
                            else:
                                ids[person][video]['num'] += 1
                        else:
                            if video not in ids[person]:
                                ids[person][video] = {'编号':len(ids[person].keys())+1,'num':1}
                            else:
                                ids[person][video]['num'] += 1
                        print(ids)
                        cv2.imencode('.png',img)[1].tofile('/home/tyx/work/triplet_network/videoshot_5_new'+'/{}_{}_{}.jpg'.format(N,ids[person][video]['编号'],ids[person][video]['num']))

