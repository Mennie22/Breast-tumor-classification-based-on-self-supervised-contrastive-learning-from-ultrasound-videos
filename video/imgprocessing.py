import os
from PIL import Image
import cv2
import numpy as np

path = ['./new5videoshot','./fuzzy','bilateral']

files = os.listdir(path[0])
for im in files:
    path1 = os.path.join(path[0],im)
#    path2 = os.path.join(path[1],im)
#    path3 = os.path.join(path[2],im)
    
    print(path1)
    img1 = Image.open(path1)
 #   img2 = Image.open(path2)
 #   img3 = Image.open(path3)
 #   img = np.stack([img1,img2,img3],axis=-1)
    img = np.array(img)
    retval, dst = cv2.threshold(img, 0, 255, cv2.THRESH_OTSU) #cv2.bilateralFilter(img, d=10, sigmaColor=15, sigmaSpace=10)
    Image.fromarray(dst).save('./fuzzy/'+im)
    #Image.fromarray(img).save('./stack/'+im)

