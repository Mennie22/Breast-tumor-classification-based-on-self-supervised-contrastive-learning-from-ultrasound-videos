import os
from PIL import Image
import cv2
import numpy as np

path = ['./new5videoshot', './fuzzy', './bilateral']
files = os.listdir(path[0])

'''obtain fuzzy and bilateral imgs'''
for im in files:
    path = os.path.join(path[0], im)
    img = np.array(Image.open(path))
    retval, dst = cv2.threshold(img, 0, 255, cv2.THRESH_OTSU)
    bilateral = cv2.bilateralFilter(img, d=10, sigmaColor=15, sigmaSpace=10)
    Image.fromarray(dst).save('./fuzzy/' + im)
    Image.fromarray(bilateral).save('./bilateral/' + im)


'''original, fuzzy and bilateral imgs stack in channels'''
for im in files:
    path1 = os.path.join(path[0], im)
    path2 = os.path.join(path[1], im)
    path3 = os.path.join(path[2], im)
    img1 = np.array(Image.open(path1))
    img2 = np.array(Image.open(path2))
    img3 = np.array(Image.open(path3))
    img = np.stack([img1, img2, img3], axis=-1)

    Image.fromarray(img).save('./stack/'+im)



