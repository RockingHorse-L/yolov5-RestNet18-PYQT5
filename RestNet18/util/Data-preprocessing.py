import os
import cv2
import numpy as np
import random
import imutils


def image_Hist(img):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(1, 1))
    equalized = clahe.apply(img)
    return equalized

#扩充：找到中心点，及最大边长，沿着最大边长的两边扩充
def paste_image_on_canvas(img3):

    img = cv2.resize(img3, (224, 224))
    return img

# 随机旋转
def img_return(img):
    num = random.randint(-45, 45)
    rot1 = imutils.rotate(img, angle=num)
    return rot1

def img_save(org_path, save_path):
    foldDic = {}
    className = os.listdir(org_path)
    train_txt = 'D:/AIdata/gutou/arthrosis/train_labels'
    for name in className:
        img_path = org_path + '\\' + name
        #print(img_path)
        img_path_name = os.listdir(img_path)
        foldDic[name] = img_path_name

    for key, value in foldDic.items():
        if not os.path.exists(train_txt):
            os.makedirs(train_txt)
        list_file = open(f'{train_txt}/{key}.txt', 'w')
        for i in foldDic[key]:
            imgPath = os.path.join(org_path, key, i)
            savePath = os.path.join(save_path, key, i)
            imgPathName = os.listdir(imgPath)
            for name in imgPathName:
                if not os.path.exists(savePath):
                    os.makedirs(savePath)
                img = cv2.imread(os.path.join(imgPath, name))
                square_img = paste_image_on_canvas(img)
                for j in range(5):
                    ro1 = img_return(square_img)
                    equalized = image_Hist(ro1)
                    cv2.imwrite(f'{savePath}/{j}{name}', equalized)
                    list_file.write(f'{savePath}/{j}{name} {i}\n')
                #print(name)


if __name__ == '__main__':
    org_path = r'D:\AIdata\gutou\arthrosis\arthrosis'
    save_path = r'D:\AIdata\gutou\arthrosis\train'
    img_save(org_path, save_path)
