import os
import random
import cv2
import numpy as np

def img_save(org_path, save_path):
    foldDic = {}
    className = os.listdir(org_path)
    #test_txt = 'D:/AIdata/gutou/arthrosis/test_labels'
    for name in className:
        img_path = org_path + '\\' + name
        #print(img_path)
        img_path_name = os.listdir(img_path)
        foldDic[name] = img_path_name

    for key, value in foldDic.items():
        # if not os.path.exists(test_txt):
        #     os.makedirs(test_txt)
        # list_file = open(f'{test_txt}/{key}.txt', 'w')
        for i in foldDic[key]:
            imgPath = os.path.join(org_path, key, i)
            savePath = os.path.join(save_path, key, i)
            imgPathName = os.listdir(imgPath)
            l =  int(len(imgPathName) * 0.1)
            if not os.path.exists(savePath):
                os.makedirs(savePath)
            for j in range(l):
                try:
                    flag = random.randint(0, len(imgPathName))
                    img = cv2.imread(os.path.join(imgPath, imgPathName[flag - 1]))
                    cv2.imwrite(f'{savePath}/{imgPathName[flag - 1]}', img)
                    #list_file.write(f'{savePath}/{imgPathName[flag - 1]}\n')
                    os.remove(os.path.join(imgPath, imgPathName[flag - 1]))
                except:
                    print(imgPathName[flag - 1])

if __name__ == '__main__':
    sets = ['train', 'test']
    path = r'D:\AIdata\gutou\arthrosis\train'
    save_path = r'D:\AIdata\gutou\arthrosis\test'
    img_save(path, save_path)