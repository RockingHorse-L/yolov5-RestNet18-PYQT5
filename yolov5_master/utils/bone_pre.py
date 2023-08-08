import os
import cv2

def imgae_Hist(path):
    image = cv2.imread(path, 0)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    equalized = clahe.apply(image)
    return equalized
    # cv2.imshow('img', equalized)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

if __name__ == '__main__':
    path = r'D:\AIdata\gutou\voc\voc\VOCdevkit\VOC2007\JPEGImages'
    save_path = r'D:\AIdata\gutou\voc\voc\VOCdevkit\VOC2007\Images_Pre'
    file = os.listdir(path)
    for img in file:
        img_path = path + '\\' + img
        img_new = imgae_Hist(img_path)
        cv2.imwrite(f'{save_path}/{img}', img_new )



