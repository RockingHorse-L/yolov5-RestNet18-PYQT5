import cv2
import numpy as np


def convert_to_square(path):
    img = cv2.imread(path)
    h, w = img.shape[:2]
    max_side = np.maximum(h, w)
    width = max_side
    height = max_side
    W = int(abs(w * 0.5 - max_side * 0.5))
    H = int(abs(h * 0.5 - max_side * 0.5))
    bg = np.zeros((height, width, 3), dtype=np.uint8)
    bg[H: h+H, W:w+W] = img
    cv2.imshow('Canvas with Image', bg)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    return bg
# import cv2
# import numpy as np

def image_Hist(img_path):
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(1, 1))
    equalized = clahe.apply(img)
    cv2.imshow('Canvas with Image', equalized)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    return equalized

def paste_image_on_canvas(image_path):
    # 读取原始图像
    img = cv2.imread(image_path)
    img = cv2.resize(img, (224, 224))
    # 显示结果
    cv2.imshow('Canvas with Image', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    return img


# 调用函数，传入图片路径



if __name__ == '__main__':
    path = 'DIP_101893.png'
    img = paste_image_on_canvas(path)
    # cv2.imshow('img', img)
    # cv2.imwrite()
    # cv2.destroyAllWindows()
