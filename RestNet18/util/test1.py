import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os, shutil, cv2, random


def save_file(list, path, name):
    myfile = os.path.join(path, name)
    if os.path.exists(myfile):
        os.remove(myfile)
    with open(myfile, "w") as f:
        f.writelines(list)


pic_path_folder = r"D:\AIdata\gutou\arthrosis\arthrosis"
for pic_folder in os.listdir(pic_path_folder):
    data_path = os.path.join(pic_path_folder, pic_folder)
    num_class = len(os.listdir(data_path))
    train_list = []
    val_list = []
    train_ratio = 0.9
    for folder in os.listdir(data_path):
        if os.path.isfile(os.path.join(data_path, folder)):
            continue
        train_nums = len(os.listdir(os.path.join(data_path, folder))) * train_ratio
        img_lists = os.listdir(os.path.join(data_path, folder))
        random.shuffle(img_lists)
        for index, img in enumerate(img_lists):
            if index < train_nums:
                train_list.append(os.path.join(data_path, folder, img) + ' ' + str(int(folder) - 1) + '\n')
            else:
                val_list.append(os.path.join(data_path, folder, img) + ' ' + str(int(folder) - 1) + '\n')

    random.shuffle(train_list)
    random.shuffle(val_list)
    save_file(train_list, data_path, 'train.txt')
    save_file(val_list, data_path, 'val.txt')
print("完成")