import os

import cv2
import numpy as np
from torch.utils.data import Dataset


class Data_set(Dataset):
    def __init__(self, root):
        super().__init__()
        self.dataset = []
        self.num = []
        img_file = os.listdir(root)
        img_path = img_file[-1]
        image_path = os.path.join(root, img_path, 'crops')
        # for nums in imgName_list:
        #     self.num.append(nums.split('.')[0])
        file = os.listdir(image_path)
        for file_name in file:
            path = os.path.join(image_path, file_name)
            img_list = os.listdir(path)
            for imgName in img_list:
                img = imgName.split('_')
                labels = img[0]
                imgname = img[1]
                self.dataset.append((f'{path}//{imgName}', labels))


    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        data = self.dataset[index]
        # number = self.num[:-2]
        img = cv2.imread(data[0])
        try:
            # imgName = data[2].split('.')
            # imgNum = imgName[0]
            img = cv2.resize(img, (224, 224))
            img = img.swapaxes(1, 2).swapaxes(0, 1)
            #print(img.shape)
            img = img.reshape(-1)
            img = img / 255
            #one_hot = np.zeros(26)
            #one_hot[int(data[1])] = 1
            return np.float32(img),  data[1]
        except Exception as e:
            print(e)
