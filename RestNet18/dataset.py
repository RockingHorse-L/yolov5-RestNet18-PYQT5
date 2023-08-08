import os

import cv2
import numpy as np
import torch
from PIL import Image
import torch.nn as nn
from torch.fx.experimental.migrate_gradual_types.constraint import Transpose
from torch.utils.data import Dataset
from torchvision.transforms import Compose, Resize, ColorJitter, RandomHorizontalFlip, Normalize



class Dataset(Dataset):
    def __init__(self,
                 root,
                 is_train,):
        #         self.train_dataset = Dataset(root=r"D:\AIdata\gutou\arthrosis", is_train=True)
        self.dataset = []
        train_or_test = "train" if is_train else "test"
        file_path = f'{root}/{train_or_test}/DIPFirst'
        for label in os.listdir(file_path):
            for img_path in os.listdir(f'{file_path}//{label}'):
                self.dataset.append((f'{file_path}//{label}//{img_path}', label))


    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        image_path, label = self.dataset[idx]
        im = cv2.imread(image_path)
        im = im.swapaxes(1, 2).swapaxes(0, 1)
        im = np.array(im) / 255
        # if self.mode  == 'train':
        #     im = mytransform(im)
        one_hot = np.zeros(11)
        one_hot[int(label)- 1] = 1
        return np.float32(im), np.float32(one_hot)

