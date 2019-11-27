import os
import os.path
import random
import cv2
import numpy as np
import torch
import torch.utils.data as data
from transforms import color_aug

def shu():
    print("hello")

class FoodDataSet(data.Dataset):
    def __init__(self, data_root, img_list, transform, is_train):

        self._data_root = data_root
        self._transform = transform
        self._is_train = is_train

        self._input_mean = torch.from_numpy(
            np.array([0.485, 0.456, 0.406]).reshape((1, 3, 1, 1))).float()
        self._input_std = torch.from_numpy(
            np.array([0.229, 0.224, 0.225]).reshape((1, 3, 1, 1))).float()

        self._load_list(img_list)

    def _load_list(self, img_list):
        self._image_list = []
        with open(img_list, 'r') as f:
            for line in f:
                image_file, label = line.strip().split()
                image_file = self._data_root + image_file
                self._image_list.append((
                    image_file,
                    int(label)))

        print('%d files loaded.' % len(self._image_list))

    def __getitem__(self, index):

        if self._is_train:
            image_path, label = random.choice(self._image_list)
        else:
            image_path, label = self._image_list[index]

        img = cv2.imread(image_path, cv2.IMREAD_COLOR|cv2.IMREAD_IGNORE_ORIENTATION)
        if img is None:
            print('Error: loading image %s failed.' % image_path)
            img = np.zeros((256, 256, 3))

        # BGR to RGB. (PyTorch uses RGB according to doc.)
        img = img[..., ::-1]

        frames = [img]
        frames = self._transform(frames)
        frames = np.array(frames)
        frames = np.transpose(frames, (0, 3, 1, 2))
        input = torch.from_numpy(frames).float() / 255.0
        input = (input - self._input_mean) / self._input_std

        return input[0], label

    def __len__(self):
        return len(self._image_list)
