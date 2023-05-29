import os
import cv2
from pathlib import Path

import numpy as np
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms.transforms import Compose, ToTensor

from utils.data_loading import BasicDataset


# class VOCDataset(Dataset):
#     def __init__(self, data_dir='dataset/VOCdevkit/VOC2012', transform=Compose([ToTensor()])):
#         self.data_dir = Path(data_dir)
#
#         self.img_prefix = 'JPEGImages'
#         self.mask_prefix = 'SegmentationClass'
#
#         self.transform = transform
#
#         self.img_list = [Path(i).stem for i in os.listdir(self.data_dir / 'SegmentationClass')]
#         self.img_list.sort()
#
#     def __len__(self):
#         return len(self.img_list)
#
#     def __getitem__(self, index):
#         img = cv2.imread(str(self.data_dir / self.img_prefix / self.img_list[index]) + '.jpg')
#         mask = cv2.imread(str(self.data_dir / self.mask_prefix / self.img_list[index]) + '.png')
#         mask = np.where(mask == (192, 224, 224), (0, 0, 0), mask).astype(np.uint8)
#
#         img = cv2.resize(img, (572, 572))
#         mask = cv2.resize(mask, (388, 388))
#
#         img = (img / 255).astype(np.float32)
#         mask = (mask / 255).astype(np.float32)
#
#         data = {'input': self.transform(img), 'label': self.transform(mask)}
#         # if self.transform:
#         #     data = self.transform(data)
#
#         return data


class VOCDataset(BasicDataset):
    def __init__(self, data_dir='dataset/VOCdevkit/VOC2012', scale=1):
        self.data_dir = Path(data_dir)
        self.img_dir = 'JPEGImages'
        self.mask_dir = 'SegmentationClassnp'
        super().__init__(str(self.data_dir / self.img_dir), str(self.data_dir / self.mask_dir), scale, mask_suffix='.png')
