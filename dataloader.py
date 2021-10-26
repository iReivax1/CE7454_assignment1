from __future__ import division
import os

import numpy as np
import torch
import torch.nn.parallel
import torch.optim
import torch.utils.data
import torchvision.transforms as transforms
from PIL import Image
from torch.utils.data.dataset import Dataset

class AttrDatasetLoader(data.Dataset):
    CLASSES = None

    def __init__(self, img_path, img_file, label_file, bbox_file, img_size):
        self.img_path = img_path

        normalize = transforms.Normalize(
            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        self.transform = transforms.Compose([
            transforms.Resize(256),
            transforms.RandomResizedCrop(img_size[0]),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ])

        # read image path names i.e. img/0001.jpg
        fp = open(img_file, 'r')
        self.img_list = [x.strip() for x in fp]
        # read attribute labels and category annotations
        self.labels = np.loadtxt(label_file, dtype=np.float32)
        # [[1,2,3,4,5,6], [1,2,3,4,5,6] ..... [...]]
        self.cat1_labels = self.labels[:, [0]]
        # y_temp = torch.eye(7)
        # self.cat1_labels = np.asarray(y_temp[self.cat1_labels]).squeeze()
        self.cat2_labels = self.labels[:, [1]]
        # y_temp = torch.eye(3)
        # self.cat2_labels = np.asarray(y_temp[self.cat2_labels]).squeeze()
        self.cat3_labels = self.labels[:, [2]]
        # y_temp = torch.eye(3)
        # self.cat3_labels = np.asarray(y_temp[self.cat3_labels]).squeeze()
        
        self.cat4_labels = self.labels[:, [3]]
        # y_temp = torch.eye(4)
        # self.cat4_labels = np.asarray(y_temp[self.cat4_labels]).squeeze()       
        self.cat5_labels = self.labels[:, [4]]
        # y_temp = torch.eye(6)
        # self.cat5_labels = np.asarray(y_temp[self.cat5_labels]).squeeze()
        self.cat6_labels = self.labels[:, [5]]
        # y_temp = torch.eye(3)
        # self.cat6_labels = np.asarray(y_temp[self.cat6_labels]).squeeze()

        self.img_size = img_size

        # load bbox
        if bbox_file:
            self.bboxes = np.loadtxt(bbox_file, usecols=(0, 1, 2, 3))
            self.with_bbox = True
        else:
            self.with_bbox = False
            self.bboxes = None



    def get_bbox_img(self, idx):
        img = Image.open(os.path.join(self.img_path,
                                      self.img_list[idx])).convert('RGB')

        width, height = img.size
        if self.with_bbox:
            bbox_cor = self.bboxes[idx]
            x1 = int(bbox_cor[0])
            y1 = int(bbox_cor[1])
            x2 = int(bbox_cor[2]) 
            y2 = int(bbox_cor[3]) 
            bbox_w = x2 - x1
            bbox_h = y2 - y1
            img = img.crop(box=(x1, y1, x2, y2))
        else:
            bbox_w, bbox_h = self.img_size[0], self.img_size[1]

        img.thumbnail(self.img_size, Image.ANTIALIAS)
        img = self.transform(img)
        img = torch.tensor(img, dtype=torch.float32)
        data = {'image': img, 'label': {
                                        'cat1_label': torch.from_numpy(self.cat1_labels[idx]),
                                        'cat2_label': torch.from_numpy(self.cat2_labels[idx]),
                                        'cat3_label': torch.from_numpy(self.cat3_labels[idx]),
                                        'cat4_label': torch.from_numpy(self.cat4_labels[idx]),
                                        'cat5_label': torch.from_numpy(self.cat5_labels[idx]),
                                        'cat6_label': torch.from_numpy(self.cat6_labels[idx]),
                                        }
                }
        return data

    def __getitem__(self, idx):
        return self.get_bbox_img(idx)

    def __len__(self):
        return len(self.img_list)
