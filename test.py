from __future__ import division

import os
import matplotlib.pyplot as plt
import torchvision.models as models
import torchvision.transforms as transforms
import torch.utils.data as data
import torch.optim as optim
import torch.nn.functional as F
import torch.nn as nn
import torch
from PIL import Image
from main import create_model_2 
import numpy as np
from torchvision.transforms.transforms import Resize
root_dir = "/Users/xavier/Documents/NTU/CE7454 Deep Learning/Assignment 1/FashionDataset"
use_cuda = torch.cuda.is_available()
device = 'cuda' if torch.cuda.is_available() else 'cpu'

categories = ['cat1', 'cat2', 'cat3', 'cat4', 'cat5', 'cat6']
cat_1 = ['floral', 'graphic', 'striped',
         'embroidered', 'pleated', 'solid', 'lattice']
cat_2 = ['long_sleeve', 'short sleeve', 'sleeveless']
cat_3 = ['maxi_length', 'mini_length', 'no_dress']
cat_4 = ['crew_neckline', 'v_neckline', 'square_neckline', 'no_neckline']
cat_5 = ['denim', 'chiffon', 'cotton', 'leather', 'faux', 'knit']
cat_6 = ['tight', 'loose', 'conventional']
img_size = (224, 224)

test_config = {
    'img_path': os.path.join(root_dir),
    'img_file': os.path.join(root_dir, 'split/test.txt'),
    'bbox_file': os.path.join(root_dir, 'split/test_bbox.txt'),
    'attr_cloth_file': os.path.join(root_dir, 'split/list_attr_cloth.txt'),
}

gpu_config = {
    'imgs_per_gpu': 4,
    'workers_per_gpu': 2,
    'train': [0, 1],
    'test': [0, 1]
} 
def load_model(path):
    model = create_model_2()
    model.load_state_dict(torch.load(path, map_location=torch.device(device)))
    model.eval()
    return model

class TestDatasetLoader(data.Dataset):
    def __init__(self, img_path, img_file, bbox_file, img_size):
        self.img_path = img_path

        normalize = transforms.Normalize(
            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        self.transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(img_size[0]),
            transforms.ToTensor(),
            normalize,
        ])
        self.img_list = [img_file]

        # read image path names i.e. img/06000.jpg
      
        self.img_size = img_size

        # load bbox
        if bbox_file:
            self.with_bbox = True
            self.bboxes = np.loadtxt(bbox_file, usecols=(0, 1, 2, 3))
        else:
            self.with_bbox = False
            self.bboxes = None

    def get_basic_item(self, idx):
        img = Image.open(os.path.join(self.img_path,
                                      self.img_list[idx])).convert('RGB')
        if self.with_bbox:
            bbox_cor = self.bboxes[idx]
            x1 = max(0, int(bbox_cor[0]))
            y1 = max(0, int(bbox_cor[1]))
            x2 = int(bbox_cor[2])
            y2 = int(bbox_cor[3])
            bbox_w = x2 - x1
            bbox_h = y2 - y1
            img = img.crop(box=(x1, y1, x2, y2))
        
        img.thumbnail(self.img_size, Image.ANTIALIAS)
        img = self.transform(img)
        img = torch.tensor(img, dtype=torch.float32)
        data = {'image': img}
        return data

    def __getitem__(self, idx):
        return self.get_basic_item(idx)

    def __len__(self):
        return len(self.img_list)

def write_result(result_arrays):
    file = open(os.path.join(root_dir,"prediction.txt"),"w")
    for results in result_arrays:
        file.write(" ".join(str(categories) for categories in results))
        file.write("\n")

def main():
    path = os.path.join(root_dir, "model.pth")
    model = load_model(path)
    prediction_array = []
    fp = open(test_config['img_file'], 'r')
    img_file_list = [x.strip() for x in fp]
    for img_file in img_file_list:
        test_img_set = TestDatasetLoader(test_config["img_path"], img_file, test_config['bbox_file'],img_size)
        test_loader = data.DataLoader(
            test_img_set,
            batch_size=128,
            num_workers=2,
            shuffle=False
        )
        
        for idx, test_data in enumerate(test_loader):   
            imgs = test_data['image']
            output_predictions = model(imgs)
            prediction_list = []
            for cat in categories:
                pred_tesnor = output_predictions[cat]
                predictions = torch.argmax(pred_tesnor)
                prediction_list.append(predictions.item())
            prediction_array.append(prediction_list)
            print(prediction_list)
        
    
    write_result(prediction_array)

if __name__ == '__main__':
    main()
