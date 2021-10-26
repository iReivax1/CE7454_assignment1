

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
import numpy as np
from torchvision.transforms.transforms import Resize


use_cuda = torch.cuda.is_available()
device = 'cuda' if torch.cuda.is_available() else 'cpu'

img_size = (224, 224)
root_dir = "/Users/xavier/Documents/NTU/CE7454 Deep Learning/Assignment 1/FashionDataset"

hyperparam_config = {
    'lr': 1e-3,
    'mom': 0.9,
    'epochs' : 16,
    'step_size' : 10,
    'gamma': 0.5,
    'weight_decay' : 1e-5,
    'dropout': 0.7,
    'loss_function': 'focal',
    'optimizer':'SGD',
    'model': 'resnet50',
    'feature_extract': True
}

train_config = {
    "img_path": root_dir,
    "img_file": os.path.join(root_dir, 'split/train.txt'),
    "label_file": os.path.join(root_dir, 'split/train_attr.txt'),
    "bbox_file": os.path.join(root_dir, 'split/train_bbox.txt'),
    "img_size": img_size
}

val_config = {
    'img_path': root_dir,
    'img_file': os.path.join(root_dir, 'split/val.txt'),
    'label_file': os.path.join(root_dir, 'split/val_attr.txt'),
    'bbox_file': os.path.join(root_dir, 'split/val_bbox.txt'),
    'img_size': img_size
}

gpu_config = {
    'num_gpus' : 8,
    'imgs_per_gpu': 8,
    'workers_per_gpu': 1,
    'train': [0, 1],
    'test': [0, 1]
}

cat_1 = ['floral', 'graphic', 'striped',
         'embroidered', 'pleated', 'solid', 'lattice']
cat_2 = ['long_sleeve', 'short sleeve', 'sleeveless']
cat_3 = ['maxi_length', 'mini_length', 'no_dress']
cat_4 = ['crew_neckline', 'v_neckline', 'square_neckline', 'no_neckline']
cat_5 = ['denim', 'chiffon', 'cotton', 'leather', 'faux', 'knit']
cat_6 = ['tight', 'loose', 'conventional']

labels_list = ['cat1_label', 'cat2_label', 'cat3_label', 'cat4_label', 'cat5_label', 'cat6_label']
labels_dict = { 'cat1': 'cat1_label', 'cat2':'cat2_label', 'cat3': 'cat3_label', 'cat4':'cat4_label', 'cat5': 'cat5_label', 'cat6':'cat6_label'}

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


def train_model_2(model, dataloaders, criterion, optimizer):
     # Set model to training mode
    model.train() 

    running_loss = 0.0

   
    if hyperparam_config['loss_function'] == 'NLLLoss':
        softmax_function = nn.LogSoftmax(dim=0)

    # Iterate over data.
    for data in dataloaders['train']:
        images  = data['image'].to(device)
        labels = data['label']
        #iterate through cat 1 to 6
        labels = {x : labels[x] for x in labels}
        for cat_label in labels_list:
                labels[cat_label] = torch.reshape(labels[cat_label] , (-1,)).type(torch.LongTensor)
        optimizer.zero_grad()

        with torch.set_grad_enabled(True):
            outputs = model(images)  
            total_loss = 0          
            for cat_keys in labels_dict:
                total_loss += criterion((outputs[cat_keys]), labels[labels_dict[cat_keys]].to(device))
            total_loss.backward()
            optimizer.step()

        # statistics
        running_loss += total_loss.item()

    epoch_loss = running_loss / len(dataloaders['train'].dataset)

    print('{} Loss: {:.4f}'.format('train', epoch_loss))

    return epoch_loss


def val_model_2(model, dataloaders, criterion, optimizer):
    # Set model to eval mode
    model.eval()  

    running_loss = 0.0
  
       
    if hyperparam_config['loss_function'] == 'NLLLoss':
        softmax_function = nn.LogSoftmax(dim=0)
    
    # Iterate over data.
    for data in dataloaders['val']:
        images  = data['image'].to(device)
        labels = data['label']
        labels = {x : labels[x] for x in labels}
        for cat_label in labels_list:
                labels[cat_label] = torch.reshape(labels[cat_label] , (-1,)).type(torch.LongTensor)
        optimizer.zero_grad()
        with torch.set_grad_enabled(False):
            
            outputs = model(images)
            total_loss = 0
            for cat_keys in labels_dict:
                total_loss += criterion((outputs[cat_keys]), labels[labels_dict[cat_keys]].to(device))
          

            
            
        # statistics
        running_loss += total_loss.item()

    epoch_loss = running_loss / len(dataloaders['val'].dataset)
    print('{} Loss: {:.4f}'.format('val', epoch_loss))
    
    return epoch_loss


class create_model_2(nn.Module):
    def __init__(self):
        super().__init__()

        if hyperparam_config['model'] == 'resnet18':
            self.model = models.resnet18(pretrained=True)
            in_features = self.model.fc.out_features
        elif hyperparam_config['model'] == "densenet":
            self.model = models.densenet121(pretrained=True)
            in_features = self.model.classifier.out_features
        elif hyperparam_config['model'] == 'resnet50':
            self.model = models.resnet50(pretrained=True)
            in_features = self.model.fc.out_features
        40
        # seperate classfiers for each categories
        self.fc1 = nn.Sequential(nn.Dropout(p=hyperparam_config['dropout']),nn.Linear(in_features, len(cat_1)))
        self.fc2 = nn.Sequential(nn.Dropout(p=hyperparam_config['dropout']),nn.Linear(in_features, len(cat_2)))
        self.fc3 = nn.Sequential(nn.Dropout(p=hyperparam_config['dropout']),nn.Linear(in_features, len(cat_3)))
        self.fc4 = nn.Sequential(nn.Dropout(p=hyperparam_config['dropout']),nn.Linear(in_features, len(cat_4)))
        self.fc5 = nn.Sequential(nn.Dropout(p=hyperparam_config['dropout']),nn.Linear(in_features, len(cat_5)))
        self.fc6 = nn.Sequential(nn.Dropout(p=hyperparam_config['dropout']),nn.Linear(in_features, len(cat_6)))

    def forward(self, x):
        x = self.model(x)
        # reshape from [batch, channels, 1, 1] to [batch, channels] to put it into classifier
        x = torch.flatten(x, start_dim=1)
        
        return {
            'cat1': self.fc1(x),
            'cat2': self.fc2(x),
            'cat3': self.fc3(x),
            'cat4': self.fc4(x),
            'cat5': self.fc5(x),
            'cat6': self.fc6(x)
        }


#class imbalance too many 5 for cat 1 for example, hence use focal loss (multiclass)
class MultiClass_FocalLoss(nn.modules.loss._WeightedLoss):
    def __init__(self, weight=None, gamma=2,reduction='mean'):
        super(MultiClass_FocalLoss, self).__init__(weight,reduction=reduction)
        self.gamma = gamma
        #weight is alpha param to counter balance class weights
        self.weight = weight

    def forward(self, input, target):
        #CE instead of BCE because multi class
        cross_entropy_loss = F.cross_entropy(input, target,reduction=self.reduction,weight=self.weight)
        loss = torch.exp(-cross_entropy_loss)
        focal_loss = ((1 - loss) ** self.gamma * cross_entropy_loss).mean()
        return focal_loss

    

def plot_graph(train_loss,val_loss):
    
    plt.plot(range(1, len(train_loss) + 1), train_loss, label='Train')
    plt.plot(range(1, len(val_loss) + 1), val_loss, label='Val')
    plt.title('Model Loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend()
    graph_name = hyperparam_config['model'] + '_' + str(hyperparam_config['optimizer'])+ str(hyperparam_config['lr']) +'_WD:'+ str(hyperparam_config['weight_decay'])+ '_' + hyperparam_config['loss_function']  +'_' + 'loss.png'
    plt.savefig(os.path.join(root_dir,graph_name))
    plt.close()



def count_data(label_file):
  
  labels = np.loadtxt(label_file, dtype=np.float32)
  cat1_labels = torch.from_numpy(labels[:, [0]])
  cat1_labels = torch.reshape(cat1_labels , (-1,)).type(torch.LongTensor)
  print(cat1_labels)
  cat2_labels = torch.from_numpy(labels[:, [1]])
  cat2_labels = torch.reshape(cat2_labels, (-1,)).type(torch.LongTensor)
  
  cat3_labels = torch.from_numpy(labels[:, [2]])
  cat3_labels = torch.reshape(cat3_labels, (-1,)).type(torch.LongTensor)
  
  cat4_labels = torch.from_numpy(labels[:, [3]])
  cat4_labels = torch.reshape(cat4_labels, (-1,)).type(torch.LongTensor)

  cat5_labels = torch.from_numpy(labels[:, [4]])
  cat5_labels = torch.reshape(cat5_labels, (-1,)).type(torch.LongTensor)
  
  cat6_labels = torch.from_numpy(labels[:, [5]])
  cat6_labels = torch.reshape(cat6_labels, (-1,)).type(torch.LongTensor)
  
  cat_1_numlabel = {'0': 0, '1': 0, '2': 0, '3': 0, '4': 0 , '5':0, '6':0}
  cat_2_numlabel = {'0': 0, '1': 0, '2': 0}
  cat_3_numlabel = {'0': 0, '1': 0, '2': 0}
  cat_4_numlabel = {'0': 0, '1': 0, '2': 0, '3': 0}
  cat_5_numlabel = {'0': 0, '1': 0, '2': 0, '3': 0, '4': 0 , '5':0}
  cat_6_numlabel = {'0': 0, '1': 0, '2': 0} 
  
  for labelY in cat1_labels:
    if labelY.item() == 0:
      cat_1_numlabel['0'] += 1
    elif labelY.item() == 1:
      cat_1_numlabel['1'] += 1
    elif labelY.item() == 2:
      cat_1_numlabel['2'] += 1
    elif labelY.item() == 3:
      cat_1_numlabel['3'] += 1
    elif labelY.item() == 4:
      cat_1_numlabel['4'] += 1
    elif labelY.item() == 5:
      cat_1_numlabel['5'] += 1
    elif labelY.item() == 6:
      cat_1_numlabel['6'] += 1
  
  for labelY in cat2_labels:
    if labelY.item() == 0:
      cat_2_numlabel['0'] += 1
    elif labelY.item() == 1:
      cat_2_numlabel['1'] += 1
    elif labelY.item() == 2:
      cat_2_numlabel['2'] += 1

  for labelY in cat3_labels:
    if labelY.item() == 0:
      cat_3_numlabel['0'] += 1
    elif labelY.item() == 1:
      cat_3_numlabel['1'] += 1
    elif labelY.item() == 2:
      cat_3_numlabel['2'] += 1

  for labelY in cat4_labels:
    if labelY.item() == 0:
      cat_4_numlabel['0'] += 1
    elif labelY.item() == 1:
      cat_4_numlabel['1'] += 1
    elif labelY.item() == 2:
      cat_4_numlabel['2'] += 1
    elif labelY.item() == 3:
      cat_4_numlabel['3'] += 1

  for labelY in cat5_labels:
    if labelY.item() == 0:
      cat_5_numlabel['0'] += 1
    elif labelY.item() == 1:
      cat_5_numlabel['1'] += 1
    elif labelY.item() == 2:
      cat_5_numlabel['2'] += 1
    elif labelY.item() == 3:
      cat_5_numlabel['3'] += 1
    elif labelY.item() == 4:
      cat_5_numlabel['4'] += 1
    elif labelY.item() == 5:
      cat_5_numlabel['5'] += 1

  for labelY in cat6_labels:
    if labelY.item() == 0:
      cat_6_numlabel['0'] += 1
    elif labelY.item() == 1:
      cat_6_numlabel['1'] += 1
    elif labelY.item() == 2:
      cat_6_numlabel['2'] += 1

  print(cat_1_numlabel)
  print(cat_2_numlabel)
  print(cat_3_numlabel)
  print(cat_4_numlabel)
  print(cat_5_numlabel)
  print(cat_6_numlabel)

def main():
    
    model = create_model_2()
    model = model.to(device)
    if hyperparam_config['feature_extract'] == True:
      parameters_to_be_updated = []
      for name, parameters in model.named_parameters():
          if parameters.requires_grad == True:
              parameters_to_be_updated.append(parameters)
    else:
        parameters_to_be_updated = model.parameters()
    batch_size = gpu_config['num_gpus'] * gpu_config['imgs_per_gpu']
    num_workers = gpu_config['num_gpus'] * gpu_config['workers_per_gpu']

    # data loader
    train_dataset = AttrDatasetLoader(train_config['img_path'], train_config['img_file'],
                                      train_config['label_file'],
                                      train_config['bbox_file'],
                                      train_config['img_size'])
    val_dataset = AttrDatasetLoader(val_config['img_path'], val_config['img_file'],
                                    val_config['label_file'],
                                    val_config['bbox_file'],
                                    val_config['img_size'])

    train_loader = data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=False,
        shuffle=True
    )
    val_loader = data.DataLoader(
        val_dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=False,
        shuffle=False
    )

    # use test set on prediction phase only

    dataloaders_dict = {'train': train_loader,
                        'val': val_loader}
    
    train_loss = []
    valid_loss = []

    epochs = hyperparam_config['epochs']
    # Loss and optimizer functions
    if hyperparam_config['optimizer'] == 'SGD':
        optimizer = optim.SGD(parameters_to_be_updated, lr=hyperparam_config['lr'], momentum=hyperparam_config['mom'], weight_decay=hyperparam_config['weight_decay'])
    elif hyperparam_config['optimizer'] == 'Adam':
        optimizer = optim.Adam(parameters_to_be_updated, lr=hyperparam_config['lr'], weight_decay=hyperparam_config['weight_decay'])
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=hyperparam_config['step_size'], gamma=hyperparam_config['gamma'])

    if hyperparam_config['loss_function'] == 'crossentropy':
        #combines Log Softmax and NLL so no need to softmax
        criterion = nn.CrossEntropyLoss()
    elif hyperparam_config['loss_function'] == 'NLLLoss':
        #should result the same as CE loss
        criterion = nn.NLLLoss()
    elif hyperparam_config['loss_function'] == 'focal':
        criterion = MultiClass_FocalLoss()
 
    for epoch in range(epochs):
        print(f"Epoch {epoch+1} of {epochs}")
        
        train_epoch_loss = train_model_2(
            model, dataloaders_dict, criterion, optimizer
        )
        if epoch % 5 == 0:
            valid_epoch_loss = val_model_2(
                  model, dataloaders_dict, criterion, optimizer
              )
        scheduler.step()
        train_loss.append(train_epoch_loss)
        valid_loss.append(valid_epoch_loss)
    
    plot_graph(train_loss, valid_loss)
    model_name = hyperparam_config['model'] + '_' + str(hyperparam_config['optimizer'])+ str(hyperparam_config['lr']) +'_WD:'+ str(hyperparam_config['weight_decay'])+ '_' + hyperparam_config['loss_function'] +'.pth'
    torch.save( model.state_dict(), os.path.join(root_dir, model_name))
    

if __name__ == '__main__':
    main()
    
