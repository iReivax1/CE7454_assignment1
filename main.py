from __future__ import division
from __future__ import print_function
import os
import time
import copy

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
import torch.utils.data as data

import matplotlib.pyplot as plt
import numpy as np


from dataloader import AttrDatasetLoader


root_dir = "/Users/xavier/Documents/NTU/CE7454 Deep Learning/Assignment 1/FashionDataset"

use_cuda = torch.cuda.is_available()
device = 'cuda' if torch.cuda.is_available() else 'cpu'

attribute_num = 26
img_size = (224, 224)
CROP_SIZE = 224
num_gpus = 1
epochs = 20

cat_1 = ['floral', 'graphic', 'striped', 'embroidered', 'pleated', 'solid', 'lattice']
cat_2 = ['long_sleeve', 'short sleeve', 'sleeveless']
cat_3 = ['maxi_length' , 'mini_length' , 'no_dress']
cat_4 = ['crew_neckline', 'v_neckline', 'square_neckline', 'no_neckline']
cat_5 = ['denim', 'chiffon', 'cotton', 'leather', 'faux', 'knit']
cat_6 = ['tight', 'loose', 'conventional']


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

test_config = {
    'img_path': os.path.join(root_dir, 'img'),
    'img_file': os.path.join(root_dir, 'split/test.txt'),
    'bbox_file': os.path.join(root_dir, 'split/test_bbox.txt'),
    'attr_cloth_file': os.path.join(root_dir, 'split/list_attr_cloth.txt'),
    'img_size': img_size
}

gpu_config = {
    'imgs_per_gpu': 8,
    'workers_per_gpu': 4,
    'train': [0, 1],
    'test': [0, 1]
}








def train_model(model, dataloaders, criterion, optimizer, num_epochs, is_inception=False):
    since = time.time()

    val_acc_history = []

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0
    lowest_loss = 99999

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0

            # Iterate over data.
            for data in dataloaders[phase]:
                images, labels = data['image'].to(device), data['label'].to(device)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    # Get model outputs and calculate loss
                    # Special case for inception because in training it has an auxiliary output. In train
                    #   mode we calculate the loss by summing the final output and the auxiliary output
                    #   but in testing we only consider the final output.
                    outputs = model(images)
                    outputs = torch.sigmoid(outputs)
                    loss = criterion(outputs, labels)

                    # _, preds = torch.max(outputs, 1)
                    
                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # statistics
                running_loss += loss.item()
                # running_corrects += torch.sum(preds == labels.data)

            epoch_loss = running_loss / len(dataloaders[phase].dataset)
            # epoch_acc = running_corrects.double() / len(dataloaders[phase].dataset)

            # print('{} Loss: {:.4f} Acc: {:.4f}'.format(
            #     phase, epoch_loss, epoch_acc))
          
            print('{} Loss: {:.4f}'.format(
                phase, epoch_loss))
     
            # deep copy the model
            # if phase == 'val' and epoch_acc > best_acc:
            #     best_acc = epoch_acc
            #     best_model_wts = copy.deepcopy(model.state_dict())
            # if phase == 'val':
            #     val_acc_history.append(epoch_acc)
            if phase == 'val' and epoch_loss < lowest_loss:
                lowest_loss = epoch_loss
                best_model_wts = copy.deepcopy(model.state_dict())
            if phase == 'val':
                val_acc_history.append(epoch_loss)


        print()

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model, val_acc_history


def create_model(to_feature_extract, out_features):
    model = models.resnet50(pretrained=True)
    # to freeze the hidden layers
    if to_feature_extract == False:
        for param in model.parameters():
            param.requires_grad = False
    # to train the hidden layers
    elif to_feature_extract == True:
        for param in model.parameters():
            param.requires_grad = True
    # make the classification layer learnable
    # we have 25 classes in total
    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, out_features)
    return model

# class Resnext50(nn.Module):
#     def __init__(self, n_classes):
#         super().__init__()
#         resnet = models.resnext50_32x4d(pretrained=True)
#         resnet.fc = nn.Sequential(
#             nn.Dropout(p=0.2),
#             nn.Linear(in_features=resnet.fc.in_features, out_features=n_classes)
#         )
#         self.base_model = resnet
#         self.sigm = nn.ReLU()

#     def forward(self, x):
#         return self.sigm(self.base_model(x))


def main():

    model = create_model(False, attribute_num)
    print(model)
    model = model.to(device)
    params_to_update = model.parameters()

    batch_size = num_gpus * gpu_config['imgs_per_gpu']
    num_workers = num_gpus * gpu_config['workers_per_gpu']

    # Loss and optimizer functions
    optimizer = optim.SGD(params_to_update, lr=0.001, momentum=0.9)
    criterion = nn.BCELoss()

    # data loader
    train_dataset = AttrDatasetLoader(train_config['img_path'], train_config['img_file'],
                                      train_config['label_file'],
                                      train_config['bbox_file'],
                                      train_config['img_size'])
    val_dataset = AttrDatasetLoader(val_config['img_path'], val_config['img_file'],
                                    val_config['label_file'],
                                    val_config['bbox_file'],
                                    val_config['img_size'])
    # test_dataset = AttrDatasetLoader(test_config['img_path'], test_config['img_file'],
    #                                  test_config['label_file'],
    #                                  test_config['bbox_file'],
    #                                  test_config['img_size'])

    #pin memory = true, if using nvidia gpu
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

    # test_loader = data.DataLoader(
    #     test_dataset,
    #     batch_size=batch_size,
    #     num_workers=num_workers,
    #     shuffle=False
    # )
    
    # use test set on prediction phase only

    dataloaders_dict = {'train': train_loader,
                        'val':val_loader}

    

    model, hist = train_model(
        model, dataloaders_dict, criterion, optimizer, num_epochs=epochs, is_inception=False)


if __name__ == '__main__':
    main()
