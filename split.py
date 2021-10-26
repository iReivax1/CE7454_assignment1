import os
import numpy as np
from shutil import copy

root_dir = "/Users/xavier/Documents/NTU/CE7454 Deep Learning/Assignment 1/FashionDataset"
split_dir = root_dir+"/split"

train_txt_dir = split_dir+'/train.txt'
test_txt_dir = split_dir+'/test.txt'
val_txt_dir = split_dir+'/val.txt'


 
def open_txt(text_file_path, file_type):

    with open(text_file_path) as fp:
        Lines = fp.readlines()

        for line in Lines:
            line = line.rstrip('\n')
            copy(os.path.join(root_dir, line), os.path.join(root_dir, file_type))
            print(line)
    

if __name__ == "__main__":
    # open_txt(train_txt_dir, "train")
    open_txt(test_txt_dir, "test")
    open_txt(val_txt_dir, "val")