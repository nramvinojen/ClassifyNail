# -*- coding: utf-8 -*-
"""
Created on Sat Mar 14 20:04:09 2020

@author: Ramvinojen
"""


import os
PROJECT_ROOT = os.path.abspath(os.path.dirname(__file__))
print(PROJECT_ROOT)

global rootdir
rootdir = PROJECT_ROOT


class dirs:
    base_dir = os.path.join(rootdir, "data")
    original_dataset_dir = os.path.join(base_dir, "whole")
    train_dir = os.path.join(base_dir, "split/train")
    validation_dir = os.path.join(base_dir, "split/validate")
    test_dir = os.path.join(base_dir, "split/test")
    model_dir = os.path.join(rootdir, "model_logs")


class params:
    batch_size = 6
    cnn_epochs = 65
    vgg_epochs = 25
    learning_rate = 0.001
    image_width = 150
    image_heigth = 150
    croped_size = (590, 150, 830, 900)   # (xmin, ymin, dx, dy)
    cropwidth = 350     # lenght of the image for crop 

class dataset: #parameters for datset creation and spliting
    split = 0.12
    seed  = 123
    crop  = 1
    clean = 1

def main():
    pass


if __name__ == '__main__':
    main()
