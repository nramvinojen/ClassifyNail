# -*- coding: utf-8 -*-
"""
Created on Sat Mar 14 07:59:45 2020

@author: Ramvinojen

this file serves as a driver to call other .py files
this fill takes in arguments to run between "make the dataset, train, test"
"""


import sys
from sys import exit
from src.data import make_dataset as mkds
from src.models import train_model as trn
from src.models import test_model as tst


def main():
    if len(sys.argv) < 2:
        print('make data set -mkds, train cnn model -train,  test model -test')
        exit(1)
    
    opts = sys.argv[1]
    if  opts == 'mkds' :
        print("starting to create dataset")   
        mkds.main()
    elif opts == 'train' :
        print("starting to train models")
        trn.main()
    elif opts == 'test':
        print("starting to test models")
        tst.main()
    else:
        print('make data set -mkds, train model -train,  test model -test')
        exit(1)
        

if __name__ == '__main__':
    main()
    
    




