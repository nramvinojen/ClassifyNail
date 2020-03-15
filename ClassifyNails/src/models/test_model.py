# -*- coding: utf-8 -*-
"""
Created on Sat Mar 14 20:21:01 2020

@author: Ramvinojen
"""

import os
import sys
from pathlib import Path
from keras.models import load_model


from keras.preprocessing.image import ImageDataGenerator
from keras import optimizers
from keras.callbacks import TensorBoard, ModelCheckpoint, EarlyStopping, CSVLogger
import tensorflow as tf
from sklearn.metrics import classification_report, confusion_matrix


import utils as ut
import src.models.model as md



def test( modelname,
          image_width,
          image_heigth):
    input_shape = (image_width, image_heigth, 3)
    
    # testing data augmentation (only rescaling)
    test_datagen = ImageDataGenerator(rescale=1./255)
    
    test_generator = test_datagen.flow_from_directory(
        ut.dirs.test_dir,
        target_size=(150, 150),
        batch_size=ut.params.batch_size,
        class_mode='binary',  
        shuffle=False)  # keep data in same order as labels

    load_model_from = os.path.join(ut.dirs.model_dir, modelname + '.h5')
    

    num_test_img = sum([len(files) for r, d, files in os.walk(ut.dirs.test_dir)])       
    model = load_model(load_model_from)
    model._make_predict_function()  
    probabilities = model.predict_generator(test_generator, num_test_img //ut.params.batch_size )
    
    probabilities[probabilities >= 0.5] = 1
    probabilities[probabilities < 0.5] = 0
  
    print("\n\n---------------Results of the ",  modelname, "model ---------------\n")
    print('-------------Confusion Matrix')
    print(confusion_matrix(test_generator.classes, probabilities))
    print('\n-------------Classification Report')
    target_names = ['good', 'bad']
    print(classification_report(test_generator.classes, probabilities, target_names=target_names))
  


def main():
    width = ut.params.image_width
    heigth = ut.params.image_heigth
    test("cnn", width, heigth)
    test("vgg16", width, heigth)


if __name__ == '__main__':
    main()