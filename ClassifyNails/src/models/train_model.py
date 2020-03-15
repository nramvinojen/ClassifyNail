# -*- coding: utf-8 -*-
"""
Created on Sat Mar 14 08:52:53 2020

@author: Ramvinojen
"""

import os
import sys
from pathlib import Path
#import settings


from keras.preprocessing.image import ImageDataGenerator
from keras import optimizers
from keras.callbacks import TensorBoard, ModelCheckpoint, EarlyStopping, CSVLogger
import tensorflow as tf


import utils as ut
import src.models.model as md


def train(modelname,
          batch_size,
          epochs,
          learning_rate,
          augment,
          image_width,
          image_heigth):
    input_shape = (image_width, image_heigth, 3)

    # training data augmentation
    train_datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=90,
        horizontal_flip=True,
        vertical_flip=True)
    # testing data augmentation (only rescaling)
    test_datagen = ImageDataGenerator(rescale=1./255)

    if(augment is False):
        train_datagen = test_datagen

    # training data generator
    train_generator = train_datagen.flow_from_directory(
        ut.dirs.train_dir,
        target_size=(image_width, image_heigth),
        color_mode='rgb',
        batch_size=batch_size,
        class_mode='binary')
    # validation data generator
    validation_generator = test_datagen.flow_from_directory(
        ut.dirs.validation_dir,
        target_size=(image_width, image_heigth),
        color_mode='rgb',
        batch_size=batch_size,
        class_mode='binary')

    mod = md.custom_models(input_shape, 1)

    if(modelname == 'vgg16'):
        model = mod.vgg16()
    elif(modelname == "cnn"):
        model = mod.CNN()
    else:
        print('invalid model selection.\n\
               please choose from one of the available models:\n\
                 vgg16, cnn')
        sys.exit()

    # Do not forget to compile it
    model.compile(
                    loss='binary_crossentropy',
                    optimizer=optimizers.RMSprop(lr=learning_rate),
                    metrics=['accuracy']
                    )

    model.summary()

    save_model_to = os.path.join(ut.dirs.model_dir, modelname + '.h5')

    Checkpoint = ModelCheckpoint(save_model_to,
                                 monitor='val_loss',
                                 verbose=0,
                                 save_best_only=False,
                                 save_weights_only=False,
                                 mode='auto',
                                 period=1)
    Earlystop = EarlyStopping(monitor='val_loss',
                              min_delta=0,
                              patience=5,
                              verbose=0,
                              mode='auto',
                              baseline=None)

    model.fit_generator(
        train_generator,
        callbacks=[
                    Checkpoint #, Earlystop
                    ],
        steps_per_epoch=150//batch_size,
        epochs=epochs,
        validation_data=validation_generator,
        validation_steps=12//batch_size
    )



def main():
    
    
    bs = ut.params.batch_size
    ep_vgg = ut.params.vgg_epochs
    ep_cnn = ut.params.cnn_epochs
    lr = ut.params.learning_rate
    augment = 1
    width = ut.params.image_width
    heigth = ut.params.image_heigth

    augmentation = True
    if(augment == 0):
        augmentation = False

    train("cnn", bs, ep_cnn, lr, augmentation, width, heigth)
    train("vgg16", bs, ep_vgg, lr, augmentation, width, heigth)


if __name__ == '__main__':
    main()
