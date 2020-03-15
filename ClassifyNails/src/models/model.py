# -*- coding: utf-8 -*-
"""
Created on Sat Mar 14 08:49:26 2020

@author: Ramvinojen
"""

from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Input, Dropout, Flatten, Dense
from keras import applications
from keras import Model


class custom_models:
    def __init__(self, input_shape, n_classes):
        self.input_shape = input_shape
        self.n_classes = n_classes

    def vgg16(self):
        input_tensor = Input(shape=self.input_shape)
        vgg_model = applications.VGG16(weights='imagenet',
                                       include_top=False,
                                       input_tensor=input_tensor)

        # map layer names to the layers
        layer_dict = dict([(layer.name, layer) for layer in vgg_model.layers])

        # output of the last VGG layer
        x = layer_dict['block5_conv3'].output
        print(x.shape)

        # appending a new simple  network on top of it
        custom_flat = Flatten()(x)
        custom_dense1 = Dense(256, activation='relu')(custom_flat)
        custom_dense = Dense(128, activation='relu')(custom_dense1)
        custom_drop = Dropout(0.5)(custom_dense)
        custom_out = Dense(self.n_classes, activation='sigmoid')(custom_drop)
        if(self.n_classes > 1):
            custom_out = Dense(self.n_classes,
                               activation='softmax')(custom_drop)

        # Creating the new model
        custom_model = Model(input=vgg_model.input, output=custom_out)

        # pre-trained bottom layers are not trainable
        for layer in vgg_model.layers:
            layer.trainable = False

        return custom_model
    

    def CNN(self):
        input = Input(shape=self.input_shape)

        conv1 = Conv2D(32, (2, 2), activation='relu')(input)
        pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)

        conv2 = Conv2D(32, (2, 2), activation='relu')(pool1)
        pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)

        conv3 = Conv2D(64, (2, 2), activation='relu')(pool2)
        pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)

        flat = Flatten()(pool3)
        dense = Dense(64, activation='relu')(flat)
        drop = Dropout(0.5)(dense)
        custom_out = Dense(self.n_classes, activation='sigmoid')(drop)
        if self.n_classes > 1:
            custom_out = Dense(self.n_classes, activation='softmax')(drop)

        custom_model = Model(input=input, output=custom_out)

        return custom_model
