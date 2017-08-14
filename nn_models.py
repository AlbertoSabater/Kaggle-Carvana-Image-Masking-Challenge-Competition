#!/usr/bin/env python2
# -*- coding: utf-8 -*-

from keras.layers import Input, Dense, Conv2D, Conv2DTranspose, MaxPooling2D, UpSampling2D, Dropout, concatenate, merge, Convolution2D
from keras.models import Model


# %%
def autoencoder(input_shape=(200,300,1)):
    # %%
    input_img = Input(shape=input_shape)  

    encoded = Conv2D(16, (3, 3), activation='relu', padding='same')(input_img)
    encoded = Conv2D(16, (3, 3), activation='relu', padding='same')(encoded)
    encoded = Dropout(0.2)(encoded)
    encoded = MaxPooling2D((2, 2), padding='same')(encoded)
    encoded = Conv2D(8, (3, 3), activation='relu', padding='same')(encoded)
    encoded = Conv2D(8, (3, 3), activation='relu', padding='same')(encoded)
    encoded = MaxPooling2D((2, 2), padding='same')(encoded)
    encoded = Conv2D(4, (3, 3), activation='relu', padding='same')(encoded)
    encoded = Dropout(0.2)(encoded)
    encoded = Conv2D(4, (3, 3), activation='relu', padding='same')(encoded)
    
    # at this point the representation is (4, 4, 8) i.e. 128-dimensional
    
    decoded = Conv2DTranspose(4, (3, 3), strides=(1,1), activation='relu', padding='same')(encoded)
    encoded = Dropout(0.2)(encoded)
    decoded = Conv2DTranspose(4, (3, 3), strides=(1,1), activation='relu', padding='same')(encoded)
    decoded = Conv2DTranspose(8, (3, 3), strides=(2,2), activation='relu', padding='same')(decoded)
    decoded = Conv2DTranspose(8, (3, 3), strides=(1,1), activation='relu', padding='same')(decoded)
    encoded = Dropout(0.2)(encoded)
    decoded = Conv2DTranspose(16, (3, 3), strides=(2,2), activation='relu', padding='same')(decoded)
    decoded = Conv2DTranspose(16, (3, 3), strides=(1,1), activation='relu', padding='same')(decoded)
    #x = UpSampling2D((2, 2))(decoded)
    decoded = Conv2D(1, (3, 3), activation='sigmoid', padding='same')(decoded)
    
#    print Model(input_img, decoded).summary()
     # %%
   
    return Model(input_img, decoded)



def autoencoder_v2(input_shape=(200,300,1)):
    # %%
    input_img = Input(shape=input_shape)  

    encoded = Conv2D(16, (3, 3), activation='relu', padding='same')(input_img)
    encoded = Conv2D(16, (3, 3), activation='relu', padding='same')(encoded)
    encoded = MaxPooling2D((2, 2), padding='same')(encoded)
    
    encoded = Conv2D(32, (3, 3), activation='relu', padding='same')(encoded)
    encoded = Conv2D(32, (3, 3), activation='relu', padding='same')(encoded)
    encoded = MaxPooling2D((2, 2), padding='same')(encoded)
    
    
    encoded = Conv2D(64, (3, 3), activation='relu', padding='same')(encoded)
    decoded = Conv2D(64, (3, 3), activation='relu', padding='same')(encoded)
    
    
    decoded = Conv2DTranspose(32, (3, 3), strides=(2,2), activation='relu', padding='same')(decoded)
    decoded = Conv2D(32, (3, 3), activation='relu', padding='same')(decoded)
    decoded = Conv2D(32, (3, 3), activation='relu', padding='same')(decoded)
    
    decoded = Conv2DTranspose(16, (3, 3), strides=(2,2), activation='relu', padding='same')(decoded)
    decoded = Conv2D(16, (3, 3), activation='relu', padding='same')(decoded)
    decoded = Conv2D(16, (3, 3), activation='relu', padding='same')(decoded)
    
    decoded = Conv2D(1, (3, 3), activation='sigmoid', padding='same')(decoded)
    
    
    print Model(input_img, decoded).summary()
     # %%
   
    return Model(input_img, decoded)



# https://github.com/jocicmarko/ultrasound-nerve-segmentation/blob/master/train.py#L19
#https://www.kaggle.com/drn01z3/end-to-end-baseline-with-u-net-keras
#    https://github.com/fchollet/keras/issues/2994
def u_net(input_shape=(200,300,1)):
    # %%
    input_img = Input(input_shape)
    conv1 = Conv2D(16, (3, 3), activation='relu', padding='same')(input_img)
    conv1 = Conv2D(16, (3, 3), activation='relu', padding='same')(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)

    conv2 = Conv2D(32, (3, 3), activation='relu', padding='same')(pool1)
    conv2 = Conv2D(32, (3, 3), activation='relu', padding='same')(conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)

    conv3 = Conv2D(64, (3, 3), activation='relu', padding='same')(pool2)
    conv3 = Conv2D(64, (3, 3), activation='relu', padding='same')(conv3)
#    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)
#
#    conv4 = Conv2D(256, (3, 3), activation='relu', padding='same')(pool3)
#    conv4 = Conv2D(256, (3, 3), activation='relu', padding='same')(conv4)
#    pool4 = MaxPooling2D(pool_size=(2, 2))(conv4)
#
#    conv5 = Conv2D(512, (3, 3), activation='relu', padding='same')(pool4)
#    conv5 = Conv2D(512, (3, 3), activation='relu', padding='same')(conv5)
#    
    print Model(input_img, conv3).summary()
##    print Conv2DTranspose(256, (2, 2), strides=(2, 2), padding='same')(conv5)
#
#    up6 = concatenate([Conv2DTranspose(256, (2, 2), strides=(2, 2), padding='same')(conv5), conv4], axis=3)
#    conv6 = Conv2D(256, (3, 3), activation='relu', padding='same')(up6)
#    conv6 = Conv2D(256, (3, 3), activation='relu', padding='same')(conv6)

#    up7 = concatenate([Conv2DTranspose(128, (2, 2), strides=(2, 2), padding='same')(conv4), conv3], axis=3)
#    conv7 = Conv2D(128, (3, 3), activation='relu', padding='same')(up7)
#    conv7 = Conv2D(128, (3, 3), activation='relu', padding='same')(conv7)

    up8 = concatenate([Conv2DTranspose(32, (2, 2), strides=(2, 2), padding='same')(conv3), conv2], axis=3)
    conv8 = Conv2D(32, (3, 3), activation='relu', padding='same')(up8)
    conv8 = Conv2D(32, (3, 3), activation='relu', padding='same')(conv8)

    up9 = concatenate([Conv2DTranspose(16, (2, 2), strides=(2, 2), padding='same')(conv8), conv1], axis=3)
    conv9 = Conv2D(16, (3, 3), activation='relu', padding='same')(up9)
    conv9 = Conv2D(16, (3, 3), activation='relu', padding='same')(conv9)

    conv10 = Conv2D(1, (1, 1), activation='sigmoid')(conv9)
    
#    print Model(input_img, conv10).summary()
     # %%

    return Model(input_img, conv10)







