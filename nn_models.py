#!/usr/bin/env python2
# -*- coding: utf-8 -*-

from keras.layers import Input, Dense, Conv2D, Conv2DTranspose, MaxPooling2D, UpSampling2D, Dropout, concatenate, merge, Convolution2D, BatchNormalization
from keras.models import Model


# %%
def autoencoder(input_shape=(200,300,1)):
    # %%
    
    saving_file = "autoencoder_e{epoch:02d}_*{val_dice_coef:.4f}*_" + str(input_shape) + '.h5'
            
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
   
    return Model(input_img, decoded), saving_file



def autoencoder_v2(input_shape=(200,300,1)):
    # %%
    saving_file = "autoencoder_2_e{epoch:02d}_*{val_dice_coef:.4f}*_" + str(input_shape) + '.h5'

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
   
    return Model(input_img, decoded), saving_file



# https://github.com/jocicmarko/ultrasound-nerve-segmentation/blob/master/train.py#L19
#https://www.kaggle.com/drn01z3/end-to-end-baseline-with-u-net-keras
#    https://github.com/fchollet/keras/issues/2994
def u_net(input_shape=(200,300,1)):
    # %%
    saving_file = "u-net_e{epoch:02d}_*{val_dice_coef:.4f}*_" + str(input_shape) + '.h5'

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
#    print Model(input_img, conv3).summary()
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

    return Model(input_img, conv10), saving_file


def u_net_v2(input_shape=(200,300,1), loss='val_dice_coef', norm=True, simple=False, monitor=['val_dice_coef']):
    # %%
    saving_file = "u-net_v2_"+loss+"_e{epoch:02d}_mm_" + str(input_shape)
    saving_file = saving_file.replace('mm', '_'.join(['*{'+m+':.5f}*' for m in monitor]))
    
    if norm: saving_file += '_norm'
    if simple: saving_file += '_simpl'
    saving_file += '.h5'

    input_img = Input(input_shape)
    
    conv1 = Conv2D(16, (3, 3), activation='relu', padding='same')(input_img)
    if not simple: conv1 = Conv2D(16, (3, 3), activation='relu', padding='same')(conv1)
    if norm: conv1 = BatchNormalization()(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)

    conv2 = Conv2D(32, (3, 3), activation='relu', padding='same')(pool1)
    if not simple: conv2 = Conv2D(32, (3, 3), activation='relu', padding='same')(conv2)
    if norm: conv2 = BatchNormalization()(conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)

    conv3 = Conv2D(64, (3, 3), activation='relu', padding='same')(pool2)
    if not simple: conv3 = Conv2D(64, (3, 3), activation='relu', padding='same')(conv3)
    if norm: conv3 = BatchNormalization()(conv3)
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)

    conv4 = Conv2D(256, (3, 3), activation='relu', padding='same')(pool3)
    if not simple: conv4 = Conv2D(256, (3, 3), activation='relu', padding='same')(conv4)
    if norm: conv4 = BatchNormalization()(conv4)
    pool4 = MaxPooling2D(pool_size=(2, 2))(conv4)

    conv5 = Conv2D(256, (3, 3), activation='relu', padding='same')(pool4)
    if not simple: conv5 = Conv2D(256, (3, 3), activation='relu', padding='same')(conv5)
#    
#    print Model(input_img, conv3).summary()
##    print Conv2DTranspose(256, (2, 2), strides=(2, 2), padding='same')(conv5)
#
    up6 = concatenate([Conv2DTranspose(256, (2, 2), strides=(2, 2), padding='same')(conv5), conv4], axis=3)
    conv6 = Conv2D(256, (3, 3), activation='relu', padding='same')(up6)
    if not simple: conv6 = Conv2D(256, (3, 3), activation='relu', padding='same')(conv6)
    if norm: conv6 = BatchNormalization()(conv6)

    up7 = concatenate([Conv2DTranspose(128, (2, 2), strides=(2, 2), padding='same')(conv6), conv3], axis=3)
    conv7 = Conv2D(128, (3, 3), activation='relu', padding='same')(up7)
    if not simple: conv7 = Conv2D(128, (3, 3), activation='relu', padding='same')(conv7)
    if norm: conv7 = BatchNormalization()(conv7)

    up8 = concatenate([Conv2DTranspose(32, (2, 2), strides=(2, 2), padding='same')(conv7), conv2], axis=3)
    conv8 = Conv2D(32, (3, 3), activation='relu', padding='same')(up8)
    if not simple: conv8 = Conv2D(32, (3, 3), activation='relu', padding='same')(conv8)
    if norm: conv8 = BatchNormalization()(conv8)

    up9 = concatenate([Conv2DTranspose(16, (2, 2), strides=(2, 2), padding='same')(conv8), conv1], axis=3)
    conv9 = Conv2D(16, (3, 3), activation='relu', padding='same')(up9)
    if not simple: conv9 = Conv2D(16, (3, 3), activation='relu', padding='same')(conv9)
    if norm: conv9 = BatchNormalization()(conv9)

    conv10 = Conv2D(1, (1, 1), activation='sigmoid')(conv9)
    
    print Model(input_img, conv10).summary()
     # %%

    return Model(input_img, conv10), saving_file







