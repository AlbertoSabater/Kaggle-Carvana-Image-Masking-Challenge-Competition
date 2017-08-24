#!/usr/bin/env python2
# -*- coding: utf-8 -*-

import pickle
import numpy as np
import matplotlib.pyplot as plt
import time
import nn_models
import nn_utils
import os
import datetime
import glob
from PIL import Image
from joblib import Parallel, delayed
import multiprocessing

from keras import backend as K
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.preprocessing.image import ImageDataGenerator

np.random.seed(123)  # for reproducibility

BASE_DIR = 'models/'
earlyStopping = EarlyStopping(monitor='val_dice_coef', min_delta=0.00001, verbose=1, mode='max', patience=7)

shape = (256,256)


# %%
# =============================================================================
# Load training data
# =============================================================================

t = time.time()
print "Loading data"


with open('data/train_car_BN'+str(shape)+'.pickle', 'r') as f:
    train_car = pickle.load(f)
    
rand = np.random.permutation(np.arange(len(train_car)))                                                                                                                                                                                                                                                                                                                                     
train_car = train_car.reshape(train_car.shape[0], train_car.shape[1], train_car.shape[2], 1)
val_car = train_car[rand[int(len(rand)*0.7):]]
train_car = train_car[rand[:int(len(rand)*0.7)]]
   
    
with open('data/train_car_mask'+str(shape)+'.pickle', 'r') as f:
    train_car_mask = pickle.load(f)

train_car_mask = train_car_mask.reshape(train_car_mask.shape[0], train_car_mask.shape[1], train_car_mask.shape[2], 1)
val_car_mask = train_car_mask[rand[int(len(rand)*0.7):]]
train_car_mask = train_car_mask[rand[:int(len(rand)*0.7)]]


#train_car = train_car[rand]
#train_car_mask = train_car_mask[rand]

print train_car.shape, train_car_mask.shape, val_car.shape, val_car_mask.shape
print "Data loaded:", (time.time()-t)/60



# %%
# =============================================================================
# Load validation data
# =============================================================================

#t = time.time()
#filelist = glob.glob('data/val_'+shape+'/*')
##filelist = filelist[:20]
#def loadValidationData(fname, img_size):
#    fname = fname.split('/')[2].split('.')[0]
#    return [np.array(Image.open('data/val_'+img_size+'/'+fname+'.jpg')).astype(float)/255,
#     np.array(Image.open('data/val_masks_'+img_size+'/'+fname+'_mask.gif')).astype(float)]
#
#pairs = np.array(Parallel(n_jobs=8)(delayed(loadValidationData)(fname, shape) for fname in filelist))
#val_car = pairs[:,0,:,:]
#val_car_mask = pairs[:,1,:,:]
#print "Time elapsed:",  (time.time()-t)/60


# %%
# =============================================================================
# Create the model
# =============================================================================

input_shape = (train_car.shape[1], train_car.shape[2], 1)

model, saving_file = nn_models.u_net_v2(input_shape=input_shape, norm=True)

model.compile(optimizer='adam', 
                    loss='binary_crossentropy',     # dice_coef_loss / mse / binary_crossentropy
                    metrics=[nn_utils.dice_coef])   # binary_accuracy

print model.summary()

model_dir = nn_utils.createModelDir()
with open(model_dir + "architecture.json", "w") as json_file:
    json_file.write(model.to_json())

batch_size = 16

# %% 
# =============================================================================
# Train the model
# =============================================================================

checkpoint = ModelCheckpoint(model_dir + saving_file, monitor='val_dice_coef', verbose=1, save_best_only=True, mode='max')

t = time.time()
model.fit(train_car, train_car_mask,
                epochs = 500,
                batch_size = batch_size,
                shuffle = True,
                validation_data = (val_car, val_car_mask),
                callbacks = [earlyStopping, checkpoint],
                verbose = 2)
training_time = (time.time()-t)/60


# %%
best_name, best_loss, model_dir = nn_utils.removeWorstModels(model_dir, rename_dir=True)

model.load_weights(model_dir + best_name)


# %%
# =============================================================================
# Tune base_score and store statistics
# =============================================================================


for prefix in ['']:     # , 'ups_' -> not RAM enogh to work

    t = time.time()
    train_preds = model.predict(train_car)
    val_preds = model.predict(val_car)
    print ' - Elapsed time in predictions:', (time.time()-t)/60
    
    if prefix == 'ups_':
        print "Upscaling data"
        shape = (train_car.shape[1], train_car.shape[2])
        train_preds = np.array([ nn_utils.upsampleArray(arr, shape) for arr in train_preds ])
        val_preds = np.array([ nn_utils.upsampleArray(arr, shape) for arr in val_preds ])
        train_car_mask = np.array([ nn_utils.upsampleArray(arr, shape) for arr in train_car_mask ])
        val_car_mask = np.array([ nn_utils.upsampleArray(arr, shape) for arr in val_car_mask ])

    base_score = nn_utils.get_best_base_score(np.copy(val_preds), np.copy(val_car_mask))
    
    train_preds = np.where(train_preds>base_score, 1, 0)
    val_preds = np.where(val_preds>base_score, 1, 0)
    
    train_preds = train_preds.astype(float)
    train_car_mask = train_car_mask.astype(float)
    val_preds = val_preds.astype(float)
    val_car_mask = val_car_mask.astype(float)

    nn_utils.storeTrainStatistics(model_dir, train_car_mask, train_preds, val_car_mask, val_preds, base_score, training_time, prefix=prefix)
    
    print " - Statistics stored"



# %%
# =============================================================================
# Predictions
# =============================================================================

i = np.random.randint(0, len(val_car))

plt.figure()
plt.imshow(val_car[i].reshape(val_car.shape[1], val_car.shape[2]))
plt.figure()
plt.imshow(val_car_mask[i].reshape(val_car_mask.shape[1], val_car_mask.shape[2]))
plt.figure()
plt.imshow(val_preds[i].reshape(val_car.shape[1], val_car.shape[2]))



# %%
#import os
#print "Suspending..."
#time.sleep(15)
#os.system('systemctl suspend') 


