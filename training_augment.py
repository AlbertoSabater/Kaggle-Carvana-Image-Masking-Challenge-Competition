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
import itertools

from keras import backend as K
from keras.callbacks import EarlyStopping, ModelCheckpoint
from image import ImageDataGenerator
#from keras.preprocessing.image import ImageDataGenerator

np.random.seed(123)  # for reproducibility

BASE_DIR = 'models/'

earlyStopping = EarlyStopping(monitor='val_dice_coef', min_delta=0.00001, verbose=1, mode='max', patience=9)

shape, batch_size = (256,256), 16
shape, batch_size = (512,512), 4
shape = (512+256,512+256)
shape = (1024,1024)

shape, batch_size = (256,256), 16

base_score = 0.5
simple = True


# -- 1. Entrenar 256x256 con image augmentation
# 1.1. Entrenar sin dobles convolucionales
# 2. Entrenar 512x512 sin image augmentation
# 3. Entrenar 512x512 con image augmentation
# 4. Probar cuál es el input más grande posible
# 5. Optimizar el image augmentation
# 6. Input en RGB
# 6. Optimizar learning rate


# %%
# =============================================================================
# Load validation data
# =============================================================================

t = time.time()
filelist = glob.glob('data/val_'+str(shape)+'/data/*')
#filelist = filelist[:20]
def loadValidationData(fname, img_size):
#    print fname
    fname = fname.split('/')[3]
    return [np.array(Image.open('data/val_'+str(shape)+'/data/'+fname)).astype(float)/255,
     np.array(Image.open('data/val_mask_'+str(shape)+'/data/'+fname)).astype(float)]

pairs = np.array(Parallel(n_jobs=8)(delayed(loadValidationData)(fname, shape) for fname in filelist))
val_car = pairs[:,0,:,:]
val_car_mask = pairs[:,1,:,:]
val_car = val_car.reshape(val_car.shape[0], shape[0], shape[1], 1)
val_car_mask = val_car_mask.reshape(val_car.shape[0], shape[0], shape[1], 1)
del pairs

print "Time elapsed:",  (time.time()-t)/60


# %%
# =============================================================================
# Create the model
# =============================================================================

input_shape = (shape[0], shape[1], 1)

model, saving_file = nn_models.u_net_v2(input_shape=input_shape, norm=True, simple=simple)

model.compile(optimizer='adam', 
                    loss='binary_crossentropy',     # dice_coef_loss / mse / binary_crossentropy
                    metrics=[nn_utils.dice_coef])   # binary_accuracy

print model.summary()

model_dir = nn_utils.createModelDir()
with open(model_dir + "architecture.json", "w") as json_file:
    json_file.write(model.to_json())
    
saving_file = saving_file.replace('.h5', '_augm.h5')


# %%
# =============================================================================
# Load training data with augmentation
# =============================================================================

#batch_size = 1 ###############

# we create two instances with the same arguments
data_gen_args = dict(rescale = 1./255,
#                    shear_range = 0.1,
                     rotation_range = 4,
                    zoom_range = 0.03,
                    horizontal_flip = True)
image_datagen = ImageDataGenerator(**data_gen_args)
mask_datagen = ImageDataGenerator(**data_gen_args)

# Provide the same seed and keyword arguments to the fit and flow methods
seed = 1
#image_datagen.fit(images, augment=True, seed=seed)
#mask_datagen.fit(masks, augment=True, seed=seed)

image_generator = image_datagen.flow_from_directory(
    'data/train_'+str(shape),
    target_size=shape,
    color_mode = 'grayscale',
    class_mode = None,
    batch_size = batch_size,
    seed = seed)

mask_generator = mask_datagen.flow_from_directory(
    'data/train_mask_'+str(shape),
    target_size=shape,
    color_mode = 'grayscale',
    class_mode = None,
    batch_size = batch_size,
    seed = seed)

# combine generators into one which yields image and masks
train_generator = itertools.izip(image_generator, mask_generator)


# %% 
# =============================================================================
# Train the model
# =============================================================================


checkpoint = ModelCheckpoint(model_dir + saving_file, monitor='val_dice_coef', verbose=1, save_best_only=True, mode='max')

t = time.time()
hist = model.fit_generator(
        generator = train_generator,
        steps_per_epoch = 3072 // batch_size,
        epochs = 120,
        validation_data = (val_car, val_car_mask),
        callbacks = [earlyStopping, checkpoint],
        use_multiprocessing = True,
        verbose = 2)

training_time = (time.time()-t)/60

#checkpoint = ModelCheckpoint(model_dir + saving_file, monitor='val_dice_coef', verbose=1, save_best_only=True, mode='max')
#
#t = time.time()
#model.fit(train_car, train_car_mask,
#                epochs = 500,
#                batch_size = batch_size,
#                shuffle = True,
#                validation_data = (val_car, val_car_mask),
#                callbacks = [earlyStopping, checkpoint],
#                verbose = 2)
#training_time = (time.time()-t)/60


# %% 

with open(model_dir+'hist.pickle', 'w') as f:
    pickle.dump(hist.history, f)
    
# %%
    
with open(model_dir+'hist.pickle', 'r') as f:
    history = pickle.load(f)

# %%
    
plt.figure()
plt.plot(history['loss'], label='loss')
plt.plot(history['val_loss'], label='val_loss')
plt.ylim([0,0.5])
plt.legend(loc=5)

plt.figure()
plt.plot(history['dice_coef'], label='dice_coef')
plt.plot(history['val_dice_coef'], label='val_dice_coef')
plt.ylim([0.75,1])
plt.legend(loc=5)

# %%

training_time = (time.time()-t)/60
best_name, best_loss, model_dir = nn_utils.removeWorstModels(model_dir, rename_dir=True)

model.load_weights(model_dir + best_name)


# %%
# =============================================================================
# Tune base_score and store statistics
# =============================================================================


for prefix in ['']:     # , 'ups_' -> not RAM enogh to work

    t = time.time()
#    train_preds = model.predict(train_car)
    val_preds = model.predict(val_car, batch_size)
#    val_preds = np.array([])
#    step = len(val_car)/10
#    for i in np.arange(0,len(val_car), step):
#        if len(val_preds) == 0: val_preds = model.predict(val_car[i:i+step], batch_size)
#        else: val_preds = np.concatenate([val_preds, model.predict(val_car[i:i+step])])
#        print i,'/',len(val_car)
    print ' - Elapsed time in predictions:', (time.time()-t)/60
    
    if prefix == 'ups_':
        print "Upscaling data"
#        shape = (train_car.shape[1], train_car.shape[2])
#        train_preds = np.array([ nn_utils.upsampleArray(arr, shape) for arr in train_preds ])
        val_preds = np.array([ nn_utils.upsampleArray(arr, shape) for arr in val_preds ])
#        train_car_mask = np.array([ nn_utils.upsampleArray(arr, shape) for arr in train_car_mask ])
        val_car_mask = np.array([ nn_utils.upsampleArray(arr, shape) for arr in val_car_mask ])

    base_score = nn_utils.get_best_base_score(np.copy(val_preds), np.copy(val_car_mask))
    
#    train_preds = np.where(train_preds>base_score, 1, 0)
    val_preds = np.where(val_preds>base_score, 1, 0)
    
#    train_preds = train_preds.astype(float)
#    train_car_mask = train_car_mask.astype(float)
    val_preds = val_preds.astype(float)
    val_car_mask = val_car_mask.astype(float)

    nn_utils.storeTrainStatistics_augm(model_dir, val_car_mask, val_preds, base_score, training_time, prefix=prefix)
    
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
#plt.imshow(val_preds[i].reshape(val_car.shape[1], val_car.shape[2]))
plt.imshow(np.where(model.predict(np.array([val_car[i]]))>base_score, 1, 0).reshape(val_car.shape[1], val_car.shape[2]))



# %%
#import os
#print "Suspending..."
#time.sleep(15)
#os.system('systemctl suspend') 


