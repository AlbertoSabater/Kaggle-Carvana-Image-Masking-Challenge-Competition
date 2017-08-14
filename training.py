#!/usr/bin/env python2
# -*- coding: utf-8 -*-

import pickle
import numpy as np
import matplotlib.pyplot as plt
import time
import nn_models
import nn_utils

from keras import backend as K
from keras.callbacks import EarlyStopping, ModelCheckpoint

np.random.seed(123)  # for reproducibility

earlyStopping = EarlyStopping(monitor='val_loss', min_delta=0.00001, verbose=1, mode='min', patience=5)




# %%
# =============================================================================
# Load training data
# =============================================================================


with open('data/train_car_BN(300, 200).pickle', 'r') as f:
    train_car = pickle.load(f)
    
rand = np.random.permutation(np.arange(len(train_car)))                                                                                                                                                                                                                                                                                                                                     
train_car = train_car.reshape(train_car.shape[0], train_car.shape[1], train_car.shape[2], 1)
val_car = train_car[rand[int(len(rand)*0.7):]]
train_car = train_car[rand[:int(len(rand)*0.7)]]
   
    
with open('data/train_car_mask(300, 200).pickle', 'r') as f:
    train_car_mask = pickle.load(f)

train_car_mask = train_car_mask.reshape(train_car_mask.shape[0], train_car_mask.shape[1], train_car_mask.shape[2], 1)
val_car_mask = train_car_mask[rand[int(len(rand)*0.7):]]
train_car_mask = train_car_mask[rand[:int(len(rand)*0.7)]]


#train_car = train_car[rand]
#train_car_mask = train_car_mask[rand]

print train_car.shape, train_car_mask.shape, val_car.shape, val_car_mask.shape




# %%
# =============================================================================
# Create the model
# =============================================================================

input_shape = (train_car.shape[1], train_car.shape[2], 1)

model = nn_models.u_net(input_shape=input_shape)

model.compile(optimizer='adam', 
                    loss='mse',     # dice_coef_loss / mse / binary_crossentropy
                    metrics=['binary_accuracy', nn_utils.dice_coef])

print model.summary()



# %% 
# =============================================================================
# Train the model
# =============================================================================

model.fit(train_car, train_car_mask,
                epochs = 500,
                batch_size = 32,
                shuffle = True,
                validation_data = (val_car, val_car_mask),
                callbacks = [earlyStopping],
                verbose = 2)



# %%
# =============================================================================
# Tune base_score
# =============================================================================

t = time.time()
preds = model.predict(val_car)
print 'Elapsed time in predictions:', (time.time()-t)/60

base_score = nn_utils.get_best_base_score(np.copy(preds), val_car_mask)



# %%

preds[preds<base_score] = 0
preds[preds>=base_score] = 1



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
plt.imshow(preds[i].reshape(val_car.shape[1], val_car.shape[2]))




