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
import re

from keras import backend as K
from keras.callbacks import EarlyStopping, ModelCheckpoint, TensorBoard, CSVLogger
from image import ImageDataGenerator
#from keras.preprocessing.image import ImageDataGenerator

np.random.seed(123)  # for reproducibility

BASE_DIR = 'models/'


shape, batch_size = (256,256), 16
shape, batch_size = (512,512), 4
shape, batch_size = (512+256,512+256), 2
shape, batch_size = (1024,1024), 1

#shape, batch_size = (512+256,512+256), 2
shape, batch_size = (1024,1024), 1

base_score = 0.5
augmentation = True
simple = False
load_validation_data = False


RGB = True
rgb_suffix = '_RGB' if RGB else ''
color_mode = 'rgb' if RGB else 'grayscale'
num_channels = 3 if RGB else 1


loss = 'bcedice'        # 'bce'
#monitor = ['val_dice_coef']


resume_training, model_to_resume = True, 15


full = True
train_prefix = 'full_' if full else 'train_'
monitor = ['val_dice_coef', 'val_loss'] if full else ['val_dice_coef']


#data_gen_args_train = dict(rescale = 1./255,
#                     shear_range = 0.08,
#                     rotation_range = 5,
#                     width_shift_range = 0.08,
#                     height_shift_range = 0.08,
#                    horizontal_flip = True
#                    )

data_gen_args_train = dict(rescale = 1./255,
                    shear_range = 0.1,
                    rotation_range = 4,
                    zoom_range = 0.03,
                    horizontal_flip = True)    

t = time.time()

# -- 1. Entrenar 256x256 con image augmentation
# -- 1.1. Entrenar sin dobles convolucionales
# -- 1.2. Entrenar sin dobles convolucionales y sin flip_horizontal -> Mejora respecto de con Flip horizontal | 243 s
# -- 2.1. Entrenar 512x512 sin image augmentation | 1280s
# -- 2.2. Entrenar 512x512 con image augmentation (sin horizontal_flip)
# -- 3.1. Loss -> binary + dice. Entrenar 512x512 con image augmentation (con horizontal_flip)
# -- 4.1 756x756 + augmenting + flip_horizontal + bcedice  | 2800
# -- 4.2 756X756. RGB. Augmented, flip_horizontal, bcedice     | 3000
# -- 5. 768x768, FULL RGB, Agmented | 4000?
# Entrenar con train+validation
# 3.2. Loss -> dice
# 4. Probar cuál es el input más grande posible
# 6. Ver que algoritmo de entrenamiento funciona mejor Adam vs. RMSProp
# 6. Optimizar learning rate
# Upsample prediction in NN and test with full image
# Entrenar a full, pero añadir una validación sin transformaciones. Elegir modelo en función de las pruebas en validation

# Implementar u-net con los pares de convs como dense blocks: https://github.com/tdeboissiere/DeepLearningImplementations/blob/master/DenseNet/densenet.py
# https://medium.com/towards-data-science/densenet-2810936aeebb
# -- Submit con Base_score = 0.5



# %%
# =============================================================================
# Create NN model
# =============================================================================


# Create new model
if not resume_training:
    print ' * Creating new model'
    input_shape = (shape[0], shape[1], num_channels)
    
    model, saving_file = nn_models.u_net_v2(input_shape=input_shape, loss=loss, norm=True, simple=simple, monitor=monitor)
    
    if loss=='bcedice': loss_metric = nn_utils.bce_dice_loss
    elif loss=='bce': loss_metric = 'binary_crossentropy'
    
    model.compile(optimizer='adam', 
                        loss=loss_metric,     # dice_coef_loss / mse / binary_crossentropy / nn_utils.bce_dice_loss
                        metrics=[nn_utils.dice_coef])   # binary_accuracy
    
    print model.summary()
    
    model_dir = nn_utils.createModelDir()
    with open(model_dir + "architecture.json", "w") as json_file:
        json_file.write(model.to_json())
        
    saving_file = saving_file.replace(')_', ')_'+str(batch_size)+'_')
    if augmentation: saving_file = saving_file.replace('.h5', '_augm.h5')
    else: saving_file = saving_file.replace('.h5', '_noaugm.h5')
    if full: saving_file = saving_file.replace('.h5', '_full.h5')
    if RGB: saving_file = saving_file.replace('.h5', '_RGB.h5')
    
    print ' - Saving file:', saving_file

# Load trained model
else:
    print ' * Loading trained model'
    
    num_model = 'model_' + str(model_to_resume)
    files = os.listdir('models/')
    model_dir = 'models/' + [f for f in files if f.startswith(num_model)][0] + '/'
    from keras.models import model_from_json
    json_file = open(model_dir+'architecture.json', 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    model = model_from_json(loaded_model_json)
    model_name = [f for f in os.listdir(model_dir) if f[-3:]=='.h5'][-1]
    model.load_weights(model_dir+model_name)
#    model_name = [f for f in os.listdir(model_dir) if f[-3:]=='.h5'][0]#[:-3]
    print 'Model loaded:', model_name
    
    
    if '_bcedice' in model_name: loss_metric = nn_utils.bce_dice_loss
    elif '_bce' in model_name: loss_metric = 'binary_crossentropy'
    
    if '_RGB' in model_name:
        RGB = True
        rgb_suffix = '_RGB' if RGB else ''
        color_mode = 'rgb' if RGB else 'grayscale'
        num_channels = 3 if RGB else 1

    if 'full' in model_name:
        full = True
        train_prefix = 'full_' if full else 'train_'
        monitor = ['dice_coef', 'loss', 'val_dice_coef', 'val_loss'] if full else ['val_dice_coef']
        
    
    initial_epoch = int(re.match(r'.*_e([0-9]+).*', model_name).group(1))
    
    model_name = re.sub(r'_e[0-9]+', '_e{epoch:02d}', model_name)
    model_name = re.sub(r'_\*[\.0-9\*_]+\*', '_'+'_'.join(['*{'+m+':.5f}*' for m in monitor]), model_name)
    shape = tuple([ int(i) for i in re.split(r'\(|\)', model_name)[1].split(',')][:-1])
    
    
    model.compile(optimizer='adam', 
                        loss=loss_metric,     # dice_coef_loss / mse / binary_crossentropy / nn_utils.bce_dice_loss
                        metrics=[nn_utils.dice_coef])   # binary_accuracy
    
    
    saving_file = model_name

    print model.summary()
    print 'Model loaded:', model_name
    print ' - Saving file:', saving_file



# %%
# =============================================================================
# Load validation data
# =============================================================================

if load_validation_data:
    print 'Loading validation data'
    
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
    
else:
    print 'Creating validation imageDataGenerators'
    
    val_data_gen_args = dict(rescale = 1./255)
    
    val_image_datagen = ImageDataGenerator(**val_data_gen_args)
    val_mask_datagen = ImageDataGenerator(**val_data_gen_args)
    
    val_dir = 'data/full_' if full else 'data/val_'
    val_dir += str(shape)+rgb_suffix
    
    # Provide the same seed and keyword arguments to the fit and flow methods
    seed = 1
    
    val_image_generator = val_image_datagen.flow_from_directory(
        'data/'+train_prefix+str(shape)+rgb_suffix,
        target_size=shape,
        color_mode = color_mode,
        class_mode = None,
        batch_size = batch_size,
        seed = seed)
    
    val_mask_generator = val_mask_datagen.flow_from_directory(
        'data/'+train_prefix+'mask_'+str(shape),
        target_size=shape,
        color_mode = 'grayscale',
        class_mode = None,
        batch_size = batch_size,
        seed = seed)
    
    # combine generators into one which yields image and masks
    val_generator = itertools.izip(val_image_generator, val_mask_generator)

    num_samples_val = val_image_generator.n


# %%
# =============================================================================
# Load training data with augmentation
# =============================================================================


# we create two instances with the same arguments
if augmentation:
    print ' - Augmentation'

else:
    print' - No augmentation'
    data_gen_args_train = dict(rescale = 1./255,)
    
image_datagen = ImageDataGenerator(**data_gen_args_train)
mask_datagen = ImageDataGenerator(**data_gen_args_train)

# Provide the same seed and keyword arguments to the fit and flow methods
seed = 1
#image_datagen.fit(images, augment=True, seed=seed)
#mask_datagen.fit(masks, augment=True, seed=seed)

image_generator = image_datagen.flow_from_directory(
    'data/'+train_prefix+str(shape)+rgb_suffix,
    target_size=shape,
    color_mode = color_mode,
    class_mode = None,
    batch_size = batch_size,
    seed = seed)

mask_generator = mask_datagen.flow_from_directory(
    'data/'+train_prefix+'mask_'+str(shape),
    target_size=shape,
    color_mode = 'grayscale',
    class_mode = None,
    batch_size = batch_size,
    seed = seed)

num_samples_train = image_generator.n

# combine generators into one which yields image and masks
train_generator = itertools.izip(image_generator, mask_generator)



# %% 
# =============================================================================
# Train the model
# =============================================================================

if full:
    checkpoint = ModelCheckpoint(model_dir + saving_file, monitor='', verbose=1, save_best_only=False, mode=None)
    callbacks = [checkpoint]
else:
    checkpoint = ModelCheckpoint(model_dir + saving_file, monitor=monitor, verbose=1, save_best_only=True, mode='max')
    earlyStopping = EarlyStopping(monitor=monitor, min_delta=0.00001, verbose=1, mode='max', patience=9)
    callbacks = [earlyStopping, checkpoint]

tensorBoard = TensorBoard(log_dir='log', histogram_freq=0, write_graph=True, write_images=True)
callbacks.append(tensorBoard)
csv_logger = CSVLogger(model_dir+'log.csv', separator=',', append=True)
callbacks.append(csv_logger)


validation_data = (val_car, val_car_mask) if load_validation_data else val_generator

t = time.time()
if not resume_training:
    print " * New training"
    hist = model.fit_generator(
            generator = train_generator,
            steps_per_epoch = num_samples_train // batch_size, #################### 3072 num_samples_train
            epochs = 120,
            validation_data = validation_data,
            validation_steps = num_samples_val // batch_size,
            callbacks = callbacks,
            use_multiprocessing = True,
            verbose = 2)
else:
    print " * Restore training -", model_to_resume
    hist = model.fit_generator(
            generator = train_generator,
            steps_per_epoch = num_samples_train // batch_size, #################### num_samples_train
            epochs = 120,
            validation_data = validation_data,
            validation_steps = num_samples_val // batch_size,
            callbacks = callbacks,
            use_multiprocessing = True,
            initial_epoch=initial_epoch+1,
            verbose = 2)

training_time = (time.time()-t)/60

print 'Training ended'


# %% 

with open(model_dir+'hist.pickle', 'w') as f:
    pickle.dump(hist.history, f)
    
# %%
    
with open(model_dir+'hist.pickle', 'r') as f:
    history = pickle.load(f)

# %%
    
# =============================================================================
# Remove worst models and load the best
# =============================================================================

training_time = (time.time()-t)/60
if full:
    best_name, best_loss, model_dir = nn_utils.removeWorstModelsFull_v2(model_dir, rename_dir=True)
else:
    best_name, best_loss, model_dir = nn_utils.removeWorstModels(model_dir, rename_dir=True)

model.load_weights(model_dir + best_name)


# %%
# =============================================================================
# Tune base_score and store statistics
# =============================================================================


for prefix in ['']:     # , 'ups_' -> not RAM enogh to work

    t = time.time()
    
    if load_validation_data:
        val_preds = model.predict(val_car, batch_size)
        
        print ' - Elapsed time in predictions:', (time.time()-t)/60
        
        if prefix == 'ups_':
            print "Upscaling data"
            val_preds = np.array([ nn_utils.upsampleArray(arr, shape) for arr in val_preds ])
            val_car_mask = np.array([ nn_utils.upsampleArray(arr, shape) for arr in val_car_mask ])
    
        base_score = nn_utils.get_best_base_score(np.copy(val_preds), np.copy(val_car_mask))
        
    #    val_preds = np.where(val_preds>base_score, 1, 0)
    #    val_preds = val_preds.astype(float)
    #    val_car_mask = val_car_mask.astype(float)
    
        nn_utils.storeTrainStatistics_augm(model_dir, val_car_mask, val_preds, base_score, training_time, prefix=prefix)
        
    else:
#        base_score = nn_utils.get_best_base_score_generator(val_generator, model, num_steps=1528, batch_size=batch_size)
        generator = train_generator if full else val_generator
        nn_utils.storeTrainStatistics_augm_generator(model_dir, generator, model, base_score=0.5, training_time=training_time, prefix=prefix)
        
    
    print " - Statistics stored"



# %%
    
# =============================================================================
# Plot story
# =============================================================================

plt.figure()
plt.plot(history['loss'], label='loss')
plt.plot(history['val_loss'], label='val_loss')
plt.ylim([0,0.5])
plt.legend(loc=5)
plt.savefig(model_dir+'loss.png')

plt.figure()
plt.plot(history['dice_coef'], label='dice_coef')
plt.plot(history['val_dice_coef'], label='val_dice_coef')
plt.ylim([0.75,1])
plt.legend(loc=5)
plt.savefig(model_dir+'dice_coef.png')

# %%
# =============================================================================
# Predictions
# =============================================================================

if load_validation_data:
    i = np.random.randint(0, len(val_car))
    
    plt.figure()
    plt.imshow(val_car[i].reshape(val_car.shape[1], val_car.shape[2]))
    plt.figure()
    plt.imshow(val_car_mask[i].reshape(val_car_mask.shape[1], val_car_mask.shape[2]))
    plt.figure()
    #plt.imshow(val_preds[i].reshape(val_car.shape[1], val_car.shape[2]))
    plt.imshow(np.where(model.predict(np.array([val_car[i]]))>base_score, 1, 0).reshape(val_car.shape[1], val_car.shape[2]))

else:
    imgs = val_generator.next() # if not full else train_generator.next()
    
    plt.figure()
    plt.imshow(imgs[0][0].reshape(imgs[0].shape[1], imgs[0].shape[2], num_channels))
    plt.title('Original')
    plt.figure()
    plt.imshow(imgs[1][0].reshape(imgs[1].shape[1], imgs[1].shape[2]))
    plt.title('Original mask')
    plt.figure()
    plt.imshow(model.predict(np.array([imgs[0][0]])).reshape(imgs[0].shape[1], imgs[0].shape[2]))
    plt.title('Predicted mask')
    plt.figure()
    plt.imshow(np.where(model.predict(np.array([imgs[0][0]]))>base_score, 1, 0).reshape(imgs[0].shape[1], imgs[0].shape[2]))
    plt.title('Final mask')


# %%

base_dir = 'post_images/'+str(shape)+'/'


for i in np.arange(2):
    
    suffix = 'train/' if i == 0 else 'val/'
    store_dir = base_dir+suffix
    if not os.path.exists(store_dir): os.makedirs(store_dir)

#
#    images_dir = 'data/'+train_prefix+str(shape)+rgb_suffix+'/data/*'
#    images_dir_mask = 'data/'+train_prefix+'mask_'+str(shape)+'/data/*'
#
#    list_imgs = glob.glob(images_dir)
#    list_imgs_mask = glob.glob(images_dir_mask)
#    list_imgs.sort()
#    list_imgs_mask.sort()
#
#    i = np.random.randint(0, len(filelist))
    
    generator = train_generator if i == 0 else val_generator
    
    print '***', suffix
    
    for j in np.arange(10):
        imgs = generator.next() # if not full else train_generator.next()
        
#         1. Store original image
        # 2. Store original image cropped
#         3. Store original mask
        # 4. Store original mask cropped
        # 5. Store prediction
        # 6. Store prediction mask
        # Repetir x veces con y sin auagment
        
        
        
        plt.figure()
        original_img = imgs[0][0].reshape(imgs[0].shape[1], imgs[0].shape[2], num_channels)
        plt.imshow(original_img)
        plt.title('Original')
        plt.savefig(store_dir+'original_img_'+str(j)+'.png',bbox_inches='tight')
        
        plt.figure()
        original_mask = imgs[1][0].reshape(imgs[1].shape[1], imgs[1].shape[2])
        plt.imshow(original_mask)
        plt.title('Original mask')
        plt.savefig(store_dir+'original_mask_'+str(j)+'.png',bbox_inches='tight')

        plt.figure()
        predicted_mask = model.predict(np.array([imgs[0][0]])).reshape(imgs[0].shape[1], imgs[0].shape[2])
        plt.imshow(predicted_mask)
        plt.title('Predicted mask')
        plt.savefig(store_dir+'predicted_mask_'+str(j)+'.png',bbox_inches='tight')
        
        plt.figure()
        final_predicted_mask = np.where(model.predict(np.array([imgs[0][0]]))>base_score, 1, 0).reshape(imgs[0].shape[1], imgs[0].shape[2])
        plt.imshow(final_predicted_mask)
        plt.title('Final mask')
        plt.savefig(store_dir+'final_predicted_mask_'+str(j)+'.png',bbox_inches='tight')
    


# %%
    
res = []
for i in np.arange(100):
    imgs = val_generator.next()
    res += model.predict(np.array([imgs[0][0]])).ravel().tolist()
    
l_1 = len(res)
res = [ r for r in res if r > 0.05 and r < 0.95 ]
l_2 = len(res)
print l_1, l_2, float(l_2)/l_1 

plt.hist(res)

# %%

if False:
# %%
    import os
    import time
    print "Suspending..."
    time.sleep(60*60*14)
    os.system('systemctl suspend') 


