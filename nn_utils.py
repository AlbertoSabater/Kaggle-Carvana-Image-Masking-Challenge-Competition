#!/usr/bin/env python2
# -*- coding: utf-8 -*-

import time
import numpy as np
import numpy as np
import matplotlib.pyplot as plt
import glob
from PIL import Image

import pickle
from joblib import Parallel, delayed
import multiprocessing
from keras import backend as K
from keras.losses import binary_crossentropy
import os
import re
from sklearn import metrics
import os

BASE_DIR = 'models/'

#img_size = (374, 250)
#img_size = (300, 200)


def diceCoefficient(y_pred, y):
    return (2*np.sum(y_pred*y))/(np.sum(y_pred)+np.sum(y))


def dice_coef(y_true, y_pred):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection + 1.) / (K.sum(y_true_f) + K.sum(y_pred_f) + 1.)

#def dice_coef_loss(y_true, y_pred):
#    return -dice_coef(y_true, y_pred)


def diceCoefficient_upsample(y_pred, y, shape):
    def aux(y_pred, y, shape):
        
        y_pred = np.array(Image.fromarray(np.array(y_pred).reshape(shape)).resize((1918, 1280), Image.ANTIALIAS))
        y = np.array(Image.fromarray(np.array(y).reshape(shape)).resize((1918, 1280), Image.ANTIALIAS))

        return (2*np.sum(y_pred*y))/(np.sum(y_pred)+np.sum(y))
        
        
    coefs = Parallel(n_jobs=8)(delayed(aux)(y_pred[i], y[i], shape) for i in np.arange(len(y)))

    return np.sum(coefs)/len(y)


# https://github.com/petrosgk/Kaggle-Carvana-Image-Masking-Challenge/blob/master/model/losses.py
def dice_loss(y_true, y_pred):
    loss = 1 - dice_coef(y_true, y_pred)
    return loss

def bce_dice_loss(y_true, y_pred):
    return binary_crossentropy(y_true, y_pred) + dice_loss(y_true, y_pred)
    


def get_best_base_score(preds, y):
    best_base_score = 0
    best_score = 0
    
    for base_score in np.arange(0.05, 1, 0.05):
        y_pred = np.copy(preds)
        y_pred[y_pred<=base_score] = 0
        y_pred[y_pred>base_score] = 1
        score = diceCoefficient(y_pred, y)
        if score > best_score: 
            best_score = score
            best_base_score = base_score
    print "Wave 1 ended", best_base_score, best_score
        
    for base_score in np.arange(best_base_score-0.05, best_base_score+0.06, 0.01):
        y_pred = np.copy(preds)
        y_pred[y_pred<=base_score] = 0
        y_pred[y_pred>base_score] = 1
        score = diceCoefficient(y_pred, y)
        if score > best_score: 
            best_score = score
            best_base_score = base_score
    print "Wave 2 ended", best_base_score, best_score
    
    for base_score in np.arange(best_base_score-0.01, best_base_score+0.01, 0.001):
        y_pred = np.copy(preds)
        y_pred[y_pred<=base_score] = 0
        y_pred[y_pred>base_score] = 1
        score = diceCoefficient(y_pred, y)
        if score > best_score: 
            best_score = score
            best_base_score = base_score
    print "Wave 3 ended", best_base_score, best_score
           
    print 'Best_score:', best_score
    print 'Best_base_score:', best_base_score
    
    return best_base_score


def get_best_base_score_generator(generator, model, num_steps, batch_size):
    best_base_score = 0
    best_score = 0
 
#    batch_size = generator.next()[0].shape[0]
#    num_steps = 1550//batch_size

    for base_score in np.arange(0.25, 0.85, 0.05):
        score = 0.0
        
        print base_score
        
        for i in np.arange(num_steps):
            aux = generator.next()
            y_pred = model.predict(aux[0], batch_size)
            y = aux[1]
            
            y_pred[y_pred<=base_score] = 0
            y_pred[y_pred>base_score] = 1
            score = diceCoefficient(y_pred, y)
                
            score += diceCoefficient(y_pred, y)
            
        score /= num_steps
                
        if score > best_score: 
            best_score = score
            best_base_score = base_score
    print "Wave 1 ended", best_base_score, best_score
        
    for base_score in np.arange(best_base_score-0.05, best_base_score+0.06, 0.01):
        score = 0.0

        print base_score
        
        for i in np.arange(num_steps):
            aux = generator.next()
            y_pred = model.predict(aux[0], batch_size)
            y = aux[1]
            
            y_pred[y_pred<=base_score] = 0
            y_pred[y_pred>base_score] = 1
            score = diceCoefficient(y_pred, y)
                
            score += diceCoefficient(y_pred, y)
                
        best_score /= num_steps

        if score > best_score: 
            best_score = score
            best_base_score = base_score
    print "Wave 2 ended", best_base_score, best_score
    
    for base_score in np.arange(best_base_score-0.01, best_base_score+0.01, 0.001):
        score = 0.0

        print base_score
        
        for i in np.arange(num_steps):
            aux = generator.next()
            y_pred = model.predict(aux[0], batch_size)
            y = aux[1]
            
            y_pred[y_pred<=base_score] = 0
            y_pred[y_pred>base_score] = 1
            score = diceCoefficient(y_pred, y)
                
            score += diceCoefficient(y_pred, y)
                
        best_score /= num_steps

        if score > best_score: 
            best_score = score
            best_base_score = base_score
    print "Wave 3 ended", best_base_score, best_score
           
    print 'Best_score:', best_score
    print 'Best_base_score:', best_base_score
    
    return best_base_score


def createModelDir():
    files = os.listdir(BASE_DIR)
    dir_name = BASE_DIR + 'model_' + str(len(files)) + '/'
    os.makedirs(dir_name)
    print '\n\n\n#########################################################################'
    print '###############################  MODEL', len(files), ' ##############################'
    print '#########################################################################'
    return dir_name


def removeWorstModels(model_dir, rename_dir=True):
    model_dir = BASE_DIR+[d for d in os.listdir(BASE_DIR) if d.startswith(model_dir.split('/')[1]) ][0] + '/'
    files = [ f for f in os.listdir(model_dir) if re.findall('.*\.h5', f) ]

    best_loss = 0
    best_name = ''
    
    # Get best model name
    for name in files:
        current_splited = name.split('*')
        if best_loss < float(current_splited[1]):
            best_loss = float(current_splited[1])
            best_name = name
                
    print "Best model name:", best_name, "-" , best_loss
    
    # Remove bad models
    for name in files:
        if name != best_name:
            os.remove(model_dir + name)
            
    if rename_dir: 
        new_model_dir = BASE_DIR+re.findall('(model_[0-9]*)_?', model_dir.split('/')[1])[0]+'_'+str(best_loss) + '/'
        print model_dir, new_model_dir
        os.rename(model_dir, new_model_dir)
            
    return best_name, best_loss, new_model_dir


def removeWorstModelsFull(model_dir, rename_dir=True):
    model_dir = BASE_DIR+[d for d in os.listdir(BASE_DIR) if d.startswith(model_dir.split('/')[1]) ][0] + '/'
    files = [ f for f in os.listdir(model_dir) if re.findall('.*\.h5', f) ]

    # Get best models name
    best_names = []
    best_vals = []
    
    best_loss = float('inf')
    best_name = ''
    for name in files:
        current_splited = name.split('*')
        if best_loss > float(current_splited[3]):
            best_loss = float(current_splited[3])
            best_name = name
            
    best_names.append(best_name)
    best_vals.append(best_loss)
    
    
    best_dice = 0
    best_name = ''
    for name in files:
        current_splited = name.split('*')
        if best_dice < float(current_splited[1]):
            best_dice = float(current_splited[1])
            best_name = name
            
    best_names.append(best_name)
    best_vals.append(best_dice)
    
    
    print best_names
    print best_vals            
    
#    print "Best model name:", best_name, "-" , best_loss
    
    # Remove bad models
    for name in files:
        if not name in best_names:
            os.remove(model_dir + name)
            
    if rename_dir: 
        new_model_dir = BASE_DIR+re.findall('(model_[0-9]*)_?', model_dir.split('/')[1])[0]+'_'+str(best_dice) + '/'
        print model_dir, new_model_dir
        os.rename(model_dir, new_model_dir)

    return best_name, best_loss, new_model_dir


# Iterate over a varible number of metrics. [dice, loss]+
def removeWorstModelsFull_v2(model_dir, rename_dir=True):

    model_dir = BASE_DIR+[d for d in os.listdir(BASE_DIR) if d.startswith(model_dir.split('/')[1]) ][0] + '/'
    files = [ f for f in os.listdir(model_dir) if re.findall('.*\.h5', f) ]

    mtr_pattern = r'\*(\d\.[\d]+)\*'
    num_metrics = len(re.split(mtr_pattern, files[-1])[1::2])
    
    
    best_names = []
    best_vals = []
    for i in np.arange(num_metrics):
        
        print "***", i
        
        # Sie s impar maximizar si no minimizar
        best_val = float('inf') if i%2 != 0 else 0.0
        best_file = ''
        for f in files:
            mtrs = re.split(mtr_pattern, f)[1::2]
            if len(mtrs)>i:
                val = float(mtrs[i])
                if i%2 != 0:    # minimizar
                    print 'min', val, best_val
                    if val < best_val:
                        best_val = val
                        best_file = f
                else:           # Maximizar
                    print 'max', val, best_val
                    if val > best_val:
                        best_val = val
                        best_file = f 
                        
        best_names.append(best_file)
        best_vals.append(best_val)
        
    print list(set(best_names))
    print best_vals
            
    
    print best_names
    print best_vals            
    
#    print "Best model name:", best_name, "-" , best_loss
    
    # Remove bad models
    for name in files:
        if not name in best_names:
            os.remove(model_dir + name)
            
    best_dice = best_vals[2] if len(best_vals)>2 else best_vals[0]
    best_name = best_names[2] if len(best_names)>2 else best_names[0]
    best_loss = best_vals[3] if len(best_vals)>2 else best_vals[1]
    
    if rename_dir:
        new_model_dir = BASE_DIR+re.findall('(model_[0-9]*)_?', model_dir.split('/')[1])[0]+'_'+str(best_dice) + '/'
        print model_dir, new_model_dir
        os.rename(model_dir, new_model_dir)

    return best_name, best_loss, new_model_dir


def storeTrainStatistics(model_dir, train_car_mask, train_preds, val_car_mask, val_preds, base_score, training_time, prefix=''):
    with open(model_dir + 'results.txt', 'a+') as f:    # a+
        f.write(prefix+'base_score: ' + str(base_score) + '\n')
        f.write(prefix+'train_acc: ' + str(metrics.accuracy_score(train_car_mask.ravel(), train_preds.ravel())) +'\n')
        f.write(prefix+'val_acc: ' + str(metrics.accuracy_score(val_car_mask.ravel(), val_preds.ravel())) +'\n')
        f.write(prefix+'train_dice: ' + str(diceCoefficient(train_preds, train_car_mask)) +'\n')
        f.write(prefix+'val_dice: ' + str(diceCoefficient(val_preds, val_car_mask)) +'\n')
        f.write('training_time: ' + str(training_time) +'\n')
    f.close()
    
def storeTrainStatistics_augm(model_dir, val_car_mask, val_preds, base_score, training_time, prefix=''):
    with open(model_dir + 'results.txt', 'a+') as f:    # a+
        f.write(prefix+'base_score: ' + str(base_score) + '\n')
        f.write('training_time: ' + str(training_time) +'\n')
        for bs in [0.5, base_score]:
            
            vp = np.where(val_preds>bs, 1, 0)
            vp = vp.astype(float)
            vcm = val_car_mask.astype(float)
    
            f.write(prefix+'val_acc'+str(bs)+': ' + str(metrics.accuracy_score(vcm.ravel(), vp.ravel())) +'\n')
            f.write(prefix+'val_dice'+str(bs)+': ' + str(diceCoefficient(vp, vcm)) +'\n')
    f.close()
    
def storeTrainStatistics_augm_generator(model_dir, generator, model, base_score, training_time, prefix):
    with open(model_dir + 'results.txt', 'a+') as f:    # a+
        f.write(prefix+'base_score: ' + str(base_score) + '\n')
        f.write('training_time: ' + str(training_time) +'\n')
        
        bs_iter = [0.5] if base_score==0.5 else [0.5, base_score]
        for bs in bs_iter:
            
            val_acc = 0.0
            val_dice = 0.0
            
            batch_size = generator.next()[0].shape[0]
            num_steps = 1550//batch_size
            
            for i in np.arange(num_steps):
                aux = generator.next()
                val_preds = model.predict(aux[0], batch_size)
                val_car_mask = aux[1]
            
                vp = np.where(val_preds>bs, 1, 0)
                vp = vp.astype(float)
                vcm = val_car_mask.astype(float)
                
                val_acc += metrics.accuracy_score(vcm.ravel(), vp.ravel())
                val_dice += diceCoefficient(vp, vcm)
            
            print 'Results:', val_acc, val_dice
            val_acc /= num_steps
            val_dice /= num_steps
            print 'Results:', val_acc, val_dice
    
            f.write(prefix+'val_acc'+str(bs)+': ' + str(val_acc) +'\n')
            f.write(prefix+'val_dice'+str(bs)+': ' + str(val_dice) +'\n')
    f.close()
    
    
def upsampleArray(arr, arr_shappe):
    arr = Image.fromarray(np.array(arr).reshape(arr_shappe))
    arr = np.array(arr.resize((1918, 1280), Image.ANTIALIAS))
    return arr


# %%

# Store two numpy arrays with images and masks
def preprocessData(img_size):
# %%
    t = time.time()
    filelist = glob.glob('data/train/*')
    #filelist = filelist[:20]
    def processCarBN(fname):
        fname = fname.split('/')[2].split('.')[0]
        return [np.array(Image.open('data/train/'+fname+'.jpg').resize(img_size, Image.ANTIALIAS).convert('L')).astype(float)/255,
         np.array(Image.open('data/train_masks/'+fname+'_mask.gif').resize(img_size, Image.ANTIALIAS)).astype(float)]
    
    pairs = np.array(Parallel(n_jobs=8)(delayed(processCarBN)(fname) for fname in filelist))
    train_car = pairs[:,0,:,:]
    train_car_mask = pairs[:,1,:,:]
    print "Time elapsed:",  (time.time()-t)/60
    
    
    t = time.time()
    with open('data/train_car_BN'+str(img_size)+'.pickle', 'w') as f:
        pickle.dump(train_car, f)
    train_car = None
    with open('data/train_car_mask'+str(img_size)+'.pickle', 'w') as f:
        pickle.dump(train_car_mask, f)
    train_car_mask = None
    print "Time elapsed:",  (time.time()-t)/60
    

# DEPRECATED
# Stores the training numpy arrays splited in several files
def preprocessData_v2(img_size):
# %%
    t = time.time()
    filelist = glob.glob('data/train/*')
    #filelist = filelist[:20]
#    def processCarBN(fname):
#        fname = fname.split('/')[2].split('.')[0]
#        return [np.array(Image.open('data/train/'+fname+'.jpg').resize(img_size, Image.ANTIALIAS).convert('L')).astype(float)/255,
#         np.array(Image.open('data/train_masks/'+fname+'_mask.gif').resize(img_size, Image.ANTIALIAS)).astype(float)]
        
    
    def processImageBN(fname):
        fname = fname.split('/')[2].split('.')[0]
        return np.array(Image.open('data/train/'+fname+'.jpg').resize(img_size, Image.ANTIALIAS).convert('L')).astype(float)/255
        
    def processImageBN_mask(fname):
        fname = fname.split('/')[2].split('.')[0]
        return np.array(Image.open('data/train_masks/'+fname+'_mask.gif').resize(img_size, Image.ANTIALIAS)).astype(float)
        
    
    step = len(filelist)/4
    for i in np.arange(0, len(filelist), step):
        t = time.time()
        train_car = np.array(Parallel(n_jobs=8)(delayed(processImageBN)(fname) for fname in filelist[i:i+step]))
        print "Done "+str(i/step)
        with open('data/train_car_BN'+str(img_size)+'_'+str(i/step)+'.pickle', 'w') as f:
            pickle.dump(train_car, f)
        train_car = None
        del train_car
        print "Time elapsed:",  (time.time()-t)/60
    
        t = time.time()
        train_car_mask = np.array(Parallel(n_jobs=8)(delayed(processImageBN_mask)(fname) for fname in filelist[i:i+step]))
        print "Done "+str(i/step)
        with open('data/train_car_mask'+str(img_size)+'_'+str(i/step)+'.pickle', 'w') as f:
            pickle.dump(train_car_mask, f)
        train_car_mask = None
        del train_car_mask
        print "Time elapsed:",  (time.time()-t)/60
    
    
    
# %%
        
# Preprocess images and mask and stores each image/mask in a folder, different for image/mask and train/val/test
def preprocessData_v3(img_size, RGB):
# %%
    
    train_perc = 0.7
    
    filelist = glob.glob('data/train/*')
    filelist_train = filelist[:int(len(filelist)*train_perc)]
    filelist_val = filelist[int(len(filelist)*train_perc):]
    
    def processImageBN(fname, store_dir, store_dir_full, source):
        fname = fname.split('/')[2].split('.')[0]
        img = Image.open(source+fname+'.jpg').resize(img_size, Image.ANTIALIAS).convert('L')
        img.save(store_dir+fname+'.png')
        if store_dir_full: img.save(store_dir_full+fname+'.png')
    
    def processImageRGB(fname, store_dir, store_dir_full, source):
        fname = fname.split('/')[2].split('.')[0]
        img = Image.open(source+fname+'.jpg').resize(img_size, Image.ANTIALIAS)
        img.save(store_dir+fname+'.png')
        if store_dir_full: img.save(store_dir_full+fname+'.png')

    def processImageBN_mask(fname, store_dir_full, store_dir):
        fname = fname.split('/')[2].split('.')[0]
        img = Image.open('data/train_masks/'+fname+'_mask.gif').resize(img_size, Image.ANTIALIAS)
        img.save(store_dir+fname+'.png')
        if store_dir_full: img.save(store_dir_full+fname+'.png')
        
    
    data_func = processImageRGB if RGB else processImageBN
    rgb_sufix = '_RGB' if RGB else ''
    
    # Get taining data
    t = time.time()
    store_dir = 'data/train_'+str(img_size)+rgb_sufix+'/data/'
    store_dir_full = 'data/full_'+str(img_size)+rgb_sufix+'/data/'
    if not os.path.exists(store_dir): os.makedirs(store_dir)
    if not os.path.exists(store_dir_full): os.makedirs(store_dir_full)
    Parallel(n_jobs=8)(delayed(data_func)(fname, store_dir, store_dir_full, 'data/train/') for fname in filelist_train)
    
    store_dir = 'data/train_mask_'+str(img_size)+'/data/'
    store_dir_full = 'data/full_mask_'+str(img_size)+'/data/'
    if not os.path.exists(store_dir): os.makedirs(store_dir)
    if not os.path.exists(store_dir_full): os.makedirs(store_dir_full)
    Parallel(n_jobs=8)(delayed(processImageBN_mask)(fname, store_dir, store_dir_full) for fname in filelist_train)
    print "Train. Time elapsed:",  (time.time()-t)/60
    

    # Get validation data
    t = time.time()
    store_dir = 'data/val_'+str(img_size)+rgb_sufix+'/data/'
    store_dir_full = 'data/full_'+str(img_size)+rgb_sufix+'/data/'
    if not os.path.exists(store_dir): os.makedirs(store_dir)
    if not os.path.exists(store_dir_full): os.makedirs(store_dir)
    Parallel(n_jobs=8)(delayed(data_func)(fname, store_dir, store_dir_full, 'data/train/') for fname in filelist_val)
    
    store_dir = 'data/val_mask_'+str(img_size)+'/data/'
    store_dir_full = 'data/full_mask_'+str(img_size)+'/data/'
    if not os.path.exists(store_dir): os.makedirs(store_dir)
    Parallel(n_jobs=8)(delayed(processImageBN_mask)(fname, store_dir, store_dir_full) for fname in filelist_val)
    print "Validation. Time elapsed:",  (time.time()-t)/60
    
    
    # Get test data
    t = time.time()
    print "Processing Test..."
    filelist = glob.glob('data/test/*')
    store_dir = 'data/test_'+str(img_size)+rgb_sufix+'/data/'
    if not os.path.exists(store_dir): os.makedirs(store_dir)
    step = len(filelist)/10
    for i in np.arange(0,len(filelist), step):
        st = time.time()
        Parallel(n_jobs=8)(delayed(data_func)(fname, store_dir, None, 'data/test/') for fname in filelist[i:i+step])
#        print i,'/', 10, '\t', time.strftime("%H:%M:%S"), '-', (time.time()-st)
        print '{0}/{1}\t{2} - {3:.2f}'.format(i, 10, time.strftime("%H:%M:%S"), (time.time()-st))
    print "Test. Time elapsed:",  (time.time()-t)/60
   
        
# %%
def preprocessDataTest(img_size):
# %%
    t = time.time()
    filelist = glob.glob('data/test/*')
    
    store_dir = 'data/test_'+str(img_size)+'/'
    if not os.path.exists(store_dir): os.makedirs(store_dir)

    def processTestCarBN(fname, store_dir):
        Image.open(fname).resize(img_size, Image.ANTIALIAS).convert('L').save(store_dir+fname.split('/')[2])
        
    Parallel(n_jobs=8)(delayed(processTestCarBN)(fname, store_dir) for fname in filelist)
     
    print "Time elapsed:",  (time.time()-t)/60


# %%
    
def shutDown():
    import os
    print "Shuting down..."
    time.sleep(15)
    os.system('systemctl poweroff') 
    
def suspend():
    import os
    print "Suspending..."
    time.sleep(15)
    os.system('systemctl suspend') 
    
def hibernate():
    import os
    print "Hibernating..."
    time.sleep(15)
    os.system('systemctl hibernate') 

## %%
## =============================================================================
## Load, normalize and store train data    
## =============================================================================
#
#
#t = time.time()
#
#filelist = glob.glob('data/train/*')
##filelist = filelist[:50]
#def processCarBN(fname):
#    return np.array(Image.open(fname).resize(img_size, Image.ANTIALIAS).convert('L')).astype(float)/255
#def processCarRGB(fname):
#    return np.array(Image.open(fname).resize(img_size, Image.ANTIALIAS)).astype(float)/255
#
##x_car = np.array([processCar(fname) for fname in filelist])
#x_car = np.array(Parallel(n_jobs=8)(delayed(processCarBN)(fname) for fname in filelist))
#
#print "Time elapsed:",  (time.time()-t)/60
#print x_car.shape
#plt.figure()
#plt.imshow(x_car[1])
#
#
##with open('data/x_car_BN'+str(img_size)+'.pickle', 'w') as f:
##    pickle.dump(x_car, f)
#
#
## %%
## =============================================================================
## Load, normalize and store train masks
## =============================================================================
#
#
#t = time.time()
#filelist = glob.glob('data/train_masks/*')
##filelist = filelist[:1000]
#def processCar(fname):
#    return np.array(Image.open(fname).resize(img_size, Image.ANTIALIAS)).astype(float)
#
##x_mask = np.array([np.array(Image.open(fname)).astype(float) for fname in filelist])
#x_mask = np.array(Parallel(n_jobs=8)(delayed(processCar)(fname) for fname in filelist))
#
#print "Time elapsed:",  (time.time()-t)/60
#print x_mask.shape
#plt.figure()
#plt.imshow(x_mask[1])
#
#
##with open('data/x_mask'+str(img_size)+'.pickle', 'w') as f:
##    pickle.dump(x_mask, f)
#    
#    
## %%
#    
#i = 1234
#
#plt.figure()
#plt.imshow(train_car[i])
#
#plt.figure()
#plt.imshow(train_car_mask[i])



