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


#img_size = (374, 250)
img_size = (300, 200)


def diceCoefficient(y_pred, y):
    return (2*np.sum(y_pred*y))/(np.sum(y_pred)+np.sum(y))


def dice_coef(y_true, y_pred):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection + 1.) / (K.sum(y_true_f) + K.sum(y_pred_f) + 1.)

def dice_coef_loss(y_true, y_pred):
    return -dice_coef(y_true, y_pred)


def get_best_base_score(preds, y):
    best_base_score = 0
    best_score = 0
    
    for base_score in np.arange(0.05, 1, 0.05):
        y_pred = np.copy(preds)
        y_pred[y_pred<base_score] = 0
        y_pred[y_pred>=base_score] = 1
        score = diceCoefficient(y_pred, y)
        if score > best_score: 
            best_score = score
            best_base_score = base_score
    print "Wave 1 ended", best_base_score, best_score
        
    for base_score in np.arange(best_base_score-0.05, best_base_score+0.06, 0.01):
        y_pred = np.copy(preds)
        y_pred[y_pred<base_score] = 0
        y_pred[y_pred>=base_score] = 1
        score = diceCoefficient(y_pred, y)
        if score > best_score: 
            best_score = score
            best_base_score = base_score
    print "Wave 2 ended", best_base_score, best_score
    
    for base_score in np.arange(best_base_score-0.01, best_base_score+0.01, 0.001):
        y_pred = np.copy(preds)
        y_pred[y_pred<base_score] = 0
        y_pred[y_pred>=base_score] = 1
        score = diceCoefficient(y_pred, y)
        if score > best_score: 
            best_score = score
            best_base_score = base_score
    print "Wave 3 ended", best_base_score, best_score
           
    print 'Best_score:', best_score
    print 'Best_base_score:', best_base_score
    
    return best_base_score


def preprocessData():
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
    with open('data/train_car_mask'+str(img_size)+'.pickle', 'w') as f:
        pickle.dump(train_car_mask, f)
    print "Time elapsed:",  (time.time()-t)/60

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



