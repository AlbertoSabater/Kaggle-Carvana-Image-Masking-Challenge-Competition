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
from PIL import Image
import glob
from joblib import Parallel, delayed
import cv2
import re


# https://www.kaggle.com/rrqqmm/even-faster-submission


from keras import backend as K
from keras.callbacks import EarlyStopping, ModelCheckpoint


input_shape = (200,300, 1)
base_score = 0.5


# Fast run length encoding
# Based on: https://www.kaggle.com/hackerpoet/even-faster-run-length-encoder
def rle(img, base_score):
    
    img = nn_utils.upsampleArray(img, shape)
    
    flat_img = img.flatten()
    flat_img = np.where(flat_img > base_score, 1, 0).astype(np.uint8)

    starts = np.array((flat_img[:-1] == 0) & (flat_img[1:] == 1))
    ends = np.array((flat_img[:-1] == 1) & (flat_img[1:] == 0))
    starts_ix = np.where(starts)[0] + 2
    ends_ix = np.where(ends)[0] + 2
    lengths = ends_ix - starts_ix
    
    return ' '.join([ str(s)+' '+str(l) for s, l in zip(starts_ix, lengths) ])


# Get best base_score calculated
def readBaseScore(model_dir):
    with open(model_dir+'results.txt', 'r') as f:
        lines = f.readlines()
        l = [ l for l in lines if l.startswith('base_score') ][0]
        return float(l.split(': ')[1].split('\n')[0])
    
    
# %%
# =============================================================================
# Load model architecture and weights
# =============================================================================
        
#model_dir = 'models/model_2_0.9793/u-net_*0.9793*_(200, 300, 1)'
#model, saving_file = nn_models.u_net(input_shape=input_shape)
#from keras.model import model_from_json
#json_file = open('model.json', 'r')
#loaded_model_json = json_file.read()
#json_file.close()
#model = model_from_json('...architecture.json')
#model.load_weights(model_dir)


num_model = 'model_4'
files = os.listdir('models/')
model_dir = 'models/' + [f for f in files if f.startswith(num_model)][0] + '/'
from keras.models import model_from_json
json_file = open(model_dir+'architecture.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
model = model_from_json(loaded_model_json)

model.load_weights(model_dir+[f for f in os.listdir(model_dir) if f[-3:]=='.h5'][0])

model_name = [f for f in os.listdir(model_dir) if f[-3:]=='.h5'][0][:-3]


base_score = readBaseScore(re.findall('(.*/.*/)', model_dir)[0])

# Get img shape
l = model.layers[0]
shape = (int(l.input.shape[1]), int(l.input.shape[2]))

# %%

filelist = glob.glob('data/test_'+str(shape)+'/data/*')

i = np.random.randint(0, len(filelist))

img = np.array(Image.open(filelist[i])).reshape(1,shape[0],shape[1],1).astype(float)/255
pred = model.predict(img)
pred = nn_utils.upsampleArray(pred, shape)

plt.figure()
plt.imshow(pred)
plt.figure()
plt.imshow(np.array(Image.open(filelist[i]).resize((1918, 1280), Image.ANTIALIAS)).reshape(1280, 1918).astype(float)/255)
plt.figure()
plt.imshow(np.where(pred>base_score, 1, 0))



# %%

t = time.time()

filelist = glob.glob('data/test_'+str(shape)+'/data/*')

f = open(model_dir+model_name+'_submission.csv', 'w')
f.write('img,rle_mask\n')

step = 2500
for i in np.arange(0, len(filelist), step):
    
    rles = Parallel(n_jobs=8)(delayed(rle)(
            model.predict(np.array(Image.open(fname)).reshape(1,shape[0],shape[1],1).astype(float)/255), 
                               base_score) for fname in filelist[i:i+step])
    
    [ f.write(fname.split('/')[3][:-4]+'.jpg' + ',' + res + '\n') for fname, res in zip(filelist[i:i+step], rles) ]
    
    print i+step, '\t', time.strftime("%H:%M:%S")
    
f.close()


print "Time elapsed:",  (time.time()-t)/60


# %%

import gzip
import shutil
with open(model_dir+model_name+'_submission.csv', 'rb') as f_in, gzip.open(model_dir+model_name+'_submission.csv.gz', 'wb') as f_out:
    shutil.copyfileobj(f_in, f_out)
    
    
    
# %%
    
#import os
#import time
#print "Shuting down..."
#time.sleep(60*75)
#os.system('systemctl poweroff') 

