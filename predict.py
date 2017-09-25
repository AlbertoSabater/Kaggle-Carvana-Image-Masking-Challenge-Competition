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


base_score = 0.5
model_number = 15


# Fast run length encoding
# Based on: https://www.kaggle.com/hackerpoet/even-faster-run-length-encoder
# Return rle with and without custom base_score
def rle(img, base_score):
    
    img = nn_utils.upsampleArray(img, shape)
    
#    flat_img = img.flatten()
#    flat_img = np.where(flat_img > base_score, 1, 0).astype(np.uint8)
#
#    starts = np.array((flat_img[:-1] == 0) & (flat_img[1:] == 1))
#    ends = np.array((flat_img[:-1] == 1) & (flat_img[1:] == 0))
#    starts_ix = np.where(starts)[0] + 2
#    ends_ix = np.where(ends)[0] + 2
#    lengths = ends_ix - starts_ix
    
    res = []

    flat_img = img.flatten()
    flat_img = np.where(flat_img > 0.5, 1, 0).astype(np.uint8)
    flat_img = np.insert(flat_img, [0, len(flat_img)], [0, 0])

    starts = np.array((flat_img[:-1] == 0) & (flat_img[1:] == 1))
    ends = np.array((flat_img[:-1] == 1) & (flat_img[1:] == 0))
    starts_ix = np.where(starts)[0] + 1
    ends_ix = np.where(ends)[0] + 1
    lengths = ends_ix - starts_ix
    
    res.append(' '.join([ str(s)+' '+str(l) for s, l in zip(starts_ix, lengths) ]))
    
    
    if base_score!=0.5:
        flat_img = img.flatten()
        flat_img = np.where(flat_img > base_score, 1, 0).astype(np.uint8)
        flat_img = np.insert(flat_img, [0, len(flat_img)], [0, 0])
    
        starts = np.array((flat_img[:-1] == 0) & (flat_img[1:] == 1))
        ends = np.array((flat_img[:-1] == 1) & (flat_img[1:] == 0))
        starts_ix = np.where(starts)[0] + 1
        ends_ix = np.where(ends)[0] + 1
        lengths = ends_ix - starts_ix
        
        res.append(' '.join([ str(s)+' '+str(l) for s, l in zip(starts_ix, lengths) ]))
    
    
    return res    
#    return ' '.join([ str(s)+' '+str(l) for s, l in zip(starts_ix, lengths) ])


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


num_model = 'model_' + str(model_number)
files = os.listdir('models/')
model_dir = 'models/' + [f for f in files if f.startswith(num_model)][0] + '/'
from keras.models import model_from_json
json_file = open(model_dir+'architecture.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
model = model_from_json(loaded_model_json)

#model.load_weights(model_dir+[f for f in os.listdir(model_dir) if f[-3:]=='.h5'][0])
#
#model_name = [f for f in os.listdir(model_dir) if f[-3:]=='.h5'][0][:-3]


model.load_weights(model_dir+[f for f in os.listdir(model_dir) if f[-3:]=='.h5'][-1])

model_name = [f for f in os.listdir(model_dir) if f[-3:]=='.h5'][-1][:-3]


print model_name


base_score = readBaseScore(re.findall('(.*/.*/)', model_dir)[0])

# Get img shape
l = model.layers[0]
num_channels = int(l.input.shape[3])
#shape = (int(l.input.shape[1]), int(l.input.shape[2])) if num_channels==1 else (int(l.input.shape[1]), int(l.input.shape[2]), int(l.input.shape[3]))
shape = (int(l.input.shape[1]), int(l.input.shape[2]))

test_dir = 'data/test_'+str(shape)+'_RGB/data/*' if num_channels==3  else 'data/test_'+str(shape)+'/data/*'

# %%

filelist = glob.glob(test_dir)

i = np.random.randint(0, len(filelist))

img = np.array(Image.open(filelist[i])).reshape(1,shape[0],shape[1],num_channels).astype(float)/255
pred = model.predict(img)
pred = nn_utils.upsampleArray(pred, shape)

plt.figure()
plt.imshow(pred)
plt.title('Prediction')
plt.figure()
plt.imshow(np.array(Image.open(filelist[i]).resize((1918, 1280), Image.ANTIALIAS)).reshape(1280, 1918, num_channels).astype(float)/255)
plt.title('Original')
plt.figure()
plt.imshow(np.where(pred>base_score, 1, 0))
plt.title('Predicted mask')
#plt.figure()
#plt.imshow(np.where(pred>0.5, 1, 0))
#plt.title('Predicted mask')



# %%

t = time.time()

filelist = glob.glob(test_dir)

subm_raw_name = model_dir+model_name+'_submission_bs0.5.csv'
f = open(subm_raw_name, 'w')
f.write('img,rle_mask\n')

if base_score!=0.5:
    subm_custom_name = model_dir+model_name+'_submission_bs'+str(base_score)+'.csv'
    g = open(subm_custom_name, 'w')
    g.write('img,rle_mask\n')


step = 2500
for i in np.arange(0, len(filelist), step):
    
    st = time.time()
    
    rles = Parallel(n_jobs=8)(delayed(rle)(
            model.predict(np.array(Image.open(fname)).reshape(1,shape[0],shape[1],num_channels).astype(float)/255), 
                               base_score) for fname in filelist[i:i+step])
    
    [ f.write(fname.split('/')[3][:-4]+'.jpg' + ',' + res[0] + '\n') for fname, res in zip(filelist[i:i+step], rles) ]
    if base_score != 0.5:
        [ g.write(fname.split('/')[3][:-4]+'.jpg' + ',' + res[1] + '\n') for fname, res in zip(filelist[i:i+step], rles) ]
    
    print i+step, '\t', time.strftime("%H:%M:%S"), '-', (time.time()-st)
    
f.close()


print "CSV predictions. Time elapsed:",  (time.time()-t)/60


# %%

import gzip
import shutil
t = time.time()
with open(subm_raw_name, 'rb') as f_in, gzip.open(subm_raw_name+'.gz', 'wb') as f_out:
    shutil.copyfileobj(f_in, f_out)
print "Raw CSV compression. Time elapsed:",  (time.time()-t)/60

if base_score != 0.5:
    t = time.time()
    with open(subm_custom_name, 'rb') as f_in, gzip.open(subm_custom_name+'.gz', 'wb') as f_out:
        shutil.copyfileobj(f_in, f_out)
    print "Custom CSV compression. Time elapsed:",  (time.time()-t)/60
    
    
# %%
    
if False:
# %%
    import os
    import time
    print "Shuting down..."
    time.sleep(60*60*17)
#    time.sleep(60)
    #os.system('systemctl suspend') 
    os.system('systemctl shutdown') 

