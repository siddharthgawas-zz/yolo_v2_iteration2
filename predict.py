# -*- coding: utf-8 -*-
"""
Created on Sun Apr 28 01:35:40 2019

@author: siddh
"""
#%%
import tensorflow as tf
from loss import yolo_loss
from dataset import Dataset, ImageDataset
import numpy as np
from network import get_custom_model, get_pre_trained_model
from config import Configuration
from non_max_supression import non_max_suppression, non_max_suppression2
import matplotlib.pyplot as plt
import util
import cv2
#%%
IMAGE_H, IMAGE_W = Configuration.IMAGE_HEIGHT, Configuration.IMAGE_WIDTH
N_CLASSES = len(Configuration.CLASS_LABELS.keys())
N_ANCHORS = Configuration.ANCHORS.shape[0]
GRID_SIZE = Configuration.GRID_SIZE
#%%
#Load  Model along with weights
model = get_pre_trained_model()
model.load_weights('trained_weights\leaf_data_v2\model_weights_16-nosoftmax.h5')
model.summary()
#%%
dataset = ImageDataset('data/leaf_data_v2/train','data/leaf_data_v2/test',
                       shuffle=True, histogram_equ=False,brightness_range=None)
#%%
x, y = dataset.next_test_batch()
#%%
y_pred = model.predict(x)
#%%
y_pred_1 = y_pred[0]
x_1 = x[0]
#%%
y_pred_1 = np.reshape(y_pred_1,(GRID_SIZE*GRID_SIZE*N_ANCHORS,N_CLASSES+5))
for i in range(y_pred_1.shape[0]):
    y_pred_1[i,0:4] = util.cvt_coord_to_diagonal(y_pred_1[i,0:4])
y_pred_1[:,0:4] = np.clip(y_pred_1[:,0:4],a_min=0.0,a_max=1.0)
#%%
y_pred_1 = non_max_suppression(y_pred_1,0.5,0.3)
#%%
img = x_1
for i in range(y_pred_1.shape[0]):
    box = y_pred_1[i,0:4]
    box =  np.ceil(IMAGE_H*box).astype(int)
    pt1 = (box[0],box[1])
    pt2 =(box[2],box[3])
    cv2.rectangle(img,pt1,pt2,(255,0,0,),3)
plt.imshow(img)
#%%
#PARAMETERS
IMAGE_H, IMAGE_W = Configuration.IMAGE_HEIGHT, Configuration.IMAGE_WIDTH
N_CLASSES = len(Configuration.CLASS_LABELS.keys())
N_ANCHORS = Configuration.ANCHORS.shape[0]
GRID_SIZE = Configuration.GRID_SIZE
#Loading Single Image
x = cv2.imread(r'test_images\DSC_0181 S9 upward curl with puckering distroted tips.JPG')
original_h, original_w , _ = x.shape
#By Default cv2 Loads in BGR format so rotate to RGB
x = cv2.cvtColor(x,cv2.COLOR_BGR2RGB)
original_image = x
#Resize Image
x = cv2.resize(x,(IMAGE_H,IMAGE_W))
#Normalize
x = x / 255.0
#Reshape into [BATCH_SIZE,H,W,C]
x = np.reshape(x,(-1,IMAGE_H,IMAGE_W,3))
#%%
y_pred = model.predict(x)
#%%
y_pred_1 = y_pred[0]
x_1 = x[0]
#%%
y_pred_1 = np.reshape(y_pred_1,(GRID_SIZE*GRID_SIZE*N_ANCHORS,N_CLASSES+5))
for i in range(y_pred_1.shape[0]):
    y_pred_1[i,0:4] = util.cvt_coord_to_diagonal(y_pred_1[i,0:4])
y_pred_1[:,0:4] = np.clip(y_pred_1[:,0:4],a_min=0.0,a_max=3.0)
#%%
#Here each row contains (x1,y1,w,h,objectness_score,class_scores)
y_pred_1 = non_max_suppression(y_pred_1,0.5,0.1)
#%%
img = original_image
for i in range(y_pred_1.shape[0]):
    box = y_pred_1[i,0:4].reshape(-1,4)
    box[:,0] = original_w*box[:,0]
    box[:,1] = original_h*box[:,1]
    box[:,2] = original_w*box[:,2]
    box[:,3] = original_h*box[:,3]
    box = box.reshape(-1)
    box = np.ceil(box).astype(int)
    pt1 = (box[0],box[1])
    pt2 =(box[2],box[3])
    cv2.rectangle(img,pt1,pt2,(255,0,0,),10)
plt.imshow(img)
#%%
img = x_1
for i in range(y_pred_1.shape[0]):
    box = y_pred_1[i,0:4]
    box =  np.ceil(IMAGE_H*util.cvt_coord_to_diagonal(box)).astype(int)
    pt1 = (box[0],box[1])
    pt2 =(box[2],box[3])
    cv2.rectangle(img,pt1,pt2,(255,0,0,),3)
plt.imshow(img)
#%%
result = []
for i in range(y_pred_1.shape[0]):
    box = y_pred_1[i,0:4]
    box = np.array(util.cvt_coord_to_diagonal(box)).reshape(-1,2)
    box[:,0] = original_w*box[:,0]
    box[:,1] = original_h*box[:,1] 
    box = box.reshape(-1)
    box = np.ceil(box).astype(int)
    obj = {
            'box': box.tolist(),
            'infected': y_pred_1[i,5],
            'not_infected': y_pred_1[i,6],
            'leafness': y_pred_1[i,4]
            }
    result.append(obj)