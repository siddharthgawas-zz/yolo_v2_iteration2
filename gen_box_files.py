# -*- coding: utf-8 -*-
"""
Created on Fri May  3 21:46:18 2019

@author: siddh
"""
#%%
import tensorflow as tf
import numpy as np
from config import Configuration as cfg
import matplotlib.pyplot as plt
import cv2
import pandas as pd
from yolov2 import (YoloNetwork, get_pre_trained_model_vgg16, get_batch,
                    np_cvt_coord_to_diagonal, Dataset)
#%%
tf.enable_eager_execution()
#%%
image_h = 346
image_w = 346
m_grid = 13
n_grid = 13
anchor_dim = (0.44752974, 0.51667883, 0.17638203, 0.16553482, 0.076274  ,
       0.10540195, 0.10295663, 0.23137522, 0.24225889, 0.3452266 )
n_anchors = int(len(anchor_dim) / 2)
class_labels = ['infected','not_infected']
n_classes = len(class_labels)

#%%
#Load  Model along with weights
model = get_pre_trained_model_vgg16(image_w,image_h,m_grid,n_grid
                                    ,n_anchors,n_classes,1,anchor_dim)

model.load_weights('trained_weights/leaf_data_v2/yolo_net_epoch_102.h5')
model.summary()
#%%
def predict(x,max_boxes,iou_thresh,obj_thresh,dataset: Dataset):
    """
    Function predicts and returns bounding boxes
    """
    y_pred = model(x).numpy()
    y_pred = y_pred.reshape(-1,5+n_classes)
    
    y_pred_coord = y_pred[:,0:4]
    y_pred_conf = y_pred[:,4]
    y_pred_classes = y_pred[:,5:]
    
    y_pred_coord = np_cvt_coord_to_diagonal(y_pred_coord.reshape(1,-1,4))[0]
    idx = tf.image.non_max_suppression(y_pred_coord,y_pred_conf,
                                       max_boxes,
                                       iou_threshold=iou_thresh,
                                       score_threshold=obj_thresh)
    if idx.shape[0] == 0:
        idx = tf.image.non_max_suppression(y_pred_coord,y_pred_conf,
                                       1,
                                       iou_threshold=iou_thresh)
    idx = idx.numpy().reshape(-1,1)
    boxes = y_pred_coord[idx].reshape(-1,4)
    boxes = np.clip(boxes,0,1.0)
    conf = y_pred_conf[idx].reshape(-1,1)
    classes = y_pred_classes[idx].reshape(-1,n_classes)
    classes = dataset.one_hot_encoder.inverse_transform(classes).astype(int)
    
    classes = dataset.label_encoder.inverse_transform(classes)
    return boxes, conf, classes
#%%
#Generate files for test data
dataset = Dataset('data/leaf_data_v2/test/',image_w,image_h,m_grid,n_grid,
                       class_labels,anchor_dim,shuffle=False)
n = len(dataset)
gnd_df = pd.DataFrame(columns=['filename','label','xmin','ymin','xmax','ymax'])
pred_df = pd.DataFrame(columns=['filename','label','xmin','ymin','xmax','ymax','objectness'])
file_id = 0
for i in range(n):
    X, y = dataset.get_sample(i)
    y = y [['label','xmin','ymin','xmax','ymax']]
    y['filename'] = str(file_id)
    gnd_df = gnd_df.append(y,sort=False,ignore_index=True)
    
    X,_,_= get_batch(dataset,i,batch_size=1)
    box,conf,labels = predict(X,max_boxes=5,iou_thresh=0.5,obj_thresh=0.2,
                              dataset=dataset)
    df = pd.DataFrame({'filename': [str(file_id) for _ in range(box.shape[0])],
                           'label': labels,
                           'xmin': box[:,0],
                           'ymin': box[:,1],
                           'xmax': box[:,2],
                           'ymax': box[:,3],
                           'objectness': conf[:,0]})
    pred_df = pred_df.append(df,ignore_index=True,sort=False)
    file_id+=1
#%%
gnd_df.to_csv('test_gnd_truth.csv',index=False)
pred_df.to_csv('test_pred.csv',index=False)
#%%
#Generate files for train data
dataset = Dataset('data/leaf_data_v2/train/',image_w,image_h,m_grid,n_grid,
                       class_labels,anchor_dim,shuffle=False)
n = len(dataset)
gnd_df = pd.DataFrame(columns=['filename','label','xmin','ymin','xmax','ymax'])
pred_df = pd.DataFrame(columns=['filename','label','xmin','ymin','xmax','ymax','objectness'])
file_id = 0
for i in range(n):
    X, y = dataset.get_sample(i)
    y = y [['label','xmin','ymin','xmax','ymax']]
    y['filename'] = str(file_id)
    gnd_df = gnd_df.append(y,sort=False,ignore_index=True)
    
    X,_,_= get_batch(dataset,i,batch_size=1)
    box,conf,labels = predict(X,max_boxes=5,iou_thresh=0.5,obj_thresh=0.2,
                              dataset=dataset)
    df = pd.DataFrame({'filename': [str(file_id) for _ in range(box.shape[0])],
                           'label': labels,
                           'xmin': box[:,0],
                           'ymin': box[:,1],
                           'xmax': box[:,2],
                           'ymax': box[:,3],
                           'objectness': conf[:,0]})
    pred_df = pred_df.append(df,ignore_index=True,sort=False)
    file_id+=1
#%%
gnd_df.to_csv('train_gnd_truth.csv',index=False)
pred_df.to_csv('train_pred.csv',index=False)
 #%%