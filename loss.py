# -*- coding: utf-8 -*-
"""
Created on Fri Feb 22 23:03:45 2019

@author: siddh
"""

import tensorflow as tf
import numpy as np
from dataset import Dataset
from config import Configuration
import pandas as pd
GRID_SIZE = Configuration.GRID_SIZE
N_ANCHORS = Configuration.ANCHORS.shape[0]
CLASS_LABELS = Configuration.CLASS_LABELS
N_CLASS_LABELS = len(CLASS_LABELS.keys())
BATCH_SIZE = Configuration.BATCH_SIZE

LAMBDA_COORD = Configuration.LAMBDA_COORD
LAMBDA_NOOBJ = Configuration.LAMBDA_NOOBJ

C_COLUMN_INDICES = []
BOX_COORD_COLUMN_INDICES = []
LABEL_INDICES = []
j = 4
for i in range(N_ANCHORS):
    C_COLUMN_INDICES.append(j)
    
    BOX_COORD_COLUMN_INDICES.append(j-4)
    BOX_COORD_COLUMN_INDICES.append(j-3)
    BOX_COORD_COLUMN_INDICES.append(j-2)
    BOX_COORD_COLUMN_INDICES.append(j-1)
    for k in range(N_CLASS_LABELS):
        LABEL_INDICES.append(j+k+1)
        
    j = j+N_CLASS_LABELS+5
    
del j

def yolo_loss(y_true, y_pred):
    """
    Shape of y_true: [batch_size,GRID_SIZE,GRID_SIZE,N_ANCHORS*(5+N_CLASS_LABELS)]
    Shape of y_pred: [batch_size,GRID_SIZE,GRID_SIZE,N_ANCHORS*(5+N_CLASS_LABELS)]
    Returns the error of the shape[batch_size,]
    """
    return tf.reduce_mean(reg_loss(y_true,y_pred) + obj_loss(y_true,y_pred) + cls_loss(y_true,y_pred))
#    return tf.reduce_mean(reg_loss(y_true,y_pred) + obj_loss(y_true,y_pred))

def reg_loss(y_true, y_pred):
    #Get the one hot encoded tensor of shape[BATCH_SIZE,GRID_SIZE,GRID_SIZE,N_ANCHORS]
    #which represents the anchors which are reponsible to detect the object
    one_ij = get_responsible_anchors(y_true)
    one_ij = tf.cast(one_ij,dtype=tf.float32)
    
    #Get all the bounding box coordinates from the ground truth and predicted tensor
    true_box_tensor = tf.gather(y_true,BOX_COORD_COLUMN_INDICES,axis=3)
    pred_box_tensor = tf.gather(y_pred,BOX_COORD_COLUMN_INDICES,axis=3)
    
    #Reshape to [BATCH_SIZE,GRID_SIZE,GRID_SIZE,N_ANCHORS,4]
    true_box_tensor = tf.reshape(true_box_tensor,shape=(-1,GRID_SIZE,GRID_SIZE,N_ANCHORS,4))
    pred_box_tensor = tf.reshape(pred_box_tensor,shape=(-1,GRID_SIZE,GRID_SIZE,N_ANCHORS,4))
    
    #Compute the square error of Xc coordinate
    xc_error = tf.square(true_box_tensor[:,:,:,:,0] - pred_box_tensor[:,:,:,:,0])
    #Compute the square error of Yc coordinate
    yc_error = tf.square(true_box_tensor[:,:,:,:,1] - pred_box_tensor[:,:,:,:,1])
    #Compute width error
    w_error = tf.square(tf.sqrt(true_box_tensor[:,:,:,:,2])-tf.sqrt(pred_box_tensor[:,:,:,:,2]))
    #Compute height error
    h_error = tf.square(tf.sqrt(true_box_tensor[:,:,:,:,3])-tf.sqrt(pred_box_tensor[:,:,:,:,3]))
    
    center_error = one_ij*(xc_error + yc_error)
    center_error = tf.reduce_sum(center_error,axis=3)
    center_error = tf.reduce_sum(center_error,axis=2)
    center_error = tf.reduce_sum(center_error,axis=1)
    
    dim_error = one_ij*(w_error+h_error)
    dim_error = tf.reduce_sum(dim_error,axis=3)
    dim_error = tf.reduce_sum(dim_error,axis=2)
    dim_error = tf.reduce_sum(dim_error,axis=1)
    total_error = LAMBDA_COORD*(center_error+dim_error)
    return total_error

def obj_loss(y_true,y_pred):
    one_ij = get_responsible_anchors(y_true)
    
    noobj_one_ij = tf.cast(one_ij,dtype=tf.bool)
    noobj_one_ij = tf.logical_not(noobj_one_ij)
    noobj_one_ij = tf.cast(noobj_one_ij,tf.float32)
    one_ij = tf.cast(one_ij,dtype=tf.float32)
    
    
    true_obj_tensor = tf.gather(y_true,C_COLUMN_INDICES,axis=3)
    pred_obj_tensor = tf.gather(y_pred,C_COLUMN_INDICES,axis=3)
    
    true_obj_tensor = tf.reshape(true_obj_tensor,shape=(-1,GRID_SIZE,GRID_SIZE,N_ANCHORS,1))
    pred_obj_tensor = tf.reshape(pred_obj_tensor, shape=(-1, GRID_SIZE,GRID_SIZE,N_ANCHORS,1))
    
    error = tf.square(true_obj_tensor[:,:,:,:,0] - pred_obj_tensor[:,:,:,:,0])
    obj_error = one_ij*error
    noobj_error = noobj_one_ij*error
    
    obj_error = tf.reduce_sum(obj_error,axis=3)
    obj_error = tf.reduce_sum(obj_error,axis=2)
    obj_error = tf.reduce_sum(obj_error,axis=1)
    
    noobj_error= tf.reduce_sum(noobj_error,axis=3)
    noobj_error= tf.reduce_sum(noobj_error,axis=2)
    noobj_error = tf.reduce_sum(noobj_error,axis=1)
    
    total_error = obj_error +  LAMBDA_NOOBJ*noobj_error
    return total_error

def cls_loss(y_true,y_pred):
    one_ij = get_responsible_anchors(y_true)
    one_ij = tf.cast(one_ij,dtype=tf.float32)
        
    true_cls_tensor = tf.gather(y_true,LABEL_INDICES,axis=3)
    pred_cls_tensor = tf.gather(y_pred,LABEL_INDICES,axis=3)
    true_cls_tensor = tf.reshape(true_cls_tensor,(-1,GRID_SIZE,GRID_SIZE,N_ANCHORS,N_CLASS_LABELS))
    pred_cls_tensor = tf.reshape(pred_cls_tensor,(-1,GRID_SIZE,GRID_SIZE,N_ANCHORS,N_CLASS_LABELS))
    
#    error = -true_cls_tensor*tf.log(pred_cls_tensor)
    error = tf.square(true_cls_tensor-pred_cls_tensor)
    error = tf.reduce_sum(error,axis=4)
    error = one_ij*error
    error = tf.reduce_sum(error,axis=3)
    error = tf.reduce_sum(error,axis=2)
    error = tf.reduce_sum(error,axis=1)
    return error

#def get_responsible_anchors(y_true):
#    """
#    Shape of y_true: [batch_size,GRID_SIZE,GRID_SIZE,N_ANCHORS*(5+N_CLASS_LABELS)]
#    Returns the sparse tensor of shape [batch_size,GRID_SIZE,GRID_SIZE,N_ANCHORS]
#    representing the anchors reponsible for detection
#    """
#    y_true = np.reshape(y_true,newshape=(-1,GRID_SIZE*GRID_SIZE,N_ANCHORS*(5+N_CLASS_LABELS)))
#    one_ij = np.zeros(shape=(y_true.shape[0],GRID_SIZE*GRID_SIZE,N_ANCHORS))
#    
#    c_tensor = np.reshape(y_true[:,:,C_COLUMN_INDICES],newshape=(-1,GRID_SIZE*GRID_SIZE,N_ANCHORS))
#    max_tensor = np.reshape(np.argmax(c_tensor,axis=2),newshape=(-1,GRID_SIZE*GRID_SIZE))
#    c_tensor_1 = np.reshape(np.sum(c_tensor,axis=2),newshape=(-1,GRID_SIZE*GRID_SIZE))
#    c_tensor_1[c_tensor_1 > 0] = 1
#    
#    indices = np.arange(0,GRID_SIZE*GRID_SIZE,1,dtype=np.int32)
#    for i in range(y_true.shape[0]):
#        one_ij[i,indices,max_tensor[i]] = np.multiply(1.0,c_tensor_1[i])
#    return np.reshape(one_ij,(-1,GRID_SIZE,GRID_SIZE,N_ANCHORS))

def get_responsible_anchors(y_true):
    """
    Shape of y_true: [batch_size,GRID_SIZE,GRID_SIZE,N_ANCHORS*(5+N_CLASS_LABELS)]
    Returns the sparse tensor of shape [batch_size,GRID_SIZE,GRID_SIZE,N_ANCHORS]
    representing the anchors reponsible for detection
    """
    
    y_true = tf.reshape(y_true,(-1,GRID_SIZE*GRID_SIZE,N_ANCHORS*(5+N_CLASS_LABELS)))    
    c_tensor = tf.gather(y_true,indices=C_COLUMN_INDICES,axis=2)
    max_tensor = tf.reduce_max(c_tensor,keepdims=True,axis=2)
    max_tensor = tf.equal(c_tensor,max_tensor)
    max_tensor = tf.cast(max_tensor,dtype=tf.int32)
    c_tensor_1 = tf.reshape(tf.reduce_sum(c_tensor,axis=2),shape=(-1,GRID_SIZE*GRID_SIZE))
    
    comparison = tf.greater(c_tensor_1,tf.constant(0.0,dtype=tf.float32))
    c_tensor_1 = tf.cast(comparison,dtype=tf.int32)
    c_tensor_1 = tf.reshape(c_tensor_1,(BATCH_SIZE,GRID_SIZE*GRID_SIZE,1))
    c_tensor_1 = tf.tile(c_tensor_1,[1,1,N_ANCHORS])
    
    return tf.reshape(tf.multiply(c_tensor_1,max_tensor),
                      (BATCH_SIZE,GRID_SIZE,GRID_SIZE,N_ANCHORS))


if __name__ == "__main__":
    dataset = Dataset('data/train','data/test')
    X1, y_true_1 = dataset.next_train_batch()
    X2, y_true_2 = dataset.next_train_batch()
    
    y_true_1 = np.reshape(y_true_1,newshape=(-1,GRID_SIZE,GRID_SIZE,N_ANCHORS*(5+N_CLASS_LABELS)))
    y_true_2 = np.reshape(y_true_2,newshape=(-1,GRID_SIZE,GRID_SIZE,N_ANCHORS*(5+N_CLASS_LABELS)))
    y_pred_1 = np.random.uniform(size=(y_true_1.shape[0],GRID_SIZE,GRID_SIZE,N_ANCHORS*(5+N_CLASS_LABELS)))
    y_pred_2 = np.random.uniform(size=(y_true_2.shape[0],GRID_SIZE,GRID_SIZE,N_ANCHORS*(5+N_CLASS_LABELS)))
    
    y_true = np.append(y_true_1,y_true_2,axis=0)
    y_pred = np.append(y_pred_1,y_pred_2,axis=0)


    y_true_placeholder = tf.placeholder(dtype=tf.float32,shape=(1,GRID_SIZE,GRID_SIZE,N_ANCHORS*(5+N_CLASS_LABELS)))
    y_pred_placeholder = tf.placeholder(dtype=tf.float32,shape=(1,GRID_SIZE,GRID_SIZE,N_ANCHORS*(5+N_CLASS_LABELS)))

    init_var = tf.global_variables_initializer()

    with tf.Session() as sess:
        out =  yolo_loss(y_true_placeholder,y_pred_placeholder)
        _,error = sess.run([init_var,out], feed_dict={y_true_placeholder: y_true_1,y_pred_placeholder: y_pred_1})

        y_true = np.reshape(y_true,(-1,GRID_SIZE*GRID_SIZE,N_ANCHORS*(5+N_CLASS_LABELS)))
        y_pred = np.reshape(y_pred,(-1,GRID_SIZE*GRID_SIZE,N_ANCHORS*(5+N_CLASS_LABELS)))












