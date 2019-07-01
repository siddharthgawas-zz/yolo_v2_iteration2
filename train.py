# -*- coding: utf-8 -*-
"""
Created on Sun Feb 24 22:32:17 2019

@author: siddh
"""

#from network import model
#%%
import tensorflow as tf
from loss import yolo_loss
from keras.optimizers import SGD, Adam
from dataset import Dataset, ImageDataset
import numpy as np
from network import get_custom_model, get_pre_trained_model
from config import Configuration
from keras.models import load_model
from network import output_layer
#%%
EPOCHS = 10
STEPS_PER_EPOCH = 1000
GRID_SIZE = Configuration.GRID_SIZE
ANCHORS = Configuration.ANCHORS
N_ANCHORS = Configuration.ANCHORS.shape[0]
CLASS_LABELS = Configuration.CLASS_LABELS
N_CLASS_LABELS = len(CLASS_LABELS.keys())
LEAKY_RELU_ALPHA = Configuration.LEAKY_RELU_ALPHA
PREDICTIONS_PER_CELL = N_ANCHORS*(5+N_CLASS_LABELS)
GRID_CELL_LOCATIONS = Configuration.GRID_CELL_LOCATIONS
IMAGE_H, IMAGE_W = Configuration.IMAGE_HEIGHT, Configuration.IMAGE_WIDTH
BATCH_SIZE = Configuration.BATCH_SIZE
LEARNING_RATE = Configuration.LEARNING_RATE
#%%
model = get_pre_trained_model()
model.summary()
model.compile(optimizer=Adam(lr=LEARNING_RATE),loss=yolo_loss)
model.load_weights('trained_weights/model_weights_16_leaf_v2.h5')
#%%
dataset = ImageDataset('data/leaf_data_v2/train','data/leaf_data_v2/test',shuffle=True)
x, y = dataset.next_train_batch()
#%%
for i in range(EPOCHS):
    print("Epoch ",i+1)
    for j in range(STEPS_PER_EPOCH):
        X, y = dataset.next_train_batch()
        X = np.reshape(X,newshape=(BATCH_SIZE,Configuration.IMAGE_HEIGHT,Configuration.IMAGE_WIDTH,3))
        y = np.reshape(y,newshape=(BATCH_SIZE,GRID_SIZE,GRID_SIZE,PREDICTIONS_PER_CELL))
        model.train_on_batch(X,y)
        if (j) % 100 == 0:
                print("Step {}".format(j))
                X_test, y_test = dataset.next_test_batch()
                X_test = np.reshape(X_test,(BATCH_SIZE,Configuration.IMAGE_HEIGHT,Configuration.IMAGE_WIDTH,3))
                y_test = np.reshape(y_test,(BATCH_SIZE,GRID_SIZE,GRID_SIZE,PREDICTIONS_PER_CELL))
                l1 = model.evaluate(X_test,y_test)
                print('Val Loss:{}'.format(l1))
    model.save_weights('trained_weights/leaf_v2_model.h5')
#%%    
print("Finished Training")        
model.save_weights('trained_weigths/leaf_v2_model.h5')
