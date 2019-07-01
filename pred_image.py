# -*- coding: utf-8 -*-
"""
Created on Fri Nov  2 19:08:52 2018

@author: NH
"""

from keras.models import load_model
from keras.preprocessing import image
import numpy as np
from keras import backend as K
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



class Predict:
    def __init__(self):
        self.fruit_index = {'Apple': 0,
         'Apricot': 1,
         'Avocado': 2,
         'Banana': 3,
         'Cocos': 4,
         'Dates': 5,
         'Grape': 6,
         'Guava': 7,
         'Kiwi': 8,
         'Lemon': 9,
         'Limes': 10,
         'Lychee': 11,
         'Mango': 12,
         'Orange': 13,
         'Papaya': 14,
         'Peach': 15,
         'Pineapple': 16,
         'Plum': 17,
         'Pomegranate': 18,
         'Raspberry': 19,
         'Strawberry': 20,
         'Walnut': 21}

    def predict_class(self,folder,file):
        # dimensions of our images
        img_width, img_height = 100, 100

        # load the model we saved
        model = load_model(r'D:\Keras Models\mdlFruitRec_v5')


        # predicting images
        file_address = folder + '\\' + file
        print(file_address)
        img = image.load_img(file_address, target_size=(img_width, img_height))
        x = image.img_to_array(img)
        x = np.expand_dims(x, axis=0)
        np.set_printoptions(precision=10)
        images = np.vstack([x])
        classes = model.predict_classes(images, batch_size=10).tolist()
        probabilities = model.predict_proba(images,batch_size=10).tolist()
        K.clear_session()
        prediction_result = {}
        prediction_result['class']=classes[0]
        prediction_result['probabilities']=probabilities
        for name,id in self.fruit_index.items():
            if classes[0]==id:
                prediction_result['name']=name
        result = []
        result.append(prediction_result)
        return result

    def predict_disease(self,path):
        model = get_pre_trained_model()
        model.load_weights('trained_weights/model_weights_16.h5')
        # model.summary()
        
        IMAGE_H, IMAGE_W = Configuration.IMAGE_HEIGHT, Configuration.IMAGE_WIDTH
        N_CLASSES = len(Configuration.CLASS_LABELS.keys())
        N_ANCHORS = Configuration.ANCHORS.shape[0]
        GRID_SIZE = Configuration.GRID_SIZE
        # Loading Single Image
        x = cv2.imread(path)
        original_h, original_w , _ = x.shape
        # By Default cv2 Loads in BGR format so rotate to RGB
        x = cv2.cvtColor(x, cv2.COLOR_BGR2RGB)
        # Resize Image
        x = cv2.resize(x, (IMAGE_H, IMAGE_W))
        # Normalize
        x = x / 255.0
        # Reshape into [BATCH_SIZE,H,W,C]
        x = np.reshape(x, (-1, IMAGE_H, IMAGE_W, 3))
        y_pred = model.predict(x)
        K.clear_session()
        y_pred_1 = y_pred[0]
        x_1 = x[0]
        y_pred_1 = non_max_suppression(np.reshape(y_pred_1, (GRID_SIZE * GRID_SIZE * N_ANCHORS, 5 + N_CLASSES)), 0.4,
                                       0.2)
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
                    'infected': float(y_pred_1[i,5]),
                    'not_infected': float(y_pred_1[i,6]),
                    'objectness': float(y_pred_1[i,4])
                    }
            result.append(obj)
        return result
