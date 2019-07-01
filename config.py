# -*- coding: utf-8 -*-
"""
Created on Thu Feb 21 10:07:40 2019

@author: siddh
"""

import numpy as np
import json

__config_main = None
with open('config.json','r') as __f:
    __config_main = json.load(__f)
        
class Configuration:
    """
    Test file configuration contains the parameters to train the model
    """
    pass
#    IMAGE_HEIGHT = 346
#    IMAGE_WIDTH = 346
#    
#    #K-means clustering algorithm parameters
#    K = 5
#    ERROR_THRESHOLD = 1e-6
#    MAX_ITERATION = 1000
#    
#    #Yolo Parameters
#    GRID_SIZE = 13
#    
#    GRID_CELL_LOCATIONS  = []
#    BATCH_SIZE = 1
#    
##Anchors
#    ANCHORS = np.array([[0.31021142, 0.29145264],
# [0.19524535, 0.40350059],
# [0.06133306, 0.09161108],
# [0.19177536, 0.18688395],
# [0.45981438, 0.4935617 ],
# [0.1103404,  0.13231001],
# [0.10667653, 0.2470207 ]])
#
#    
#    LAMBDA_COORD = 5.0
#    LAMBDA_NOOBJ = 0.5
#    LEAKY_RELU_ALPHA = 0.1
#    
#    LEARNING_RATE = 0.0001
#    CLASS_LABELS = {'infected': 0, 'not_infected': 1}

__config = __config_main['CLUSTERING_PARAMS']
Configuration.K = __config['K']
Configuration.ERROR_THRESHOLD = __config['ERROR_THRESHOLD']
Configuration.MAX_ITERATION =  __config['MAX_ITERATION']

__config = __config_main['YOLO_PARAMS']
Configuration.IMAGE_HEIGHT = __config['IMAGE_H']
Configuration.IMAGE_WIDTH = __config['IMAGE_W']
Configuration.GRID_SIZE = __config['GRID_SIZE']
Configuration.BATCH_SIZE = __config['BATCH_SIZE']
Configuration.ANCHORS = np.array(__config['ANCHORS']).reshape(-1,2)
Configuration.LAMBDA_COORD = __config['LAMBDA_COORD']
Configuration.LAMBDA_NOOBJ = __config['LAMBDA_NOOBJ']
Configuration.LEAKY_RELU_ALPHA = __config['LEAKY_RELU_ALPHA']
Configuration.LEARNING_RATE = __config['LEARNING_RATE']
Configuration.CLASS_LABELS = __config['CLASS_LABELS']

 
__grid_cell_len = 1.0 / Configuration.GRID_SIZE
__cx = 0.0
__cy = 0.0
Configuration.GRID_CELL_LOCATIONS = np.array(np.zeros((Configuration.GRID_SIZE,Configuration.GRID_SIZE,2)))
for i in range(Configuration.GRID_SIZE):
    __cx = 0.0
    for j in range(Configuration.GRID_SIZE):
        Configuration.GRID_CELL_LOCATIONS[i,j,:] = [__cx,__cy]
        __cx += __grid_cell_len
    __cy+= __grid_cell_len