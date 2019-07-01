# -*- coding: utf-8 -*-
"""
Created on Thu Feb 21 12:36:18 2019
This is the script to generate csv file of bounding boxes for train and test data
@author: siddh
"""

#%%
import os
from yolov2 import Dataset
import csv
import sys
from config import Configuration as cfg
#%%
__arguments = sys.argv[1:]
TRAIN_DATA_PATH = __arguments[0]
# TEST_DATA_PATH = __arguments[1]
TRAIN_CSV_NAME = __arguments[1]
# TEST_CSV_NAME = __arguments[3]
#%%
dataset = Dataset(TRAIN_DATA_PATH,cfg.IMAGE_WIDTH,cfg.IMAGE_HEIGHT,
                  cfg.GRID_SIZE,cfg.GRID_SIZE,
                  cfg.CLASS_LABELS,[1,2],shuffle=False,img_extensions=['jpg','jpeg','JPG'])
#%%
#Make train csv file
print("Writing csv file..........")
with open(TRAIN_CSV_NAME,'w') as file:
    column_names = ['image_path','label','xmin','ymin','xmax','ymax','xc','yc','w','h']
    writer = csv.DictWriter(file,fieldnames=column_names)
    writer.writeheader()
    for i in range(len(dataset)):
        image_path = os.path.join(TRAIN_DATA_PATH,dataset.data_files.iloc[i,0])
        _, annotation = dataset.get_sample(i)
        row = {}
        for j in range(annotation.shape[0]):
            row['image_path'] = image_path
            row['label'] = annotation.iloc[j,0]
            row['xmin'] = annotation.iloc[j,1]
            row['ymin'] = annotation.iloc[j,2]
            row['xmax'] = annotation.iloc[j,3]
            row['ymax'] = annotation.iloc[j,4]
            row['xc'] = annotation.iloc[j,5]
            row['yc'] = annotation.iloc[j,6]
            row['w'] = annotation.iloc[j,7]
            row['h'] = annotation.iloc[j,8]
            writer.writerow(row)
        print('Processed ',i+1,'out of ',len(dataset),' files')
print('Finished writing csv')
#%%
# dataset = Dataset(TEST_DATA_PATH,cfg.IMAGE_WIDTH,cfg.IMAGE_HEIGHT,
#                   cfg.GRID_SIZE,cfg.GRID_SIZE,
#                   cfg.CLASS_LABELS,[1,2],shuffle=False)
# #%%
# #Make test csv file
# print('Writing test csv')
# with open(TEST_CSV_NAME,'w') as file:
#     column_names = ['image_path','label','xmin','ymin','xmax','ymax','xc','yc','w','h']
#     writer = csv.DictWriter(file,fieldnames=column_names)
#     writer.writeheader()
#     for i in range(len(dataset)):
#         image_path = os.path.join(TRAIN_DATA_PATH,dataset.data_files.iloc[i,0])
#         _, annotation = dataset.get_sample(i)
#         row = {}
#         for j in range(annotation.shape[0]):
#             row['image_path'] = image_path
#             row['label'] = annotation.iloc[j,0]
#             row['xmin'] = annotation.iloc[j,1]
#             row['ymin'] = annotation.iloc[j,2]
#             row['xmax'] = annotation.iloc[j,3]
#             row['ymax'] = annotation.iloc[j,4]
#             row['xc'] = annotation.iloc[j,5]
#             row['yc'] = annotation.iloc[j,6]
#             row['w'] = annotation.iloc[j,7]
#             row['h'] = annotation.iloc[j,8]
#             writer.writerow(row)
#         print('Processed ',i+1,'out of ',len(dataset),' files')
# print('Finished test csv')    

        
        
    


