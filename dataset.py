# -*- coding: utf-8 -*-
"""
Created on Mon Feb 18 22:48:47 2019

@author: siddh
"""
import os
import pandas as pd
import re
import cv2
from config import Configuration
import xml.etree.ElementTree as ET
import matplotlib.pyplot as plt
from util import cvt_coord_to_mid_point, scale_boxes, cvt_coord_to_diagonal,intersection_over_union
from util import change_brightness
import numpy as np
from sklearn.utils import shuffle

class Dataset:
    
    def __init__(self,train_path,test_path,shuffle=True,
                 brightness_range=None, histogram_equ = False):
        self.histogram_equ = histogram_equ
        self.brightness_range = brightness_range
        self.train_path = train_path
        self.test_path = test_path
        self.train_data_files = pd.DataFrame(columns=['image','annotation'])
        self.test_data_files = pd.DataFrame(columns=['image','annotation'])
        self.shuffle = shuffle
        #Create pandas data frame containing image filename and corresponding
        #annotation file name
        self.__create_data_file_frame()
        
        #Initiazlize the file pointers to 0
        self.cur_train_file = 0
        self.cur_test_file = 0
        
    def __create_data_file_frame(self):
        file_list = os.listdir(self.train_path)
        data = []
        for file_name in file_list:
            if file_name.endswith('.jpg'):
                name = re.search('(.*)(.jpg)$',file_name).groups()[0]
                if name+'.xml' in file_list:
                    data.append({'image':file_name,'annotation':name+'.xml'})
                else:
                    data.append({'image':file_name,'annotation':''})
                
        self.train_data_files =self.train_data_files.append(data)
        
        file_list = os.listdir(self.test_path)
        data = []
        for file_name in file_list:
            if file_name.endswith('.jpg'):
                name = re.search('(.*)(.jpg)$',file_name).groups()[0]
                if name+'.xml' in file_list:
                    data.append({'image':file_name,'annotation':name+'.xml'})
                else:
                    data.append({'image':file_name,'annotation':''})
                
        self.test_data_files =self.test_data_files.append(data)
        
        #Shufle dataframes
        if self.shuffle:
            self.train_data_files = shuffle(self.train_data_files)
            self.test_data_files = shuffle(self.test_data_files)
        
    
    def next_train_sample(self):
        #Load the image from the training data 
        img = cv2.imread(os.path.join(
                self.train_path,
                self.train_data_files.iloc[self.cur_train_file,0]))
        #Load the corresponding annotation file from training data
        annotations = read_annotation_file(
                os.path.join(self.train_path,
                             self.train_data_files.iloc[self.cur_train_file,1]
                             ))
        #increment train file index
        self.increment_train_file_index()
        #return the next sample image and corresponding pandas frame of annotations
        return self.__next_sample(img,annotations)
     
    
    def next_test_sample(self):
         #Load the image from the test data 
        img = cv2.imread(os.path.join(
                self.test_path,
                self.test_data_files.iloc[self.cur_test_file,0]))
        #Load the corresponding annotation file from test data
        annotations = read_annotation_file(
                os.path.join(self.test_path,
                             self.test_data_files.iloc[self.cur_test_file,1]
                             ))
        #increment test file index
        self.increment_test_file_index()
        #return the next sample image and corresponding pandas frame of annotations
        return self.__next_sample(img,annotations)
    
    
    def __next_sample(self,img,annotations):
        #Get the image height and width
        height, width, _ = img.shape
        #Convert image from BGR to RGB space
        img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
        #Resize image as per the height in configuration file
        img = cv2.resize(img,(Configuration.IMAGE_WIDTH,
                                Configuration.IMAGE_HEIGHT))
        #histogram equalization
        if self.histogram_equ:
            img = cv2.cvtColor(img,cv2.COLOR_RGB2Lab)
            img[:,:,0] = cv2.equalizeHist(img[:,:,0])
            img = cv2.cvtColor(img,cv2.COLOR_Lab2RGB)
        
        #chnage brightness
        if self.brightness_range is not None:
            value = np.random.randint(self.brightness_range[0],
                                  self.brightness_range[1]+1)
            img = change_brightness(img,value)
        #Data frame to hold coordinates in xc, yc, w, h format
        df = pd.DataFrame(columns=['xc','yc','w','h'])
        
        #Iterate over each bounding box to normalize it and convert to xc,yc, w, h format
        for i in range(annotations.shape[0]):
            #This is done since some files have wrong values of height and width
            annotations.iloc[i,5] = width
            annotations.iloc[i,6] = height
            #normalize with respect to width and height
            annotations.iloc[i,1] = annotations.iloc[i,1] / annotations.iloc[i,5]
            annotations.iloc[i,2] = annotations.iloc[i,2] / annotations.iloc[i,6]
            annotations.iloc[i,3] = annotations.iloc[i,3] / annotations.iloc[i,5]
            annotations.iloc[i,4] = annotations.iloc[i,4] / annotations.iloc[i,6]
            #convert bounding box to xc, yc, w, h format
            cvt_coordinates = cvt_coord_to_mid_point(annotations.iloc[i,1:5])
            #append it to df
            df = df.append(cvt_coordinates,ignore_index=True)
        #Append converted coordinates to original annotations
        annotations = pd.concat([annotations,df],axis=1)
        #Drop img_width and img_height as they are not needed
        annotations = annotations.drop(columns=['img_width','img_height'])
        
        return img / 255.0, annotations
    
    def next_train_batch(self):
        img, annotations = self.next_train_sample()
        predictions_per_cell = Configuration.ANCHORS.shape[0]*(len(Configuration.CLASS_LABELS.keys())+5)
        #annotations = annotations[['label','xc','yc','w','h']]
        y_true = pd.DataFrame(np.zeros(shape=(
                Configuration.GRID_SIZE*Configuration.GRID_SIZE,predictions_per_cell)))
        #Get the size of the cell
        cell_size = 1.0 / Configuration.GRID_SIZE
        for i in range(annotations.shape[0]):
            l = 0
            m = 0
            cx = 0.0
            cy = 0.0
            gnd_truth_box = annotations.iloc[i]
            while cx <= gnd_truth_box[5]:
                cx+=cell_size
                l+=1
            cx -= cell_size
            l-=1
            while cy <= gnd_truth_box[6]:
                cy+=cell_size
                m+=1
            cy -= cell_size
            m-=1
            
            xc = gnd_truth_box[5]
            yc = gnd_truth_box[6]
#            xc = gnd_truth_box[5] - cx
#            yc = gnd_truth_box[6] - cy
            w = gnd_truth_box[7]
            h = gnd_truth_box[8]
            labels = []
            for label in Configuration.CLASS_LABELS.keys():
                if label == gnd_truth_box[0]:
                    labels.append(1)
                else:
                    labels.append(0)
            data = []
            for j in range(Configuration.ANCHORS.shape[0]):
                anchor_box = Configuration.ANCHORS[j]
                anchor_box =np.append(gnd_truth_box[5:7],anchor_box,axis=0)
                anchor_box = cvt_coord_to_diagonal(anchor_box)
                c = intersection_over_union(anchor_box,gnd_truth_box[1:5])
#                c = 1.0
                data = data  + [xc,yc,w,h,c]+labels
            y_true.iloc[Configuration.GRID_SIZE*m+l] = pd.Series(data)
        
        #y true contains [xc,yc,w,h,c]+one_hot_labels
        return img, np.reshape(np.array(y_true),newshape=(
                Configuration.GRID_SIZE,
                Configuration.GRID_SIZE,
                predictions_per_cell))
        
    def next_test_batch(self):
        img, annotations = self.next_test_sample()
        predictions_per_cell = Configuration.ANCHORS.shape[0]*(len(Configuration.CLASS_LABELS.keys())+5)
        #annotations = annotations[['label','xc','yc','w','h']]
        y_true = pd.DataFrame(np.zeros(shape=(
                Configuration.GRID_SIZE*Configuration.GRID_SIZE,predictions_per_cell)))
        #Get the size of the cell
        cell_size = 1.0 / Configuration.GRID_SIZE
        for i in range(annotations.shape[0]):
            l = 0
            m = 0
            cx = 0.0
            cy = 0.0
            gnd_truth_box = annotations.iloc[i]
            while cx <= gnd_truth_box[5]:
                cx+=cell_size
                l+=1
            cx -= cell_size
            l-=1
            while cy <= gnd_truth_box[6]:
                cy+=cell_size
                m+=1
            cy -= cell_size
            m-=1
            
            xc = gnd_truth_box[5]
            yc = gnd_truth_box[6]
#            xc = gnd_truth_box[5] - cx
#            yc = gnd_truth_box[6] - cy
            w = gnd_truth_box[7]
            h = gnd_truth_box[8]
            labels = []
            for label in Configuration.CLASS_LABELS.keys():
                if label == gnd_truth_box[0]:
                    labels.append(1)
                else:
                    labels.append(0)
            data = []
            for j in range(Configuration.ANCHORS.shape[0]):
                anchor_box = Configuration.ANCHORS[j]
                anchor_box =np.append(gnd_truth_box[5:7],anchor_box,axis=0)
                anchor_box = cvt_coord_to_diagonal(anchor_box)
                c = intersection_over_union(anchor_box,gnd_truth_box[1:5])
#                c = 1.0
                data = data  + [xc,yc,w,h,c]+labels
            y_true.iloc[Configuration.GRID_SIZE*m+l] = pd.Series(data)
        
        #y true contains [xc,yc,w,h,c]+one_hot_labels
        return img, np.reshape(np.array(y_true),newshape=(
                Configuration.GRID_SIZE,
                Configuration.GRID_SIZE,
                predictions_per_cell))
    
    def increment_train_file_index(self):
        self.cur_train_file = (self.cur_train_file+1)%self.train_data_files.shape[0]
    
    def increment_test_file_index(self):
        self.cur_test_file = (self.cur_test_file+1)%self.test_data_files.shape[0]
        
    def get_train_data_size(self):
        return self.train_data_files.shape[0]
    def get_test_data_size(self):
        return self.test_data_files.shape[0]
    
    def cur_train_image_path(self):
        return os.path.join(self.train_path,
                            self.train_data_files.iloc[self.cur_train_file][0])
    def cur_test_image_path(self):
        return os.path.join(self.test_path,
                            self.test_data_files.iloc[self.cur_test_file][0])

def read_annotation_file(file_path):
    """
    @Param file_path: Pascal VOC xml file path
    @Return panda frame containing class name and bounding boxes and original 
    height and width of image
    ['label','xmin','ymin','xmax','ymax', 'img_width','img_height']
    """
    xml = ET.parse(file_path)
    root = xml.getroot()
    data = []
    height = int(root.find('size').find('height').text,10)
    width = int(root.find('size').find('width').text,10)
    for object in root.findall('object'):
        name = object.find('name').text
        xmin = int(object.find('bndbox').find('xmin').text,10)
        ymin = int(object.find('bndbox').find('ymin').text,10)
        xmax = int(object.find('bndbox').find('xmax').text,10)
        ymax = int(object.find('bndbox').find('ymax').text,10)
        data.append([name,xmin,ymin,xmax,ymax,height,width])
    df = pd.DataFrame(data=data,columns=['label','xmin','ymin','xmax','ymax',
                                         'img_width','img_height'])
    return df

class ImageDataset:
    def __init__(self,train_path,test_path,shuffle = True, 
                 brightness_range=None, histogram_equ=False):
        self.dataset = Dataset(train_path,test_path,shuffle,brightness_range,
                               histogram_equ)
    
    def next_train_batch(self):
        X = []
        y = []
        for i in range(Configuration.BATCH_SIZE):
            img, ann = self.dataset.next_train_batch()
            X.append(img)
            y.append(ann)
        X = np.array(X)
        y = np.array(y)
        return X, y
        
    def next_test_batch(self):
        X = []
        y = []
        for i in range(Configuration.BATCH_SIZE):
            img, ann = self.dataset.next_test_batch()
            X.append(img)
            y.append(ann)
        X = np.array(X)
        y = np.array(y)
        return X, y
    
    
if __name__ == '__main__':
    dataset = Dataset('data/pascal_voc/train','data/pascal_voc/test',shuffle=True,
                      brightness_range=[-30,30],histogram_equ=False)
    img, ann = dataset.next_train_batch()
    ann_2 = ann[0,0,:]
    plt.imshow(img)






























