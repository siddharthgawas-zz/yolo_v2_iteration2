# -*- coding: utf-8 -*-
"""
Created on Thu Feb 21 13:45:12 2019

@author: siddh
"""
#%%
from config import Configuration
import pandas as pd
import numpy as np
from util import cvt_coord_to_diagonal, intersection_over_union, cvt_coord_to_mid_point
import matplotlib.pyplot as plt
#%%
#TRAIN_CSV_NAME = "train_bndbox.csv"
#TEST_CSV_NAME = "test_bndbox.csv"

#%%
    
class KMeans:
    
    def __init__(self,k,e,max_iteration):
        self.k = k
        self.e = e
        self.max_iteration = max_iteration
        #Initialize clusters
        self.cluster_vectors = np.random.uniform(size=(self.k,2))
        
    def train(self,data):
        #Initialization
        iteration = 1
        avg_error = 10000000000
        
        #Initialize cluster vectors to random points from data
        idx = np.random.randint(low=0,high=data.shape[0],size=self.k)
        self.cluster_vectors = data[idx,:]
        
        #Cluster Labels for data
        cluster_labels = np.zeros(shape=(data.shape[0]),dtype=np.int32)
        
        while True:
            #divide the set of training vectors into K clusters using
            #minimum error criteria
            print('Iteration:',iteration)
            for i in range(data.shape[0]):
                bnd_box = data[i]
                bnd_box = np.concatenate(([0,0],bnd_box))
                bnd_box = cvt_coord_to_diagonal(bnd_box)
                error_iou = 0
                #assign bnd_box a cluster based on iou
                for j in range(self.k):
                    training_vector = np.concatenate(([0,0],
                                                      self.cluster_vectors[j]))
                    
                    training_vector = cvt_coord_to_diagonal(training_vector)
                    
                    e_iou = 1.0 - intersection_over_union(bnd_box, training_vector)
                    if j == 0:
                        error_iou = e_iou
                        cluster_labels[i] = j
                    elif e_iou < error_iou:
                        error_iou = e_iou
                        cluster_labels[i] = j
            
            #compute average distortion
            error = 0.0
            xc = [0 for _ in range(self.k)]
            yc = [0 for _ in range(self.k)]
            coordinate_counts = [0 for _ in range(self.k)]
            average_iou = 0
            for i in range(data.shape[0]):
                bnd_box = data[i]
                
                xc[cluster_labels[i]] += bnd_box[0]
                yc[cluster_labels[i]] += bnd_box[1]
                coordinate_counts[cluster_labels[i]]+=1
                
                bnd_box = np.concatenate(([0,0],bnd_box))
                bnd_box = cvt_coord_to_diagonal(bnd_box)
                
                training_vector = self.cluster_vectors[cluster_labels[i]]
                training_vector = np.concatenate(([0,0],training_vector))
                training_vector = cvt_coord_to_diagonal(training_vector)
                iou = intersection_over_union(bnd_box,training_vector)
                average_iou+=iou
                error += (1.0 - iou)
                
            error = error / data.shape[0]
            average_iou = average_iou / data.shape[0]
            
            #Make new cluster vectors ie find new centroids 
            new_clusters = []
            for i in range(self.k):
                if coordinate_counts[i] == 0:
                    new_clusters.append([self.cluster_vectors[i][0],self.cluster_vectors[i][1]])
                else:
                    new_clusters.append([xc[i]/coordinate_counts[i],
                                     yc[i]/coordinate_counts[i]])
            self.cluster_vectors = np.array(new_clusters)
            
            if  np.abs(avg_error - error) / error < self.e or iteration >= self.max_iteration:
                break
            else:
                avg_error = error
                iteration+=1
                
        print("K: ",self.k,' AvgError: ', np.abs(avg_error - error) / error,
              'Iterations: ',iteration,'AvgIou: ',average_iou)
        return self.cluster_vectors, cluster_labels, average_iou
