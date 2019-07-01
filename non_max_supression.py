# -*- coding: utf-8 -*-
"""
Created on Mon Feb 25 21:30:52 2019

@author: siddh
"""
import numpy as np
from config import Configuration
from util import intersection_over_union

# GRID_SIZE = Configuration.GRID_SIZE
# ANCHORS = Configuration.ANCHORS
# N_ANCHORS = Configuration.ANCHORS.shape[0]
# CLASS_LABELS = Configuration.CLASS_LABELS
# N_CLASS_LABELS = len(CLASS_LABELS.keys())
# LEAKY_RELU_ALPHA = Configuration.LEAKY_RELU_ALPHA
# PREDICTIONS_PER_CELL = N_ANCHORS*(5+N_CLASS_LABELS)
# GRID_CELL_LOCATIONS = Configuration.GRID_CELL_LOCATIONS
# BATCH_SIZE = Configuration.BATCH_SIZE


def non_max_suppression2(bounding_boxes,iou_threshold):
    """
    The bounding boxes of shape [GRID_SIZE*GRID_SIZE*N_ANCHORS,5+N_CLASS_LABELS]
    """
    #bounding_boxes = __discard_boxes_by_obj(bounding_boxes,objectness_threshold)
    bounding_boxes = __choose_max_obj_boxes(bounding_boxes,200)
    result = []
    while bounding_boxes.shape[0] > 0:
        bnd_box_max,  bounding_boxes = __pop_highest_objectness_bndbox(bounding_boxes)
        result.append(bnd_box_max)
        bounding_boxes = __discard_boxes_by_iou(bnd_box_max,bounding_boxes,iou_threshold)
    return np.array(result)

def non_max_suppression(bounding_boxes,iou_threshold,objectness_threshold):
    """
    The bounding boxes of shape [GRID_SIZE*GRID_SIZE*N_ANCHORS,5+N_CLASS_LABELS]
    """
    bounding_boxes = __discard_boxes_by_obj(bounding_boxes,objectness_threshold)
    #bounding_boxes = __choose_max_obj_boxes(bounding_boxes,100)
    result = []
    while bounding_boxes.shape[0] > 0:
        bnd_box_max,  bounding_boxes = __pop_highest_objectness_bndbox(bounding_boxes)
        result.append(bnd_box_max)
        bounding_boxes = __discard_boxes_by_iou(bnd_box_max,bounding_boxes,iou_threshold)
    return np.array(result)

def __discard_boxes_by_obj(bounding_boxes, objectness_threshold):
    temp = []
    for i in range(bounding_boxes.shape[0]):
        if bounding_boxes[i,4] > objectness_threshold:
            x = bounding_boxes[i]
            temp.append(x)
    return np.array(temp)

def __pop_highest_objectness_bndbox(bounding_boxes):
    indx_highest = 0
    highest_obj = bounding_boxes[0,4]
    for i in range(bounding_boxes.shape[0]):
        bnd_box = bounding_boxes[i]
        if bnd_box[4] > highest_obj:
            indx_highest = i
            highest_obj = bnd_box[4]
    bnd_box = bounding_boxes[indx_highest]
    bounding_boxes = np.delete(bounding_boxes,indx_highest,axis=0)
    return bnd_box, bounding_boxes

def __choose_max_obj_boxes(bounding_boxes,max_num):
    max_boxes = []
    for i in range(max_num):
        max_obj = 0
        max_bnd = 0
        for j in range(bounding_boxes.shape[0]):
            if bounding_boxes[j,4] > max_obj:
                max_obj = bounding_boxes[j,4]
                max_bnd = j
        max_boxes.append(bounding_boxes[max_bnd])
        bounding_boxes = np.concatenate((bounding_boxes[:j,:],bounding_boxes[j+1:,:]))
        
    return np.array(max_boxes)
def __discard_boxes_by_iou(bnd_box, bounding_boxes,threshold):
    result = []
    for i in range(bounding_boxes.shape[0]):
        bnd_box2 = bounding_boxes[i]
        if intersection_over_union(bnd_box[:4],bnd_box2[:4]) < threshold:
            result.append(bnd_box2)
    return np.array(result)



if __name__ == "__main__":  
    x = np.random.uniform(size=(GRID_SIZE*GRID_SIZE*N_ANCHORS,5+N_CLASS_LABELS))
    y = non_max_suppression(x,0.1,0.6)