# -*- coding: utf-8 -*-
"""
Created on Thu Feb 21 10:53:31 2019

@author: siddh
"""
import numpy as np
import pandas as pd
import cv2
def cvt_coord_to_mid_point(coordinates):
    """
    [xmin,ymin,xmax,ymax] -> [xc, yc, w, h]
    """
    xc = (coordinates[0] + coordinates[2]) / 2.0
    yc = (coordinates[1] + coordinates[3]) / 2.0
    w = np.abs(coordinates[0] - coordinates[2])
    h = np.abs(coordinates[1] - coordinates[3])
    return pd.Series([xc,yc,w,h],index=['xc','yc','w','h'])

def cvt_coord_to_diagonal(coordinates):
    """
    [xc, yc, w, h]->[xmin,ymin,xmax,ymax]
    """
    xmin = coordinates[0] - coordinates[2]/2.0
    xmax = coordinates[0] + coordinates[2]/2.0
    ymin = coordinates[1] - coordinates[3]/2.0
    ymax = coordinates[1] + coordinates[3]/2.0
    return pd.Series([xmin,ymin,xmax,ymax],index=['xmin','ymin','xmax','ymax'])

def intersection_over_union(boxA, boxB):
    """
    Finds the intersection over union between boxes boxA and boxB
    Parameters
    ---------------------------------------------------------------------------
    boxA: list of 4 coordinates of bounding boxA x1,y1,x2,y2
    boxB: list of 4 coordinates of bounding boxB x1, y1, x2, y2
    Returns
    ---------------------------------------------------------------------------
    Intersection over union of two bounding boxes
    """
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
    interArea = max(0, xB - xA ) * max(0, yB - yA )
    boxAArea = (boxA[2] - boxA[0] ) * (boxA[3] - boxA[1] )
    boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])
    iou = interArea / float(boxAArea + boxBArea - interArea)
    return iou

def scale_boxes(coordinates,sx,sy):
    """
    Scale coordinates [xmin,ymin,xmax,ymax]
    """
    xmin = coordinates[0] *sx
    ymin = coordinates[1]*sy
    xmax = coordinates[2] *sx
    ymax = coordinates[3]*sy
    return pd.Series([xmin,ymin,xmax,ymax],
                     index=['xmin','ymin','xmax','ymax'])

def change_brightness(img, value):
    hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    h, s, v = cv2.split(hsv)

    if value >=0:
        lim = 255 - value
        v[v > lim] = 255
        v[v <= lim] += value
    else:
        lim = -value
        v[v < lim] = 0
        v[v >= lim] -= -value

    final_hsv = cv2.merge((h, s, v))
    img = cv2.cvtColor(final_hsv, cv2.COLOR_HSV2RGB)
    return img
    
    