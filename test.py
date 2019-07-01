# -*- coding: utf-8 -*-
import unittest

from util import (cvt_coord_to_diagonal, cvt_coord_to_mid_point,
                  intersection_over_union, scale_boxes
                  )

class TestUtil(unittest.TestCase):
    
    def setUp(self):
        pass
    
    def tearDown(self):
        pass
    
    def test_cvt_coord_to_diagonal(self):
        x = [3,8,5,1]
        y_true = [0.5,7.5,5.5,8.5]
        y = cvt_coord_to_diagonal(x).tolist()
        
        self.assertEqual(y_true,y)
        
    def test_cvt_coord_to_mid_point(self):
        x = [0.5,7.5,5.5,8.5]
        y_true = [3,8,5,1]
        y = cvt_coord_to_mid_point(x).tolist()
        self.assertEqual(y_true,y)
    
    def test_iou(self):
        
        a = [3,5,10,8]
        b = [4,6,8,15]
        iou_true = 0.1633
        iou = intersection_over_union(a, b)
        self.assertAlmostEqual(iou_true,iou,places=4)
        iou = intersection_over_union(b,a)
        self.assertAlmostEqual(iou_true,iou,places=4)
        
        a = [3,5,10,8]
        b = [11,5,15,8]
        iou = intersection_over_union(a, b)
        iou_true = 0.0
        self.assertAlmostEqual(iou_true,iou,places=4)
        
        a = [0,1,13,12]
        b = [3,5,10,8]
        iou_true = 0.1469
        iou = intersection_over_union(a,b)
        self.assertAlmostEqual(iou_true,iou,places=4)
                
        a = [6,1,8,12]
        b = [3,5,10,8]
        iou_true = 0.1622
        iou = intersection_over_union(a,b)
        self.assertAlmostEqual(iou_true,iou,places=4)
        
        a = [3,5,10,8]
        b = [3.5,5.5,10.5,8.5]
        iou_true = 0.6311
        iou = intersection_over_union(a,b)
        self.assertAlmostEqual(iou_true,iou,places=4)
        
    
    def test_scale_boxes(self):
        sx = 2
        sy = 3
        box = [3,5,10,8]
        true_scaled_box = [6,15,20,24]
        scaled_box = scale_boxes(box,sx,sy).tolist()
        self.assertEqual(scaled_box,true_scaled_box)
        
        
if __name__ == '__main__':
    unittest.main()