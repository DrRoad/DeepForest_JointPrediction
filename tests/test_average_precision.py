#Test average precision
import pytest
import pandas as pd
import numpy as np

from DeepForestJointPrediction import average_precision
from shapely.geometry import box

@pytest.fixture()
def boxes():
    """Generate two boxes for train and test. First set are good matches, the second set are not"""
    true_boxes = pd.DataFrame({"xmin": [3,10],"ymin": [5,12], "xmax":[7,14],"ymax": [9,16],"label":["Tree","Tree"]})
    predicted_boxes = pd.DataFrame({"xmin": [3,12,20],"ymin": [5,15,20], "xmax":[7,19,25],"ymax": [9,20,25],"label":["Tree","Tree","Tree"],"score":[0.75,0.85,0.05]})
    
    predicted_boxes['geometry'] = predicted_boxes.apply(
        lambda x: box(x.xmin, x.ymin, x.xmax, x.ymax), axis=1)
    
    true_boxes['geometry'] = true_boxes.apply(
        lambda x: box(x.xmin, x.ymin, x.xmax, x.ymax), axis=1)    
    
    return true_boxes, predicted_boxes

def test_IoU(boxes):
    true_boxes, predicted_boxes = boxes
    
    #Exact match
    iou = average_precision.IoU(true_boxes.geometry.iloc[0], true_boxes.geometry.iloc[0])
    assert iou == 1.0
    
    # Similiar boxes
    iou = average_precision.IoU(true_boxes.geometry.iloc[0], predicted_boxes.geometry.iloc[0])
    assert iou == 1.0
    
    # Different boxes
    iou = average_precision.IoU(true_boxes.geometry.iloc[1], predicted_boxes.geometry.iloc[1])
    assert iou < 0.5    
    
def test_calculate_IOU(boxes):
    """Return a list of IoU equal to the length of the true boxes"""
    true_boxes, predicted_boxes = boxes    
    iou_list = average_precision.calculate_IoU(predicted_boxes.geometry.iloc[0], true_boxes.geometry.values)
    
    assert len(iou_list) == 2
    assert np.argmax(iou_list) == 0

def test_calculate_AP(boxes):
    true_boxes, predicted_boxes = boxes    
    AP = average_precision.calculate_AP(true_boxes, predicted_boxes)
    
    assert (AP > 0) & (AP < 1)
    