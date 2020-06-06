"""
Calculate mAP for a set of predicted and ground truth boxes
Inspired by https://github.com/fizyr/keras-retinanet/blob/45db917de178f2409f69698ac984a4fa9a6c5f3a/keras_retinanet/utils/eval.py#L30
"""
import pandas as pd
import numpy as np

from shapely.geometry import box
from rtree.index import Index as rtree_index

def IoU(true_box, predicted_box):
    """Calculate intersection-over-union scores for a pair of boxes
    Args:
        true_box: shapely box of the true bounding box position
        predicted_box: predicted shapely box
    Returns:
        iou: float numeric intersection-over-union
    """
    intersection = true_box.intersection(predicted_box).area
    union = true_box.union(predicted_box).area
    iou = intersection/float(union)
    
    return iou

def calculate_IoU(predicted_box, true_boxes):
    """given a predicted box, find IoU to all matching true boxes"""
    IoU_list = [ ]
    for true_box in true_boxes:
        IoU_list.append(IoU(true_box, predicted_box))
        
    return IoU_list

def _compute_ap(recall, precision):
    """ Compute the average precision, given the recall and precision curves.
    Code originally from https://github.com/rbgirshick/py-faster-rcnn.
    # Arguments
        recall:    The recall curve (list).
        precision: The precision curve (list).
    # Returns
        The average precision as computed in py-faster-rcnn.
    """
    # correct AP calculation
    # first append sentinel values at the end
    mrec = np.concatenate(([0.], recall, [1.]))
    mpre = np.concatenate(([0.], precision, [0.]))

    # compute the precision envelope
    for i in range(mpre.size - 1, 0, -1):
        mpre[i - 1] = np.maximum(mpre[i - 1], mpre[i])

    # to calculate area under PR curve, look for points
    # where X axis (recall) changes value
    i = np.where(mrec[1:] != mrec[:-1])[0]

    # and sum (\Delta recall) * prec
    ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])
    
    return ap

def calculate_mAP(true_boxes, predicted_boxes, iou_threshold=0.5):
    labels = true_boxes.label.unique()
    APs = [ ]
    for label in labels:
        AP = calculate_AP(true_boxes[true_boxes.label == label], predicted_boxes[predicted_boxes.label == label])
        print("Average Precision for {} is {}".format(label, AP))
        APs.append(AP)
    mAP = np.mean(AP)
    print("Mean Average Precision is {}".format(label, mAP))
    
    return mAP
    
def calculate_AP(true_boxes, predicted_boxes, iou_threshold=0.5):
    """Calculate average precision given two pandas frames of predictions"""
    
    #Create shapely bounding box objects for both dataframes
    true_boxes['geometry'] = true_boxes.apply(
        lambda x: shapely.geometry.box(x.xmin, x.ymin, x.xmax, x.ymax), axis=1)
    
    predicted_boxes['geometry'] = predicted_boxes.apply(
        lambda x: shapely.geometry.box(x.xmin, x.ymin, x.xmax, x.ymax), axis=1)
    
    #create rtree index for fast lookup of ground truth
    idx = rtree_index()
    for pos, cell in enumerate(true_boxes["geometry"]):
        idx.insert(pos, cell.bounds)    
    
    for index, row in predicted_boxes.iterrows():
        #Find all ground truth that match prediction
        matched_truth=[true_boxes["geometry"][x] for x in idx.intersection(row["geometry"])]
        
        #Calculate IoU
        overlaps = calculate_IoU(row["geometry"], matched_truth)
        assigned_annotation = np.argmax(overlaps)
        max_overlap         = overlaps[0, assigned_annotation]
    
        if max_overlap >= iou_threshold and assigned_annotation not in detected_annotations:
            false_positives = np.append(false_positives, 0)
            true_positives  = np.append(true_positives, 1)
            detected_annotations.append(assigned_annotation)
        else:
            false_positives = np.append(false_positives, 1)
            true_positives  = np.append(true_positives, 0)    
    
    # sort by score
    indices         = np.argsort(-scores)
    false_positives = false_positives[indices]
    true_positives  = true_positives[indices]

    # compute false positives and true positives
    false_positives = np.cumsum(false_positives)
    true_positives  = np.cumsum(true_positives)

    # compute recall and precision
    recall    = true_positives / num_annotations
    precision = true_positives / np.maximum(true_positives + false_positives, np.finfo(np.float64).eps)

    # compute average precision
    average_precision  = _compute_ap(recall, precision)
    
    return average_precision