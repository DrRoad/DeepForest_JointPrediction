"""
Calculate mAP for a set of predicted and ground truth boxes
Inspired by https://github.com/fizyr/keras-retinanet/blob/45db917de178f2409f69698ac984a4fa9a6c5f3a/keras_retinanet/utils/eval.py#L30
"""
import pandas as pd
import numpy as np
from shapely.geometry import box

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
        AP = calculate_AP(true_boxes[true_boxes.label == label], predicted_boxes[predicted_boxes.label == label], iou_threshold=iou_threshold)
        APs.append(AP)
    mAP = np.mean(APs)
    print("Mean Average Precision is {:.3f}".format(mAP))
    
    return mAP
    
def calculate_AP(true_boxes, predicted_boxes, iou_threshold=0.5):
    """
    Calculate average precision given two pandas frames of predictions
    Args:
        true_boxes: a pandas dataframe with the columns plot_name, xmin, xmax, ymin, ymax, label
        predicted_boxes: a pandas dataframe with the columns plot_name, xmin, xmax, ymin, ymax, label, score
    """
    
    #TODO check args
    
    #Create shapely bounding box objects for both dataframes
    true_boxes['geometry'] = true_boxes.apply(
        lambda x: box(x.xmin, x.ymin, x.xmax, x.ymax), axis=1)
    
    predicted_boxes['geometry'] = predicted_boxes.apply(
        lambda x: box(x.xmin, x.ymin, x.xmax, x.ymax), axis=1)
    
    #Sort predictions by score, such that higher scoring boxes get matching first
    predicted_boxes = predicted_boxes.sort_values(by=["score"])
    
    #Counters
    false_positives = np.zeros((0,))
    true_positives  = np.zeros((0,))   
    num_annotations = 0.0
    scores = []
    
    #For each image
    images = true_boxes.plot_name.unique()
    for plot_name in images:
        #Filter detections and annotations
        predicted_image_boxes = predicted_boxes[predicted_boxes.plot_name == plot_name]
        true_image_boxes = true_boxes[true_boxes.plot_name == plot_name]
        
        #add to annotations
        num_annotations +=true_image_boxes.shape[0]
        
        #holder to make sure annotations are not double detected for an image
        detected_annotations = []
    
        for index, row in predicted_image_boxes.iterrows():
            
            #Add score to counter
            scores.append(row["score"])
                                            
            #Calculate IoU
            overlaps = calculate_IoU(row["geometry"], true_boxes["geometry"])
            assigned_annotation = np.argmax(overlaps)
            max_overlap         = overlaps[assigned_annotation]
        
            if max_overlap >= iou_threshold and assigned_annotation not in detected_annotations:
                false_positives = np.append(false_positives, 0)
                true_positives  = np.append(true_positives, 1)
                detected_annotations.append(assigned_annotation)
            else:
                false_positives = np.append(false_positives, 1)
                true_positives  = np.append(true_positives, 0)    
    
    # sort by score, 
    indices         = np.argsort(-np.array(scores))
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
    
    print("{} instances of class {} with Average Precision: {:.3f}".format(num_annotations, true_boxes.label.unique(), average_precision))
    
    return average_precision