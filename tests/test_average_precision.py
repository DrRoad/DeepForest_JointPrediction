#Test average precision
import os
import pytest
import pandas as pd
import numpy as np
from deepforest import deepforest
from deepforest import utilities
from keras_retinanet.preprocessing import csv_generator
from keras_retinanet.utils.eval import _get_detections
from keras_retinanet.utils.eval import _get_annotations
from keras_retinanet.utils.anchors import compute_overlap

from deepforest.retinanet_train import parse_args

from DeepForestJointPrediction import average_precision
from DeepForestJointPrediction import generate_benchmark
from shapely.geometry import box
import geopandas

@pytest.fixture()
def boxes():
    """Generate two boxes for train and test. First set are good matches, the second set are not"""
    true_boxes = pd.DataFrame({"plot_name" : ["1.jpg", "1.jpg"], "xmin": [3,10],"ymin": [5,12], "xmax":[7,14],"ymax": [9,16],"label":["Tree","Tree"]})
    predicted_boxes = pd.DataFrame({"plot_name": ["1.jpg", "1.jpg", "1.jpg"], "xmin": [3,12,20],"ymin": [5,15,20], "xmax":[7,19,25],"ymax": [9,20,25],"label":["Tree","Tree","Tree"],"score":[0.75,0.85,0.05]})
    
    predicted_boxes['geometry'] = predicted_boxes.apply(
        lambda x: box(x.xmin, x.ymin, x.xmax, x.ymax), axis=1)
    
    true_boxes['geometry'] = true_boxes.apply(
        lambda x: box(x.xmin, x.ymin, x.xmax, x.ymax), axis=1)    
    
    return true_boxes, predicted_boxes

@pytest.fixture()
def multi_label_boxes():
    """Generate two boxes for train and test. First set are good matches, the second set are not"""
    true_boxes = pd.DataFrame({"plot_name" : ["1.jpg", "1.jpg"],"xmin": [3,10],"ymin": [5,12], "xmax":[7,14],"ymax": [9,16],"label":["Tree","Bird"]})
    predicted_boxes = pd.DataFrame({"plot_name" : ["1.jpg", "1.jpg", "1.jpg"], "xmin": [3,12,20],"ymin": [5,15,20], "xmax":[7,19,25],"ymax": [9,20,25],"label":["Tree","Bird","Tree"],"score":[0.75,0.85,0.05]})
    
    predicted_boxes['geometry'] = predicted_boxes.apply(
        lambda x: box(x.xmin, x.ymin, x.xmax, x.ymax), axis=1)
    
    true_boxes['geometry'] = true_boxes.apply(
        lambda x: box(x.xmin, x.ymin, x.xmax, x.ymax), axis=1)    
    
    return true_boxes, predicted_boxes

@pytest.fixture()
def annotations_csv():
    current_folder = os.path.dirname(os.path.abspath(__file__))
    rgb_dir = os.path.join(current_folder,"data/benchmark/RGB/")

    annotations_csv = generate_benchmark.run(rgb_dir=rgb_dir, savedir=rgb_dir)

    return annotations_csv

def test_exact_iou():
    
    #One box entire inside another
    prediction = box(minx=0, miny=0, maxx=10, maxy=10)
    truth = box(minx=0, miny=0, maxx=5, maxy=5)
    source_iou = average_precision.IoU(truth, prediction)
    
    assert source_iou == 25/100
    
    true_array = np.expand_dims(np.array(truth.bounds),axis=0)
    prediction_array = np.expand_dims(np.array([*prediction.bounds]), axis=0)
    retinanet_iou = compute_overlap(prediction_array,true_array)
    
    assert retinanet_iou[0][0] == source_iou
    
    #Poorly overlapping
    prediction = box(minx=0, miny=0, maxx=5, maxy=5)
    truth = box(minx=4, miny=4, maxx=9, maxy=9)
    source_iou = average_precision.IoU(truth, prediction)
    
    assert source_iou == (1/49)
    
    true_array = np.expand_dims(np.array(truth.bounds),axis=0)
    prediction_array = np.expand_dims(np.array([*prediction.bounds]), axis=0)
    retinanet_iou = compute_overlap(prediction_array,true_array)
    
    assert retinanet_iou == source_iou
        
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

def test_calculate_mAP(multi_label_boxes):
    true_boxes, predicted_boxes = multi_label_boxes    
    mAP = average_precision.calculate_mAP(true_boxes, predicted_boxes)
    
    assert (mAP > 0) & (mAP < 1)

@pytest.fixture()
def from_retinanet(annotations_csv):
    model = deepforest.deepforest()
    model.use_release()    
    
    """use the keras retinanet source to create detections"""
    ### Keras retinanet
    # Format args for CSV generator
    classes_file = utilities.create_classes(annotations_csv)
    arg_list = utilities.format_args(annotations_csv, classes_file, model.config)
    args = parse_args(arg_list)

    # create generator
    validation_generator = csv_generator.CSVGenerator(
        args.annotations,
        args.classes,
        image_min_side=args.image_min_side,
        image_max_side=args.image_max_side,
        config=args.config,
        shuffle_groups=False,
    )
    
    all_detections     = _get_detections(validation_generator, model.prediction_model, score_threshold=args.score_threshold, max_detections=100)
    all_annotations = _get_annotations(validation_generator)
    
    return all_detections, all_annotations, validation_generator

@pytest.fixture()
def from_source(annotations_csv):
    """Use the code in this repo to get detections"""
    model = deepforest.deepforest()
    model.use_release()    
    predicted_boxes = model.predict_generator(annotations_csv)
    true_boxes = pd.read_csv(annotations_csv, names=["image_path","xmin","ymin","xmax","ymax","label"])
    true_boxes["plot_name"] = true_boxes.image_path.apply(lambda x: os.path.splitext(x)[0])
    
    return predicted_boxes, true_boxes

def test_equal_detections(from_retinanet, from_source, annotations_csv):
    """Assert in the input data is the same for both eval metrics"""
    all_detections, all_annotations, _ = from_retinanet
    predicted_boxes, true_boxes = from_source

    average_precision_deepforest = average_precision.calculate_AP(true_boxes, predicted_boxes)
    
    #Parse into a pandas frame
    evalboxes = []
    for i in all_detections:
        for j in i:
            for k in j:
                evalboxes.append(k)
            
    evalboxes = np.array(evalboxes)
    evaldf = pd.DataFrame(evalboxes)
    evaldf.columns = ["xmin","xmax","ymin","ymax","score"]
    evaldf["label"] = "Tree"
    
    #Assert same number of predictions
    assert predicted_boxes.shape[0] == evaldf.shape[0]
    
    #assert same scores and first position
    assert evaldf.score.sort_values().reset_index(drop=True).equals(predicted_boxes.score.sort_values().reset_index(drop=True)) 
    assert evaldf.xmin.sort_values().reset_index(drop=True).equals(predicted_boxes.xmin.sort_values().reset_index(drop=True))

@pytest.fixture()
def source_true_positives(from_source):
    predicted_boxes, true_boxes = from_source
    iou_threshold = 0.5
    
    ##Source
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
    boxes = []
    iou_score = []
    
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
            boxes.append(row["geometry"])
                                            
            #Calculate IoU
            overlaps = average_precision.calculate_IoU(row["geometry"], true_boxes["geometry"])
            assigned_annotation = np.argmax(overlaps)
            max_overlap         = overlaps[assigned_annotation]
            iou_score.append(max_overlap)
            if max_overlap >= iou_threshold and assigned_annotation not in detected_annotations:
                false_positives = np.append(false_positives, 0)
                true_positives  = np.append(true_positives, 1)
                detected_annotations.append(assigned_annotation)
            else:
                false_positives = np.append(false_positives, 1)
                true_positives  = np.append(true_positives, 0)    

    result = pd.DataFrame({"box":boxes,"match":true_positives,"iou":iou_score})

    return result

@pytest.fixture()
def retinant_true_positive(from_retinanet):
    all_detections, all_annotations, validation_generator = from_retinanet 
    iou_threshold = 0.5
    boxes = []
    
    ##Retinanet
    #walk through entire eval to find why the next test fails
    # process detections and annotations
    for label in range(validation_generator.num_classes()):
        if not validation_generator.has_label(label):
            continue

        false_positives = np.zeros((0,))
        true_positives  = np.zeros((0,))
        scores          = np.zeros((0,))
        num_annotations = 0.0
        iou_score = []

        for i in range(validation_generator.size()):
            detections           = all_detections[i][label]
            annotations          = all_annotations[i][label]
            num_annotations     += annotations.shape[0]
            detected_annotations = []

            for d in detections:
                scores = np.append(scores, d[4])
                boxes.append(box(*d[:4]))

                if annotations.shape[0] == 0:
                    false_positives = np.append(false_positives, 1)
                    true_positives  = np.append(true_positives, 0)
                    continue
                
                overlaps            = compute_overlap(np.expand_dims(d, axis=0), annotations)
                assigned_annotation = np.argmax(overlaps, axis=1)
                max_overlap         = overlaps[0, assigned_annotation]
                iou_score.append(max_overlap)

                if max_overlap >= iou_threshold and assigned_annotation not in detected_annotations:
                    false_positives = np.append(false_positives, 0)
                    true_positives  = np.append(true_positives, 1)
                    detected_annotations.append(assigned_annotation)
                else:
                    false_positives = np.append(false_positives, 1)
                    true_positives  = np.append(true_positives, 0)    
                    
    result = pd.DataFrame({"box":boxes,"match":true_positives,"iou":iou_score})
        
    return result

def test_equal_true_positives(retinant_true_positive, source_true_positives):
    """
    compare the number of matches
    """
    retinant_true_positive["bounds"] = retinant_true_positive.box.apply(lambda x: x.bounds)
    source_true_positives["bounds"] = source_true_positives.box.apply(lambda x: x.bounds)
    
    retinant_true_positive = retinant_true_positive.sort_values(["bounds"]).reset_index(drop=True)
    source_true_positives = source_true_positives.sort_values(["bounds"]).reset_index(drop=True)
    
    assert retinant_true_positive.box.equals(source_true_positives.box)
    assert retinant_true_positive.shape == source_true_positives.shape

    merged = pd.merge(retinant_true_positive, source_true_positives, suffixes=["_retinanet","_source"], on="bounds")    
    
    incorrect = merged[~(merged.match_retinanet==merged.match_source)]
    assert incorrect.empty 
    
    assert np.sum(retinant_true_positive.match) == np.sum(source_true_positives.match)

def test_mAP_deepforest(annotations_csv):
    model = deepforest.deepforest()
    model.use_release()
    
    #Original retinanet implementation
    mAP_retinanet  = model.evaluate_generator(annotations=annotations_csv)
    
    #This repo implementation
    predicted_boxes = model.predict_generator(annotations_csv)
    true_boxes = pd.read_csv(annotations_csv, names=["image_path","xmin","ymin","xmax","ymax","label"])
    true_boxes["plot_name"] = true_boxes.image_path.apply(lambda x: os.path.splitext(x)[0])
    mAP = average_precision.calculate_mAP(true_boxes, predicted_boxes, iou_threshold=0.5)
    
    assert mAP_retinanet == mAP
    
    