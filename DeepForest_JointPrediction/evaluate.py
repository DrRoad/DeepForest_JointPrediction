"""Evaluate joint temporal predictions on benchmark data"""
import pandas as pd
import numpy as np
import os
import rasterstats
import tensorflow as tf

from deepforest import deepforest
from deepforest import predict
from .utilities import project, non_zero_99_quantile


def load_model(saved_model=None):
    if saved_model:
        model = deepforest.deepforest(weights=saved_model)
    else:
        model = deepforest.deepforest()
        model.use_release()

    return model


def predict_images(model, annotation_csv):
    """Generate single year predictions"""
    boxes = model.predict_generator(annotation_csv)
    return boxes


def joint_prediction(sess, predicted_boxes):
    #tensorflow sessions
    new_boxes, new_scores, new_labels = predict.non_max_suppression(
        sess,
        predicted_boxes[["xmin", "ymin", "xmax", "ymax"]].values,
        predicted_boxes.score.values,
        predicted_boxes.label.values,
        max_output_size=predicted_boxes.shape[0],
        iou_threshold=0.1)

    #Recreate box dataframe
    image_detections = np.concatenate([
        new_boxes,
        np.expand_dims(new_scores, axis=1),
        np.expand_dims(new_labels, axis=1)
    ],
                                      axis=1)

    mosaic_df = pd.DataFrame(image_detections,
                             columns=["xmin", "ymin", "xmax", "ymax", "score", "label"])
    mosaic_df.label = mosaic_df.label.str.decode("utf-8")

    print("{} predictions kept after non-max suppression".format(mosaic_df.shape[0]))

    return mosaic_df


def cross_year_prediction(boxes):
    """Group images by plot and perform joint prediction"""
    grouped = boxes.groupby('plot_name')
    jointdf = []

    #Start a tensorflow session for non-max suppression
    sess = tf.Session()
    for name, group in grouped:
        prediction = joint_prediction(sess, group)
        jointdf.append(prediction)
    jointdf = pd.concat(jointdf)

    return jointdf


def filter_by_height(boxes, rgb_dir, CHM_dir, min_height=3):
    """Limit predictions to crowns that have a LiDAR-derived height of atleast 3m
    Args:
        boxes: pandas dataframe from deepforest predict_generator
    Returns:
        threshold_boxes: a pandas dataframe filtered by height
    """
    boxes_grouped = boxes.groupby('plot_name')
    plot_groups = [boxes_grouped.get_group(x) for x in boxes_grouped.groups]

    #Project
    threshold_boxes = []
    for x in plot_groups:
        plot_name = x.plot_name.unique()[0]

        #Look up RGB image for projection
        image_path = "{}/{}.tif".format(rgb_dir, plot_name)
        result = project(image_path, x)

        #Extract heights
        chm_path = "{}_CHM.tif".format(os.path.join(CHM_dir, plot_name))
        try:
            height_dict = rasterstats.zonal_stats(result,
                                                  chm_path,
                                                  stats="mean",
                                                  add_stats={'q99': non_zero_99_quantile})
        except Exception as e:
            print("{} raises {}".format(plot_name, e))
            continue

        x["height"] = [g["q99"] for g in height_dict]

        #Merge back to the original frames
        threshold_boxes.append(x)

    threshold_boxes = pd.concat(threshold_boxes)
    threshold_boxes = threshold_boxes[threshold_boxes.height > min_height]

    threshold_boxes = threshold_boxes[[
        "plot_name", "xmin", "ymin", "xmax", "ymax", "score", "label"
    ]]

    return threshold_boxes


def run(annotation_csv, chm_dir, saved_model=None, joint=True):
    """Run evaluation on benchmark data using a joint year model
    Args:
        annotation_csv: a csv file following the keras-retinanet predict generator format
        chm_dir: path to canopy-height model rasters
        joint: run cross year predictions
        model: a saved keras-retinanet model
    Returns
        filtered_boxes: a pandas dataframe with predicted bounding boxes
    """
    model = load_model(saved_model)

    boxes = predict_images(model, annotation_csv)

    if joint:
        joint_boxes = cross_year_prediction(boxes)
    else:
        filtered_boxes = boxes

    rgb_dir = os.path.dirname(annotation_csv)
    filtered_boxes = filter_by_height(boxes, rgb_dir, chm_dir)

    return filtered_boxes


if __name__ == "__main__":
    benchmark_csv = "/home/b.weinstein/NeonTreeEvaluation/evaluation/RGB/benchmark_annotations.csv"
    CHM_dir = "/home/b.weinstein/NeonTreeEvaluation/evaluation/CHM/"

    run(benchmark_csv=benchmark_csv, CHM_dir=CHM_dir)
