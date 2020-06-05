"""Evaluate joint temporal predictions on benchmark data"""
import pandas as pd
import numpy as np
import os

from deepforest import deepforest
from deepforest import predict
from .joint import cross_year_prediction, filter_by_height
from .utilities import load_model


def predict_tile(model, tilelist):
    """Generate single year predictions for a list of tiles"""
    allboxes = []
    for tile in tilelist:
        boxes = model.predict_tile(tile)
        allboxes.append(boxes)

    return allboxes


def run(tile_list, chm_dir, weights=None, joint=True):
    """Run evaluation on benchmark data using a joint year model
    Args:
        tile_list: a list of tiles to predict
        chm_dir: path to canopy-height model rasters
        joint: run cross year predictions
        model: a saved keras-retinanet model
    Returns
        filtered_boxes: a pandas dataframe with predicted bounding boxes
    """
    model = load_model(weights)
    boxes = predict_tile(model, tile_list)

    if joint:
        joint_boxes = cross_year_prediction(boxes)
    else:
        filtered_boxes = boxes

    rgb_dir = os.path.dirname(image_csv)
    filtered_boxes = filter_by_height(boxes, rgb_dir, chm_dir)

    return filtered_boxes
