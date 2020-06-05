"""
Evaluate joint temporal predictions on benchmark data
"""
import pandas as pd
import os
from .joint import cross_year_prediction, filter_by_height
from .utilities import load_model

    
def predict_images(model, image_csv):
    """Generate single year predictions"""
    boxes = model.predict_generator(image_csv)
    return boxes

def run(image_csv, chm_dir, saved_model=None, joint=True):
    """Run evaluation on benchmark data using a joint year model
    Args:
        image_csv: a csv file following the keras-retinanet predict generator format, see generate_benchmark
        chm_dir: path to canopy-height model rasters
        joint: run cross year predictions
        model: a saved keras-retinanet model
    Returns:
        filtered_boxes: a pandas dataframe with predicted bounding boxes
    """
    model = load_model(saved_model)
    boxes = predict_images(model, image_csv)
    if joint:
        joint_boxes = cross_year_prediction(boxes)
    else:
        filtered_boxes = boxes

    rgb_dir = os.path.dirname(image_csv)
    filtered_boxes = filter_by_height(boxes, rgb_dir, chm_dir)

    return filtered_boxes


if __name__ == "__main__":
    benchmark_csv = "/home/b.weinstein/NeonTreeEvaluation/evaluation/RGB/benchmark_annotations.csv"
    CHM_dir = "/home/b.weinstein/NeonTreeEvaluation/evaluation/CHM/"

    run(benchmark_csv=benchmark_csv, CHM_dir=CHM_dir)
