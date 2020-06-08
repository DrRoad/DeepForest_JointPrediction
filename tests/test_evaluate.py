#test submission document
import pytest
import os

from DeepForestJointPrediction import evaluate
from DeepForestJointPrediction import generate_benchmark

@pytest.fixture()
def model():
    model = evaluate.load_model()
    return model


@pytest.fixture()
def image_csv():
    current_folder = os.path.dirname(os.path.abspath(__file__))
    rgb_dir = os.path.join(current_folder,"data/benchmark/RGB/")
        
    image_csv = generate_benchmark.run(rgb_dir=rgb_dir, savedir=rgb_dir)
    
    return image_csv

@pytest.fixture()
def chm_dir():
    current_folder = os.path.dirname(os.path.abspath(__file__))
    return os.path.join(current_folder, "data/benchmark/CHM/")

def test_load_model():
    model = evaluate.load_model()
    assert model.prediction_model

def test_predict_images(model, image_csv):
    boxes = evaluate.predict_images(model, image_csv)
    assert not boxes.empty
    assert all(
        boxes.columns == ["plot_name", "xmin", "ymin", "xmax", "ymax", "score", "label"])

def test_run_single_year(model, image_csv, chm_dir):
    mAP = evaluate.run(image_csv=image_csv, chm_dir=chm_dir, joint=False)
    assert mAP > 0 & mAP < 1

def test_run_joint(model, image_csv, chm_dir):
    boxes = evaluate.run(image_csv=image_csv, chm_dir=chm_dir, joint=True)
    assert mAP > 0 & mAP < 1
