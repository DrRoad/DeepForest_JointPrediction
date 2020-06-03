#test submission document
import pytest
import os

from DeepForestJointPrediction import evaluate
from DeepForestJointPrediction.data import generate_benchmark

@pytest.fixture()
def model():
    model = evaluate.load_model()
    return model


@pytest.fixture()
def annotation_csv():
    current_folder = os.path.dirname(os.path.abspath(__file__))
    rgb_dir = os.path.join(current_folder,"data/benchmark/RGB/")
        
    annotation_csv = generate_benchmark.run(rgb_dir=rgb_dir, savedir=rgb_dir)
    
    return annotation_csv


@pytest.fixture()
def chm_dir():
    current_folder = os.path.dirname(os.path.abspath(__file__))
    return os.path.join(current_folder, "data/benchmark/CHM/")


def test_load_model():
    model = evaluate.load_model()
    assert model.prediction_model


def test_predict_images(model, annotation_csv):
    boxes = evaluate.predict_images(model, annotation_csv)
    assert not boxes.empty
    assert all(
        boxes.columns == ["plot_name", "xmin", "ymin", "xmax", "ymax", "score", "label"])


def test_run_single_year(model, annotation_csv, chm_dir):
    boxes = evaluate.run(annotation_csv=annotation_csv, chm_dir=chm_dir, joint=False)
    assert all(
        boxes.columns == ["plot_name", "xmin", "ymin", "xmax", "ymax", "score", "label"])

    assert not boxes.empty


def test_run_joint(model, annotation_csv, chm_dir):
    boxes = evaluate.run(annotation_csv=annotation_csv, chm_dir=chm_dir, joint=True)
    assert all(
        boxes.columns == ["plot_name", "xmin", "ymin", "xmax", "ymax", "score", "label"])
