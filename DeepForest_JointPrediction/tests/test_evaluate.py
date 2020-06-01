#test submission document
import os
import sys
sys.path.append(os.path.dirname(os.getcwd()))
import pytest

from .. import evaluate
from ..data import generate_benchmark


@pytest.fixture()
def model():
    model = evaluate.load_model()
    return model


@pytest.fixture()
def annotation_csv():
    annotation_csv = generate_benchmark.run(rgb_dir="data/benchmark/RGB/",
                                            savedir="data/benchmark/RGB/")
    return annotation_csv


@pytest.fixture()
def CHM_dir():
    return "data/benchmark/CHM_dir/"


def test_load_model():
    model = evaluate.load_model()
    assert model.prediction_model


def test_predict_images(model, annotation_csv):
    boxes = evaluate.predict_images(model, annotation_csv)
    assert not boxes.empty
    assert all(
        boxes.columns == ["plot_name", "xmin", "ymin", "xmax", "ymax", "score", "label"])


def test_run_single_year(model, annotation_csv, CHM_dir):
    boxes = evaluate.run(model=model,
                         annotation_csv=annotation_csv,
                         CHM_dir=CHM_dir,
                         joint=False)
    assert all(
        boxes.columns == ["plot_name", "xmin", "ymin", "xmax", "ymax", "score", "label"])


def test_run_joint(model, annotation_csv, CHM_dir):
    boxes = evaluate.run(model=model,
                         annotation_csv=annotation_csv,
                         CHM_dir=CHM_dir,
                         joint=True)
    assert all(
        boxes.columns == ["plot_name", "xmin", "ymin", "xmax", "ymax", "score", "label"])
