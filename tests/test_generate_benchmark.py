#test submission document
import os
import sys
sys.path.append(os.path.dirname(os.getcwd()))
import pandas as pd

from ..DeepForest_JointPrediction.data import generate_benchmark


def test_generate_benchmark():
    generate_benchmark.run(rgb_dir="data/benchmark/RGB/", savedir="output/")

    fname = os.path.join("output/benchmark_annotations.csv")
    assert os.path.exists(fname)

    boxes = pd.read_csv(fname)
    assert not boxes.empty
    assert boxes.shape[1] == 6
