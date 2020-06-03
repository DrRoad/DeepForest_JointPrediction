#test submission document
import os
import sys
import pandas as pd

from DeepForestJointPrediction.data import generate_benchmark


def test_generate_benchmark():
    current_folder = os.path.dirname(os.path.abspath(__file__))
    rgb_dir = os.path.join(current_folder,"data/benchmark/RGB/")

    generate_benchmark.run(rgb_dir=rgb_dir, savedir=os.path.join(current_folder, "output/"))

    fname = os.path.join(current_folder, "output/benchmark_annotations.csv")
    assert os.path.exists(fname)

    boxes = pd.read_csv(fname)
    assert not boxes.empty
    assert boxes.shape[1] == 6
