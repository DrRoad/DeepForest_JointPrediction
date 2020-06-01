#test submission document
import os
import sys
sys.path.append(os.path.dirname(os.getcwd()))
import pandas as pd

from ..data.generate_benchmark import run


def test_generate_benchmark():
    run(rgb_dir="../../ext_data/raw/", savedir="output/")

    fname = os.path.join("output/benchmark_annotations.csv")
    assert os.path.exists(fname)

    boxes = pd.read_csv(fname)
    assert not boxes.empty
    assert boxes.shape[1] == 6
