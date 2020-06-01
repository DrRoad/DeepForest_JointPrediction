#test submission document
import os
import sys
sys.path.append(os.path.dirname(os.getcwd()))
from .. import evaluate

def test_run(model):
    eval_path = "/Users/ben/Documents/NeonTreeEvaluation/evaluation/RGB/benchmark_annotations.csv"
    CHM_dir = "/Users/ben/Documents/NeonTreeEvaluation/evaluation/CHM/"
    boxes = evaluate.run(annotation_csv=eval_path,CHM_dir)
    assert all(boxes.columns == ["plot_name","xmin","ymin","xmax","ymax","score","label"])