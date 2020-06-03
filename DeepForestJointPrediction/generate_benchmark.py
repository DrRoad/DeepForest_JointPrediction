import os
import glob
import pandas as pd
from deepforest import utilities


def run(rgb_dir, savedir):
    """Generate a DeepForest annotation file for a directory of xml and rgb tifs"""
    tifs = glob.glob(rgb_dir + "*.tif")
    xmls = [os.path.splitext(x)[0] for x in tifs]
    xmls = ["{}.xml".format(x) for x in xmls]

    if len(xmls) == 0:
        raise IOError("There are no matching .xml files in {}".format(
            os.path.abspath(rgb_dir)))
    #Load and format xmls, not every RGB image has an annotation
    annotation_list = []
    for xml_path in xmls:
        try:
            annotation = utilities.xml_to_annotations(xml_path)
            annotation_list.append(annotation)
        except:
            pass
    benchmark_annotations = pd.concat(annotation_list, ignore_index=True)

    #save evaluation annotations
    fname = os.path.join(savedir + "benchmark_annotations.csv")
    benchmark_annotations.to_csv(fname, index=False, header=None)

    return fname
