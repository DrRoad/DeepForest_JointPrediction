import rasterio
import geopandas
import shapely
import numpy as np


def project(raster_path, boxes):
    """Project image-coordinate boxes into utm based on the .tif file metadata"""

    with rasterio.open(raster_path) as dataset:
        bounds = dataset.bounds
        pixelSizeX, pixelSizeY = dataset.res

    #subtract origin. Recall that numpy origin is top left! Not bottom left.
    boxes["left"] = (boxes["xmin"] * pixelSizeX) + bounds.left
    boxes["right"] = (boxes["xmax"] * pixelSizeX) + bounds.left
    boxes["top"] = bounds.top - (boxes["ymin"] * pixelSizeY)
    boxes["bottom"] = bounds.top - (boxes["ymax"] * pixelSizeY)

    # combine column to a shapely Box() object, save shapefile
    boxes['geometry'] = boxes.apply(
        lambda x: shapely.geometry.box(x.left, x.top, x.right, x.bottom), axis=1)
    boxes = geopandas.GeoDataFrame(boxes, geometry='geometry')

    #set projection, (see dataset.crs) hard coded here
    boxes.crs = {'init': "{}".format(dataset.crs)}

    #Select columns
    boxes = boxes[["left", "bottom", "right", "top", "score", "label", "geometry"]]
    return boxes


def non_zero_99_quantile(x):
    """Get height quantile of all cells that are no zero in a numpy array"""
    mdata = np.ma.masked_where(x < 0.5, x)
    mdata = np.ma.filled(mdata, np.nan)
    percentile = np.nanpercentile(mdata, 99)
    return (percentile)
