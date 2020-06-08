DeepForest_JointPrediction
==============================

[![Build Status](https://travis-ci.org/weecology/DeepForest_JointPrediction.svg?branch=master)](https://travis-ci.org/weecology/DeepForest)

# Roadmap

The aim of this repo is to provide reproducible analysis for analysis of tree mortality and forest dynamics at the Orway-Swisher Biological Station surveyed by the National Ecological Observatory Network's Airborne Platform. NEON's data archive potentially goes back to 2013, but with varying degrees of quality.

* Individual tree detection at the site level using the DeepForest (https://deepforest.readthedocs.io/) algorithm. We will also assess the value of joint temporal predictions for the same area across years.
* Evaluation of individual level methods against multiple years of image-annotated data, as well as individual field stems from the OSBS megaplot.
* A two class deep learning model (RGB (+ Hyperspetral?)) to classify each tree crown as 'Alive' or "Dead".
* A multi-class ensemble model to classify each "Alive" crown to species.
* A between year tree-fall detector

# Potential Questions

* How many trees fall each year? What was the estimated height and crown area by species?
* How many trees died/fell after major hurricane events such as Hurricane Irma 2017
* How many trees died/fell during proscribed burns on the site?

# Installation

```
https://github.com/weecology/DeepForest_JointPrediction.git
cd DeepForest_JointPrediction
conda env create -f=environment.yml
```

