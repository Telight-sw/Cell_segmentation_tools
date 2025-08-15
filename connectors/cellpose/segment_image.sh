#!/bin/bash

PATH_TO_CELLPOSE_MODEL=$1
INPUT_PATH_TO_TIF=$2
OUTPUT_PATH_TO_TIF=$3

# How to get and install pixi:
# -> https://pixi.sh/v0.46.0/

pixi run --frozen python segment_image.py $PATH_TO_CELLPOSE_MODEL $INPUT_PATH_TO_TIF $OUTPUT_PATH_TO_TIF

