echo off
set PATH_TO_CELLPOSE_MODEL=%1
set INPUT_PATH_TO_TIF=%2
set OUTPUT_PATH_TO_TIF=%3

REM How to get and install pixi:
REM -> https://pixi.sh/v0.46.0/

echo on
pixi run --frozen python segment_image.py %PATH_TO_CELLPOSE_MODEL% %INPUT_PATH_TO_TIF% %OUTPUT_PATH_TO_TIF%

