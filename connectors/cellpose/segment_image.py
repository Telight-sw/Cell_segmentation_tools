# how to run me:
# python segment_image.py G361.t500_bs16_e300.cellpose.model /home/ulman/data/Telight_EDIH/Segmentace_bunek/labeled/G361/00002_G361_img.tif result.tif

# cellpose v3 vs v4 note:
# =======================
#
# This code is immune to the versions of cellpose, the code functions for both of them.
# Only make sure the proper model is loaded: 26 MB for v3, 1.2 GB for v4;
# and that a proper library is loaded (see below the environments).
#
# Quick re-cap of the pixi environments:
# pixi s                 -- set's up environment for GPU with v4
# pixi s -e gpu-v3       -- set's up environment for GPU with v3
# pixi s -e cpu-v4       -- set's up environment for CPU-only with v4
# pixi s -e cpu-v3       -- set's up environment for CPU-only with v3


import sys

if len(sys.argv) == 4:
    sys.path.append('../../segmentation/cellpose')
    sys.path.append('../../images_loaders')
    import cellpose_wrapper as C
    import images_manipulators as I
    import tifffile as TIFF

    model = C.load_model(sys.argv[1])

    img = TIFF.imread(sys.argv[2])
    img = I.normalize_img_auto_range_to_0_1(img)

    res = C.apply_model(model, img)

    TIFF.imwrite(sys.argv[3], res)

else:
    print("Please provide three (3) parameters: model_name path_to_input_tiff path_to_output_tiff")

