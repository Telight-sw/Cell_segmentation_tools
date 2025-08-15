# how to run me:
# python train_two_folders.py myOwnData.bs16_e300.cellpose.model 16 300 folder_train folder_eval

# cellpose v3 vs v4 note:
# =======================
#
# This code is immune to the versions of cellpose, except for how a model is initialized/created.
# Choose one of the four options below, in the section marked with V3V4CHOOSER;
# and make sure the proper library is loaded (see below the environments).
#
# Quick re-cap of the pixi environments:
# pixi s                 -- set's up environment for GPU with v4
# pixi s -e gpu-v3       -- set's up environment for GPU with v3
# pixi s -e cpu-v4       -- set's up environment for CPU-only with v4
# pixi s -e cpu-v3       -- set's up environment for CPU-only with v3

import sys

if len(sys.argv) == 6:
    sys.path.append('../../segmentation/cellpose')
    sys.path.append('../../images_loaders')
    import cellpose_wrapper as C
    import images_manipulators as I
    import image_pairs_provider as IPP
    import image_pairs_provider_presets as PRESETS

    model_name = sys.argv[1]
    batch_size = int(sys.argv[2])
    epochs     = int(sys.argv[3])
    t_folder   = sys.argv[4]
    e_folder   = sys.argv[5]

    raw_filenames_pattern = 'raw.*'
    mask_filenames_from_raw_filenames = lambda imagename : "mask"+imagename[3:]
    #
    # obtains a list of all files matching the pattern 'raw_filenames_pattern' from the folder 't_folder'
    _, raw_files = IPP.aux__folder_and_its_files(t_folder, optional_filenames_filter_regexp = raw_filenames_pattern)
    #
    # obtains a loader/feeder of image pairs that will
    #   - take one (raw image) file after another from the list 'raw_files'
    #   - and for each it will also fetch its mask image whose name
    #     is derived from the raw image's name
    t_provider = IPP.ImagePairsProvider(*IPP.one_folder_files_and_renamer(t_folder, mask_filenames_from_raw_filenames, optional_files_list = raw_files))
    # the loader will now feed images (in fact, image pairs) as they are, unaltered;
    #
    # let's configure the loader such that provided image are:
    #   - normalized
    #   - no extra channels are explicitly added
    #   - remains as numpy data
    t_provider.set_raw_img_normalize_function( lambda img : I.normalize_img_auto_range_to_0_1(img) )
    t_provider.set_num_of_out_channels(0)
    t_provider.set_torch_transform_for_raw_images(None)
    # BTW: there's a shortcut configuration for exactly this,
    #      see PRESETS.setup_provider_to_autoNRaws_keepMasks_defaultNoTorch()

    # the same for the evaluation pairs:
    _, raw_files = IPP.aux__folder_and_its_files(e_folder, optional_filenames_filter_regexp = raw_filenames_pattern)
    e_provider = IPP.ImagePairsProvider(*IPP.one_folder_files_and_renamer(e_folder, mask_filenames_from_raw_filenames, optional_files_list = raw_files))
    PRESETS.setup_provider_to_autoNRaws_keepMasks_defaultNoTorch(e_provider)

    C.batch_size = batch_size

    ## <V3V4CHOOSER>
    ## for v3, absolutely empty model (this is available only for v3):
    model = C.models.CellposeModel(device=C.device)
    ##
    ## for v3, the latest pre-SAM 'cyto3' model:
    model = C.models.CellposeModel(model_type='cyto3', device=C.device)
    ##
    ## for v4, the current SAM-based model:
    model = C.create_sampretrained_model()
    ##
    ## for v4, some my previous model:
    model = C.load_model('somepath/some_my_previous.model')
    ## </V3V4CHOOSER>

    model_path = C.train_model(model, provider_train=t_provider, provider_eval=e_provider, new_model_name=model_name, epochs=epochs)
    print("done.")

    ## HOW TO load this model:
    # import os
    # model = C.load_model(os.path.join(model_name, 'models', model_name))
    #
    # (notice that cellpose saves its models into a 'models' subfolder)

else:
    print("Please provide five (5) parameters: model_name.model  batch_size  number_of_epochs  folder_with_training_pairs  folder_with_eval_pairs")

