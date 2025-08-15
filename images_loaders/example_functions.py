from sklearn.model_selection import train_test_split


# examples of ready-made translating functions
def example_renamer_to_the_same_name():
    '''
    Technically an identity function, i.e., returns its argument.
    '''
    return lambda filename : filename

def example_renamer_to_a_different_name():
    '''
    Replaces 'raw' prefix for 'mask', e.g., raw004.tif to mask004.tif
    '''
    return lambda rawTTTtif : 'mask'+rawTTTtif[3:]


def example_split_files_lists_to_train_test_batches(raw_files, mask_files, train_test_split_params = {}):
    '''
    Using the train_test_split() the two files lists are split,
    and quadruple is returned: train_raw_files, train_mask_files, test_raw_files, test_mask_files

    The 'train_test_split_params' are passed as-is to the underlying sklearn.model_selection.train_test_split().
    '''
    splitlist = train_test_split(raw_files, mask_files, **train_test_split_params)
    return splitlist[0],splitlist[2], splitlist[1],splitlist[3]


import image_pairs_provider as IPP
import image_pairs_provider_presets as PRESETS

import sys
sys.path.append("../analyzing_datasets")
import analyze_image_pair as A

def do_stats_cellpose_v2():
    cellpose_train_folder = '/mnt/proj2/dd-24-22/data/cellpose/train_cyto2'
    provider = PRESETS.create_provider_for_cellpose_v3(cellpose_train_folder)
    PRESETS.setup_provider_for_data_analysis(provider)
    analyzer = A.ImagePairsAnalyzer()
    qq = analyzer.objs_props_from_images_provider(provider)
    analyzer.save_stats_to_a_folder(qq, cellpose_train_folder+"/stats")

def do_stats_cellpose_v3(do_test = False):
    cellpose_folder = '/mnt/proj2/dd-24-22/data/cellpose/'
    cellpose_folder += 'v3_test' if do_test else 'v3_train'
    provider = PRESETS.create_provider_for_cellpose_v3(cellpose_folder)
    PRESETS.setup_provider_for_data_analysis(provider)
    analyzer = A.ImagePairsAnalyzer()
    qq = analyzer.objs_props_from_images_provider(provider)
    analyzer.save_stats_to_a_folder(qq, cellpose_folder+"/stats")

def do_stats_stardist():
    stardist_train_root_folder = '/mnt/proj2/dd-24-22/data/StarDist2D_original'
    provider = PRESETS.create_provider_for_stardist(stardist_train_root_folder)
    PRESETS.setup_provider_for_data_analysis(provider)
    analyzer = A.ImagePairsAnalyzer()
    qq = analyzer.objs_props_from_images_provider(provider)
    analyzer.save_stats_to_a_folder(qq, stardist_train_root_folder+"/Training_stats")

def do_stats_embedseg():
    embedseg_train_root_folder = '/mnt/proj2/dd-24-22/data/EmbedSeg-dsb-2018/train'
    provider = PRESETS.create_provider_for_dsb2018(embedseg_train_root_folder)
    PRESETS.setup_provider_for_data_analysis(provider)
    analyzer = A.ImagePairsAnalyzer()
    qq = analyzer.objs_props_from_images_provider(provider)
    analyzer.save_stats_to_a_folder(qq, embedseg_train_root_folder+"/stats")

def do_stats_telight(subfolder: str):
    """
    'subfolder' must be exactly one from: A2058  G361  HOB  PC3  PNT1A
    """
    telight_root_folder = '/mnt/proj2/dd-24-22/data/telight_labeled'+'/'+subfolder
    provider = PRESETS.create_provider_for_telight(telight_root_folder)
    PRESETS.setup_provider_for_data_analysis(provider)
    analyzer = A.ImagePairsAnalyzer()
    qq = analyzer.objs_props_from_images_provider(provider)
    analyzer.save_stats_to_a_folder(qq, telight_root_folder+"/stats")

