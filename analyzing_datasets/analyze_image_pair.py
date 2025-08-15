from tqdm import tqdm
import numpy.typing as npt
import numpy as np
from PIL import Image
from skimage import measure
import pickle
import matplotlib.pyplot as plt
import math
import os

import sys
sys.path.append("../images_loaders")
import images_manipulators as I


class BgProps:
    def __init__(self, file_id, area, extent, intensity_min, intensity_max, intensity_mean, intensity_std):
        self.file_id = file_id
        self.area = area
        self.extent = extent
        self.intensity_min = intensity_min
        self.intensity_max = intensity_max
        self.intensity_mean = intensity_mean
        self.intensity_std = intensity_std

class ObjProps:
    def __init__(self, file_id, bg_props:BgProps, label, centroid, area, extent, intensity_mean, intensity_std):
        self.file_id = file_id
        self.bg_stats = bg_props
        self.label = label
        self.centroid = centroid
        self.area = area
        self.extent = extent
        self.intensity_mean = intensity_mean
        self.intensity_std = intensity_std


class ImagePairsAnalyzer:
    def __init__(self):
        self.reset_cumulated_int_histogram()
        self.raw_img_normalization = lambda i : I.normalize_img_auto_range_to_0_1(i)
        self.folder_for_individual_histograms = None

    def set_raw_img_normalizing_function(self, transform):
        """
        Provide a one-image-in-one-image-out function that anyhow normalizes its input,
        or just 'None' to disable images normalization before taking their histograms.
        """
        self.raw_img_normalization = transform

    def reset_cumulated_int_histogram(self):
        self.last_used_int_histogram = None
        self.cumulated_int_histogram = None
        self.histogram_min_edge = 0.0
        self.histogram_max_edge = 1.0
        self.histogram_bins_cnt = 100

    def clear_cumulated_int_histogram(self):
        self.last_used_int_histogram = None
        self.cumulated_int_histogram = None

    def update_cumulated_int_histogram(self, img: npt.NDArray):
        nimg = self.raw_img_normalization(img) if self.raw_img_normalization else img
        #
        # get histogram
        self.last_used_int_histogram = get_histogram(nimg, self.histogram_min_edge, self.histogram_max_edge, self.histogram_bins_cnt)
        #
        # update the cumulated one
        if self.cumulated_int_histogram is None:
            self.cumulated_int_histogram = self.last_used_int_histogram.copy()
        else:
            for pxVal in self.cumulated_int_histogram.keys():
                self.cumulated_int_histogram[pxVal] += self.last_used_int_histogram[pxVal]


    def analyze(self, raw_img: npt.NDArray, mask_img: npt.NDArray, data_src_id: str = "unknown") -> list[ObjProps]:
        """
        Returns a list of mask properties, a list of the length that
        happens to be the number of masks in the 'mask_img'.
        """
        res = list()
        if raw_img.shape != mask_img.shape:
            print(f"Image sizes mismatch: {raw_img.shape} vs {mask_img.shape}")
            return res

        # get background values statistics
        # NB: raw_bg_img will be 1D array, with just bg pixels, so calling
        # e.g. mean() on it indeed gives the mean across background region
        # only (no bleeding of "zeroed foreground" or anything like that)
        raw_bg_img = raw_img[ mask_img == 0 ]
        bg_stats = BgProps(data_src_id, raw_bg_img.size, float(raw_bg_img.size) / float(raw_img.size), # the later is the "extent"
                           raw_bg_img.min(), raw_bg_img.max(), raw_bg_img.mean(), raw_bg_img.std()) \
                   if raw_bg_img.size > 0 else \
                   BgProps(data_src_id, raw_bg_img.size, float(raw_bg_img.size) / float(raw_img.size), # the later is the "extent"
                           0.0, 0.0, 0.0, 0.0)

        if raw_bg_img.size == 0:
            print(f"FYI: {data_src_id} has no background pixels.")

        regions = measure.regionprops(mask_img, raw_img)
        for region in regions:
            res.append( ObjProps( data_src_id, bg_stats,
                    region['label'], region['centroid'],
                    region['area'], region['extent'],
                    region['intensity_mean'], region['intensity_std'] ) )

        self.update_cumulated_int_histogram(raw_img)
        if self.folder_for_individual_histograms:
            create_plot(self.last_used_int_histogram)
            save_plot_as_png(self.folder_for_individual_histograms+"/"+data_src_id+".png")

        return res


    def analyze_from_paths(self, path_to_raw: str, path_to_mask: str) -> list[ObjProps]:
        r,m = read_pair(path_to_raw, path_to_mask)
        return self.analyze(r,m, os.path.basename(path_to_raw))

    def objs_props_from_files(self, list_of_paths: list[tuple[str,str]]) -> list[ObjProps]:
        """
        Hint on how to build paths:
        imgSrc=''
        paths = [ ["{}/{:03d}_img.png".format(imgSrc,i), "{}/{:03d}_masks.png".format(imgSrc,i)] for i in range(540) ]

        or, folder content listing + filtering...
        """
        objs_props = list()
        for raw_path,mask_path in tqdm(list_of_paths):
            objs_props += self.analyze_from_paths(raw_path,mask_path)
        return objs_props

    def objs_props_from_maskOnly_files(self, list_of_paths: list[str]) -> list[ObjProps]:
        """
        Just like 'objs_props_from_files()' but it uses the mask image also in place for the raw image.
        """
        objs_props = list()
        for mask_path in tqdm(list_of_paths):
            objs_props += self.analyze_from_paths(mask_path,mask_path)
        return objs_props

    def objs_props_from_images_provider(self, provider) -> list[ObjProps]:
        """
        The argument 'provider' is of the 'ImagePairsProvider' type which
        comes from "../images_loaders/image_pairs_provider.py".
        """
        # prepare provider for silent images loading,
        # silent -- not to mess up with tqdm print outs
        backup_silent_files_loading = provider.silent_files_loading
        provider.silent_files_loading = True
        it = iter(provider)

        objs_props = list()
        for r,m in tqdm(it):
            objs_props += self.analyze(r,m, provider.last_used_img_filename())

        provider.silent_files_loading = backup_silent_files_loading
        return objs_props


    def save_stats_to_a_folder(self, objects_properties: list[ObjProps], folder_path: str) -> None:
        with open(folder_path+"/stats.pkl","wb") as f:
            pickle.dump(objects_properties,f)

        areas = list_of_some_quantity(objects_properties, lambda o: o.area)
        pixel_vals_mean = list_of_some_quantity(objects_properties, lambda o: o.intensity_mean)
        pixel_vals_std  = list_of_some_quantity(objects_properties, lambda o: o.intensity_std)
        areas_median = sorted(areas)[len(areas)//2]

        background_properties = list_of_bg_stats(objects_properties)
        bg_pixel_vals_mean = list_of_some_quantity(background_properties, lambda o: o.intensity_mean)
        bg_pixel_vals_std  = list_of_some_quantity(background_properties, lambda o: o.intensity_std)

        with open(folder_path+"/stats.txt","w") as f:
            f.write(f"objects'    area    median    : {areas_median}\n")
            f.write(f"objects'  area   mean  pm  std: {areas.mean()} pm {areas.std()}\n")
            f.write(f"objects' intensity mean pm std: {pixel_vals_mean.mean()} pm {pixel_vals_std.mean()}\n")
            f.write(f"background intnsty mean pm std: {bg_pixel_vals_mean.mean()} pm {bg_pixel_vals_std.mean()}\n")
            f.write(f"num of objects: {len(objects_properties)}\n")
            f.write(f"num of backgrounds (files): {len(background_properties)}\n")

        create_plot( get_histogram(areas) )
        save_plot_as_png(folder_path+"/stats_FG_areas_histogram.png")
        create_plot( get_histogram(pixel_vals_mean) )
        save_plot_as_png(folder_path+"/stats_FG_intensities_histogram.png")
        create_plot( get_histogram(bg_pixel_vals_mean) )
        save_plot_as_png(folder_path+"/stats_BG_intensities_histogram.png")

        create_plot( self.cumulated_int_histogram )
        save_plot_as_png(folder_path+"/stats_cumulative_intensities_histogram.png")


def load_stats_from_a_folder(folder_path: str) -> list[ObjProps]:
    with open(folder_path+"/stats.pkl","rb") as f:
        objs_props = pickle.load(f)
        return objs_props


def read_pair(path_to_raw: str, path_to_mask: str) -> tuple[npt.NDArray,npt.NDArray]:
    r = np.array( Image.open(path_to_raw) )
    m = np.array( Image.open(path_to_mask) )
    return r,m


def list_of_some_quantity(objects_properties: list[any], prop_selector_lambda ):
    """
    Example use: list_of_some_quantity(objs_in_a_list, lambda o: o.area)

    Since it gives numpy array, one can easily do mean(), std(), min(), etc.
    """
    return np.array( [ prop_selector_lambda(o) for o in objects_properties ] )

def list_of_bg_stats(objects_properties: list[ObjProps]) -> list[BgProps]:
    return list( dict.fromkeys( [ o.bg_stats for o in objects_properties ] ) )


def get_percentile(values: npt.NDArray, percentile: float) -> float:
    sorted_values = sorted(values.flat)
    return sorted_values[ int( percentile * float(len(sorted_values)) ) ]

def get_histogram(values: npt.NDArray, min_edge:float = None, max_edge:float = None, num_bins:int = None) -> dict[float,int]:
    minVal = min(0.0,values.min()) if min_edge is None else min_edge
    maxVal = get_percentile(values, 0.98) if max_edge is None else max_edge
    bins_cnt = 100 if num_bins is None else num_bins
    bin_size = float(maxVal-minVal)/float(bins_cnt)

    hist = dict()
    for idx in range(bins_cnt+1):
        hist[minVal + float(idx)*bin_size] = 0
    for v in values.flat:
        binIdx = math.floor(float(v-minVal) / bin_size)
        if binIdx < 0 or binIdx >= len(hist):
            continue
        hist[minVal + float(binIdx)*bin_size] += 1

    return hist

def create_plot(hist: dict[float,int]) -> None:
    it = iter(hist.keys())
    bin_size = it.__next__() - it.__next__()
    plt.close()
    plt.bar(hist.keys(), hist.values(), width=-bin_size, align='edge')

def save_plot_as_png(filename):
    """
    Requires that `create_plot() has been called just before.
    """
    plt.savefig(filename)

