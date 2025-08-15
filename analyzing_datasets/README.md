# Annotated Images Characterisation

For the purpose of understanding of what are typical sizes of annotated cells and
their typical pixel brightness, as well as average intensity of the background pixels,
a dedicated [codebase for raw+mask images](analyze_image_pair.py) has been developed, and
is [exemplified below](#image-metrics-example). It is also possible to [use the codebase
only with mask images](analyze_image_pair.py#L132), in which case the intensity statistics
will of course be missing. Indeed, the mask images (instance segmentation) are vital but
the best use of this codebase is on the pairs, such as it is in the training datasets.

<a name="image-metrics"></a>
## Image Statistics Metrics

### Collecting Statistics

The [statistics per each cell (foreground mask)](analyze_image_pair.py#L26) consists of
its label and a file name in which this cell has been found (and [statistics of the background
in that image](analyze_image_pair.py#L29)), its area (size in pixels), centroid coordinate,
mean/std intensity etc. Similar [statistics for the background](analyze_image_pair.py#L16)
are collected, such as area, min/max/mean/std intensity, or an extent (essentially, a
ratio of how much the image is filled).

The statistics for each image pair are computed in the method
[`ImagePairsAnalyzer.analyze()`](analyze_image_pair.py#L76) that returns
a list of [statistics of objects](analyze_image_pair.py#L26) for all cells (masks)
found in the pair. Iteratively calling this method over [each image pair from a
folder](analyze_image_pair.py#L115), or [just using](analyze_image_pair.py#L141)
a [*Provider*](../images_loaders/README.md#loaders-providers) to obtain pairs one after
another, a cumulative statistics of a dataset can be computed.

This cumulative statistics over a dataset is in fact an [intensity histogram over all pixels
over all images visited so far](analyze_image_pair.py#L62), which is an attribute of the
`ImagePairsAnalyzer` class. Notice that the `analyze()` method is a member of this class.
For the cumulative intensity histogram, the class [normalizes the analyzed
images](analyze_image_pair.py#L41) only for the purpose of adding to this histogram,
and this [normalization can be disabled](analyze_image_pair.py#L63).

### Reporting Statistics

Note that the `ImagePairsAnalyzer.analyze()` returns a list of cells (masks)
properties, and so do also all other methods that are wrapping around it such as
[`ImagePairsAnalyzer.objs_props_from_images_provider()`](analyze_image_pair.py#L141).

The returned list and the cumulative histogram can be saved into a folder with
[`ImagePairsAnalyzer.save_stats_to_a_folder()`](analyze_image_pair.py#L160).
Besides a PNG with the cumulative histogram, this method is going to save `stats.txt`
plain text file with several reports, and three PNGs with histograms of cell areas,
average cell pixel intensities, and average background intensities, respectively.

Nevertheless, individual reports can be created too, e.q., with a [helper function to
obtain list of selected parameters](analyze_image_pair.py#L204). An [own histogram can be
created](analyze_image_pair.py#L220) on that and [plotted](analyze_image_pair.py#L237-L247).

What follows is an example of files that are created, showing again the histograms of only
foreground (cells) and background intensities, histograms of pixel area of the foreground,
and "normal image histogram" taken together over all input raw images that [are normalized
prior contributing](analyze_image_pair.py#L63) to this *cumulative* histogram. Also the
`stats.txt` and [pickle](https://docs.python.org/3/library/pickle.html) `stats.pkl` to be
able later to restore all Python variables and objects and re-analyze and re-plot.

```
ulman@localhost ~/data/Telight_EDIH/new_data
$ tree -L 2
.
├── A2058
│   ├── images.tif
│   ├── masks_test.tif
│   ├── stats_BG_intensities_histogram.png
│   ├── stats_cumulative_intensities_histogram.png
│   ├── stats_FG_areas_histogram.png
│   ├── stats_FG_intensities_histogram.png
│   ├── stats.pkl
│   └── stats.txt
├── _A8780
│   ├── images.tif
│   ├── masks.tif
│   ├── stats_BG_intensities_histogram.png
│   ├── stats_cumulative_intensities_histogram.png
│   ├── stats_FG_areas_histogram.png
│   ├── stats_FG_intensities_histogram.png
│   ├── stats.pkl
│   └── stats.txt
├── _FaDu
│   ├── images.tif
│   ├── masks.tif
│   ├── stats_BG_intensities_histogram.png
│   ├── stats_cumulative_intensities_histogram.png
│   ├── stats_FG_areas_histogram.png
│   ├── stats_FG_intensities_histogram.png
│   ├── stats.pkl
│   └── stats.txt
├── G361
(shortened)
```

Example of the `stats.txt` file content:

```
objects'    area    median    : 569.0
objects'  area   mean  pm  std: 650.9776951672862 pm 374.64184023974514
objects' intensity mean pm std: 0.6948047280311584 pm 0.2916724979877472
background intnsty mean pm std: -0.019461235031485558 pm 0.06312871724367142
num of objects: 2959
num of backgrounds (files): 14
```

👉 The intensities distributions as well as cell sizes are a useful clue to understand
homogeneity with a dataset as well as differences between datasets.


<a name="image-metrics-stats"></a>
## Image Statistics on Real Data

### Properties of the Telight Datasets

![FG areas histogram on G361](../../doc/imgs/stats_FG_areas_histogram__G361.png)

*Distribution of pixel areas of cells in `G361` dataset, this is a dataset with only 23
annotated images.*

![FG areas histogram on PC3](../../doc/imgs/stats_FG_areas_histogram__PC3.png)

*Distribution of pixel areas of cells in `PC3` dataset, this is a dataset with 243
annotated images. Notice that the distribution is rather similar to that above for `G361`,
despite ten-times more annotated images is available here. Very similar outcomes are there
for the remaining Telight datasets.*

![FG areas histogram on HOB](../../doc/imgs/stats_FG_areas_histogram__HOB.png)

*Distribution of pixel areas of cells in `HOB` dataset, this is a dataset with 31 annotated
images. Notice that the tail of cell sizes shows 25x times larger cells than in the above
histograms. This is suggesting that substantially larger cells are displayed in this dataset.*

![cumulative intensities histogram on PNT1A](../../doc/imgs/stats_cumulative_intensities_histogram__PNT1A.png)

*Histogram of intensities from all pixels from all _normalized_ images from the Telight `PNT1A` dataset.
This is also an example of a typical shape in the Telight datasets.*

![cumulative intensities histogram on HOB](../../doc/imgs/stats_cumulative_intensities_histogram__HOB.png)

*Histogram of intensities from all pixels from all _normalized_ images from the Telight `HOB` dataset.
Notice the second "camel bump" in higher intensities range as well as the apparent
discontinuity around intensities 0.2.*

👉 Note that for the cumulative histograms the [images are normalized](analyze_image_pair.py#L41)
exactly the same way as they are prepared for the training and inference. So this
histogram is showing how a network would be seeing it.

👉 The intensity profile as well as cell sizes in the Telight `HOB` dataset is different
to the rest of the datasets, which are nevertheless somewhat self-similar. This is also
accented in the [segmentation performance of models](../doc/RESULTS_ON_TRAINED.md).


### Properties of Public Datasets

![FG areas histogram on Kaggle's DSB 2018](../../doc/imgs/stats_FG_areas_histogram__DSB2018.png)

*Distribution of pixel areas of cells in [Kaggle's DSB 2018 dataset](https://www.kaggle.com/competitions/data-science-bowl-2018).
Notice the average area sizes are quarter of cell sizes in Telight data. For the mode of
the distribution the difference is even more pronounced.*

![FG areas histogram on Cellpose v3 Cyto3](../../doc/imgs/stats_FG_areas_histogram__CP3_CYTO3.png)

*Distribution of pixel areas of cells in [Cellpose v3 dataset for the Cyto3 model](https://www.cellpose.org/dataset).
Also here, the average area sizes are quarter of cell sizes in Telight data.*

![cumulative intensities histogram on Kaggle's DSB 2018](../../doc/imgs/stats_cumulative_intensities_histogram__DSB2018.png)

*Histogram of intensities from all pixels from all _normalized_ images from the DSB 2018
dataset. The shape is actually similar to that obtained on Telight data.*

![cumulative intensities histogram on Cellpose v3 Cyto3](../../doc/imgs/stats_cumulative_intensities_histogram__CP3_CYTO3.png)

*Histogram of intensities from all pixels from all _normalized_ images from the Cellpose
v3 dataset for the Cyto3 model. The shape here is similar to that obtained on Telight data.*

👉 Interestingly, both intensity histograms are showing increased presence of very bright
pixels. It is not clear, however, if this is a (negative) feature of the used
normalization algorithm, or whether indeed the datasets feature an increased number of
"very bright" pixels. In any case, this is not the feature of the Telight data.

👉 It is apparent that the both popular datasets with annotated biomedical images of cells
show smaller cells compared to the content of Telight datasets. This is the kind of input
that the popular segmentation models were designed for.


<a name="image-metrics-example"></a>
## Example on Computing the Statistics

As [outlined above](#image-metrics), the statistics can be obtained using higher-level
API, here operating on a list of pairs of paths.

```python
import analyze_image_pair as A
import images_manipulators as I

# choose a folder with raw+mask images pairs
imgSrc='/home/ulman/data/Telight_EDIH/Segmentace_bunek/labeled/HOB'

# build a list of tuples of full paths by providing file name templates and indexing range
paths = [["{}/{:05d}_HOB_img.tif".format(imgSrc,i), "{}/{:05d}_HOB_mask.png".format(imgSrc,i)] for i in range(2,33)]

analyzer = A.ImagePairsAnalyzer()
#
# optionally disable the normalization
# analyzer.set_raw_img_normalizing_function(None)

list_of_props = analyzer.objs_props_from_files(paths)
analyzer.save_stats_to_a_folder(list_of_props, '/temp/yuiop/')
```

Or, operating from data fed by some [*Provider*](../images_loaders/README.md#loaders-providers).

```python
import os
import analyze_image_pair as A
import image_pairs_provider as IPP
import image_pairs_provider_presets as PRESETS

folder = '/home/ulman/data/Telight_EDIH/new_data/'
dataset = 'PC3'

stacked_data_provider = IPP.ImagePairsProviderFromStack( \
                        os.path.join(folder,dataset,'images.tif'), \
                        os.path.join(folder,dataset,'masks.tif') )
PRESETS.setup_provider_for_data_analysis(stacked_data_provider)

analyzer = A.ImagePairsAnalyzer()
list_of_props = analyzer.objs_props_from_images_provider(stacked_data_provider)
analyzer.save_stats_to_a_folder(list_of_props, os.path.join(folder,dataset))
```

Finally, a plain list of certain quantity can be extracted, expressed in a histogram and
saved as PNG.

<a name="histogram-printing"></a>
```python
import analyze_image_pair as A

# extract only areas of the cells (masks)
sizes = A.list_of_some_quantity(list_of_props, lambda o: o.area)

# should be of the same lenghts
len(list_of_props), len(sizes)

hist_of_sizes = A.get_histogram(sizes)
A.create_plot(hist_of_sizes)
A.save_plot_as_png('somewhere/hist_of_cell_sizes.png')
```

Note that several high-level "collectors" of statistics of several "standard public" datasets are available in
[`../images_loaders/example_functions.py`](../images_loaders/example_functions.py) as the `do_stats_....()` functions.


<a name="segmentation-metrics"></a>
# Performance Metrics

## SD3 Metric

This is a special metric designed [by Telight](https://telight.eu/) for measuring similarity
of two instance segmentation (masks) images. The metric internally establishes matching pairs
of masks, between the images, one mask from one image. If the two masks from a pair should be
overlapped, the distance between their boundary is aggregated and normalized. The distance
should be small if both masks overlap nicely. As a result, metric's value starts at 0.0
(the same instance segmentation pixel-wise, label values may not be synchronized) and has
no upper bound.

The *metric is symmetric* in the sense that it has no notion of *tested* and *reference*
masks. Its two arguments (each being an image with masks) can be swapped and the return
value is always the same.

It is understood that if this metric returns a value higher than 0.05 then two inputs are
different "really a lot" 😃, so much different that is it actually highly likely that the
two inputs are segmentation of different raw images.

In the realm of this project, the numbers obtained were usually one order of magnitude
lower (for less great results) up to the order of 10^-4.

The original implementation of this [metric is in C code](https://github.com/xulman/Telight-SD3)
and it became accessible to Python with [`pybind11`](https://github.com/pybind/pybind11).

👉 The source code of SD3 metric needs to be compiled into a C-language dynamic library
(`.so` on Linuxes, probably `.lib` for Windows) according to the instructions in
[its repository](https://github.com/xulman/Telight-SD3), and the compiled library file
must be copied into the folder `path_to_this_repository/analyzing_datasets` so that it is
found next to [`Jaccard.py`](Jaccard.py) source file.


### Negative log10 of SD3

The [metric values were seen at different orders of magnitude](../doc/RESULTS_ON_TRAINED.md),
especially different orders were achieved between different datasets, between experiments.
This has made a direct comparison of values between experiments somewhat not-straightforward.
In order to make the comparisons more comfortable and metric values more stable, values of
`-1 * log_10( original_SD3 )` were also reported. Additionally, the negative one factor
turns the metric into the higher-is-better regime, just like it is typical for the Jaccard
metric (see below).

<a name="segmentation-metrics-jaccard"></a>
## Jaccard

This metric, often known also as [Jaccard index](https://en.wikipedia.org/wiki/Jaccard_index)
or as *Intersection over Union (IoU)*, is a de-facto standard metric for comparing two
segmentation results (masks). Another well-known metric would be the [*Dice coefficient*,
which is not too different from the Jaccard](https://en.wikipedia.org/wiki/Dice-S%C3%B8rensen_coefficient#Difference_from_Jaccard).
The metric works in the higher-is-better regime, and returns values between 0.0 (two masks
are not even overlapping) to 1.0 (identical masks).

The [*metric is not symmetric*](Jaccard.py#L25) (unlike SD3), it recognizes
that [always the second argument in all implementations here](Jaccard.py#L25)
represents the *image with the reference result (masks)*.

It is understood that if this metric returns a value lower than 0.7 then the test image
(mask image) is noticeably not identical to the reference (mask) image, and the downstream
image analysis is could yield different results. On the other hand, Jaccard values above 0.8
are in general obtained for good quality results. As a third opinion, a citation from [NM paper
from Olaf Ronneberger](https://www.nature.com/articles/s41592-018-0261-2) has it: "In our
experiments, a *(IoU = Jaccard)* value of ∼0.7 indicates a good segmentation result, and a
value of ∼0.9 is close to human annotation accuracy."

<a name="segmentation-metrics-jaccard-matching"></a>
When two images are provided for comparison, the metric [also must establish matching
pairs](#segmentation-metrics) between the tested and reference masks. It is [mapping all
reference labels to at most one test label](Jaccard.py#L27-L29) provided that
[the test label intersects "great deal" of its reference label](Jaccard.py#L64-L71).
Such criterion for intersection warrants that a reference label cannot be matched to two
or more test labels. It also means that a test label may not "satisfy" any reference
label.

For follow-up statistics and visualizations, the [workhorse Jaccard routine](Jaccard.py#L25)
returns the intersection of unions ratios in two dictionaries, one that maps reference
labels to ratios achieved on them, and one that maps test labels to ratios they "deliver".
To obtain the final Jaccard value for a pair of images, [average values from the former
dictionary, e.g. with `Jaccard_average()`](Jaccard.py#L90).

<a name="segmentation-metrics-coverage"></a>
### Coverage

As mentioned just above, using the return value of the workhorse Jaccard function, it is
possible to calculate how many reference labels were "covered" by some test label and how
many reference labels were thus not sufficiently segmented. For an image pair, the number
of "covered" (matched) reference labels over the number of all of them is termed here as
the __coverage__ metric. It returns values between 0.0 and 1.0, with higher-is-better. The
value indeed is a percentage if multiplied with `100%`.

### Visualization

Next to calculating Jaccard (float) values, [the package here](Jaccard.py) provides also means
to graphically represent the quality of segmentation.

- [Create a float image where all reference masks are drawn with pixel value of the
  Jaccard that was achieved for them, and with value 0.1 if no test label has been found
  for the reference label](Jaccard.py#L104). Consider opening it in Fiji with LUT *Fire*;
  there's LUT toolbar button for it.

- [Create a histogram over Jaccard values](Jaccard.py#L123), and possibly [plot and save
  it](#histogram-printing). Depending if all Jaccard from the entire dataset are provided,
  or just Jaccard values of masks from one image, histograms of two different statistics
  can be obtained, and visualized. Example for the former is here: [add Jaccard values
  from one image pair](../segmentation/test_model.py#L88) and, after all were added,
  [save PNG with the histogram](../segmentation/test_model.py#L106-L107).

## Metrics in Action

Results obtained by applying these metrics can be found in the [comparison of vanilla
models on Telight data](../doc/RESULTS_ON_VANILLA.md) or [comparison of trained own
models on Telight data](../doc/RESULTS_ON_TRAINED.md).

The results over obtained using [the codebase from this repository](../SLURM/testing_evaluation/README.md).

Another example how to test various networks (models) on various datasets:

```python
# run from the 'segmentation' folder
# create results_networkName/datasetName_AutoN for each combination

import eval_telight_data as EVAL
models = EVAL.get_telightdata_NN_trained_on_telightdata()
provider = EVAL.get_telight_mixedTrain700_provider()
experiments = EVAL.get_telight_mixedTrain700_experiments()

EVAL.eval_telightdata(models, provider, experiments)
```

