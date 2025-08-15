# Images Loaders

This folder gathers routines using which images are collected from a storage (hard drive),
matched to form pairs (raw + mask images), loaded (on demand), "massaged" and presented to
the network either for its training or inference.

These functionalities are spread over several files.
And a [commented example is given below](#loaders-usage-example).

### `image_pairs_provider.py`

<a name="loaders-providers"></a>
This is the home of the several *Providers* of image pairs, most of them derived from the
[main workhorse class `ImagePairsProvider`](image_pairs_provider.py#L91) that is itself
derived from PyTorch [`Dataset`](https://docs.pytorch.org/tutorials/beginner/basics/data_tutorial.html#creating-a-custom-dataset-for-your-files)
class. Currently available classes are:

- `ImagePairsProvider`
  - Provides images pairs from [two same-length lists](#loaders-images-lists) that could
    be possibly [somehow altered](#loaders-images-massaging) as they are loaded and
    provided down-stream.
  - The only way to construct this class is to provide a folder where raw images are,
    and a list of their names (basically as paths that are relative to that folder, so a
    plain list of file names), and similarly a folder where are mask images (can be the
    same as the folder with raw images) and the list of them (again as paths relative to
    this folder).
  - The images are expected to be 2D. They can be of any types, and the [massaging](#loaders-images-massaging)
    will make sure they end up as expected (for the networks).
  - There are [helper functions](#loaders-images-lists) to construct such lists, and there
    are [presets functions](#loaders-presets) that typically just return a fully
    configured Provider.

- `ImagePairsProvidersChained`
  - If the to-be-read images are scattered over multiple folders, and even different
    massaging would be required across the different folders, this class comes to help.
  - Using its [append()](image_pairs_provider.py#L427) additional Providers can be added,
    increasing this way the number of available/providable image pairs of this Provider.
  - This class provides the images pairs sequentially from the first added (sub)Provider
    until it is fully "consumed", then it switches to the second added (sub)Provider etc.
    Needless to say, that (sub)Provider's specific massaging is applied, this class
    doesn't have its own massaging capabilities.
  - Still the assumption holds that each provided image is 2D.

- `ImagePairsProviderFromStack`
  - Similarly to the `ImagePairsProvider`, but it is assuming that the input image is
    actually a stack of 2D images, one stack for raw images and another stack for their
    masks.
  - The class is thus constructed by providing to full paths to the two stacks.
  - The same massaging capabilities are of course supported.

- `AugmentedImagePairsProviders`
  - Equipped with an augmentation function, this class wraps a Provider and uses its
    data (and the setup for the data massaging) to deliver requested (also a user
    parameter) number of augmented image pairs.
  - The underlying Provider is welcome to define its [massaging parameters](#loaders-images-massaging),
    just like above. This [class code just makes sure all is applied](image_pairs_provider.py#L369-L394),
    which is a little non-trivial this time [as discussed here](image_pairs_provider.py#L254).
  - [Example of how to setup and use this Provider is given below](#loaders-usage-example).

All the Providers offer convenience information like:

- Reporting the last used file name (even when the image was actually taken from a cache),
  see e.g. [`last_used_img_filename()`](image_pairs_provider.py#L241).
- Such reporting is anyway, by default, carried out on the console/terminal. But it can
  [be changed with the `max_verbosely_loaded_files` attribute](image_pairs_provider.py#L102),
  which defines up to how many loadings can be reported on the console/terminal.  
  👉 It is useful to reset it with some small value (e.g., 6) to read reports from only
  the first three pairs, only to be confirmed that the wanted files/folders are taken, and
  afterward stop polluting the console/terminal.

---

👉 The __Providers were meant to be used with [`DataLoader`](https://docs.pytorch.org/tutorials/beginner/basics/data_tutorial.html#preparing-your-data-for-training-with-dataloaders)__
from PyTorch, like for example [here](../segmentation/Unet/segmenter.py#L115-L116).

👉 These *Providers* need not to be the final products. It is of course possible to wrap
them, like [it was used](../segmentation/MaskRCNN/backend.py#L64-L119) with the
[Mask R-CNN network](../segmentation/README.md#maskrcnn):

- [Provider that returns derived data](../segmentation/MaskRCNN/backend.py#L223), such as
  a triplet "box","label","mask", instead of the image+mask pair,
- [Provider that returns an image and its index](../segmentation/MaskRCNN/backend.py#L246)
  within the underlying Provider (instead of image+mask pair).

---

<a name="loaders-images-lists"></a>
#### Aux Functions to Collect Lists of Available Images

[The file](image_pairs_provider.py) includes also helper functions:

- `aux__list_of_renamed_files()`
  - Converts a provided list of file names into a list with renamed file names, using the
    provided renaming functions (e.g., replacing `raw` with `mask` substrings).

- `aux__folder_and_its_files()`
  - This one lists files from a provider folder, taking potentially only those that
    satisfy a certain [regular expression pattern](https://en.wikipedia.org/wiki/Regular_expression).
    Returns also the provided folder to keep the information together, a folder and its (filtered) content.

- `two_folders_same_files()`
  - This is a higher-level function that returns four values that are expected to be fed
    as they are into the `ImagePairsProvider`, see above.
  - It designed to serve the training datasets where both raw and mask images are named
    the same but each is stored in different folder, so a folder with raw images and
    another folder with same-named mask images.
  - If not all available files are wished to be used, a list with a subset of them can be provided.

- `one_folder_files_and_renamer()`
  - This one is also expected to feed `ImagePairsProvider` directly, just like the above.
  - This one is designed for one folder holding together raw and similarly-named mask images,
    for example pairs like this one `raw001.png` and `mask001.png`.
  - If not all available files are wished to be used, a list with a subset of them can be provided.


<a name="loaders-images-massaging"></a>
#### Overview of Images Modifications Available

All the Providers can be configured to additionally modify the loaded images. A [complete
list of such methods](image_pairs_provider.py#L115-L175) is documented in the source code.

The methods `set_` a modification to either `_raw_img` or `_mask_img` to either
`_resize_and_to_float` convert it or to `_normalize` it with a provided `_function`.
From these four name items a configuration setter name can be established, like for
example, `set_raw_img_resize_and_to_float_function()`.

That way, the loaded image can be spatially adjusted (e.g., resized, cropped) while
returning the new image already with float pixel type. Yes, these two things are combined
in order to save memory by not creating transformed copies of the images. Afterward,
the loaded (and to-float-converted) images can be intensities-adjusted (e.g., normalized
to values between 0.0 and 1.0).

👉 The raw images can be returned as a numpy array or a Torch tensor. Default is to return a
Torch tensor but this can be disabled with [`set_torch_transform_for_raw_images(None)`](image_pairs_provider.py#L158).

👉 Setting a setter to `None` is a general way to disable it, works for all of them (I think).

👉 Finally, it may be required that the mask image is loaded as if with another number
of channels, use [`set_num_of_out_channels(N)`](image_pairs_provider.py#L149) for it.
Set `N = 0` to prevent from touching anything about the mask channels.


<a name="loaders-presets"></a>
### `image_pairs_provider_presets.py`

The previous section showed that the Providers can be configured in a number of ways. To
keep the code, that establishes a new provider, relatively concise, several shortcut
functions, termed as *Presets*, are provided and they are expected to live in this file.

They are split into two groups, those that

- `create_provider_for_....()` and return a provider that can cherry pick specifically
  organized files,
- `setup_provider_for_....()` to specifically [massage](#loaders-images-massaging) the
  loaded files.

In fact, the `create....()` understands a file plan of a particular dataset and takes care
of taking files (doing the pairs) correctly from there, while the `setup....()`
understands needs of a down-stream consumer (network) and takes care that the images are
prepared correctly for it.

👉 A special [`setup_provider_to_autoNRaws_keepMasks_defaultNoTorch()`](image_pairs_provider_presets.py#L79)
exists to prepare the image pairs as [normalized](#loaders-normalization-example) float
numpy arrays that is not altering the mask images in any way. This one is especially good
for testing existing models as it essentially only normalizes the raw images.


<a name="loaders-manipulators"></a>
### `images_manipulators.py`

This library file provides various image modification functions, be it modifications in
the image size or pixel values (intensities) for raw and mask images, as well as their
supporting functions.

Worth mentioning are functions `normalize_img_....()` that change pixel values to end up
between 0.0 and 1.0 by applying linear transformation, possibly after clipping.

👉 The main normalization function that's used in the entire project is
[`normalize_img_auto_range_to_0_1()`](images_manipulators.py#L120). This is basically a
percentile stretch where the boundaries (low and high percentile values for clipping) are
computed for each image individually by analysing its histogram. In particular, the pixel
values are first [`normalize_img_full_range_to_0_1()`](images_manipulators.py#L40)
stretched and then non-decreasingly sorted. If this would be plotted (see below), the
outlaying pixel values, which are rare in the image, are displayed as the close-to-vertical
segments. Finally, the boundaries for the percentile stretch are determined as the bending
points on such curve (see [here](images_manipulators.py#L76) and [here](images_manipulators.py#L98)).

<a name="loaders-normalization-example"></a>
![Example of the sorted pixel intensities of the original image](../doc/imgs/intensities_normalization_before_auto.png)

![Example of the sorted pixel intensities after the normalization](../doc/imgs/intensities_normalization_after_auto.png)

*The upper figure shows the non-decreasingly sorted pixel intensities of an original image
(from Telight dataset). The "verticality" of the segments show "rarity" of the data values.
The normalization used in this project finds the low and upper bending points on such
plot, clip values outside and linearly stretches the rest. That way, the original image
was normalized and its values are plotted in the bottom figure.*


### `example_functions.py`

This is the last and the least important file, which serves the purpose of keeping relevant
code snippets and, in fact, it needs not to be here at all. But it was useful for collecting
[input images characteristics](../analyzing_datasets/README.md#collecting-statistics).


# Commented Example on How to Use Providers

The following commented example illustrates the principle that [`AugmentedImagePairsProviders`](image_pairs_provider.py#L304)
is actually a wrapper around some [`ImagePairsProvider`](image_pairs_provider.py#L91) married with a transformation function.

<a name="loaders-usage-example"></a>
```python
import image_pairs_provider as IPP
import image_pairs_provider_presets as PRESETS
import albumentations as A

# get a rudimentary ImagePairsProvider (for a specific "Telight" naming scheme)
provider = PRESETS.create_provider_for_telight('/home/ulman/data/Telight_EDIH/Segmentace_bunek/labeled/PC3/')

# set it up to auto-normalize loaded images,
# and present them as numpy arrays (not as Torch tensors)
PRESETS.setup_provider_to_autoNRaws_keepMasks_defaultNoTorch(provider)

# a trick to narrow down the list of available/providable images by taking only the first
# ten discovered images; note that there are more internal attributes in the Providers, but
# this one is the driving one when it comes to iterating over available/providable images,
# and it is thus enough to modify this one (if that is really needed because this is rather
# unsupported/unintended way of doing it)
provider.img_files = provider.img_files[:10]

# some augmentation recipe; it could be, I think, anything that acts as function
transforms = A.OneOf([ \
    A.Compose([ \
            A.RandomScale(scale_limit=(0.0, 0.5), always_apply=True), \
            A.RandomCrop(height=608, width=608, always_apply=True) \
    ]), \
    A.Rotate(limit=(-30,30),p = 0.9), \
    A.RandomGamma(p = 0.9) \
], p = 1.0)

# wrap the Provider (loader) with the augmentation recipe,
# and ask it to provide (on demand, lazily) 300 images pairs
aprovider = IPP.AugmentedImagePairsProviders(provider, transforms, 300)
```

One can always [read the underlying (and massaged)](image_pairs_provider.py#L217), not
augmented images. This works, for example, to retrieve the 9th (indexing is zero-based)
image pair:

```
orig_raw_image, orig_mask_image = provider[8]
```

Btw, if the same command is executed again, it shall give exactly the same images,
this time without reading them again from hard drive (owing to [caches
available](image_pairs_provider.py#L100-L101) in the Providers).

In the same way, an [augmented image can be obtained](image_pairs_provider.py#L354) with, e.g.,

```
augmented_raw_image, augmented_mask_image = aprovider[8]
```

and that chooses some image pair from the underlying `provider` (not necessary
the pair at index `8`), [augments it](image_pairs_provider.py#L387) and the result is
returned, [and cached](image_pairs_provider.py#L397). Here the caches are vital in order
to be able to return the same result when the same index is asked.

However, this behaviour of returning rigidly the same augmented images at a given index,
and instead returning always a new augmentation of otherwise the same original input
whenever the same index is asked, [can be changed with `aprovider.use_cache_for_augmented
= False`](image_pairs_provider.py#L348).

