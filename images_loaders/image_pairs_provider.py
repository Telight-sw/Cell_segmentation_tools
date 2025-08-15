from torch.utils.data import Dataset
from torchvision import transforms
from skimage import io
import os
import numpy as np
from numpy.random import randint
from sklearn.model_selection import train_test_split
import re
import images_manipulators as I

# two prevailing use-cases:
# FolderA,FolderB            two folders, same filenames in these
# Folder/imgA Folder/imgB    same folder, two filenames
#
# can be treated with this symmetric generic pattern:
# (note that getFileXfromY() could also provide the path)
#
# folderA, filesA, getFileBfromA()
# folderB, filesB, getFileAfromB()
#
# to yield the quadruple: folderA, filesA, folderB, filesB


# helper functions
def aux__list_of_renamed_files(files_list: list[str], file_renaming_function) -> list[str]:
    return [ file_renaming_function(file) for file in files_list ]

def aux__folder_and_its_files(imgs_folder: str, optional_files_list: list[str] = None, optional_filenames_filter_regexp: str = None) -> tuple[str,list[str]]:
    files_list = os.listdir(imgs_folder) if optional_files_list is None else optional_files_list
    if optional_filenames_filter_regexp is not None:
        regexp = re.compile(optional_filenames_filter_regexp)
        filtered_files_list = list()
        for filename in files_list:
            if regexp.match(filename):
                filtered_files_list.append(filename)
        files_list = filtered_files_list
    return imgs_folder, files_list


# for the two common use-cases above
def two_folders_same_files(raw_imgs_folder: str, mask_imgs_folder: str, optional_files_list: list[str] = None) -> tuple[str,list[str],str,list[str]]:
    _,raw_files = aux__folder_and_its_files(raw_imgs_folder, optional_files_list)
    return raw_imgs_folder,raw_files, mask_imgs_folder,raw_files

def one_folder_files_and_renamer(imgs_folder: str, file_renaming_function, optional_files_list: list[str] = None) -> tuple[str,list[str],str,list[str]]:
    _,raw_files = aux__folder_and_its_files(imgs_folder, optional_files_list)
    mask_files = aux__list_of_renamed_files(raw_files, file_renaming_function)
    return imgs_folder,raw_files, imgs_folder,mask_files


# obtain ImagePairsProvider(s) for the two common use-cases above
def get_all_files_provider(raw_imgs_folder: str, raw_imgs_files: list[str], mask_imgs_folder: str, mask_imgs_files: list[str]):
    '''
    Returns ImagePairsProvider for the given folders and files.
    '''
    provider = ImagePairsProvider(raw_imgs_folder,raw_imgs_files, mask_imgs_folder,mask_imgs_files)
    return provider

def get_train_test_providers(raw_imgs_folder: str, raw_imgs_files: list[str], mask_imgs_folder: str, mask_imgs_files: list[str], train_test_split_params = {}):
    '''
    Returns two ImagePairsProviders for the given folders and files.
    The files are first split into train and test batches and so two ImagePairsProviders,
    for the train batch (files subset) and test batch, are created and returned in this order.

    The 'train_test_split_params' are passed as-is to the underlying sklearn.model_selection.train_test_split().
    '''
    splitlist = train_test_split(raw_imgs_files, mask_imgs_files, **train_test_split_params)
    train_provider = ImagePairsProvider(raw_imgs_folder,splitlist[0], mask_imgs_folder,splitlist[2])
    test_provider  = ImagePairsProvider(raw_imgs_folder,splitlist[1], mask_imgs_folder,splitlist[3])
    return train_provider, test_provider


def imread(path, clean_up_mask: bool = False):
    img = io.imread(path)

    if len(img.shape) == 2 or img.shape[2] != 3:
        if clean_up_mask: I.clean_up_stack_of_masks(img)
        return img

    # re-arrange to (3,H,W)
    nimg = np.zeros((3,img.shape[0],img.shape[1]), dtype = img.dtype)

    nimg[0,:,:] = img[:,:,0]
    nimg[1,:,:] = img[:,:,1]
    nimg[2,:,:] = img[:,:,2]
    if clean_up_mask: I.clean_up_stack_of_masks(nimg)
    return nimg



class ImagePairsProvider(Dataset):
    def __init__(self, raw_imgs_folder: str, raw_imgs_files: list[str], mask_imgs_folder: str, mask_imgs_files: list[str]):
        self.img_dir = raw_imgs_folder
        self.mask_dir = mask_imgs_folder
        self.img_files = raw_imgs_files
        self.mask_files = mask_imgs_files
        self.last_used_idx = -1

        # images caches
        self.imgs = dict()
        self.masks = dict()
        self.max_verbosely_loaded_files = float('inf') #NB: means report all of them

        # "own" online modifications
        self.img_resize_and_to_float = None
        self.img_normalize = None
        self.mask_resize_and_to_float = None
        self.mask_normalize = None

        # "for-torch" online modifications
        self.out_channels = 1
        self.torch_transform = transforms.ToTensor()


    def set_raw_img_resize_and_to_float_function(self, transform):
        '''
        Provide a function that takes numpy array and returns numpy array
        of possibly different shape, and for sure of 'float32' pixel type.
        Set to 'None' if this step (during the loading of images) should be disabled.
        '''
        self.img_resize_and_to_float = transform
        self.imgs.clear()

    def set_raw_img_normalize_function(self, transform):
        '''
        Provide a function that takes numpy array and returns numpy array, possibly somehow different.
        Set to 'None' if this step (during the loading of images) should be disabled.
        '''
        self.img_normalize = transform
        self.imgs.clear()

    def set_mask_img_resize_and_to_float_function(self, transform):
        '''
        Provide a function that takes numpy array and returns numpy array
        of possibly different shape, and for sure of 'float32' pixel type.
        Set to 'None' if this step (during the loading of images) should be disabled.
        '''
        self.mask_resize_and_to_float = transform
        self.masks.clear()

    def set_mask_img_normalize_function(self, transform):
        '''
        Provide a function that takes numpy array and returns numpy array, possibly somehow different.
        Set to 'None' if this step (during the loading of images) should be disabled.
        '''
        self.mask_normalize = transform
        self.masks.clear()

    def set_num_of_out_channels(self, num_of_out_channels):
        '''
        This extends the loaded mask images with one extra dimension whose size
        would be exactly 'num_of_out_channels'. No extra dimension is created
        if 0 (zero) is given, this disables it.
        '''
        self.out_channels = num_of_out_channels
        self.masks.clear()

    def set_torch_transform_for_raw_images(self, transform):
        '''
        Possibly a composition of Pytorch vision (v2) transforms can be plugged in, they would be
        applied *only* on the raw images, not on masks (as masks are here only for the (loss) evaluation
        that happens on the CPU using custom code typically operating on the "usual" numpy arrays...).
        See: https://pytorch.org/vision/stable/auto_examples/transforms/plot_transforms_getting_started.html

        Set to 'None' if this step (during the loading of images) should be disabled.

        Example of transforms pipeline that could be used with ImagePairsProvider:
        trans = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]) # imagenet
        ])

        !! Default value is 'ToTensor()'.
        '''
        self.torch_transform = transform


    def apply_raw_image_spatial_transforms(self, img):
        return self.img_resize_and_to_float(img) if self.img_resize_and_to_float is not None else img

    def apply_raw_image_intensity_transforms(self, img):
        return self.img_normalize(img) if self.img_normalize is not None else img

    def apply_raw_image_torch_transform(self, img):
        return self.torch_transform(img) if self.torch_transform is not None else img

    def apply_mask_image_spatial_transforms(self, mask):
        return self.mask_resize_and_to_float(mask) if self.mask_resize_and_to_float is not None else mask

    def apply_mask_image_intensity_transforms(self, mask):
        return self.mask_normalize(mask) if self.mask_normalize is not None else mask

    def apply_mask_image_channel_transforms(self, mask):
        return np.reshape(mask, [self.out_channels,*mask.shape]) if self.out_channels > 0 else mask


    def io_raw_image_read(self, idx):
        fullPath = os.path.join(self.img_dir, self.img_files[idx])
        if self.max_verbosely_loaded_files > 0:
            self.max_verbosely_loaded_files -= 1
            print("I/O: reading raw image from file: "+fullPath)
        img = imread(fullPath)
        return img

    def io_mask_image_read(self, idx):
        fullPath = os.path.join(self.mask_dir, self.mask_files[idx])
        if self.max_verbosely_loaded_files > 0:
            self.max_verbosely_loaded_files -= 1
            print("I/O: reading mask image from file: "+fullPath)
        msk = imread(fullPath, clean_up_mask = True)
        return msk


    def __len__(self):
        return len(self.img_files)

    def __getitem__(self, idx):
        if idx >= len(self.img_files):
            raise StopIteration

        img = self.imgs.get(idx)
        if img is None:
            img = self.io_raw_image_read(idx)
            img = self.apply_raw_image_spatial_transforms(img)
            img = self.apply_raw_image_intensity_transforms(img)
            self.imgs[idx] = img
        img = self.apply_raw_image_torch_transform(img)

        msk = self.masks.get(idx)
        if msk is None:
            msk = self.io_mask_image_read(idx)
            msk = self.apply_mask_image_spatial_transforms(msk)
            msk = self.apply_mask_image_intensity_transforms(msk)
            msk = self.apply_mask_image_channel_transforms(msk)
            self.masks[idx] = msk

        self.last_used_idx = idx
        return img, msk


    def last_used_img_filename(self):
        return self.img_files[self.last_used_idx]

    def last_used_mask_filename(self):
        return self.mask_files[self.last_used_idx]

    def get_imgs_folder(self):
        return self.img_dir

    def get_masks_folder(self):
        return self.mask_dir


# Analysis of the behaviour and relationship between ImagePairsProvider and AugmentedImagePairsProviders
#
# ORIGINAL - current flow                     ORIGINAL - modified order                                Augmenter - aka a needed order
#
# use raw image from cache if it's there    # use raw image from cache if it's there                 # use raw image from cache if it's there
# else:                                     # else:                                                  # else:
#    load raw image from disk               #    load raw image from disk                            #    load raw image from disk
#    img_resize_and_to_float                #    img_resize_and_to_float                             #    img_resize_and_to_float
#    img_normalize                          #                                                        #    put raw image into cache
#    put raw image into cache               #                                                        #
# apply torch ops on the raw image          #                                                        #
#                                           #                                                        #
# use mask image from cache if it's there   # use mask image from cache if it's there                # use mask image from cache if it's there
# else:                                     # else:                                                  # else:
#    load mask image from disk              #    load mask image from disk                           #    load mask image from disk
#    mask_resize_and_to_float               #    mask_resize_and_to_float                            #    mask_resize_and_to_float
#    mask_normalize                         #                                                        #    put mask image into cache
#    mask_channels                          #                                                        #
#    put mask image into cache              #                                                        #
#                                           #                                                        #
#                                           #                                                        #
#                                           # if Augmenter:                                          # augment both raw and mask images at the same time
#                                           #    augment both raw and mask images at the same time   # (to assure the same op(s) is applied to both)
#                                           # else identity function...                              #
#                                           #                                                        #
#                                           # img_normalize                                          # img_normalize
#                                           # put/update (original) raw image into cache             # apply torch ops on the raw image
#                                           # apply torch ops on the raw image                       #
#                                           #                                                        #
#                                           # mask_normalize                                         # mask_normalize
#                                           # mask_channels                                          # mask_channels
#                                           # put/update (original) mask image into cache            #
#
# Analysis: Since the augmentation is an action that requires to work _in the middle_ of the processing
# and requires _both raw and mask_ images to have at the same time (right panel), it really complicates
# the current natural flow (left panel) significantly. That's because the augmentation can do some spatial
# transforms (e.g. rotation) that nevertheless preserve the image geometry (channels, size), but it can do
# changes in _intensities_ (e.g. contrast) too -- so normalization needs to happen after the augmentation.
# Consequently, while a fully-processed and ready-to-be-used image, that is resized & normalized, can be kept
# in caches in the natural flow, only resized version (because everything else should be randomly changed
# during the augmentation) is worth keeping when augmentations are in place. Thus, besides the flow itself,
# also the semantics of what is stored in the caches differ significantly.
#
# Conclusion: To obtain an "augmented" provider that's focused mostly on the augmentation itself and borrowing
# the most from (outsourcing to) a provided/underlying "normal" provider, the __getitem__() part of the normal
# provider will need to be split into smaller parts (functions, e.g., '|' section of the middle panel). This
# would allow the normal and augmented providers to setup (middle panel) their own variants of their
# __getitem__() to achieve their tasks.


class AugmentedImagePairsProviders(Dataset):
    def __init__(self, provider: ImagePairsProvider, augmenter, target_count_of_image_pairs: int):
        """
        This (Augmented)ImagePairsProvider basically brings together the standard
        ImagePairsProvider and the augmentation, which is provided by Albumentations
        Compose object (see https://albumentations.ai/docs/examples/showcase/),
        in order to blow up the amount of image pairs that can be here delivered
        (whose number is given in the last parameter 'target_count_of_image_pairs').

        Technically, this class heavily leverages on the provided ImagePairsProvider
        'provider'. It uses all of its setup, its loading and transform routines,
        but it is not using its caches (here own caches are provided because images
        at different stages of preparations are kept here compared to the normal
        underlying provider). This results here in a new __getitem__() function
        that basically intertwines differently the processing of raw and mask images
        to enable the augmentations to take place.

        While this class is not a full substitute of the ImagePairsProvider (for
        example, it is not derived from it), it offers/exposes the usual functions
        (similarly to what ImagePairsProvidersChained does). The image processing
        is, nevertheless, fully under the control of the provider underlying provider.

        .. code-block:: python
        provider = PRESETS.create_provider_for_NN("somepath")
        provider = AugmentedImagePairsProviders(provider, aug_transform, 10000)
        raw_img,mask_img = provider[10]
        #
        # NB: notice the original variable 'provider' was replaced with a new class that
        #     fully substitutes the original one w.r.t. where 'provider' could be used
        """

        self.src_provider = provider
        self.albumentations_transform = augmenter
        self.no_of_image_pairs = target_count_of_image_pairs

        # images caches:
        # cache of underlying images indexed by the underlying provider's index
        self.imgs_underlying = dict()
        self.masks_underlying = dict()
        self.usage_heat_map = [ 0 for i in range(len(provider)) ]

        # cache of created augmented images indexed by this augmentation index
        self.imgs_augmented = dict()
        self.masks_augmented = dict()
        self.use_cache_for_augmented = True


    def __len__(self):
        return self.no_of_image_pairs

    def __getitem__(self, idx):
        if idx >= self.no_of_image_pairs:
            raise StopIteration

        # Notice: own caches are used! (never that from the src_provider)
        img = self.imgs_augmented.get(idx)
        msk = self.masks_augmented.get(idx)

        # apply round-robin to choose the original image pair
        src_idx = idx % len(self.src_provider)

        # if anything is missing, redo completely this particular index
        if img is None or msk is None:
            self.usage_heat_map[src_idx] += 1

            # get that pair from the "underlying cache" or the underlying provider
            img = self.imgs_underlying.get(src_idx)
            if img is None:
                img = self.src_provider.io_raw_image_read(src_idx)
                img = self.src_provider.apply_raw_image_spatial_transforms(img)
                # make sure intensities are [0:1] (e.q. gamma() cares about it);
                # any "data-driven" normalization shall not mind that we have
                # rescaled the intensity range -- the values histogram should
                # be preserved (shape-wise) after the rescaling
                img = I.normalize_img_full_range_to_0_1(img)
                self.imgs_underlying[src_idx] = img

            msk = self.masks_underlying.get(src_idx)
            if msk is None:
                msk = self.src_provider.io_mask_image_read(src_idx)
                msk = self.src_provider.apply_mask_image_spatial_transforms(msk)
                self.masks_underlying[src_idx] = msk

            aug_images = self.albumentations_transform(image=img, mask=msk)
            img, msk = aug_images['image'], aug_images['mask']

            img = self.src_provider.apply_raw_image_intensity_transforms(img)
            img = self.src_provider.apply_raw_image_torch_transform(img)

            msk = self.src_provider.apply_mask_image_intensity_transforms(msk)
            msk = self.src_provider.apply_mask_image_channel_transforms(msk)

            # store the final product into the "augmented cache"
            if self.use_cache_for_augmented:
                self.imgs_augmented[idx] = img
                self.masks_augmented[idx] = msk

        self.src_provider.last_used_idx = src_idx
        return img, msk


    # convenience shortcuts
    def last_used_img_filename(self):
        return self.src_provider.last_used_img_filename()

    def last_used_mask_filename(self):
        return self.src_provider.last_used_mask_filename()

    def get_imgs_folder(self):
        return self.src_provider.get_imgs_folder()

    def get_masks_folder(self):
        return self.src_provider.get_masks_folder()



class ImagePairsProvidersChained(Dataset):
    def __init__(self):
        self.providers = list()
        self.boundaries = list()
        self.last_used_idx = -1
        self.last_used_provider = None

    def append(self, provider):
        self.providers.append(provider)
        size = self.boundaries[-1] if len(self.boundaries) > 0 else 0
        self.boundaries.append( size+len(provider) )


    def get_provider_for_idx(self, idx):
        if len(self.providers) == 0:
            return None, 0

        i = 0
        while i < len(self.boundaries) and idx >= self.boundaries[i]:
            i += 1

        if i >= len(self.providers):
            return None, 0

        bi = self.boundaries[i-1] if i > 0 else 0
        return self.providers[i], bi


    def __len__(self):
        return self.boundaries[-1] if len(self.boundaries) > 0 else 0

    def __getitem__(self, idx):
        p,bi = self.get_provider_for_idx(idx)
        if not p:
            print(f"ImagePairsProvidersChained is requested for idx {idx} when there is no provider available.")
            return None, None

        self.last_used_idx = idx
        self.last_used_provider = p
        return p.__getitem__(idx - bi)


    def last_used_img_filename(self):
        return self.last_used_provider.last_used_img_filename() if p else "unknownFile"

    def last_used_mask_filename(self):
        return self.last_used_provider.last_used_mask_filename() if p else "unknownFile"

    def get_imgs_folder(self):
        return self.last_used_provider.get_imgs_folder() if p else "unknownFolder"

    def get_masks_folder(self):
        return self.last_used_provider.get_masks_folder() if p else "unknownFolder"



class ImagePairsProviderFromStack(ImagePairsProvider):
    def __init__(self, stack_raw_image_path, stack_mask_image_path):
        super().__init__(None,None,None,None)
        self.img_stack = imread(stack_raw_image_path)
        self.mask_stack = imread(stack_mask_image_path, clean_up_mask = True)
        self.last_used_idx = -1

        stack_size = self.img_stack.shape[0]
        # create fake lists of input files that are, importantly, of the correct length
        # (as the lengths are queried at multiple places in the upstream class)
        self.img_files =  [ f"{os.path.basename(stack_raw_image_path)}_slice{z}.tif" for z in range(stack_size) ]
        self.mask_files = [ f"{os.path.basename(stack_mask_image_path)}_slice{z}.tif" for z in range(stack_size) ]

    def io_raw_image_read(self, idx):
        if self.max_verbosely_loaded_files > 0:
            self.max_verbosely_loaded_files -= 1
            print(f"I/O: reading raw image from slice: {idx}")
        return self.img_stack[idx,:,:]

    def io_mask_image_read(self, idx):
        if self.max_verbosely_loaded_files > 0:
            self.max_verbosely_loaded_files -= 1
            print(f"I/O: reading mask image from slice: {idx}")
        return self.mask_stack[idx,:,:]

