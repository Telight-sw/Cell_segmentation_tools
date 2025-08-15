import numpy as np
import numpy.typing as npt
from skimage.transform import resize

# Here are examples of ready-made image "massaging" functions

def reduce_potentially_last_dim(src_img: npt.NDArray, keep_min_dims: int, fix_last_dim_to_this_value: int) -> npt.NDArray:
    if len(src_img.shape) <= keep_min_dims:
        return src_img
    return src_img[..., fix_last_dim_to_this_value]

def downsize_and_to_float_img(src_img: npt.NDArray, new_x_size: int, new_y_size: int) -> npt.NDArray:
    img = np.zeros((new_y_size,new_x_size), dtype='float32')
    # make sure the size doesn't get outside any of the two images
    new_y_size = min(new_y_size, src_img.shape[0])
    new_x_size = min(new_x_size, src_img.shape[1])
    img[0:new_y_size,0:new_x_size] = src_img[0:new_y_size,0:new_x_size]
    return img

def upsize_and_to_float_img(src_img: npt.NDArray, new_x_size: int, new_y_size: int) -> npt.NDArray:
    img = np.zeros((new_y_size,new_x_size), dtype='float32')
    old_y_size = src_img.shape[0]
    old_x_size = src_img.shape[1]
    # make sure the size doesn't get outside any of the two images
    old_y_size = min(new_y_size, old_y_size)
    old_x_size = min(new_x_size, old_x_size)
    img[0:old_y_size,0:old_x_size] = src_img[0:old_y_size,0:old_x_size]
    return img

def scale_to_and_to_float_img(src_img: npt.NDArray, new_x_size: int, new_y_size: int, doing_mask: bool = False) -> npt.NDArray:
    # NB: preserves the dimensionality of the input image
    new_size = [ new_x_size, new_y_size, *src_img.shape[2:] ]
    return np.array(resize(src_img, new_size, preserve_range=True, order=0), dtype='float32') \
        if doing_mask else np.array(resize(src_img, new_size, preserve_range=True), dtype='float32')

def to_float_img(src_img: npt.NDArray) -> npt.NDArray:
    return src_img.astype('float32')


def normalize_img_full_range_to_0_1(float_img: npt.NDArray) -> npt.NDArray:
    m = float_img.min()
    r = float_img.max() - m
    float_img = (float_img - m) / r
    return float_img

def normalize_img_given_range_to_0_1(float_img: npt.NDArray, old_min: float, old_max: float) -> npt.NDArray:
    float_img[ float_img < old_min ] = old_min
    float_img[ float_img > old_max ] = old_max
    r = old_max - old_min
    float_img = (float_img - old_min) / r
    return float_img

def normalize_img_mean_std_both_to_0point5(float_img: npt.NDArray) -> npt.NDArray:
    float_img -= float_img.mean()
    float_img /= 5.6*float_img.std() # nearly 6*sigma the width
    float_img += 0.5
    return clip_to_0_1(float_img)


def get_percentiles(image: npt.NDArray, list_of_percentiles: list[float]) -> list[any]:
    values = sorted(image.flat)
    cnt = float(len(values))
    return [ values[ int(p*cnt) ] for p in list_of_percentiles ]

def normalize_img_percentiles_range_to_0_1(float_img: npt.NDArray, low_percentile: float, high_percentile: float) -> npt.NDArray:
    low_value, high_value = get_percentiles(float_img, [low_percentile, high_percentile])
    return normalize_img_given_range_to_0_1(float_img, low_value, high_value)


def clip_to_0_1(float_img: npt.NDArray) -> npt.NDArray:
    float_img[ float_img < 0.0 ] = 0.0
    float_img[ float_img > 1.0 ] = 1.0
    return float_img


def find_bg_value(sorted_px_values: list[float]) -> float:
    """
    The input list shall be pixel values within the range of [0;1], and sorted in the ascending order.
    The function returns the index to the 'sorted_px_values', it thus points at the value that could
    # be used as the bottom cut-off in some further normalization.
    """
    diff = [ v for v in sorted_px_values[1:] ]
    for i in range(len(diff)):
        diff[i] -= sorted_px_values[i]
    #NB: this has created a list of forward differences

    # search from the beginning (from the lowest px values) and find the first "plateau",
    # which is when 10 consecutive differences have average value below the "threshold"
    for i in range(len(diff)-10):
        sum_diff = sum(diff[i:i+10])
        if sum_diff < 0.0005:
            return i #,diff

    # safe default...
    return 0 #,diff


def find_rareHigh_value(sorted_px_values: list[float]) -> float:
    """
    The input list shall be pixel values within the range of [0;1], and sorted in the ascending order.
    The function returns the index to the 'sorted_px_values', it thus points at the value that could
    # be used as the top cut-off in some further normalization.
    """
    diff = [ v for v in sorted_px_values[1:] ]
    for i in range(len(diff)):
        diff[i] -= sorted_px_values[i]
    #NB: this has created a list of forward differences

    # search from the end (from the highest px values) and find the first "plateau",
    # which is when 10 consecutive differences have average value below the "threshold"
    for i in range(len(diff)-1,10,-1):
        sum_diff = sum(diff[i-10:i])
        if sum_diff < 0.0005:
            return i #,diff

    # safe default...
    return len(sorted_px_values)-1 #,diff


def normalize_img_auto_range_to_0_1(float_img: npt.NDArray) -> npt.NDArray:
    float_img01 = normalize_img_full_range_to_0_1(float_img)
    values = sorted( float_img01.flat )

    idxL = find_bg_value(values)
    idxH = find_rareHigh_value(values)

    return normalize_img_given_range_to_0_1(float_img01, values[idxL],values[idxH])


def normalize_and_binarize_mask(float_mask: npt.NDArray) -> npt.NDArray:
    float_mask[ float_mask.nonzero() ] = 1.0
    return float_mask


from skimage.measure import label, regionprops

def clean_up_masks(mask_image):
    """
    After the augmentation, due to boundary flipping, it can happen that a mask
    gets mirrored resulting in two individual/disconnected masks of the _same_ label.
    Similarly, due to cropping, a "border line" of a cropped-out mask can be left
    in the image resulting in (again, multiple disconnected masks of the same
    label) very narrow (1px wide) and/or small patches (less than 25 pixels, 5x5)
    that this routine is also detecting and removing.
    """
    labels = label(mask_image)
    for region in regionprops(labels):
        if region.area < 25:
            # remove this small blob
            for y,x in region.coords:
                mask_image[y,x] = 0
        b = region.bbox
        if b[2]-b[0] == 1 or b[3]-b[1] == 1:
            # remove the thin stripe
            for y,x in region.coords:
                mask_image[y,x] = 0
        else:
            # relabel (with an unique label) as there could
            # have been multiple CCAs of the same label, ...to be on the safe side
            for y,x in region.coords:
                mask_image[y,x] = region.label


def clean_up_stack_of_masks(mask_stack):
    """
    The function works also with a plain 2D image. If indeed a stack is provided,
    it calls 'clean_up_masks()' on each slice individually.
    """
    if len(mask_stack.shape) < 3:
        clean_up_masks(mask_stack)
        return

    for z in range(mask_stack.shape[0]):
        clean_up_masks(mask_stack[z])


def has_more_than_two_fg_values(mask_roi: npt.NDArray) -> int:
    """
    Returns 1 if the `mask_roi` contains at least two different non-zero pixels, otherwise it returns 0.
    """
    observed_fg_px = 0
    for px in mask_roi.flat:
        if px > 0:
            # first fg pixel observed?
            if observed_fg_px == 0:
                # yes, remember it
                observed_fg_px = px
            # else, is this some another fg pixel
            elif observed_fg_px != px:
                return 1
    return 0

def mark_touching_mask_boundary(mask: npt.NDArray, width:int = 5) -> npt.NDArray:
    from scipy.ndimage import generic_filter
    from scipy.ndimage import grey_dilation

    return grey_dilation( \
        generic_filter(mask, has_more_than_two_fg_values, size = 3, mode = 'constant'), \
        size=width )


def has_at_least_one_bg_and_fg_values(mask_roi: npt.NDArray) -> int:
    """
    Returns 1 if the `mask_roi` contains at least one zero and one non-zero pixel, otherwise it returns 0.
    """
    observed_fg = False
    observed_bg = False
    for px in mask_roi.flat:
        if px > 0:
            observed_fg = True
        else:
            observed_bg = True
        if observed_bg and observed_fg:
            return 1
    return 0


def has_either_bg_and_fg_or_two_fg_values(mask_roi: npt.NDArray) -> int:
    """
    Returns 1 if the `mask_roi` is showing an interface of a mask instance to another mask or boundary, otherwise it returns 0.
    """
    observed_fg_px = 0
    observed_bg = False
    for px in mask_roi.flat:
        if px == 0:
            observed_bg = True
        elif observed_fg_px == 0:
            observed_fg_px = px
        elif observed_fg_px != px:
            return 1
        if observed_bg and observed_fg_px > 0:
            return 1
    return 0


def mark_mask_boundary(mask: npt.NDArray, width:int = 5) -> npt.NDArray:
    from scipy.ndimage import generic_filter
    from scipy.ndimage import grey_dilation

    return grey_dilation( \
        generic_filter(mask, has_either_bg_and_fg_or_two_fg_values, size = 3, mode = 'constant'), \
        size=width )


def mark_mask_inner_boundary(mask: npt.NDArray, width:int = 7) -> npt.NDArray:
    mB = mark_mask_boundary(mask, width)
    return mB * (mask > 0)


#import skimage.morphology as MORPHO
import scipy.ndimage as SNDI
def augbg_transition_multiplicative_mask(fg_mask: npt.NDArray, transition_width:int = 10) -> npt.NDArray:
    '''
    - binarize (fg = 0, bg = 1) and distance transform
    - threshold above 'transition_width' to 'transition_width'
    - divide pixel values by `transition_width` to push them to the domain [0,1]
    Returned float image acts as "mask of weights".
    '''
    dt = SNDI.distance_transform_edt( 1 - fg_mask.clip(0,1) )
    dt[ dt > transition_width ] = transition_width
    dt = dt/float(transition_width)
    return dt


def augbg_sample_background(raw: npt.NDArray, mask: npt.NDArray, step: int = 30):
    xcoords = []
    ycoords = []
    raw_values = []
    for y in range(0,raw.shape[0],step):
        for x in range(0,raw.shape[1],step):
            if mask[y,x] == 0:
                ycoords.append(y)
                xcoords.append(x)
                raw_values.append(raw[y,x])
    return xcoords,ycoords,raw_values


bg_sampling_step = 30
fitting_poly_order = 4
transition_width = 15

def augbg_get_bg_model_img(raw: npt.NDArray, mask: npt.NDArray) -> npt.NDArray:
    '''
    Returns the smoothed background image (a hypothetical noise-free model of
    the background signal) and corresponding transition multiplicative mask.

    - poly-fit to real background (while ignoring fg rois)
    - apply poly with mixed-transition to fg rois only
        - removes them by replacing them with the fitted background
    - intensity clip non-fg bright regions
    - smoothout with Gaussian
    '''

    # INFO: x = 1D list of x-coordinates
    # INFO: y = 1D list of y-coordinates
    # INFO:            coordinate is thus [ x[idx],y[idx] ]
    # INFO:
    # INFO: basis = get_basis(x,y, order)
    # INFO: 1D list (over polynom's terms, the number of them is given from 'order')
    # INFO:     of 1D lists (value of the polynom's term at the coordinate)
    # INFO:
    # INFO: A = np.vstack(basis).T
    # INFO:   makes it a pure numpy array,
    # INFO:   no. of rows = no. of coordinates,
    # INFO:   no. of columns = no. of polynom's terms
    # INFO:
    # INFO: b = 1D numpy of values to fit into
    # INFO:
    # INFO: to be solved:
    # INFO:
    # INFO: A*c = b
    # INFO:
    # INFO: len(c) is number of polynom's terms... A*c creates no. of coords cases, polynom with its coordinates

    # get poly
    xcoords,ycoords,raw_values = augbg_sample_background(raw,mask, step=bg_sampling_step)
    print(f"sampled background with {len(xcoords)} values using steps of {bg_sampling_step} pixels")

    basis = get_basis(np.array(xcoords,dtype='float32'), np.array(ycoords,dtype='float32'), fitting_poly_order)
    A = np.vstack(basis).T
    b = np.array(raw_values,dtype='float32')
    c, r, rank, s = np.linalg.lstsq(A, b, rcond=None)
    ## NB: len(basis) > rank -> problem
    ## NB: fitted_values = np.matmul(A,c)
    print(f"fitted polynom of order {fitting_poly_order} that has {len(basis)} terms, rank={rank}")

    dt = augbg_transition_multiplicative_mask(mask, transition_width)
    fg_y, fg_x = np.where( dt < 1.0 ) # non-clean bg areas
    print(f"found {len(fg_x)} pixels in fg+transition areas, transition width={transition_width}")

    basis = get_basis(fg_x, fg_y, fitting_poly_order)
    fitted_values = np.matmul(np.vstack(basis).T, c)

    new_raw = raw.copy()
    for i,x in enumerate(fg_x):
        y = fg_y[i]
        dt_val = dt[y,x]
        new_raw[y,x] = dt_val*raw[y,x] + (1.0-dt_val)*fitted_values[i]

    # NB: 'new_raw' is now the original image in the bg area (where mask == 0);
    # based on pixel values from exactly that bg area a polynom is fitted
    # and sampled in the fg area (where mask > 0); in this way, fg cells are
    # replaced with artificial background

    # since not everything that could possibly be fg is marked in the masks,
    # so substantially brighter spots are clipped too -- to further smooth the signal
    max_bg_val = new_raw.mean() + new_raw.std()
    new_raw[ new_raw > max_bg_val ] = max_bg_val

    # finally, smoothen it out
    return SNDI.gaussian_filter(new_raw, 15), dt


def augbg_subtract_bg(raw, mask):
    bg_model,dt = augbg_get_bg_model_img(raw, mask)

    # find transition zone + pure background
    fg_y, fg_x = np.where( dt > 0.0 ) # outside fg areas

    rawfg = raw.copy()
    for i,x in enumerate(fg_x):
        y = fg_y[i]
        rawfg[y,x] -= dt[y,x] * bg_model[y,x]

    return rawfg


def augbg_add_bg(raw, mask, bg_model):
    dt = augbg_transition_multiplicative_mask(mask, transition_width)

    # find transition zone + pure background
    fg_y, fg_x = np.where( dt > 0.0 ) # outside fg areas

    rawfg = raw.copy()
    for i,x in enumerate(fg_x):
        y = fg_y[i]
        rawfg[y,x] += dt[y,x] * bg_model[y,x]

    return rawfg


def augbg_bg_from_A_implant_to_B(rawA,maskA, rawB,maskB):
    bg_modelA,_   = augbg_get_bg_model_img(rawA, maskA)
    return augbg_bg_model_implant_to_B(bg_modelA, rawB,maskB)


def augbg_bg_model_implant_to_B(bg_modelA, rawB,maskB):
    bg_modelB,dtB = augbg_get_bg_model_img(rawB, maskB)

    # find transition zone + pure background
    fg_y, fg_x = np.where( dtB > 0.0 )

    rawfg = rawB.copy()
    for i,x in enumerate(fg_x):
        y = fg_y[i]
        rawfg[y,x] += dtB[y,x] * (bg_modelA[y,x] - bg_modelB[y,x])

    return rawfg


# ======================== polynoms ========================
def get_basis(x, y, max_order=4):
    """
    Return the fit basis polynomials: 1, x, x^2, ..., xy, x^2y, ... etc.

    If x,y are arrays of lengths N (same length both), the output is also
    an array of N-long arrays.

    It is called `basis` because it returns the polynomial without coefficients
    for each term, as if all the coefficients were 1.0.
    """
    basis = []
    for i in range(max_order+1):
        for j in range(max_order - i +1):
            basis.append(x**j * y**i)
    return basis


def eval_poly_at(coefs,max_order, at_x,at_y):
    base = get_basis(at_x,at_y, max_order)

    val = 0.0
    for i in range(len(coefs)):
        val += coefs[i] * base[i]
    return val


def show_fitting(xcoords,ycoords,raw_values,fitted_values):
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    ax.plot3D(xcoords,ycoords,raw_values,'bo')
    ax.plot3D(xcoords,ycoords,fitted_values,'ro')
    plt.show()
# ======================== polynoms ========================

