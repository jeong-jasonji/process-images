import torch
import numpy as np
from PIL import Image


# windowing code
def CT_windowing(array_raw, hu_min, hu_max, intercept):
    """
    Function to change the intercept and window the pixel array of CT images

    inputs:
        array_raw: raw pixel array from a dicom file
        hu_min: minimum HU for windowing
        hu_max: maximum HU for windowing
        intercept: the intercept to adjust the array values
    returns:
        array_windowed: HU windowed pixel array
    """
    # linear transformation of raw pixel values from a read dicom pixel array
    array_windowed = array_raw * 1 + intercept
    # def min and max pixel values
    array_windowed[array_windowed > hu_max] = hu_max
    array_windowed[array_windowed < hu_min] = hu_min

    return array_windowed


# change code so that it just uses options not intputs
def interpolate_to_PIL(np_array, window, n_channels_out=3):
    """
    Function to change the raw dicom pixel array into a windowed image

    inputs:
        np_array: raw pixel array from a dicom file
        window: the min, max, and intercept of the HU windows
        n_channels_out: channels of the output image
    returns:
        array_windowed: HU windowed pixel array
    """

    # get the window values
    window_min = window[0]
    window_max = window[1]
    intercept = window[2]

    # convert to float 32
    np_32 = np.float32(np_array)

    # change the windows
    np_32 = CT_windowing(np_32, window_min, window_max, intercept)

    # add the min to the array so that we can normalize to 0-1
    np_32 += np.abs(np_32.min())
    # make image normalized
    normal_np_32 = np_32 / np_32.max()

    if n_channels_out == 1:  # should train with one channel images
        # change to 8 bit
        np_8 = normal_np_32 * 255
        np_8 = np_8.astype('uint8')
        # expand dimension channel last
        image = Image.fromarray(np_8)
    elif n_channels_out == 3:
        # expand dimension channel last
        np_32_3ch = np.stack((normal_np_32,) * 3, axis=0)
        image = torch.tensor(np_32_3ch)
    else:
        print('input channels must be 1 or 3')

    return image