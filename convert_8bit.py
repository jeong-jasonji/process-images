import numpy as np

def convertto8bit(array):
    """
    Function to convert 16bit or more images to 8 bit through normalization

    inputs:
        array: an array containing more than 8 bit information
    returns:
        array_255: an 8bit image [0, 255]
        array_1: a normalized image from [0, 1]
    """

    pos_array = array + np.abs(array.min())
    array_1 = pos_array / pos_array.max()
    array_255 = array_1 * 255
    array_255 = array_255.astype('uint8')

    return array_255, array_1