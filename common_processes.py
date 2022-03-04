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

def grayscale2rgb(array, image_out=False):
    """
    Function to convert any bitdepth grayscale image to 8 bit RGB image

    inputs:
        array: a grayscale array
        image_out: a boolian to retrieve an PIL image our or not
    returns:
        array_rgb: a 3channel RGB 8bit array
    """

    pos_array = array + np.abs(array.min())
    array_1 = pos_array / pos_array.max()
    array_255 = array_1 * 255
    array_3c = np.stack((array_255, ) * 3, axis=-1)
    if image_out:
        array_rgb = Image.fromarray(array_3c, mode='RGB')
    else:
        array_rgb = array_3c.astype('uint8')
        
    return array_rgb