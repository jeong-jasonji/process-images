import numpy as np
import SimpleITK as sitk


def segment_breast(image_np):
    """
    Function to crop the breast only from mammograms using sitk's connected components

    inputs:
        image_np: input image array, in numpy format
    returns:
        img_crop: numpy array cropped to the breast
        coord: coordinates for the numpy array to crop the breast
    """

    img = sitk.GetImageFromArray(image_np)

    boundary_bg = sitk.BinaryThreshold(img, 0, int(image_np.min()), 1, 0)
    seg_bg = sitk.BinaryMorphologicalClosing(boundary_bg, (1, 1, 1), sitk.sitkBall)
    seg_bg_close_open = sitk.BinaryMorphologicalOpening(boundary_bg, (30, 30, 30), sitk.sitkBall)
    seg_image_bg = seg_bg_close_open
    seg_image_bg_np = sitk.GetArrayFromImage(seg_image_bg)

    # invert the image so that background is 0 and objects are 1
    where_0 = np.where(seg_image_bg_np == 0)
    where_1 = np.where(seg_image_bg_np == 1)
    seg_image_bg_np[where_0] = 1
    seg_image_bg_np[where_1] = 0
    inv_seg_image_bg_np = seg_image_bg_np
    inv_seg_image_bg = sitk.GetImageFromArray(inv_seg_image_bg_np)

    # get bounding boxes of the objects (bb coordinates: x_start, y_start, x_length, y_length)
    image_cc = sitk.ConnectedComponent(inv_seg_image_bg)
    stats = sitk.LabelShapeStatisticsImageFilter()
    stats.Execute(image_cc)

    obj_size = 0
    coord = []
    for l in stats.GetLabels():
        if obj_size < stats.GetPhysicalSize(l):
            obj_size = stats.GetPhysicalSize(l)
            coord = stats.GetBoundingBox(l)

    img_crop = image_np[coord[1]:coord[1] + coord[3], coord[0]:coord[0] + coord[2]]

    return img_crop, coord