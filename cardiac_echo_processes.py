import os
import cv2
import random
import numpy as np
import SimpleITK as sitk
import matplotlib.pyplot as plt

def segment_cardiac_echo(frame, standardized=False):
    """
    code to segment just the ultrasound image of the cardiac ultrasound frames

    input:
        frame: the raw ultrasound array frame
        standardized: option for a standardized dataset input
    output:
        frame_crop: segmented cardiac ultrasound
    """

    if standardized:  # if it's a standarized ultrasound, just crop a rectangle
        # adding a normalization step in the middle so that the ultrasound isn't as dark
        frame_h = int(frame.shape[0] * 0.1)
        frame_w = int(frame.shape[1] * 0.25)
        frame_crop = frame[frame_h:-frame_h, frame_w:-frame_w]
    else:  # if not standardized, use James' code - need some cleaning up
        ret2, threshhold = cv2.threshold(frame, 29, 255, 0)
        contours, hierarchy = cv2.findContours(threshhold, 1, 2)  # need to change this so it picks up more things
        # Approx contour
        cnt = contours[0]
        largest = cv2.contourArea(cnt)
        contours = sorted(contours, key=lambda x: cv2.contourArea(x), reverse=True)
        cnt = contours[0]
        # Central points and area
        moments = cv2.moments(cnt)
        cent_x = int(moments['m10'] / moments['m00'])
        cent_y = int(moments['m01'] / moments['m00'])
        shape_area = cv2.contourArea(cnt)
        shape_perim = cv2.arcLength(cnt, True)
        epsilon = 0.01 * shape_perim
        approximation = cv2.approxPolyDP(cnt, epsilon, True)
        convex_hull = cv2.convexHull(cnt)
        contour_mask = np.zeros(frame.shape, np.uint8)
        contour_mask = cv2.drawContours(contour_mask, [convex_hull], 0, 255, -1)

        frame_crop = frame * contour_mask

    return frame_crop

def ablate_cardiac_echo(frame_crop, quadrant=None):
    """
    code to remove different quadrants of the cardiac ultrasound frames

    input:
        frame_crop: the raw ultrasound array to crop a quadrant from
        quadrant: option to remove a quadrant of the image for ablation study
    output:
        frame_crop: segmented cardiac ultrasound (w/w0 quadrant ablation)
    """

    if quadrant != None:
        frame_center_h = int(frame_crop.shape[0] * 0.6)
        frame_center_w = int(frame_crop.shape[1] * 0.6)
        if quadrant == 1:
            frame_crop[:frame_center_h, :frame_center_w] = 0
        elif quadrant == 2:
            frame_crop[frame_center_h:, :frame_center_w] = 0
        elif quadrant == 3:
            frame_crop[:frame_center_h, frame_center_w:] = 0
        else:
            frame_crop[frame_center_h:, frame_center_w:] = 0

    return frame_crop