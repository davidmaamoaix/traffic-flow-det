import cv2 as cv
import numpy as np


def crop_image(image, box, shape=None):
    '''
    crops out the given area

    shape: tuple, changes the width of the box according to this
        shape (makes no sense I know, but putting this in for reproducibility)
    '''
    if shape is not None:
        ratio = shape[1] / shape[0]
        new_width = ratio * box[3]
        box[0] -= (new_width - box[2]) // 2
        box[2] = new_width
    else:
        shape = box[2], box[3]

    patch = image[box[1] : box[1] + box[3], box[0] : box[0] + box[2]]

    return cv.resize(patch, tuple(shape[:: -1]))