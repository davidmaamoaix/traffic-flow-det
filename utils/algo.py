import cv2 as cv
import numpy as np


def crop_image(image, box):
    '''crops out the given area'''

    return image[box[1] : box[1] + box[3], box[0] + box[0] + box[2]]