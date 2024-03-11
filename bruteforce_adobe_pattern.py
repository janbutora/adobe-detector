# -*- coding: UTF-8 -*-
"""
@author: Jan Butora (jan.butora@univ-lille.fr)
@author: Patrick Bas (patrick.bas@univ-lille.fr)
CNRS, University of Lille 2024
"""

import tifffile as tiff
import numpy as np

def estimate_pattern():
    """
    Estimates the Adobe pattern by brute-forcing it on constant images developed with Adobe Lightroom Classic. Each image contains only 1 value in the 16-bit domain, but in the 8-bit domain, can contain up to 2 values due to the Adobe pattern effect.
    :return: The expected Adobe pattern of size (128,128)
    """
    im_path = './const_images/'
    W = np.zeros((128,128)) - 300
    for i in range(0,256):
        j = i//256
        t = tiff.imread(im_path + str(i) + '_16_8.tif')[:,:,0]
        W[(t!=j)*(W==-300)] = 256-i

    for i in range(256+128,256,-1):
        j = i//256
        t = tiff.imread(im_path + str(i) + '_16_8.tif')[:,:,0]
        W[(t!=j)*(W==-300)] = 256-i-1

    W = (W-128)/256

    return W
