#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 23 11:58:23 2021

@author: omar
"""

import numpy as np
from PIL import Image
import os
import scipy.ndimage
from matplotlib import pyplot as plt

directory = "./Data/"

# load the .tif files into numpy arrays
# cloud top height
CTH_tif = Image.open(f'{directory}/mod06_CTH.tif')

# cloud optical thickness
COT_tif = Image.open(f'{directory}/mod06_COT.tif')

# cloud top pressure
CTP_tif = Image.open(f'{directory}/mod06_CTP.tif')

# cloud phase optical properties
CPOP_tif = Image.open(f'{directory}/mod06_CPOP.tif')

# get the shape of the arrays
width, height = CTH_tif.size 

# crop and upsample the arrays
crop_factor = 16
upsampling_factor = 10

CTH_cropped = CTH_tif.crop((0, 0, width//crop_factor, height//crop_factor))
COT_cropped = COT_tif.crop((0, 0, width//crop_factor, height//crop_factor))
CTP_cropped = CTP_tif.crop((0, 0, width//crop_factor, height//crop_factor))
CPOP_cropped = CPOP_tif.crop((0, 0, width//crop_factor, height//crop_factor))

width, height = CPOP_cropped.size 

CTH_zoomed = CTH_cropped.resize((width*upsampling_factor, height*upsampling_factor))
COT_zoomed = COT_cropped.resize((width*upsampling_factor, height*upsampling_factor))
CTP_zoomed = CTP_cropped.resize((width*upsampling_factor, height*upsampling_factor))
CPOP_zoomed = CPOP_cropped.resize((width*upsampling_factor, height*upsampling_factor))

# write the arrays to .npy files
np.save(f'{directory}/mod06_upsampled/mod06_CTH.npy', np.asarray(CTH_zoomed))
np.save(f'{directory}/mod06_upsampled/mod06_COT.npy', np.asarray(COT_zoomed))
np.save(f'{directory}/mod06_upsampled/mod06_CTP.npy', np.asarray(CTP_zoomed))
np.save(f'{directory}/mod06_upsampled/mod06_CPOP.npy', np.asarray(CPOP_zoomed))

# load the RGB image
rgb = Image.open(f'{directory}/rgb.jpg')
width, height = rgb.size 

# crop it 
rgb_cropped = rgb.crop((0, 0, width//(crop_factor*2), height//(crop_factor*2)))
width, height = rgb_cropped.size 

# upsample it
rgb_up = rgb_cropped.resize((width*upsampling_factor, height*upsampling_factor))

# write it
rgb_up.save(f'{directory}/mod06_upsampled/rgb.jpg')

