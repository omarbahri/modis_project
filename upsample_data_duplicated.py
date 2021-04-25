<<<<<<< HEAD

=======
>>>>>>> d7ca14f7bc1a1bd627bdbff1f392c9dc001e1956
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
from scipy.ndimage.filters import gaussian_filter

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

# cloud temperature
CTT_tif = Image.open(f'{directory}/mod06_CTT.tif')

# get the shape of the arrays
width, height = CTH_tif.size 

# crop, upsample and blur the arrays
crop_factor = 32
upsampling_factor = 20

n = 0

left = n*width//crop_factor
top = n*height//crop_factor
right = (n+1)*width//crop_factor
bottom = (n+1)*height//crop_factor

CTH_cropped = CTH_tif.crop((left, top, right, bottom))
COT_cropped = COT_tif.crop((left, top, right, bottom))
CTP_cropped = CTP_tif.crop((left, top, right, bottom))
CPOP_cropped = CPOP_tif.crop((left, top, right, bottom))
CTT_cropped = CTT_tif.crop((left, top, right, bottom))

width, height = CPOP_cropped.size 

CTH_zoomed = CTH_cropped.resize((width*upsampling_factor, height*upsampling_factor))
COT_zoomed = COT_cropped.resize((width*upsampling_factor, height*upsampling_factor))
CTP_zoomed = CTP_cropped.resize((width*upsampling_factor, height*upsampling_factor))
CPOP_zoomed = CPOP_cropped.resize((width*upsampling_factor, height*upsampling_factor))
CTT_zoomed = CTT_cropped.resize((width*upsampling_factor, height*upsampling_factor))

blur_sdev = 37

CTH_blurred = gaussian_filter(np.asarray(CTH_zoomed), sigma=blur_sdev)
COT_blurred = gaussian_filter(np.asarray(COT_zoomed), sigma=blur_sdev)
CTP_blurred = gaussian_filter(np.asarray(CTP_zoomed), sigma=blur_sdev)
CPOP_blurred = gaussian_filter(np.asarray(CPOP_zoomed), sigma=blur_sdev)
CTT_blurred = gaussian_filter(np.asarray(CTT_zoomed), sigma=blur_sdev)


# blur and write the arrays to .npy files
np.save(f'{directory}/mod06_upsampled/mod06_CTH.npy', CTH_blurred)
np.save(f'{directory}/mod06_upsampled/mod06_COT.npy', COT_blurred)
np.save(f'{directory}/mod06_upsampled/mod06_CTP.npy', CTP_blurred)
np.save(f'{directory}/mod06_upsampled/mod06_CPOP.npy', CPOP_blurred)
np.save(f'{directory}/mod06_upsampled/mod06_CTT.npy', CTT_blurred)

# load the RGB image
rgb = Image.open(f'{directory}/rgb.jpg')
width, height = rgb.size 

# crop it 
rgb_cropped = rgb.crop((left, top, right, bottom))
width, height = rgb_cropped.size 

# upsample it
rgb_up = rgb_cropped.resize((width*upsampling_factor, height*upsampling_factor))

# write it
rgb_up.save(f'{directory}/mod06_upsampled/rgb.jpg')

