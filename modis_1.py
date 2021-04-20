# -*- coding: utf-8 -*-
"""
Created on Thu Mar 25 19:54:45 2021

@author: bahri
"""

# Import packages
import os
import warnings

import matplotlib.pyplot as plt
import numpy.ma as ma
import xarray as xr
import rioxarray as rxr
# from shapely.geometry import mapping, box
# import geopandas as gpd
# import earthpy as et
# import earthpy.plot as ep
import numpy as np
from glob import glob

warnings.simplefilter('ignore')

# Get the MODIS data
# et.data.get_data('cold-springs-modis-h4')

# # This download is for the fire boundary
# et.data.get_data('cold-springs-fire')

def combine_tifs(tif_list):
    """A function that combines a list of tifs in the same CRS
    and of the same extent into an xarray object

    Parameters
    ----------
    tif_list : list
        A list of paths to the tif files that you wish to combine.

    Returns
    -------
    An xarray object with all of the tif files in the listmerged into 
    a single object.

    """

    out_xr = []
    for i, tif_path in enumerate(tif_list):
        out_xr.append(rxr.open_rasterio(tif_path, masked=True).squeeze())
        out_xr[i]["band"] = i+1

    return xr.concat(out_xr, dim="band")


#Load HDF file


# Set working directory
os.chdir('/home/omar/Data/MODIS/')

pecs_pre_path = os.path.join('Pecs', 'pecs_MOD09_08_18_11_25.hdf')


# Create a path to the pre-fire MODIS h4 data
# modis_pre_path = os.path.join("'earth-analytics',
#                       'data'cold-springs-modis-h4",
#                               "07_july_2016",
#                               "MOD09GA.A2016189.h09v05.006.2016191073856.hdf")

# Open entire data with rioxarray
modis_pre = rxr.open_rasterio(pecs_pre_path, masked=True)

# Open just the bands that you want to process and masking the no-data values
desired_bands = ["sur_refl_b01_1",
                 "sur_refl_b02_1",
                 "sur_refl_b03_1",
                 "sur_refl_b04_1",
                 "sur_refl_b05_1",
                 "sur_refl_b06_1",
                 "sur_refl_b07_1"]
# Notice that here, you get a single xarray object with just the bands that
# you want to work with
modis_pre_bands = rxr.open_rasterio(pecs_pre_path,
                                    masked=True,
                                    variable=desired_bands).squeeze()


# Plot band one of the data
ep.plot_bands(modis_pre_bands.sur_refl_b01_1)
plt.show()

# Note that you can also call the data variable by name
ep.plot_bands(modis_pre_bands["sur_refl_b01_1"])
plt.show()

##Plot all bands with earthpy

# You can plot each band easily using a data array object
modis_pre_bands.to_array()

# Plot the data as a DataArray
# This is only a data exploration step
test = ep.plot_bands(modis_pre_bands.to_array().values,
              figsize=(10, 6))
plt.show()


# Select the rgb bands only
rgb_bands = ['sur_refl_b01_1',
             'sur_refl_b03_1',
             'sur_refl_b04_1']
# Turn the data into a DataArray
modis_rgb_xr = modis_pre_bands[rgb_bands].to_array()

def clean_array_plot(xr_obj):
    # This function takes a single xarray object as an input and produces a
    # cleaned numpy array output for plotting
    # BEGIN SOLUTION
    """
    Take an  xarray object and replace null  values with a mask for plotting

    Parameters
    ----------
    xr_obj : xarray object

    Returns
    -------
    A masked numpy array 

    """
    # END SOLUTION

    return ma.masked_array(xr_obj.values,  xr_obj.isnull())

# For plotting you will want to clean up masked values
# Note that this is not a numpy array
modis_arr = clean_array_plot(modis_rgb_xr)
modis_arr

ep.plot_rgb(modis_arr,
            rgb=[0, 2, 1],
            title='RGB Image of MODIS Data')

plt.show()

#Converting and saving all the bands to tif files
for i in range(7):
    stacked_file_path = os.path.join("Pecs",
                                     "tifs",
                                     "band_" + str(i+1)+ ".tif")
    
    name = "sur_refl_b0" + str(i+1) + "_1"
    
    modis_pre_bands[name].astype(np.int16).rio.to_raster(stacked_file_path)
 
#Load them again    
modis_bands_pre_list = glob(os.path.join("Pecs",
                                         "tifs",
                                         "*.tif"))

# Sort the list of bands
modis_bands_pre_list.sort()

# Open file list with function
modis_bands_pre = combine_tifs(modis_bands_pre_list)
   
rgb_bands = [0,3,2]

# Scale values of MODIS imagery stack
modis_bands_pre_scaled = modis_bands_pre * 0.0001   
# Actually scale them to [0-1]
modis_bands_pre_scaled_01 = (modis_bands_pre_scaled-modis_bands_pre_scaled.min())/(modis_bands_pre_scaled.max()-modis_bands_pre_scaled.min()) 

modis_rgb_01 = modis_bands_pre_scaled_01[rgb_bands]
 
modis_rgb_01_s = np.swapaxes(modis_rgb_01, 0, 2)
modis_rgb_01_s = np.swapaxes(modis_rgb_01_s, 0, 1)

#Save as JPG
plt.imsave(os.path.join("Pecs", "rgb.jpg"), modis_rgb_01_s)

    
    
    
    

# # Get the directory needed for the defined path
# modis_dir_path = os.path.dirname(stacked_file_path)
# print("Directory to save path:", modis_dir_path)

# # Create the directory if it does not exist
# if not os.path.exists(modis_dir_path):
#     os.mkdir(modis_dir_path)
#     print("The directory", modis_dir_path, "does not exist - creating it now.")

# # Here you decide how much of the data you want to export.
# # A single layer vs a stacked / array
# # Export a single band to a geotiff
# modis_rgb_xr.rio.to_raster(stacked_file_path)

# clean_modis_band_1 = clean_array_plot(modis_pre_bands["sur_refl_b01_1"])