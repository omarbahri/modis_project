# -*- coding: utf-8 -*-
"""
Created on Tue Mar 30 16:22:23 2021

@author: bahri
"""

from glob import glob
import os

import numpy as np
import numpy.ma as ma
import matplotlib.pyplot as plt
import geopandas as gpd
import rioxarray as rxr
import xarray as xr
from rasterio.plot import plotting_extent
import earthpy as et
import earthpy.spatial as es
import earthpy.plot as ep
from shapely.geometry import box


# Create list of MODIS rasters for surface reflectance
modis_bands_pre_list = glob(os.path.join("C:\\Users\\bahri\\Documents\\Data\\Satellite\\MODIS\\earth-analytics\\data\\"
                                        "cold-springs-fire",
                                         "modis",
                                         "reflectance",
                                         "07_july_2016",
                                         "crop",
                                         "*_sur_refl_b*.tif"))

# Sort the list of bands
modis_bands_pre_list.sort()
modis_bands_pre_list

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

# Open file list with function
modis_bands_pre = combine_tifs(modis_bands_pre_list)

# Plot MODIS RGB
# Take out the masked values to plot with ep.plot_rgb
modis_bands_pre_plot = ma.masked_array(
    modis_bands_pre.values, modis_bands_pre.isnull())

ep.plot_rgb(modis_bands_pre_plot,
            rgb=[0, 3, 2],
            title="Surface Reflectance \n MODIS RGB Bands")
plt.show()

# Identify minimum and maximum values of band 1 (red)
print(modis_bands_pre[1].min(), modis_bands_pre[1].max())

# Create a list of titles
titles = ["Red Band", "Near Infrared (NIR) Band", "Blue/Green Band", "Green Band",
          "Near Infrared (NIR) Band", "Mid-infrared Band", "Mid-infrared Band"]

# Plot all bands individually
ep.plot_bands(modis_bands_pre,
              cols=3,
              title=titles,
              figsize=(10, 6))
plt.show()


# Create a colors and titles list to use in the histogram
colors = ['r', 'k', 'b', 'g', 'k', 'y', 'y']
titles = ["Red Band", "Near Infrared (NIR) Band", "Blue/Green Band",
          "Green Band", "Near Infrared (NIR) Band",
          "Mid-infrared Band", "Mid-infrared Band"]

# Plot histogram
ep.hist(modis_bands_pre.values,
        colors=colors,
        title=titles,
        cols=2)
plt.show()

# Scale values of MODIS imagery stack
modis_bands_pre_scaled = modis_bands_pre * 0.0001

# Identify minimum and maximum values of scaled band 1 (red)
print(modis_bands_pre_scaled[1].min(), modis_bands_pre_scaled[1].max())

# Create a colors and titles list to use in the histogram
colors = ['r', 'k', 'b', 'g', 'k', 'y', 'y']
titles = ["Red Band", "Near Infrared (NIR) Band", "Blue/Green Band", "Green Band",
          "Near Infrared (NIR) Band", "Mid-infrared Band", "Mid-infrared Band"]

# Plot histogram
ep.hist(modis_bands_pre_scaled.values,
        colors=colors,
        title=titles,
        cols=2)
plt.show()

#**************************************************************************************

modis_pre_path = os.path.join("cold-springs-modis-h4",
                              "07_july_2016",
                              "MOD09GA.A2016189.h09v05.006.2016191073856.hdf")

stacked_file_path = os.path.join("C:\\Users\\bahri\\Documents\\Data\\Satellite\\MODIS\\earth-analytics\\data",
                                 os.path.dirname(modis_pre_path),
                                 "final_output",
                                 "modis_my_rgb_01_s.tif")

modis_bands_pre[0].rio.to_raster(stacked_file_path)


rgb_bands = [0,2,3]

# Turn the data into a DataArray
modis_rgb_xr = modis_bands_pre[rgb_bands]

modis_rgb_xr.rio.to_raster(stacked_file_path)

rgb =  65536/255. * modis_bands_pre[2] + 256/255. * modis_bands_pre[3] + modis_bands_pre[0]/255.
rgb.rio.to_raster(stacked_file_path)

plt.imshow(rgb)

modis_bands_pre_rgb = modis_bands_pre_scaled * 255.
#HERE MIN-MAX SCALING
modis_bands_pre_rgb_255 =  np.clip(modis_bands_pre_rgb, 0, 255)
modis_rgb_255 = modis_bands_pre_rgb_255[rgb_bands]

modis_rgb_255_s= np.swapaxes(modis_rgb_255, 0, 2)
modis_rgb_255_s = np.swapaxes(modis_rgb_255_s, 0, 1)

modis_rgb_255_temp = modis_rgb_255_s
modis_rgb_255_s[:,:,1] = modis_rgb_255_temp[:,:,2]
modis_rgb_255_s[:,:,2] = modis_rgb_255_temp[:,:,1]

plt.imshow(modis_rgb_255_s.astype(int))

#plt.imsave(stacked_file_path+"modis.jpg", modis_rgb_255_vf)

modis_rgb_255_vf.rio.to_raster(stacked_file_path)


modis_bands_pre_scaled_01 =  np.clip(modis_bands_pre_scaled, 0, 1) #HERE MIN-MAX SCALING
modis_rgb_01 = modis_bands_pre_scaled_01[rgb_bands]
#modis_rgb_01.rio.to_raster(stacked_file_path)

modis_rgb_01_s = np.swapaxes(modis_rgb_01, 0, 2)
modis_rgb_01_s = np.swapaxes(modis_rgb_01_s, 0, 1)

modis_rgb_01_s_temp = modis_rgb_01_s
modis_rgb_01_s[:,:,1] = modis_rgb_01_s_temp[:,:,2]
modis_rgb_01_s[:,:,2] = modis_rgb_01_s_temp[:,:,1]

plt.imshow(modis_rgb_01_s)
plt.imsave(stacked_file_path+'.jpg',modis_rgb_01_s)
#modis_rgb_01_s.rio.to_raster(stacked_file_path)

