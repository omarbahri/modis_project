import numpy as np
from tvtk.api import tvtk, write_data
from PIL import Image
import os
import time
import random
import math

def classify_cloud(opt_thickness, pressure):
    if 50 <= pressure < 440:  # high elevation
        if 0 <= opt_thickness < 3.6:
            return 'cumulus'
        elif 3.6 <= opt_thickness < 23:
            return 'stratocumulus'
        elif 23 <= opt_thickness:
            return 'stratus'
    elif 440 <= pressure < 680:  # mid elevation
        if 0 <= opt_thickness < 3.6:
            return 'altocumulus'
        elif 3.6 <= opt_thickness < 23:
            return 'altostratus'
        elif 23 <= opt_thickness:
            return 'nimbostratus'
    elif 680 <= pressure:  # low elevation
        if 0 <= opt_thickness < 3.6:
            return 'cirrus'
        elif 3.6 <= opt_thickness < 23:
            return 'cirrostratus'
        elif 23 <= opt_thickness:
            return 'deep_convection'

    return 'unknown'

# dictionary storing the cloud type and average total height
cloud_types_height = {
    'cumulus': [1, 1750],
    'stratocumulus': [2, 300],
    'stratus': [3, 750],
    'altocumulus': [4, 450],
    'altostratus': [5, 1100],
    'nimbostratus': [6, 750],
    'cirrus': [7, 300],
    'cirrostratus': [8, 300],
    'deep_convection': [9, 2000],
    'unknown': [0, 0]
}

directory = "./Data/"
height_file_name = "cloud_height"
rain_file_name = "rain_cloud"
ice_file_name = "ice_cloud"
file_extension = ".vti"

# delete the data files if they already exists
for file in os.listdir(directory):
    if file.endswith(file_extension):
        os.remove(os.path.join(directory, file))

# load the .tif files into numpy arrays
# cloud top height
CTH_tif = Image.open(f'{directory}/mod06_CTH.tif')
CTH_arr = np.array(CTH_tif, dtype='uint16')

# cloud optical thickness
COT_tif = Image.open(f'{directory}/mod06_COT.tif')
COT_arr = np.array(COT_tif, dtype='uint16')
COT_scale_factor = 0.00999999977648258

# cloud top pressure
CTP_tif = Image.open(f'{directory}/mod06_CTP.tif')
CTP_arr = np.array(CTP_tif, dtype='uint16')
CTP_scale_factor = 0.100000001490116

# cloud phase optical properties
CPOP_tif = Image.open(f'{directory}/mod06_CPOP.tif')
CPOP_arr = np.array(CPOP_tif, dtype='uint16')

# Initialize the ranges of the cloud elevations
res_factor = 50
low_elev = (500 // res_factor, 2000 // res_factor)
mid_elev = ((2000 // res_factor) + 1, 7000 // res_factor)
high_elev = ((7000 // res_factor) + 1, np.amax(CTH_arr) // res_factor)

# Get the x, y, and z dimensions
x_shrink_factor = 1
y_shrink_factor = 1
z_shrink_factor = 1
x_dim = np.shape(CTH_arr)[0] // x_shrink_factor
y_dim = np.shape(CTH_arr)[1] // y_shrink_factor
z_dim = (math.ceil((np.amax(CTH_arr) * 1.025)) // res_factor) // z_shrink_factor
print(f"Minimum Elevation: {np.amin(CTH_arr)}")
print(f"Maximum Elevation: {np.amax(CTH_arr)}")
print(f"Dimensions: ({x_dim}, {y_dim}, {z_dim})")

row = 0
col = 0
height = 0
file_counter = 0
while row < x_shrink_factor and col < y_shrink_factor and height < z_shrink_factor:
    # Initialize the 3D array that will hold the cloud data
    cloud_height_arr = np.zeros((y_dim, x_dim, z_dim), dtype='uint8')
    cloud_height_arr.fill(np.nan)
    ice_cloud_arr = np.zeros((y_dim, x_dim, z_dim), dtype='uint8')
    ice_cloud_arr.fill(np.nan)
    rain_cloud_arr = np.zeros((y_dim, x_dim, z_dim), dtype='uint8')
    rain_cloud_arr.fill(np.nan)

    start_x = col * x_dim
    start_y = row * y_dim

    start = time.time()
    for y in range(y_dim):
        if y % 100 == 0:
            print(f"Progress: {y} / {y_dim} (Updates every 100 iterations)")
        for x in range(x_dim):
            top_height = CTH_arr[start_x + x][start_y + y] // res_factor
            opt_thickness = COT_arr[start_x + x][start_y + y] * COT_scale_factor
            pressure = CTP_arr[start_x + x][start_y + y] * CTP_scale_factor

            type_and_base = cloud_types_height[classify_cloud(opt_thickness, pressure)]
            cloud_type = type_and_base[0]
            base_height = top_height - (type_and_base[1] // res_factor)

            # We can randomly vary the base_height to get more realistic clouds
            rand_offset = random.uniform(-0.2, 0.2)
            base_height *= (1 + rand_offset)
            base_height = math.ceil(base_height)

            # push the clouds up that have a negative base_height
            if base_height < 0:
                z_transform = (-1 * base_height)
                base_height += z_transform
                top_height += z_transform

            cloud_height_arr[y][x][base_height:top_height] = cloud_type

            # decide if the cloud is liquid or ice
            if CPOP_arr[start_x + x][start_y + y] == 2:
                rain_cloud_arr[y][x][base_height:top_height] = 2
            elif CPOP_arr[start_x + x][start_y + y] == 3:
                ice_cloud_arr[y][x][base_height:top_height] = 3

    # flip the array upside down so it renders right side up in Paraview
    np.flipud(cloud_height_arr)
    np.flipud(rain_cloud_arr)
    np.flipud(ice_cloud_arr)

    # Basic logging information
    print(f"Shape: {np.shape(cloud_height_arr)}")
    print(f"Size of array in MB: {(cloud_height_arr.size * cloud_height_arr.itemsize) // 1000000}")
    end = time.time()
    print(f"Time taken to compute {end - start}")

    # Write the data to files
    print(f"Writing cloud height data to file {height_file_name}{file_counter}{file_extension}")
    cloud_height_data = tvtk.ImageData(spacing=(1, 1, 1), origin=(0, 0, 0), dimensions=cloud_height_arr.shape)
    cloud_height_data.point_data.scalars = cloud_height_arr.ravel(order='F')
    cloud_height_data.point_data.scalars.name = 'Cloud_Heights'
    write_data(cloud_height_data, f"./Data/{height_file_name}{file_counter}")
    print(f"Finished writing to {height_file_name}{file_counter}{file_extension}")
    if x_shrink_factor == 1 and y_shrink_factor == 1 and z_shrink_factor == 1:
        del cloud_height_arr
        del cloud_height_data

    print(f"Writing rain cloud data to file {rain_file_name}{file_counter}{file_extension}")
    rain_cloud_data = tvtk.ImageData(spacing=(1, 1, 1), origin=(0, 0, 0), dimensions=rain_cloud_arr.shape)
    rain_cloud_data.point_data.scalars = rain_cloud_arr.ravel(order='F')
    rain_cloud_data.point_data.scalars.name = 'Cloud_Heights'
    write_data(rain_cloud_data, f"./Data/{rain_file_name}{file_counter}")
    print(f"Finished writing to {rain_file_name}{file_counter}{file_extension}")
    if x_shrink_factor == 1 and y_shrink_factor == 1 and z_shrink_factor == 1:
        del rain_cloud_arr
        del rain_cloud_data

    print(f"Writing ice cloud data to file {ice_file_name}{file_counter}{file_extension}")
    ice_cloud_data = tvtk.ImageData(spacing=(1, 1, 1), origin=(0, 0, 0), dimensions=ice_cloud_arr.shape)
    ice_cloud_data.point_data.scalars = ice_cloud_arr.ravel(order='F')
    ice_cloud_data.point_data.scalars.name = 'Cloud_Heights'
    write_data(ice_cloud_data, f"./Data/{ice_file_name}{file_counter}")
    print(f"Finished writing to {ice_file_name}{file_counter}{file_extension}")
    if x_shrink_factor == 1 and y_shrink_factor == 1 and z_shrink_factor == 1:
        del ice_cloud_arr
        del ice_cloud_data

    file_counter += 1
    print(f"{(x_shrink_factor * y_shrink_factor * z_shrink_factor) -  file_counter} files to go.")

    # update the current row and column and height
    row += 1
    if row == x_shrink_factor:
        row = 0
        col += 1
    if col == y_shrink_factor:
        col = 0
        height += 1
