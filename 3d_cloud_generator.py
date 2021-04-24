import numpy as np
from tvtk.api import tvtk, write_data
from PIL import Image
import os
import time
import random
import math
import sys


# dictionary storing the cloud type and max total height
cloud_types_height = {
    'cumulus': [1, 1200],
    'stratocumulus': [2, 650],
    'stratus': [3, 150],
    'altocumulus': [4, 550],
    'altostratus': [5, 250],
    'nimbostratus': [6, 6000],
    'cirrus': [7, 100],
    'cirrostratus': [8, 100],
    'deep_convection': [9, 12000],
    'unknown': [0, 0]
}

# dictionary storing the cloud type and min height
cloud_types_min_height = {
    'cumulus': [1, 200],
    'stratocumulus': [2, 200],
    'stratus': [3, 50],
    'altocumulus': [4, 100],
    'altostratus': [5, 50],
    'nimbostratus': [6, 300],
    'cirrus': [7, 50],
    'cirrostratus': [8, 50],
    'deep_convection': [9, 2000],
    'unknown': [0, 0]
}

# dictionary storing the cloud type and min base height (10% added for randomness later)
cloud_types_min = {
    'cumulus': [1, 331],
    'stratocumulus': [2, 331],
    'stratus': [3, 331],
    'altocumulus': [4, 2201],
    'altostratus': [5, 2201],
    'nimbostratus': [6, 331],
    'cirrus': [7, 7701],
    'cirrostratus': [8, 7701],
    'deep_convection': [9, 331],
    'unknown': [0, 0]
}

# dictionary storing the cloud type and range base height
cloud_types_base_heights = {
    'cumulus': [1, 300, 1500],
    'stratocumulus': [2, 300, 1800],
    'stratus': [3, 300, 1000],
    'altocumulus': [4, 2000, 6000],
    'altostratus': [5, 2000, 6500],
    'nimbostratus': [6, 300, 2000],
    'cirrus': [7, 7000, 11000],
    'cirrostratus': [8, 7000, 11000],
    'deep_convection': [9, 300, 1500],
    'unknown': [0, 0, 0]
}


directory = "./Data/"
rain_file_name = "rain_cloud"
ice_file_name = "ice_cloud"
file_extension = ".vti"

# delete the data files if they already exists
for file in os.listdir(directory):
    if file.endswith(file_extension):
        os.remove(os.path.join(directory, file))

#upsampling_factor = 1
## load the .tif files into numpy arrays
## cloud top height
#CTH_tif = Image.open(f'{directory}/mod06_CTH.tif')
#CTH_arr = np.array(CTH_tif, dtype='uint16')
#
## cloud optical thickness
#COT_tif = Image.open(f'{directory}/mod06_COT.tif')
#COT_arr = np.array(COT_tif, dtype='uint16')
#COT_scale_factor = 0.00999999977648258
#
## cloud top pressure
#CTP_tif = Image.open(f'{directory}/mod06_CTP.tif')
#CTP_arr = np.array(CTP_tif, dtype='uint16')
#CTP_scale_factor = 0.100000001490116
#
## cloud phase optical properties
#CPOP_tif = Image.open(f'{directory}/mod06_CPOP.tif')
#CPOP_arr = np.array(CPOP_tif, dtype='uint16')
        
# load the upsampled .npy files
upsampling_factor = 20
directory = "./Data/mod06_upsampled/"
CTH_arr = np.load(f'{directory}/mod06_CTH.npy')
COT_arr = np.load(f'{directory}/mod06_COT.npy')
COT_scale_factor = 0.00999999977648258
CTP_arr = np.load(f'{directory}/mod06_CTP.npy')
CTP_scale_factor = 0.100000001490116
CPOP_arr = np.load(f'{directory}/mod06_CPOP.npy')

# Initialize the ranges of the cloud elevations
res_factor = 1000//upsampling_factor
low_elev = (500 // res_factor, 2000 // res_factor)
mid_elev = ((2000 // res_factor) + 1, 7000 // res_factor)
high_elev = ((7000 // res_factor) + 1, np.amax(CTH_arr) // res_factor)

# Get the x, y, and z dimensions
x_shrink_factor = 1
y_shrink_factor = 1
z_shrink_factor = 1
x_dim = np.shape(CTH_arr)[0] // x_shrink_factor
y_dim = np.shape(CTH_arr)[1] // y_shrink_factor
z_dim = max((math.ceil((np.amax(CTH_arr) * 1.025)) // res_factor) // z_shrink_factor, 13000// res_factor// z_shrink_factor)
print(f"Minimum Elevation: {np.amin(CTH_arr)}")
print(f"Maximum Elevation: {np.amax(CTH_arr)}")
print(f"Dimensions: ({x_dim}, {y_dim}, {z_dim})")

def classify_cloud(opt_thickness, top_height):
    if low_elev[0] <= top_height < low_elev[1]:  # high elevation
        if 0 <= opt_thickness < 3.6:
            return 'cumulus'
        elif 3.6 <= opt_thickness < 23:
            return 'stratocumulus'
        elif 23 <= opt_thickness:
            return 'stratus'
    elif mid_elev[0] <= top_height < mid_elev[1]:  # mid elevation
        if 0 <= opt_thickness < 3.6:
            return 'altocumulus'
        elif 3.6 <= opt_thickness < 23:
            return 'altostratus'
        elif 23 <= opt_thickness:
            return 'nimbostratus'
    elif high_elev[0] <= top_height:  # low elevation
        if 0 <= opt_thickness < 3.6:
            return 'cirrus'
        elif 3.6 <= opt_thickness < 23:
            return 'cirrostratus'
        elif 23 <= opt_thickness:
            return 'deep_convection'

    return 'unknown'


row = 0
col = 0
height = 0
file_counter = 0
while row < x_shrink_factor and col < y_shrink_factor and height < z_shrink_factor:
    # Initialize the 3D array that will hold the cloud data
    cumulus_height_arr = np.zeros((y_dim, x_dim, z_dim), dtype='uint8')
    cumulus_height_arr.fill(np.nan)
    stratocumulus_height_arr = np.zeros((y_dim, x_dim, z_dim), dtype='uint8')
    stratocumulus_height_arr.fill(np.nan)
    stratus_height_arr = np.zeros((y_dim, x_dim, z_dim), dtype='uint8')
    stratus_height_arr.fill(np.nan)
    altocumulus_height_arr = np.zeros((y_dim, x_dim, z_dim), dtype='uint8')
    altocumulus_height_arr.fill(np.nan)
    altostratus_height_arr = np.zeros((y_dim, x_dim, z_dim), dtype='uint8')
    altostratus_height_arr.fill(np.nan)
    nimbostratus_height_arr = np.zeros((y_dim, x_dim, z_dim), dtype='uint8')
    nimbostratus_height_arr.fill(np.nan)
    cirrus_height_arr = np.zeros((y_dim, x_dim, z_dim), dtype='uint8')
    cirrus_height_arr.fill(np.nan)
    cirrostratus_height_arr = np.zeros((y_dim, x_dim, z_dim), dtype='uint8')
    cirrostratus_height_arr.fill(np.nan)
    deep_convection_height_arr = np.zeros((y_dim, x_dim, z_dim), dtype='uint8')
    deep_convection_height_arr.fill(np.nan)
    ice_cloud_arr = np.zeros((y_dim, x_dim, z_dim), dtype='uint8')
    ice_cloud_arr.fill(np.nan)
    rain_cloud_arr = np.zeros((y_dim, x_dim, z_dim), dtype='uint8')
    rain_cloud_arr.fill(np.nan)
    
    base_height_arr = np.zeros((y_dim, x_dim), dtype='uint8')
    top_height_arr = np.zeros((y_dim, x_dim), dtype='uint8')
    
    neighbor_base_height = 0
    neighbor_top_height = 0
    margin = 3
    
    start_x = col * x_dim
    start_y = row * y_dim
    
    
    start = time.time()
    for y in range(y_dim):
        if y % 100 == 0:
            print(f"Progress: {y} / {y_dim} (Updates every 100 iterations)")
        for x in range(x_dim):
            top_height = CTH_arr[start_x + x][start_y + y] // res_factor
            opt_thickness = COT_arr[start_x + x][start_y + y] * COT_scale_factor
            #pressure = CTP_arr[start_x + x][start_y + y] * CTP_scale_factor
            
            cloud_type_name = classify_cloud(opt_thickness, top_height)
            type_and_base = cloud_types_base_heights[cloud_type_name]
            cloud_type = type_and_base[0]
            
            min_height = cloud_types_min_height[cloud_type_name][1] // res_factor
            
            base_height_min = type_and_base[1] // res_factor
            base_height_max = type_and_base[2] // res_factor
            base_height = math.floor(random.uniform(min(base_height_min,top_height-min_height), min(base_height_max, top_height-min_height)))
            
#            cloud_height = cloud_types_height[cloud_type_name][1] // res_factor
#            min_base_height = cloud_types_min[cloud_type_name][1] // res_factor
            
            #The cloud height should not be larger than the maximum cloud height
#            base_height = max(base_height, top_height-cloud_height)
#            base_height = max(base_height, min_base_height)
            
#            if base_height > top_height:  
#                type_and_base = cloud_types_height[classify_cloud(opt_thickness, pressure)]
#                cloud_type = type_and_base[0]
#                base_height = top_height - (type_and_base[1] // res_factor)
#    
#                # We can randomly vary the base_height to get more realistic clouds
##                rand_offset = random.uniform(-0.2, 0.2)
##                base_height *= (1 + rand_offset)
##                base_height = math.ceil(base_height)
#
#                # push the clouds up that have a negative base_height
#                if base_height < 0:
#                    base_height = 300//res_factor
##                    z_transform = (-1 * base_height)
##                    base_height += z_transform + 90
##                    top_height += z_transform
            
            rand_offset = random.uniform(-0.1, 0.1)
            
            if cloud_type == 1:
                # if the previous neighbors were of the same type, make the base height closer
                if cumulus_height_arr[y-1][x].any():
                    neighbor_top_height = top_height_arr[y-1][x]
                    if np.absolute(neighbor_top_height-top_height) < margin:
                        neighbor_base_height = base_height_arr[y-1][x]
                        base_height =  math.floor(neighbor_base_height * (1 + rand_offset))
                elif cumulus_height_arr[y-1][x-1].any():
                    neighbor_top_height = top_height_arr[y-1][x]
                    if np.absolute(neighbor_top_height-top_height) < margin:
                        neighbor_base_height = base_height_arr[y-1][x-1]
                        base_height  = math.floor(neighbor_base_height * (1 + rand_offset))
                elif cumulus_height_arr[y][x-1].any():
                    neighbor_top_height = top_height_arr[y-1][x]
                    if np.absolute(neighbor_top_height-top_height) < margin:
                        neighbor_base_height = base_height_arr[y][x-1]
                        base_height = math.floor(neighbor_base_height * (1 + rand_offset)) 
                cumulus_height_arr[y][x][base_height:top_height] = cloud_type
                if neighbor_base_height:
                    base_height_arr[y][x] = neighbor_base_height
                else:
                    base_height_arr[y][x] = base_height
                
            elif cloud_type == 2:
                # if the previous neighbors were of the same type, make the base height closer
                if stratocumulus_height_arr[y-1][x].any():
                    neighbor_top_height = top_height_arr[y-1][x]
                    if np.absolute(neighbor_top_height-top_height) < margin:
                        neighbor_base_height = base_height_arr[y-1][x]
                        base_height =  math.floor(neighbor_base_height * (1 + rand_offset))
                elif stratocumulus_height_arr[y-1][x-1].any():
                    neighbor_top_height = top_height_arr[y-1][x]
                    if np.absolute(neighbor_top_height-top_height) < margin:
                        neighbor_base_height = base_height_arr[y-1][x-1]
                        base_height  = math.floor(neighbor_base_height * (1 + rand_offset))
                elif stratocumulus_height_arr[y][x-1].any():
                    neighbor_top_height = top_height_arr[y-1][x]
                    if np.absolute(neighbor_top_height-top_height) < margin:
                        neighbor_base_height = base_height_arr[y][x-1]
                        base_height = math.floor(neighbor_base_height * (1 + rand_offset)) 
                stratocumulus_height_arr[y][x][base_height:top_height] = cloud_type
                if neighbor_base_height:
                    base_height_arr[y][x] = neighbor_base_height
                else:
                    base_height_arr[y][x] = base_height
                
            elif cloud_type == 3:
                if stratus_height_arr[y-1][x].any():
                    neighbor_top_height = top_height_arr[y-1][x]
                    if np.absolute(neighbor_top_height-top_height) < margin:
                        neighbor_base_height = base_height_arr[y-1][x]
                        base_height =  math.floor(neighbor_base_height * (1 + rand_offset))
                elif stratus_height_arr[y-1][x-1].any():
                    neighbor_top_height = top_height_arr[y-1][x]
                    if np.absolute(neighbor_top_height-top_height) < margin:
                        neighbor_base_height = base_height_arr[y-1][x-1]
                        base_height  = math.floor(neighbor_base_height * (1 + rand_offset))
                elif stratus_height_arr[y][x-1].any():
                    neighbor_top_height = top_height_arr[y-1][x]
                    if np.absolute(neighbor_top_height-top_height) < margin:
                        neighbor_base_height = base_height_arr[y][x-1]
                        base_height = math.floor(neighbor_base_height * (1 + rand_offset)) 
                stratus_height_arr[y][x][base_height:top_height] = cloud_type
                if neighbor_base_height:
                    base_height_arr[y][x] = neighbor_base_height
                else:
                    base_height_arr[y][x] = base_height
                
            elif cloud_type == 4:
                if altocumulus_height_arr[y-1][x].any():
                    neighbor_top_height = top_height_arr[y-1][x]
                    if np.absolute(neighbor_top_height-top_height) < margin:
                        neighbor_base_height = base_height_arr[y-1][x]
                        base_height =  math.floor(neighbor_base_height * (1 + rand_offset))
                elif altocumulus_height_arr[y-1][x-1].any():
                    neighbor_top_height = top_height_arr[y-1][x]
                    if np.absolute(neighbor_top_height-top_height) < margin:
                        neighbor_base_height = base_height_arr[y-1][x-1]
                        base_height  = math.floor(neighbor_base_height * (1 + rand_offset))
                elif altocumulus_height_arr[y][x-1].any():
                    neighbor_top_height = top_height_arr[y-1][x]
                    if np.absolute(neighbor_top_height-top_height) < margin:
                        neighbor_base_height = base_height_arr[y][x-1]
                        base_height = math.floor(neighbor_base_height * (1 + rand_offset)) 
                altocumulus_height_arr[y][x][base_height:top_height] = cloud_type
                if neighbor_base_height:
                    base_height_arr[y][x] = neighbor_base_height
                else:
                    base_height_arr[y][x] = base_height
                
            elif cloud_type == 5:
                if altostratus_height_arr[y-1][x].any():
                    neighbor_top_height = top_height_arr[y-1][x]
                    if np.absolute(neighbor_top_height-top_height) < margin:
                        neighbor_base_height = base_height_arr[y-1][x]
                        base_height =  math.floor(neighbor_base_height * (1 + rand_offset))
                elif altostratus_height_arr[y-1][x-1].any():
                    neighbor_top_height = top_height_arr[y-1][x]
                    if np.absolute(neighbor_top_height-top_height) < margin:
                        neighbor_base_height = base_height_arr[y-1][x-1]
                        base_height  = math.floor(neighbor_base_height * (1 + rand_offset))
                elif altostratus_height_arr[y][x-1].any():
                    neighbor_top_height = top_height_arr[y-1][x]
                    if np.absolute(neighbor_top_height-top_height) < margin:
                        neighbor_base_height = base_height_arr[y][x-1]
                        base_height = math.floor(neighbor_base_height * (1 + rand_offset)) 
                altostratus_height_arr[y][x][base_height:top_height] = cloud_type
                if neighbor_base_height:
                    base_height_arr[y][x] = neighbor_base_height
                else:
                    base_height_arr[y][x] = base_height
                
            elif cloud_type == 6:
                if nimbostratus_height_arr[y-1][x].any():
                    neighbor_top_height = top_height_arr[y-1][x]
                    if np.absolute(neighbor_top_height-top_height) < margin:
                        neighbor_base_height = base_height_arr[y-1][x]
                        base_height =  math.floor(neighbor_base_height * (1 + rand_offset))
                elif nimbostratus_height_arr[y-1][x-1].any():
                    neighbor_top_height = top_height_arr[y-1][x]
                    if np.absolute(neighbor_top_height-top_height) < margin:
                        neighbor_base_height = base_height_arr[y-1][x-1]
                        base_height  = math.floor(neighbor_base_height * (1 + rand_offset))
                elif nimbostratus_height_arr[y][x-1].any():
                    neighbor_top_height = top_height_arr[y-1][x]
                    if np.absolute(neighbor_top_height-top_height) < margin:
                        neighbor_base_height = base_height_arr[y][x-1]
                        base_height = math.floor(neighbor_base_height * (1 + rand_offset)) 
                nimbostratus_height_arr[y][x][base_height:top_height] = cloud_type
                if neighbor_base_height:
                    base_height_arr[y][x] = neighbor_base_height
                else:
                    base_height_arr[y][x] = base_height
                
                if base_height>=top_height or base_height<=0:
                    print("ERROR: Base height cannot be greater than top height\n")
                    sys.exit()
                    
            elif cloud_type == 7:
                if cirrus_height_arr[y-1][x].any():
                    neighbor_top_height = top_height_arr[y-1][x]
                    if np.absolute(neighbor_top_height-top_height) < margin:
                        neighbor_base_height = base_height_arr[y-1][x]
                        base_height =  math.floor(neighbor_base_height * (1 + rand_offset))
                elif cirrus_height_arr[y-1][x-1].any():
                    neighbor_top_height = top_height_arr[y-1][x]
                    if np.absolute(neighbor_top_height-top_height) < margin:
                        neighbor_base_height = base_height_arr[y-1][x-1]
                        base_height  = math.floor(neighbor_base_height * (1 + rand_offset))
                elif cirrus_height_arr[y][x-1].any():
                    neighbor_top_height = top_height_arr[y-1][x]
                    if np.absolute(neighbor_top_height-top_height) < margin:
                        neighbor_base_height = base_height_arr[y][x-1]
                        base_height = math.floor(neighbor_base_height * (1 + rand_offset)) 
                cirrus_height_arr[y][x][base_height:top_height] = cloud_type
                if neighbor_base_height:
                    base_height_arr[y][x] = neighbor_base_height
                else:
                    base_height_arr[y][x] = base_height
                
            elif cloud_type == 8:
                if cirrostratus_height_arr[y-1][x].any():
                    neighbor_top_height = top_height_arr[y-1][x]
                    if np.absolute(neighbor_top_height-top_height) < margin:
                        neighbor_base_height = base_height_arr[y-1][x]
                        base_height =  math.floor(neighbor_base_height * (1 + rand_offset))
                elif cirrostratus_height_arr[y-1][x-1].any():
                    neighbor_top_height = top_height_arr[y-1][x]
                    if np.absolute(neighbor_top_height-top_height) < margin:
                        neighbor_base_height = base_height_arr[y-1][x-1]
                        base_height  = math.floor(neighbor_base_height * (1 + rand_offset))
                elif cirrostratus_height_arr[y][x-1].any():
                    neighbor_top_height = top_height_arr[y-1][x]
                    if np.absolute(neighbor_top_height-top_height) < margin:
                        neighbor_base_height = base_height_arr[y][x-1]
                        base_height = math.floor(neighbor_base_height * (1 + rand_offset)) 
                cirrostratus_height_arr[y][x][base_height:top_height] = cloud_type
                if neighbor_base_height:
                    base_height_arr[y][x] = neighbor_base_height
                else:
                    base_height_arr[y][x] = base_height
                
            elif cloud_type == 9:
                if deep_convection_height_arr[y-1][x].any():
                    neighbor_top_height = top_height_arr[y-1][x]
                    if np.absolute(neighbor_top_height-top_height) < margin:
                        neighbor_base_height = base_height_arr[y-1][x]
                        base_height =  math.floor(neighbor_base_height * (1 + rand_offset))
                elif deep_convection_height_arr[y-1][x-1].any():
                    neighbor_top_height = top_height_arr[y-1][x]
                    if np.absolute(neighbor_top_height-top_height) < margin:
                        neighbor_base_height = base_height_arr[y-1][x-1]
                        base_height  = math.floor(neighbor_base_height * (1 + rand_offset))
                elif deep_convection_height_arr[y][x-1].any():
                    neighbor_top_height = top_height_arr[y-1][x]
                    if np.absolute(neighbor_top_height-top_height) < margin:
                        neighbor_base_height = base_height_arr[y][x-1]
                        base_height = math.floor(neighbor_base_height * (1 + rand_offset)) 
                deep_convection_height_arr[y][x][base_height:top_height] = cloud_type
                if neighbor_base_height:
                    base_height_arr[y][x] = neighbor_base_height
                else:
                    base_height_arr[y][x] = base_height
            
            if base_height>=top_height:
                print("ERROR: Base height cannot be greater than top height\n")
                sys.exit()
            
            if base_height<=0 and cloud_type>0:
                print("ERROR: Base height cannot be negative\n")
                sys.exit()
                                    
            # decide if the cloud is liquid or ice
            if CPOP_arr[start_x + x][start_y + y] == 2:
                rain_cloud_arr[y][x][base_height:top_height] = 2
            elif CPOP_arr[start_x + x][start_y + y] == 3:
                ice_cloud_arr[y][x][base_height:top_height] = 3
                
            

    # flip the array upside down so it renders right side up in Paraview
    np.flipud(cumulus_height_arr)
    np.flipud(stratocumulus_height_arr)
    np.flipud(stratus_height_arr)
    np.flipud(altocumulus_height_arr)
    np.flipud(altostratus_height_arr)
    np.flipud(nimbostratus_height_arr)
    np.flipud(cirrus_height_arr)
    np.flipud(cirrostratus_height_arr)
    np.flipud(deep_convection_height_arr)
    np.flipud(rain_cloud_arr)
    np.flipud(ice_cloud_arr)


    # Basic logging information
    print(f"Shape: {np.shape(cumulus_height_arr)}")
    print(f"Size of array in MB: {(cumulus_height_arr.size * cumulus_height_arr.itemsize) // 1000000}")
    end = time.time()
    print(f"Time taken to compute {end - start}")

    # Write the data to files
    cumulus_file_name = "cumulus_height"
    print(f"Writing cloud height data to file {cumulus_file_name}{file_counter}{file_extension}")
    cloud_height_data = tvtk.ImageData(spacing=(1, 1, 1), origin=(0, 0, 0), dimensions=cumulus_height_arr.shape)
    cloud_height_data.point_data.scalars = cumulus_height_arr.ravel(order='F')
    cloud_height_data.point_data.scalars.name = 'Cloud_Heights'
    write_data(cloud_height_data, f"{directory}/{cumulus_file_name}{file_counter}")
    print(f"Finished writing to {cumulus_file_name}{file_counter}{file_extension}")
    if x_shrink_factor == 1 and y_shrink_factor == 1 and z_shrink_factor == 1:
        del cumulus_height_arr
        del cloud_height_data

    stratocumulus_file_name = "stratocumulus_height"
    print(f"Writing cloud height data to file {stratocumulus_file_name}{file_counter}{file_extension}")
    cloud_height_data = tvtk.ImageData(spacing=(1, 1, 1), origin=(0, 0, 0), dimensions=stratocumulus_height_arr.shape)
    cloud_height_data.point_data.scalars = stratocumulus_height_arr.ravel(order='F')
    cloud_height_data.point_data.scalars.name = 'Cloud_Heights'
    write_data(cloud_height_data, f"{directory}/{stratocumulus_file_name}{file_counter}")
    print(f"Finished writing to {stratocumulus_file_name}{file_counter}{file_extension}")
    if x_shrink_factor == 1 and y_shrink_factor == 1 and z_shrink_factor == 1:
        del stratocumulus_height_arr
        del cloud_height_data

    stratus_file_name = "stratus_height"
    print(f"Writing cloud height data to file {stratus_file_name}{file_counter}{file_extension}")
    cloud_height_data = tvtk.ImageData(spacing=(1, 1, 1), origin=(0, 0, 0), dimensions=stratus_height_arr.shape)
    cloud_height_data.point_data.scalars = stratus_height_arr.ravel(order='F')
    cloud_height_data.point_data.scalars.name = 'Cloud_Heights'
    write_data(cloud_height_data, f"{directory}/{stratus_file_name}{file_counter}")
    print(f"Finished writing to {stratus_file_name}{file_counter}{file_extension}")
    if x_shrink_factor == 1 and y_shrink_factor == 1 and z_shrink_factor == 1:
        del stratus_height_arr
        del cloud_height_data

    altocumulus_file_name = "altocumulus_height"
    print(f"Writing cloud height data to file {altocumulus_file_name}{file_counter}{file_extension}")
    cloud_height_data = tvtk.ImageData(spacing=(1, 1, 1), origin=(0, 0, 0), dimensions=altocumulus_height_arr.shape)
    cloud_height_data.point_data.scalars = altocumulus_height_arr.ravel(order='F')
    cloud_height_data.point_data.scalars.name = 'Cloud_Heights'
    write_data(cloud_height_data, f"{directory}/{altocumulus_file_name}{file_counter}")
    print(f"Finished writing to {altocumulus_file_name}{file_counter}{file_extension}")
    if x_shrink_factor == 1 and y_shrink_factor == 1 and z_shrink_factor == 1:
        del altocumulus_height_arr
        del cloud_height_data

    altostratus_file_name = "altostratus_height"
    print(f"Writing cloud height data to file {altostratus_file_name}{file_counter}{file_extension}")
    cloud_height_data = tvtk.ImageData(spacing=(1, 1, 1), origin=(0, 0, 0), dimensions=altostratus_height_arr.shape)
    cloud_height_data.point_data.scalars = altostratus_height_arr.ravel(order='F')
    cloud_height_data.point_data.scalars.name = 'Cloud_Heights'
    write_data(cloud_height_data, f"{directory}/{altostratus_file_name}{file_counter}")
    print(f"Finished writing to {altostratus_file_name}{file_counter}{file_extension}")
    if x_shrink_factor == 1 and y_shrink_factor == 1 and z_shrink_factor == 1:
        del altostratus_height_arr
        del cloud_height_data

    nimbostratus_file_name = "nimbostratus_height"
    print(f"Writing cloud height data to file {nimbostratus_file_name}{file_counter}{file_extension}")
    cloud_height_data = tvtk.ImageData(spacing=(1, 1, 1), origin=(0, 0, 0), dimensions=nimbostratus_height_arr.shape)
    cloud_height_data.point_data.scalars = nimbostratus_height_arr.ravel(order='F')
    cloud_height_data.point_data.scalars.name = 'Cloud_Heights'
    write_data(cloud_height_data, f"{directory}/{nimbostratus_file_name}{file_counter}")
    print(f"Finished writing to {nimbostratus_file_name}{file_counter}{file_extension}")
    if x_shrink_factor == 1 and y_shrink_factor == 1 and z_shrink_factor == 1:
        del nimbostratus_height_arr
        del cloud_height_data

    cirrus_file_name = "cirrus_height"
    print(f"Writing cloud height data to file {cirrus_file_name}{file_counter}{file_extension}")
    cloud_height_data = tvtk.ImageData(spacing=(1, 1, 1), origin=(0, 0, 0), dimensions=cirrus_height_arr.shape)
    cloud_height_data.point_data.scalars = cirrus_height_arr.ravel(order='F')
    cloud_height_data.point_data.scalars.name = 'Cloud_Heights'
    write_data(cloud_height_data, f"{directory}/{cirrus_file_name}{file_counter}")
    print(f"Finished writing to {cirrus_file_name}{file_counter}{file_extension}")
    if x_shrink_factor == 1 and y_shrink_factor == 1 and z_shrink_factor == 1:
        del cirrus_height_arr
        del cloud_height_data

    cirrostratus_file_name = "cirrostratus_height"
    print(f"Writing cloud height data to file {cirrostratus_file_name}{file_counter}{file_extension}")
    cloud_height_data = tvtk.ImageData(spacing=(1, 1, 1), origin=(0, 0, 0), dimensions=cirrostratus_height_arr.shape)
    cloud_height_data.point_data.scalars = cirrostratus_height_arr.ravel(order='F')
    cloud_height_data.point_data.scalars.name = 'Cloud_Heights'
    write_data(cloud_height_data, f"{directory}/{cirrostratus_file_name}{file_counter}")
    print(f"Finished writing to {cirrostratus_file_name}{file_counter}{file_extension}")
    if x_shrink_factor == 1 and y_shrink_factor == 1 and z_shrink_factor == 1:
        del cirrostratus_height_arr
        del cloud_height_data

    deep_convection_file_name = "deep_convection_height"
    print(f"Writing cloud height data to file {deep_convection_file_name}{file_counter}{file_extension}")
    cloud_height_data = tvtk.ImageData(spacing=(1, 1, 1), origin=(0, 0, 0), dimensions=deep_convection_height_arr.shape)
    cloud_height_data.point_data.scalars = deep_convection_height_arr.ravel(order='F')
    cloud_height_data.point_data.scalars.name = 'Cloud_Heights'
    write_data(cloud_height_data, f"{directory}/{deep_convection_file_name}{file_counter}")
    print(f"Finished writing to {deep_convection_file_name}{file_counter}{file_extension}")
    if x_shrink_factor == 1 and y_shrink_factor == 1 and z_shrink_factor == 1:
        del deep_convection_height_arr
        del cloud_height_data

    print(f"Writing rain cloud data to file {rain_file_name}{file_counter}{file_extension}")
    rain_cloud_data = tvtk.ImageData(spacing=(1, 1, 1), origin=(0, 0, 0), dimensions=rain_cloud_arr.shape)
    rain_cloud_data.point_data.scalars = rain_cloud_arr.ravel(order='F')
    rain_cloud_data.point_data.scalars.name = 'Cloud_Heights'
    write_data(rain_cloud_data, f"{directory}/{rain_file_name}{file_counter}")
    print(f"Finished writing to {rain_file_name}{file_counter}{file_extension}")
    if x_shrink_factor == 1 and y_shrink_factor == 1 and z_shrink_factor == 1:
        del rain_cloud_arr
        del rain_cloud_data

    print(f"Writing ice cloud data to file {ice_file_name}{file_counter}{file_extension}")
    ice_cloud_data = tvtk.ImageData(spacing=(1, 1, 1), origin=(0, 0, 0), dimensions=ice_cloud_arr.shape)
    ice_cloud_data.point_data.scalars = ice_cloud_arr.ravel(order='F')
    ice_cloud_data.point_data.scalars.name = 'Cloud_Heights'
    write_data(ice_cloud_data, f"{directory}/{ice_file_name}{file_counter}")
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
        
