import numpy as np
from pyevtk.hl import gridToVTK
from PIL import Image
import os
import time

directory = "./Data/"
file_name = "cloud_height"
file_extension = ".vtr"

# delete the data file if it already exists
for file in os.listdir(directory):
    if file.startswith(file_name) and file.endswith(file_extension):
        os.remove(os.path.join(directory, file))

# load the .tif files into a numpy array
CTH_tif = Image.open(f'{directory}/mod06_CTH.tif')
CTH_arr = np.array(CTH_tif, dtype='uint16')

# Initialize the ranges of the cloud elevations
low_elev = (500 // 500, 2000 // 500)
mid_elev = ((2000 // 500) + 1, 7000 // 500)
high_elev = ((7000 // 500) + 1, np.amax(CTH_arr) // 500)

# Initialize the pixel values for each cloud type
low_cloud = 1
mid_cloud = 2
high_cloud = 3

# Get the x, y, and z dimensions
x_shrink_factor = 2
y_shrink_factor = 2
z_shrink_factor = 1
x_dim = np.shape(CTH_arr)[0] // x_shrink_factor
y_dim = np.shape(CTH_arr)[1] // y_shrink_factor
z_dim = (np.amax(CTH_arr) // 500) // z_shrink_factor
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

    start_x = col * x_dim
    start_y = row * y_dim

    start = time.time()
    for y in range(y_dim):
        if y % 100 == 0:
            print(f"Progress: {y} / {y_dim} (Updates every 100 iterations)")
        for x in range(x_dim):
            for z in range(z_dim):
                pixel_value = CTH_arr[start_x + x][start_y + y] // 500
                if low_elev[0] <= pixel_value <= low_elev[1] and low_elev[0] <= z <= low_elev[1]:
                    cloud_height_arr[y][x][z] = low_cloud
                if mid_elev[0] <= pixel_value // 500 <= mid_elev[1] and mid_elev[0] <= z <= mid_elev[1]:
                    cloud_height_arr[y][x][z] = mid_cloud
                if high_elev[0] <= pixel_value <= high_elev[1] and high_elev[0] <= z <= high_elev[1]:
                    cloud_height_arr[y][x][z] = high_cloud
                else:
                    cloud_height_arr[y][x][z] = 0

    # Write the data to a new file
    x = np.arange(x_dim + 1)
    y = np.arange(y_dim + 1)
    z = np.arange(z_dim + 1,)

    print(f"Shape: {np.shape(cloud_height_arr)}")
    print(f"Size of array in MB: {(cloud_height_arr.size * cloud_height_arr.itemsize) // 1000000}")
    end = time.time()
    print(f"Time taken to compute {end - start}")

    print(f"Writing cloud data to file {file_name}{file_counter}.vtr")
    gridToVTK(f"./Data/{file_name}{file_counter}", y, x, z, cellData={'cloud_type': cloud_height_arr})
    print(f"Finished writing to {file_name}{file_counter}.vtr")
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
