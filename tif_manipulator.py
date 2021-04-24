import gdal

input_file_name = "Data/pecs_tifs/pressure/09_03_10_50_CTP.tif"
output_file_name = "Data/pecs_tifs/pressure/09_03_10_50_CTP_scaled.tif"
tiff_file = gdal.Open(input_file_name)

# Store the GeoTiff data that we will insert back into the scaled file
geotransform = tiff_file.GetGeoTransform()
projection = tiff_file.GetProjection()
band = tiff_file.GetRasterBand(1)
xsize = band.XSize
ysize = band.YSize

# Get the data from the file
array = band.ReadAsArray()
tiff_file = None
band = None

# Manipulate that data however you'd like
array = array * 0.100000001490116

# Create a new tiff using the data we just manipulated and the old GeoTiff data we stored
driver = gdal.GetDriverByName('GTiff')
new_tiff = driver.Create(output_file_name, xsize, ysize, 1, gdal.GDT_Int16)
new_tiff.SetGeoTransform(geotransform)
new_tiff.SetProjection(projection)
new_tiff.GetRasterBand(1).WriteArray(array)
new_tiff.FlushCache()
new_tiff = None