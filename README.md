# LineOfSide
This project implements a CUDA-accelerated Line of Sight (LOS) analysis on a digital elevation model (DEM) stored as a GeoTIFF file. It uses GDAL to read geospatial raster data, runs a ray-based LOS computation on the GPU, and exports the detected hilltops and visibility polygon into a KML file for visualization in tools like Google Earth.
