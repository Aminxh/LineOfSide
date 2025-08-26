# CUDA Line of Sight (LOS) Hill Detection  

This project implements a **GPU-accelerated Line of Sight (LOS) analysis** on a digital elevation model (DEM).  
The program assumes an **observer at the center of the raster map** and performs **ray tracing** across a given number of rays (`MAX_RAYS`).  

Any terrain point that has a higher elevation than the observer is considered a **hill**.  
Detected hills are exported into a **KML file**, which can be opened in Google Earth or similar tools for visualization.  

---

## Features  
- **GPU Acceleration (CUDA):**  
  The LOS computation is performed entirely on the GPU, making it significantly faster than a CPU-based approach.  

- **Ray Tracing:**  
  Rays are cast radially from the center point up to `MAX_RAYS` (e.g., 360,000 rays for ~0.001° precision).  

- **Hill Detection:**  
  A point is considered a *hill* if its elevation is greater than the observer’s elevation.  

- **KML Export:**  
  - 🔴 Red markers = detected hills  
  - 🟢 Green marker = observer (center point)  
  - 🔵 Blue polygon = boundary of visibility/hill hits  

- **Scalable:**  
  Works with large raster datasets (`.tif`) by reading them with GDAL.  

---

## Performance  
GPU acceleration enables **massively parallel ray tracing**, reducing runtime from minutes or hours (CPU) to seconds (GPU), depending on raster size and ray count.  

Execution time is automatically measured and displayed at the end of each run.  

---

## 📂 Workflow  
1. Load DEM raster (GeoTIFF) with **GDAL**.  
2. Compute observer location (center of raster).  
3. Launch CUDA kernel:  
   - Each thread handles one ray.  
   - Rays advance pixel by pixel until:  
     - They leave the raster bounds, or  
     - They hit terrain higher than the observer.  
4. Collect all detected hill points.  
5. Write results into a **KML file** for visualization.  

---

## Requirements  
- **CUDA Toolkit**  
- **GDAL** (for raster input and coordinate transforms)  
- **C++17 or later**  

---

## Example Outputs  

### 🔹 Center location (Google Earth View)  
  <img width="1650" height="1295" alt="map" src="https://github.com/user-attachments/assets/929bddc9-7acc-4ef7-85b0-27ea0b8c12ca" />


### 🔹 Output of kernel (The red area represents the region where, if you continue along that direction, you will eventually hit a hill.)  
<img width="2074" height="1434" alt="p1" src="https://github.com/user-attachments/assets/ff53db11-3e11-467c-92b2-9abe335374ec" />


### The .KML file will be availabe alongside other files.

