#include "gdal_priv.h"
#include "cpl_conv.h"
#include <iostream>
#include <vector>
#include <cmath>
#include <fstream>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <chrono>

#define DEG2RAD(x) ((x) * 3.14159265f / 180.0f)
#define MAX_RAYS 360  // Safe with heap allocation

using namespace std;

// ---------------- CUDA Kernel ----------------
struct HillHit {
    int x, y;
    bool valid;
};

__global__ void computeLOS(
    const float* tile, int width, int height,
    int cx, int cy, float centerElev,
    const float* angles, int numRays,
    HillHit* hits, int maxSteps)
{
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx >= numRays) return;

    float angle = DEG2RAD(angles[idx]);
    float dx = cosf(angle);
    float dy = sinf(angle);

    for (int step = 1; step < maxSteps; ++step) {
        int x = roundf(cx + dx * step);
        int y = roundf(cy + dy * step);

        if (x < 0 || x >= width || y < 0 || y >= height) break;

        float elev = tile[y * width + x];

        if (elev > centerElev) {
            hits[idx] = { x, y, true };
            break;
        }
    }
}

// ---------------- Utilities ----------------

void checkCuda(cudaError_t err, const char* msg) {
    if (err != cudaSuccess) {
        cerr << "CUDA Error: " << msg << ": " << cudaGetErrorString(err) << endl;
        exit(1);
    }
}

void PixelToWorld(double gt[6], int px, int py, double& x, double& y) {
    x = gt[0] + px * gt[1] + py * gt[2];
    y = gt[3] + px * gt[4] + py * gt[5];
}

void prepareAngles(float* angles, int numRays, float step) {
    for (int i = 0; i < numRays; ++i) {
        angles[i] = i * step;
    }
}

void writeKMLWithPolygon(
    const vector<HillHit>& hits,
    int width, int height,
    double gt[6], int cx, int cy,
    const string& filename)
{
    ofstream kml(filename);
    kml << "<?xml version=\"1.0\" encoding=\"UTF-8\"?>\n";
    kml << "<kml xmlns=\"http://www.opengis.net/kml/2.2\">\n<Document>\n";

    // Style for hill points
    kml << "  <Style id=\"hillStyle\">\n"
        << "    <IconStyle><color>ff0000ff</color><scale>1.0</scale>\n"
        << "    <Icon><href>http://maps.google.com/mapfiles/kml/shapes/placemark_circle.png</href></Icon>\n"
        << "    </IconStyle>\n  </Style>\n";

    // Style for center point
    kml << "  <Style id=\"centerStyle\">\n"
        << "    <IconStyle><color>ff00ff00</color><scale>1.2</scale>\n"
        << "    <Icon><href>http://maps.google.com/mapfiles/kml/shapes/placemark_circle.png</href></Icon>\n"
        << "    </IconStyle>\n  </Style>\n";

    // Hill point markers
    for (const auto& h : hits) {
        if (!h.valid) continue;
        double lon, lat;
        PixelToWorld(gt, h.x, h.y, lon, lat);
        kml << "<Placemark><styleUrl>#hillStyle</styleUrl><Point><coordinates>"
            << lon << "," << lat << ",0</coordinates></Point></Placemark>\n";
    }

    // Center marker
    double centerLon, centerLat;
    PixelToWorld(gt, cx, cy, centerLon, centerLat);
    kml << "<Placemark><styleUrl>#centerStyle</styleUrl><Point><coordinates>"
        << centerLon << "," << centerLat << ",0</coordinates></Point></Placemark>\n";

    // Polygon around hill hits
    kml << "<Placemark><name>LOS Polygon</name>\n"
        << "  <Style><LineStyle><color>ff0000ff</color><width>2</width></LineStyle>\n"
        << "  <PolyStyle><color>7f0000ff</color></PolyStyle></Style>\n"
        << "  <Polygon><outerBoundaryIs><LinearRing><coordinates>\n";

    for (const auto& h : hits) {
        if (!h.valid) continue;
        double lon, lat;
        PixelToWorld(gt, h.x, h.y, lon, lat);
        kml << lon << "," << lat << ",0\n";
    }

    // Close polygon
    for (const auto& h : hits) {
        if (h.valid) {
            double lon, lat;
            PixelToWorld(gt, h.x, h.y, lon, lat);
            kml << lon << "," << lat << ",0\n";
            break;
        }
    }

    kml << "</coordinates></LinearRing></outerBoundaryIs></Polygon></Placemark>\n";
    kml << "</Document>\n</kml>\n";
}

// ---------------- Main ----------------

int main() {

    auto start = std::chrono::high_resolution_clock::now();


    GDALAllRegister();
    string tifPath = "E:/tiles/all_merged.tif";         // Place your own path here

    GDALDataset* ds = (GDALDataset*)GDALOpen(tifPath.c_str(), GA_ReadOnly);
    if (!ds) {
        cerr << "Failed to open file!\n";
        return 1;
    }

    int width = ds->GetRasterXSize();
    int height = ds->GetRasterYSize();
    GDALRasterBand* band = ds->GetRasterBand(1);
    double gt[6];
    ds->GetGeoTransform(gt);

    vector<float> tile(width * height);
    band->RasterIO(GF_Read, 0, 0, width, height, tile.data(), width, height, GDT_Float32, 0, 0);

    int cx = width / 2;
    int cy = height / 2;
    float centerElev = tile[cy * width + cx];

    // Allocate angles on heap to avoid stack overflow
    vector<float> angles(MAX_RAYS);
    prepareAngles(angles.data(), MAX_RAYS, 360.0f / MAX_RAYS);

    int maxSteps = (int)sqrtf((float)(width * width + height * height));

    // Allocate CUDA memory
    float* d_tile;
    float* d_angles;
    HillHit* d_hits;
    checkCuda(cudaMalloc(&d_tile, width * height * sizeof(float)), "alloc tile");
    checkCuda(cudaMalloc(&d_angles, MAX_RAYS * sizeof(float)), "alloc angles");
    checkCuda(cudaMalloc(&d_hits, MAX_RAYS * sizeof(HillHit)), "alloc hits");

    checkCuda(cudaMemcpy(d_tile, tile.data(), width * height * sizeof(float), cudaMemcpyHostToDevice), "copy tile");
    checkCuda(cudaMemcpy(d_angles, angles.data(), MAX_RAYS * sizeof(float), cudaMemcpyHostToDevice), "copy angles");
    checkCuda(cudaMemset(d_hits, 0, MAX_RAYS * sizeof(HillHit)), "clear hits");

    computeLOS << <(MAX_RAYS + 255) / 256, 256 >> > (
        d_tile, width, height, cx, cy, centerElev,
        d_angles, MAX_RAYS, d_hits, maxSteps
        );
    checkCuda(cudaDeviceSynchronize(), "kernel");

    vector<HillHit> hits(MAX_RAYS);
    checkCuda(cudaMemcpy(hits.data(), d_hits, MAX_RAYS * sizeof(HillHit), cudaMemcpyDeviceToHost), "copy hits");

    writeKMLWithPolygon(hits, width, height, gt, cx, cy, "E:/tiles/LOS_1000p.kml");     // Place your own path here
    cout << "Exported KML to E:/tiles/output_LOS_polygon.kml\n";                        // Place your own path here

    GDALClose(ds);
    cudaFree(d_tile);
    cudaFree(d_angles);
    cudaFree(d_hits);

    // Stop timer
    auto end = std::chrono::high_resolution_clock::now();

    // Calculate duration
    std::chrono::duration<double> duration = end - start;

    std::cout << "Execution time: " << duration.count() << " seconds" << std::endl;

    return 0;
}
