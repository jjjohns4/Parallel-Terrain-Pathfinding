#include <iostream>
#include <vector>
#include <limits>
#include <cmath>

#include "gdal_priv.h"
#include "cpl_conv.h"  

#include <opencv2/opencv.hpp>
#include <opencv2/highgui.hpp>

int main() {
    GDALAllRegister();

    const char* filename = "SanFran.tif";

    GDALDataset *poDataset = (GDALDataset *) GDALOpen(filename, GA_ReadOnly);
    if (!poDataset) {
        std::cerr << "Failed to open dataset: " << filename << std::endl;
        return 1;
    }

    GDALRasterBand *poBand = poDataset->GetRasterBand(1);
    int nX = poBand->GetXSize();
    int nY = poBand->GetYSize();

    std::vector<float> data(nX * nY);
    if (poBand->RasterIO(GF_Read, 0, 0, nX, nY, data.data(), nX, nY, GDT_Float32, 0, 0) != CE_None) {
        std::cerr << "Failed to read raster data." << std::endl;
        GDALClose(poDataset);
        return 1;
    }

    for (auto& value : data) {
        if (value < 0)
            value = std::numeric_limits<float>::quiet_NaN();
    }

    float data_min = std::numeric_limits<float>::infinity();
    float data_max = -std::numeric_limits<float>::infinity();
    size_t nanCount = 0;
    for (const auto& value : data) {
        if (std::isnan(value)) {
            nanCount++;
            continue;
        }
        if (value < data_min)
            data_min = value;
        if (value > data_max)
            data_max = value;
    }
    std::cout << "Min: " << data_min << ", Max: " << data_max << std::endl;
    std::cout << "Total pixels: " << data.size() << ", NaN count: " << nanCount << std::endl;

    cv::Mat image(nY, nX, CV_32FC1, data.data());

    cv::Mat normImage(nY, nX, CV_8UC1);
    float range = data_max - data_min;
    if (range == 0.0f) {
        range = 1.0f;
    }

    for (int i = 0; i < nY; ++i) {
        for (int j = 0; j < nX; ++j) {
            float v = image.at<float>(i, j);
            if (std::isnan(v)) {
                normImage.at<uchar>(i, j) = 255;
            } else {
                int scaled = static_cast<int>(((v - data_min) / range) * 255.0f);
                if (scaled < 0) scaled = 0;
                if (scaled > 255) scaled = 255;
                normImage.at<uchar>(i, j) = static_cast<uchar>(scaled);
            }
        }
    }

    cv::namedWindow("Georeferenced Aspect", cv::WINDOW_NORMAL);
    cv::imshow("Georeferenced Aspect", normImage);
    cv::waitKey(0);

    cv::imwrite("aspect.png", normImage);

    GDALClose(poDataset);
    return 0;
}
