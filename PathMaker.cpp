#include <iostream>
#include <vector>
#include <limits>
#include <cmath>
#include <algorithm>
#include <string>

#include <omp.h>
#include "gdal_priv.h"
#include "cpl_conv.h"

#include <opencv2/opencv.hpp>
#include <opencv2/highgui.hpp>

static const double INF = std::numeric_limits<double>::infinity();
static const int MAX_PIXELS = 10000;

void floyd_warshall_parallel(int n, double *dist, int *nxt)
{
    const int B = 32;          
    const int barW = 50;       
    int next_tick = 0;

    for (int k = 0; k < n; ++k) {
        double *Dk = dist + (size_t)k * n;

        #pragma omp parallel for schedule(static)
        for (int ii = 0; ii < n; ii += B) {
            int i_max = std::min(ii + B, n);
            for (int jj = 0; jj < n; jj += B) {
                int j_max = std::min(jj + B, n);
                for (int i = ii; i < i_max; ++i) {
                    double *Di = dist + (size_t)i * n;
                    double dik = Di[k];
                    int in = i * n;
                    for (int j = jj; j < j_max; ++j) {
                        double alt = dik + Dk[j];
                        if (alt < Di[j]) {
                            Di[j] = alt;
                            nxt[in + j] = nxt[in + k];
                        }
                    }
                }
            }
        }

        if (k >= next_tick || k == n - 1) {
            next_tick = k + std::max(1, n / 100);
            int pct = (k + 1) * 100 / n;
            int fill = pct * barW / 100;
            std::cerr << '\r' << '[' << std::string(fill, '-')
                      << std::string(barW - fill, ' ') << "] " << pct << '%' << std::flush;
        }
    }
    std::cerr << '\n';
}

int reconstruct_path(int u, int v, int n, const int *nxt, int *out)
{
    if (nxt[(size_t)u * n + v] < 0) return 0;
    int len = 0;
    out[len++] = u;
    while (u != v) {
        u = nxt[(size_t)u * n + v];
        out[len++] = u;
    }
    return len;
}

static std::vector<cv::Point> clicks;
static void onMouse(int e, int x, int y, int, void*)
{
    if (e == cv::EVENT_LBUTTONDOWN) clicks.emplace_back(x, y);
}

int main(int argc, char **argv)
{
    if (argc != 2) {
        std::cerr << "Usage: " << argv[0] << " <terrain.tif>\n";
        return 1;
    }

    GDALAllRegister();
    GDALDataset *ds = static_cast<GDALDataset *>(GDALOpen(argv[1], GA_ReadOnly));
    if (!ds) {
        std::cerr << "Cannot open " << argv[1] << "\n";
        return 1;
    }
    GDALRasterBand *band = ds->GetRasterBand(1);
    int nX = band->GetXSize(), nY = band->GetYSize();
    std::vector<float> raw((size_t)nX * nY);
    band->RasterIO(GF_Read, 0, 0, nX, nY, raw.data(), nX, nY, GDT_Float32, 0, 0);
    GDALClose(ds);

    float mn = INFINITY, mx = -INFINITY;
    for (auto &v : raw) {
        if (v < 0)
            v = std::numeric_limits<float>::quiet_NaN();
        else {
            mn = std::min(mn, v);
            mx = std::max(mx, v);
        }
    }

    cv::Mat gray(nY, nX, CV_8UC1);
    float range = mx - mn;
    if (range == 0) range = 1;
    for (int y = 0; y < nY; ++y)
        for (int x = 0; x < nX; ++x) {
            float v = raw[(size_t)y * nX + x];
            if (std::isnan(v))
                gray.at<uchar>(y, x) = 255;
            else {
                int s = int((v - mn) / range * 255.0f + 0.5f);
                gray.at<uchar>(y, x) = static_cast<uchar>(std::clamp(s, 0, 255));
            }
        }

    if ((size_t)nX * nY > MAX_PIXELS) {
        double scale = std::sqrt(double(MAX_PIXELS) / (nX * 1.0 * nY));
        int newX = std::max(1, int(nX * scale + 0.5));
        int newY = std::max(1, int(nY * scale + 0.5));
        cv::resize(gray, gray, cv::Size(newX, newY), 0, 0, cv::INTER_AREA);
        nX = newX;
        nY = newY;
        std::cout << "Downsampled to " << nX << "×" << nY << "\n";
    }

    int N = nX * nY;
    std::cout << "Graph nodes: " << N << "\n";

    std::vector<double> dist((size_t)N * N, INF);
    std::vector<int> nxt((size_t)N * N, -1);
    auto idx = [&](int x, int y) { return y * nX + x; };

    for (int i = 0; i < N; ++i) {
        dist[(size_t)i * N + i] = 0.0;
        nxt[(size_t)i * N + i] = i;
    }

    int dx[8] = {1, 1, 0, -1, -1, -1, 0, 1};
    int dy[8] = {0, 1, 1, 1, 0, -1, -1, -1};

    for (int y = 0; y < nY; ++y)
        for (int x = 0; x < nX; ++x) {
            int u = idx(x, y);
            if (gray.at<uchar>(y, x) == 255) continue;
            for (int k = 0; k < 8; ++k) {
                int nx = x + dx[k], ny = y + dy[k];
                if (nx < 0 || nx >= nX || ny < 0 || ny >= nY) continue;
                if (gray.at<uchar>(ny, nx) == 255) continue;
                int v = idx(nx, ny);
                double w = std::hypot(dx[k], dy[k]);
                dist[(size_t)u * N + v] = w;
                nxt[(size_t)u * N + v] = v;
            }
        }

    std::cout << "Running Floyd–Warshall (may take time) ...\n";
    double t0 = omp_get_wtime();

    floyd_warshall_parallel(N, dist.data(), nxt.data());

    double t1 = omp_get_wtime();
    std::cout << "Done.  Elapsed: " << (t1 - t0) << " s\n";

    cv::Mat base, vis;
    cv::cvtColor(gray, base, cv::COLOR_GRAY2BGR);
    vis = base.clone();

    cv::namedWindow("paths", cv::WINDOW_NORMAL);
    cv::setMouseCallback("paths", onMouse, nullptr);
    cv::imshow("paths", vis);

    bool needPath = false;
    while (true) {
        int key = cv::waitKey(15) & 0xFF;

        if (key == 27)
            break;

        if (key == 'c') {
            clicks.clear();
            needPath = false;
            vis = base.clone();
            cv::imshow("paths", vis);
            continue;
        }

        if (clicks.size() == 2 && !needPath) {
            needPath = true;
        }

        if (needPath) {
            int u = idx(clicks[0].x, clicks[0].y);
            int v = idx(clicks[1].x, clicks[1].y);

            std::vector<int> path(N);
            int L = reconstruct_path(u, v, N, nxt.data(), path.data());
            if (L == 0) {
                std::cerr << "No path!\n";
            } else {
                for (int i = 0; i + 1 < L; ++i) {
                    int a = path[i], b = path[i + 1];
                    cv::line(vis,
                             {a % nX, a / nX},
                             {b % nX, b / nX},
                             cv::Scalar(0, 0, 255),
                             1);
                }
                cv::circle(vis, clicks[0], 3, cv::Scalar(0, 255, 0), -1);
                cv::circle(vis, clicks[1], 3, cv::Scalar(255, 0, 0), -1);
            }

            cv::imshow("paths", vis);
            clicks.clear();
            needPath = false;
        }
    }

    cv::destroyAllWindows();
    return 0;
}