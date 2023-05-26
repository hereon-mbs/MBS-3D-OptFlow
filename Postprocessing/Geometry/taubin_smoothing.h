#ifndef TAUBIN_SMOOTHING_H
#define TAUBIN_SMOOTHING_H

#include <vector>
#include <cstdint>
#include <iostream>
#include <algorithm>
#include <omp.h>

namespace mesh_filter
{
    class TaubinFilter
    {
    public:

        /* References:
         * the simple approach: Gabriel Taubin 1995: "Curve and Surface Smoothing without Shrinkage"
         * the fairing approach: Gabriel Taubin: "A Signal Processing Approach To Fair Surface Design"
         *
         * --> we'll just run the filter based on triangle-area
         */

        float lmbda = 0.5f;
        float mu = -0.53f;
        int iterations = 10;

        void run_areafilter(std::vector<std::vector<float>> &vertices, std::vector<std::vector<int64_t>> &triangles, std::pair<float,float> zrange = {-1e9, 1e9});

    private:
    };
}

#endif //TRIANGLE_QUERY
