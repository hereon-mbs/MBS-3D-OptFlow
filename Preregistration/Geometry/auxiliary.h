#ifndef AUXILIARY_H
#define AUXILIARY_H

#include <iostream>
#include <string.h>
#include <cstdint>
#include <vector>

namespace aux
{
    /*String-Manipulation
    *********************************************************/
    std::string zfill_int2string(int inint, const unsigned int &zfill);

    /*Numpy-like
    *********************************************************/
    std::vector<float> linspace(float startval, float endval, uint64_t bins);
    std::vector<double> linspace(double startval, double endval, uint64_t bins);

    float* reduce2activevoxels(float *image, float *mask, int shape[3], long long int &nactive_out);

    float* project_maximum(int axis, float* imagestack, int shape[3]);
    float* project_average(int axis, float* imagestack, int shape[3]);
    float* project_masked_average(int axis, float* imagestack, float *mask, int shape[3], bool mask2D = true);


}

#endif // AUXILIARY_H
