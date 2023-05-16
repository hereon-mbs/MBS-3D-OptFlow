#ifndef RESAMPLING_H
#define RESAMPLING_H

#include <iostream>
#include <string.h>
#include <cstdint>
#include <vector>

namespace resample
{
    //Pointwise functions:
    ////////////////////////////////////////////////////////////////////////
    float interpolate_cubic(float y0, float y1, float y2, float y3, float mu);
    ////////////////////////////////////////////////////////////////////////

    //Resampling with Coons-patches (recommended)
    ////////////////////////////////////////////////////////////////////////
    void linear_coons(float* input, int inshape[3], float* output, int outshape[3], int vector_dims = 1, bool scale_intensity = false);
    void cubic_coons(float* input, int inshape[3], float* output, int outshape[3], int vector_dims = 1, bool scale_intensity = false);
    ////////////////////////////////////////////////////////////////////////

    //Resampling with interpolation
    ////////////////////////////////////////////////////////////////////////
    void downsample_linear(float* input, int inshape[3], float* output, int outshape[3], bool average);
    void upsample_linear(float* input, int inshape[3], float* output, int outshape[3]);
    void upsample_cubic(float* input, int inshape[3], float* output, int outshape[3]);
    ////////////////////////////////////////////////////////////////////////

    //Resampling by majority
	////////////////////////////////////////////////////////////////////////
	float* downsample_majority_bin2(float* input, int inshape[3], bool keep_bigger_value = false);
	////////////////////////////////////////////////////////////////////////

    //Upscaling a flow vector
    ////////////////////////////////////////////////////////////////////////
    void upscalevector(float *&u, int last_shape[3], int next_shape[3], int ndims, std::string interpolation_mode);
    ////////////////////////////////////////////////////////////////////////
}

#endif //RESAMPLING_H
