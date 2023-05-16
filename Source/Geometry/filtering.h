#ifndef FILTERING_H
#define FILTERING_H

#include <iostream>
#include <string.h>
#include <cstdint>
#include <vector>

namespace filter
{
	float* apply_1Dconvolution(int dim, float* image, int shape[3], std::vector<float> &kernel1D);
	std::vector<float> create_gaussiankernel(float sigma);

	//applies a kernel
	void apply_3Dconvolution_splitdims(float* image, int shape[3], std::vector<float> &kernel1D, float* &outimg);
	void apply_3Dconvolution_splitdims(float* image, int shape[3], std::vector<float> &kernel_dim0, std::vector<float> &kernel_dim1, std::vector<float> &kernel_dim2, float* &outimg);
    void apply_3Dconvolution_precise(float* image, int shape[3], std::vector<float> &kernel1D, float* outimg);

    //calculates and applies a Gaussian kernel
    void apply_3DGaussianFilter(float* &image, int shape[3], int fsize);
    void apply_3DGaussianFilter(float* &image, int shape[3], float sigma);
    void apply_3DGaussianFilter2Vector(float* vectorimg, int shape[3], float sigma, int ndims);

    //anisotropic Gaussian and Gabor filter
    void apply_2DanisotropicGaussian_spatialdomain(float* image, int shape[3], float sigma0, float sigma1, float theta, float* outimg);

    //and the median
    void apply_3DMedianFilter_cubic(float* &image, int shape[3], float radius);
    void apply_3DMedianFilter_spheric(float* &image, int shape[3], float radius);
    void apply2vector_3DMedianFilter_spheric(float* &vectorimage, int ndim, int shape[3], float radius);
    void apply2vector_3DMedianFilter_spheric(float* &vectorimage, float *mask, int ndim, int shape[3], float radius);

    //filters both frames
    void apply_3DImageFilter_2frame(float* &image0, float *&image1, int shape[3], float sigma, std::string filtername);

    //blurred derivatives
    float* calc_blurred_derivative(int dim, float* image, int shape[3], int stencil, int fd_order = 2);
    float* calc_Farid_derivative(float* image, int shape[3], int dim, int radius = 4, int order = 1, bool use_interpolator = true);
}

#endif // FILTERING_H
