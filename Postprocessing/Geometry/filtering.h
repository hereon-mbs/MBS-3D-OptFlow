#ifndef FILTERING_H
#define FILTERING_H

#include <iostream>
#include <string.h>
#include <cstdint>

namespace filter
{
	//applies a kernel
	void apply_2Dconvolution_splitdims(float* image2D, int shape[2], std::vector<float> &kernel1D, float* &outimg);
	void apply_3Dconvolution_splitdims(float* image, int shape[3], std::vector<float> &kernel1D, float* &outimg);
	void apply_3Dconvolution_splitdims_labelonly(float* image, int shape[3], std::vector<float> &kernel1D, float* &outimg, float* labelimage, float label); //only filter in mask region
	void apply_3Dconvolution_splitdims(float* image, int shape[3], std::vector<float> &kernel_dim0, std::vector<float> &kernel_dim1, std::vector<float> &kernel_dim2, float* &outimg);
    void apply_3Dconvolution_splitdims_interfaceonly(float* image, int shape[3], std::vector<float> &kernel1D, float* &outimg, uint8_t* phase_image);
    void apply_3Dconvolution_precise(float* image, int shape[3], std::vector<float> &kernel1D, float* outimg);

    //calculates and applies a Gaussian kernel
    void apply_2DGaussianFilter(float* &image, int shape[2], float sigma);
    //void apply_3DGaussianFilter(float* &image, int shape[3], int fsize);
    void apply_3DGaussianFilter(float* &image, int shape[3], float sigma);
    void apply_3DGaussianFilter2Label(float* &image, int shape[3], float sigma, float* labelimage, float label); //only filter over ROI in mask
    void apply_3DGaussianFilter2Vector(float* vectorimg, int shape[3], float sigma, int ndims);
    void apply_3DGaussianFilter2Vector(float* &ux, float* &uy, float* &uz, int shape[3], float sigma);
    void apply_3DGaussianFilter2Interface(float* &image, int shape[3], float sigma, uint8_t* phase_image);

    //anisotropic Gaussian and Gabor filter
    void apply_2DanisotropicGaussian_spatialdomain(float* image, int shape[3], float sigma0, float sigma1, float theta, float* outimg);

    //and the median
    void apply_3DMedianFilter_cubic(float* &image, int shape[3], float radius);
    void apply_3DMedianFilter_spheric(float* &image, int shape[3], float radius);
    void apply2vector_3DMedianFilter_spheric(float* &vectorimage, int ndim, int shape[3], float radius);
    void apply2vector_3DMedianFilter_spheric(float* &vectorimage, float *mask, int ndim, int shape[3], float radius);

    //filters both frames
    void apply_3DImageFilter_2frame(float* &image0, float *&image1, int shape[3], float sigma, std::string filtername);

    void maxfilter27neighbourhood(float* &image, int shape[3]);

    //blurred derivatives
    float* calc_blurred_derivative(int dim, float* image, int shape[3], int stencil, int fd_order = 2);
    float* calc_Farid_derivative(float* image, int shape[3], int dim, int radius, int order = 1, bool use_interpolator = true);

}

#endif // FILTERING_H

