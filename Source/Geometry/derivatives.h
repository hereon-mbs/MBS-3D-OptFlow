#ifndef DERIVATIVES_H
#define DERIVATIVES_H

#include <iostream>
#include <string.h>
#include <cstdint>

namespace derive
{
	typedef float imgtype;

	imgtype* firstDerivative_fourthOrder(imgtype *imgstack, int shape[3]);
	imgtype* firstDerivative_fourthOrder(int dim, imgtype *imgstack, int shape[3]);
	imgtype* firstDerivativeMagnitude_fourthOrder(imgtype *imgstack, int shape[3]);

	void add_gradientmask(imgtype *confidencemap, imgtype *frame0, imgtype *frame1, int shape[3], float p_used, float sigma_blur, int n_histobins = 256);
	void add_gradientweightedconfidence(imgtype *confidencemap, imgtype *frame0, int shape[3], float sigma_blur);
	void add_intensity_and_gradientweightedconfidence(imgtype *confidencemap, imgtype *frame0, int shape[3], float sigma_blur);
	imgtype* calculate_edgeorientation(imgtype *imgstack, int shape[3], std::string mode, float sigma);
}

#endif // DERIVATIVES_H
