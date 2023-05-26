#ifndef DISPLACEMENT_DERIVATIVES_H
#define DISPLACEMENT_DERIVATIVES_H

#include <iostream>
#include <vector>
#include <string.h>
#include <omp.h>

namespace derive
{
	float* calc_divergence(float* ux, float *uy, float* uz, int shape[3], std::string finitedifference_type="fourthorder");
	float* calc_volumetric_strain(float* ux, float *uy, float* uz, int shape[3], std::string finitedifference_type="fourthorder");
	float* calc_maximum_shear_strain(float* ux, float *uy, float* uz, int shape[3], std::string finitedifference_type="fourthorder");
	float* calc_from_green_strain(float* ux, float *uy, float* uz, int shape[3], std::string outval, std::string finitedifference_type="fourthorder");
	uint8_t* calc_localdisplacement_maxima(float* ux, float *uy, float* uz, int shape[3]);

	float* calc_directional_stretch(float* ui, int shape[3]);
}

#endif //DISPLACEMENT_DERIVATIVES_H
