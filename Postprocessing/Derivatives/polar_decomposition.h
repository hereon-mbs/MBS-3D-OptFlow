#ifndef POLAR_DECOMPOSITION_H
#define POLAR_DECOMPOSITION_H

#include <iostream>
#include <cstdlib>
#include <time.h>
using namespace std;

namespace polardec
{
	//Reference: https://www.continuummechanics.org/polardecomposition.html

	float _determinant3x3(float M[9]);
	int _invert3x3_Cramer(float M[9], float out[9]);
	void _transpose3x3(float M[9]);

	int _iterative_decomposition(float F[9], float out[9], float epsilon = 1.e-6f);
	int _strain_matrix(float F[9], float R[9], float U[9]);

	float get_divergence_of_strain(float deformation_gradient[9], float epsilon=1.e-6f);
	float get_volumetricstrain(float deformation_gradient[9], float epsilon=1.e-6f);
}


#endif //POLAR_DECOMPOSITION_H
