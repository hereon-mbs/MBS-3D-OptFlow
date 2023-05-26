#include <iostream>
#include <cstdlib>
#include <time.h>
#include <algorithm>
#include <numeric>
#include <math.h>

using namespace std;

namespace polardec
{
	//Reference: https://www.continuummechanics.org/polardecomposition.html

	float _determinant3x3(float M[9])
	{
		//assuming row first
		return (M[0]*M[4]*M[8]) + (M[1]*M[5]*M[6]) + (M[2]*M[3]*M[7]) - (M[6]*M[4]*M[2]) - (M[7]*M[5]*M[0]) - (M[8]*M[3]*M[1]);
	}
	int _invert3x3_Cramer(float M[9], float out[9])
	{
		//returns 0 if there is no inverse

		float D = _determinant3x3(M);
		if (D == 0) return 0;

		//first row
		float Mtmp[9] = {1.f, M[1], M[2], 0.f, M[4], M[5], 0.f, M[7], M[8]};
		out[0] = _determinant3x3(Mtmp)/D;
		Mtmp[0] = 0.f; Mtmp[3] = 1.f;
		out[1] = _determinant3x3(Mtmp)/D;
		Mtmp[3] = 0.f; Mtmp[6] = 1.f;
		out[2] = _determinant3x3(Mtmp)/D;

		//second row
		Mtmp[0] = M[0]; Mtmp[3] = M[3]; Mtmp[6] = M[6];
		Mtmp[1] = 1.f; Mtmp[4] = 0.f; Mtmp[7] = 0.f;
		out[3] = _determinant3x3(Mtmp)/D;
		Mtmp[1] = 0.f; Mtmp[4] = 1.f;
		out[4] = _determinant3x3(Mtmp)/D;
		Mtmp[4] = 0.f; Mtmp[7] = 1.f;
		out[5] = _determinant3x3(Mtmp)/D;

		//third row
		Mtmp[1] = M[1]; Mtmp[4] = M[4]; Mtmp[7] = M[7];
		Mtmp[2] = 1.f; Mtmp[5] = 0.f; Mtmp[8] = 0.f;
		out[6] = _determinant3x3(Mtmp)/D;
		Mtmp[2] = 0.f; Mtmp[5] = 1.f;
		out[7] = _determinant3x3(Mtmp)/D;
		Mtmp[5] = 0.f; Mtmp[8] = 1.f;
		out[8] = _determinant3x3(Mtmp)/D;

		return 1;
	}
	void _transpose3x3(float M[9])
	{
		float val1 = M[1]; float val3 = M[3];
		M[3] = val1; M[1] = val3;

		val1 = M[2]; val3 = M[6];
		M[6] = val1; M[2] = val3;

		val1 = M[5]; val3 = M[7];
		M[7] = val1; M[5] = val3;
		return;
	}

	int _iterative_decomposition(float F[9], float out[9], float epsilon = 1.e-6f)
	{
		//takes the deformation gradient and calculates the rotation matrix
		//return 0 on fail

		float A_it[9];

		//prepare inverse transpose of deformation gradient
		int fail = _invert3x3_Cramer(F, A_it);
		if (fail == 0) return 0;
		_transpose3x3(A_it);

		//A1 = 0.5*(A+A_it) := initialize result
		for (int i = 0; i < 9; i++) out[i] = 0.5f*(F[i]+A_it[i]);

		//iterate
		float maxchange = epsilon+1.f;
		while (maxchange > epsilon)
		{
			maxchange = 0.0f;
			_invert3x3_Cramer(out, A_it);
			_transpose3x3(A_it);

			for (int i = 0; i < 9; i++)
			{
				float oldval = out[i];
				float newval = 0.5f*(oldval+A_it[i]);
				out[i] = newval;

				if(fabs(newval-oldval) > maxchange) maxchange = fabs(newval-oldval);
			}
		}

		return 1;
	}
	int _strain_matrix(float F[9], float R[9], float U[9])
	{
		float R_i[9];
		int fail = _invert3x3_Cramer(R, R_i);
		if (fail == 0) return 0;

		U[0] = R_i[0]*F[0] + R_i[1]*F[3] + R_i[2]*F[6];
		U[1] = R_i[0]*F[1] + R_i[1]*F[4] + R_i[2]*F[7];
		U[2] = R_i[0]*F[2] + R_i[1]*F[5] + R_i[2]*F[8];

		U[3] = R_i[3]*F[0] + R_i[4]*F[3] + R_i[5]*F[6];
		U[4] = R_i[3]*F[1] + R_i[4]*F[4] + R_i[5]*F[7];
		U[5] = R_i[3]*F[2] + R_i[4]*F[5] + R_i[5]*F[8];

		U[6] = R_i[6]*F[0] + R_i[7]*F[3] + R_i[8]*F[6];
		U[7] = R_i[6]*F[1] + R_i[7]*F[4] + R_i[8]*F[7];
		U[8] = R_i[6]*F[2] + R_i[7]*F[5] + R_i[8]*F[8];

		return 1;
	}

	float get_divergence_of_strain(float deformation_gradient[9], float epsilon=1.e-6f)
	{
		float rotation_matrix[9];
		int fail = _iterative_decomposition(deformation_gradient, rotation_matrix, epsilon);

		if (fail == 0) return 0.0f;

		float strain_matrix[9];
		fail = _strain_matrix(deformation_gradient, rotation_matrix, strain_matrix);
		if (fail == 0) return 0.0f;

		return strain_matrix[0]+strain_matrix[4]+strain_matrix[8]-3.f;
	}
	float get_volumetricstrain(float deformation_gradient[9], float epsilon=1.e-6f)
	{
		float rotation_matrix[9];
		int fail = _iterative_decomposition(deformation_gradient, rotation_matrix, epsilon);

		if (fail == 0) return 0.0f;

		float U[9];
		fail = _strain_matrix(deformation_gradient, rotation_matrix, U);
		if (fail == 0) return 0.0f;

		//volumetric strain calculated as det(U)-1
		float volstrain = _determinant3x3(U)-1.f;

		return volstrain;
	}
}
