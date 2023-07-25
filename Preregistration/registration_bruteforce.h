#ifndef REGISTRATION_BRUTEFORCE_H
#define REGISTRATION_BRUTEFORCE_H

#include <iostream>

/*********************************************************************************************************************************************************
 * Location: Helmholtz-Zentrum fuer Material und Kuestenforschung, Max-Planck-Strasse 1, 21502 Geesthacht
 * Author: Stefan Bruns
 * Contact: bruns@nano.ku.dk
 *
 * License: TBA
 *********************************************************************************************************************************************************/

namespace registrate
{
	class BruteForce
	{
	public:
		int configure_device(int maxshape[3], int deviceID_, bool use_mask, int interpolation_order);
		void free_device();

		void set_frames(float* frame0, float *frame1, int shape[3]);
		void set_frames(float* frame0, float *frame1, float *mask, int shape[3]);

		float execute_correlation(float dx, float dy, float dz, int shape[3]); //floating point translation
		float execute_correlation(float jaw, float pitch, float roll, float dx, float dy, float dz, int shape[3], float rotcenter[3], bool use_mask);

		float next_gradientascentstep_translation(float result[6], float h, float gamma, int shape[3], float out_step[3]);
		float next_gradientascentstep(float result[6], float h_trans, float h_rot, float gamma, int dofflag[6],int shape[3],
				float out_step[6], float rotcenter[3], bool use_mask);
		float ascent_translation_singledimension(int dim, float best_corr, float stepsize, float result[3], int shape[3], int max_extensions);
		float ascent_singledimension(int dim, float best_corr, float stepsize, float result[3], int shape[3], int max_extensions, float rotcenter[3], bool use_mask);

	private:
		void prepare_rotation_coefficients(float *out_coefficients, float jaw, float roll, float pitch);

		float *devframe0, *devframe1, *devmask;
		float *gridreduce0, *gridreduce1;
		float *rotationmatrix, *dummy;

		int threadsPerBlock = 128;
		int deviceID = 0;
	};
}

#endif //REGISTRATION_BRUTEFORCE_H
