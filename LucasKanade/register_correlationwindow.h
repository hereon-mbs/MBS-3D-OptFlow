#ifndef REGISTER_CORRELATIONWINDOW_H
#define REGISTER_CORRELATIONWINDOW_H

#include <iostream>
#include <vector>

/*********************************************************************************************************************************************************
 * Location: Helmholtz-Zentrum fuer Material und Kuestenforschung, Max-Planck-Strasse 1, 21502 Geesthacht
 * Author: Stefan Bruns
 * Contact: bruns@nano.ku.dk
 *
 * License: TBA
 *********************************************************************************************************************************************************/

namespace corrwindow
{
	class NaiveOptimizer
	{
	public:
		int interpolation_order = 2; //1 = linear, 2 = cubic
		int shape[3]; int window_shape[3];

		float step_scaling = 0.5f;
		float max_step_translation = 1.0;
		float min_step_translation = 0.1;

		NaiveOptimizer(float* frame0, float* frame1, int shape_[3], int window_shape_[3], int deviceID)
		{
			shape[0] = shape_[0]; shape[1] = shape_[1]; shape[2] = shape_[2];
			window_shape[0] = window_shape_[0]; window_shape[1] = window_shape_[1]; window_shape[2] = window_shape_[2];
			int errorcode = configure_device(shape, deviceID, interpolation_order);
			if (errorcode != 0) std::cout << "Configuration error!" << std::endl;

			set_frames(frame0, frame1, shape);
		}

		int configure_device(int maxshape[3], int deviceID_, int interpolation_order);
		void free_device();
		void set_frames(float* frame0, float *frame1, int shape[3]);

		std::vector<std::vector<float>> run_integertranslation_prestrained_cpu(std::vector<std::vector<int>> &support_points, float strain_guess, float* frame0, float* frame1);

	private:

		float *devframe0, *devframe1;
		float *gridreduce0, *gridreduce1;
		float *rotationmatrix, *dummy;

		int threadsPerBlock = 128;
		int deviceID = 0;

		std::vector<float> prepare_stepsizes_(float min_stepsize, float max_stepsize, float scaling = 0.5f);

		//CPU for testing:
		float* get_correlation_window_cpu(float* frame, int x0, int y0, int z0);
		float* get_correlation_window_cpu(float* frame, int x0, int y0, int z0, int dx, int dy, int dz);
		float get_correlation_cpu(float* window0, float* window1);
	};
}

#endif //REGISTER_CORRELATIONWINDOW_H
