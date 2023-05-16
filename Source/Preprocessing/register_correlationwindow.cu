#include <iostream>
#include <cuda.h>
#include <omp.h>
#include <typeinfo>
#include <limits>
#include <vector>
#include <math.h>
#include "register_correlationwindow.h"

/*********************************************************************************************************************************************************
 * Location: Helmholtz-Zentrum fuer Material und Kuestenforschung, Max-Planck-Strasse 1, 21502 Geesthacht
 * Author: Stefan Bruns
 * Contact: bruns@nano.ku.dk
 *
 * License: TBA
 *********************************************************************************************************************************************************/

namespace corrwindow
{
	namespace gpu_const
	{
		__constant__ int nx_c, ny_c, nz_c;
		__constant__ int interpolation_order_c = 1; //1 = linear, 2 = cubic
	}

	namespace gpu_solve
	{
	}

	int NaiveOptimizer::configure_device(int maxshape[3], int deviceID_, int interpolation_order)
	{
		deviceID = deviceID_;
		cudaSetDevice(deviceID);

		long long int nslice = maxshape[0]*maxshape[1];
		long long int nstack = maxshape[2]*nslice;
		long long int blocksPerGrid = (nstack + threadsPerBlock - 1) / (threadsPerBlock);

		//check memory requirements
		////////////////////////////////////////////////////
		size_t free_byte, total_byte ;
		cudaMemGetInfo( &free_byte, &total_byte ) ;

		double free_db = (double)free_byte ;
		double expected_usage = 2.*nstack*sizeof(float);
		expected_usage += 6*(2*blocksPerGrid)*sizeof(float);

		if (expected_usage > free_db){std::cout << "\033[1;31mError! Expected to run out of GPU memory!\033[0m" << std::endl;return 2;}
		////////////////////////////////////////////////////

		//allocate memory and set constant memory
		////////////////////////////////////////////////////
		(float*) cudaMalloc((void**)&devframe0, nstack*sizeof(*devframe0));
		(float*) cudaMalloc((void**)&devframe1, nstack*sizeof(*devframe1));
		(float*) cudaMalloc((void**)&gridreduce0, 6*blocksPerGrid*sizeof(*gridreduce0));
		(float*) cudaMalloc((void**)&gridreduce1, 6*blocksPerGrid*sizeof(*gridreduce1));

		cudaMemcpyToSymbol(gpu_const::interpolation_order_c, &interpolation_order, sizeof(gpu_const::interpolation_order_c));

		cudaDeviceSynchronize();
		////////////////////////////////////////////////////

		std::string error_string = (std::string) cudaGetErrorString(cudaGetLastError());
		if (error_string != "no error")
		{
			std::cout << "Device Variable Copying: " << error_string << std::endl;
			return 1;
		}

		return 0;
	}
	void NaiveOptimizer::free_device()
	{
		cudaSetDevice(deviceID);

		cudaFree(devframe0);
		cudaFree(devframe1);
		cudaFree(gridreduce0);
		cudaFree(gridreduce1);
	}
	void NaiveOptimizer::set_frames(float* frame0, float *frame1, int shape[3])
	{
		cudaSetDevice(deviceID);

		int nx = shape[0]; int ny = shape[1]; int nz = shape[2];
		long long int nslice = shape[0]*shape[1];
		long long int nstack = shape[2]*nslice;
		long long int asize = nstack*sizeof(*devframe0);

		cudaMemcpyAsync(devframe0, frame0, asize, cudaMemcpyHostToDevice);
		cudaMemcpyAsync(devframe1, frame1, asize, cudaMemcpyHostToDevice);

		cudaMemcpyToSymbol(gpu_const::nx_c, &nx, sizeof(gpu_const::nx_c));
		cudaMemcpyToSymbol(gpu_const::ny_c, &ny, sizeof(gpu_const::ny_c));
		cudaMemcpyToSymbol(gpu_const::nz_c, &nz, sizeof(gpu_const::nz_c));
		cudaDeviceSynchronize();

		return;
	}

	std::vector<std::vector<float>> NaiveOptimizer::run_integertranslation_prestrained_cpu(std::vector<std::vector<int>> &support_points, float strain_guess, float* frame0, float* frame1)
	{
		long long int nslice = shape[0]*shape[1];
		long long int nstack = shape[2]*nslice;

		//we have an idea of the strain and search for local deviations
		std::vector<std::vector<float>> guess;


		//set initial strain guess for the points to be evaluated
		//////////////////////////////////////////////////////////////
		float zorigin = shape[2]/2.-0.5f;

		for (long long int idx = 0; idx < support_points.size(); idx++)
		{
			int z = support_points[idx][2];
			float shift = (z-zorigin)*strain_guess;
			guess.push_back({0.0,0.0,shift});
		}
		//////////////////////////////////////////////////////////////

		std::vector<float> stepsizes = prepare_stepsizes_(min_step_translation, max_step_translation, step_scaling);

		//optimize each point
		//#pragma omp parallel for
		//for (long long int idx = 0; idx < support_points.size(); idx++)
		//std::cout << support_points.size() << std::endl;
		#pragma omp parallel for
		for (long long int idx = 0; idx < support_points.size(); idx++)
		{
			int first_guess_z = ((int) std::round(guess[idx][2]));
			int x0 = support_points[idx][0];
			int y0 = support_points[idx][1];
			int z0 = support_points[idx][2];

			float* window0 = get_correlation_window_cpu(frame0, x0, y0, z0);
			float* window1 = get_correlation_window_cpu(frame1, x0, y0, z0, 0, 0, first_guess_z);
			free(window1);

			float best_corr = get_correlation_cpu(window0, window1);
			int best_guess[3] = {0, 0, first_guess_z};
			//std::cout << best_corr << ": " << best_guess[0] << "," << best_guess[1] << "," << best_guess[2] << std::endl;

			for (int dz = -10+first_guess_z; dz <= 10+first_guess_z; dz++)
			for (int dy = -10; dy <= 10; dy++)
			for (int dx = -10; dx <= 10; dx++)
			{
				float* window2 = get_correlation_window_cpu(frame1, x0, y0, z0, dx, dy, dz);
				float this_corr = get_correlation_cpu(window0, window2);
				free(window2);

				if (this_corr > best_corr)
				{
					best_corr = this_corr;
					best_guess[0] = dx; best_guess[1] = dy; best_guess[2] = dz;
				//std::cout << best_corr << ": " << best_guess[0] << "," << best_guess[1] << "," << best_guess[2] << std::endl;
				}
			}
			//std::cin.get();

			if (omp_get_thread_num() == 0 && (idx+1)%100 == 0)
			{
			std::cout << (idx+1) << "/" << support_points.size() << "    \r";
				std::cout.flush();
			}

			guess[idx][0] = best_guess[0];
			guess[idx][1] = best_guess[1];
			guess[idx][2] = best_guess[2];
		}

		return guess;
	}

	float* NaiveOptimizer::get_correlation_window_cpu(float* frame, int x0, int y0, int z0)
	{
		int nz = window_shape[2]; int ny = window_shape[1]; int nx = window_shape[0];
		long long int windownslice = nx*ny;
		long long int windowsize = windownslice*nz;
		long long int framenslice = shape[0]*shape[1];
		float* window = (float*) malloc(windowsize*sizeof(*window));

		for (int z1 = -nz/2; z1 <= nz/2; z1++)
		{
			int z2 = z0+z1;
			if (z2 < 0) z2 = -z2;
			else if (z2 > shape[2]-1) z2 = 2*shape[2]-z2-2;

		for (int y1 = -ny/2; y1 <= ny/2; y1++)
		{
			int y2 = y0+y1;
			if (y2 < 0) y2 = -y2;
			else if (y2 > shape[1]-1) y2 = 2*shape[1]-y2-2;

		for (int x1 = -nx/2; x1 <= nx/2; x1++)
		{
			int x2 = x0+x1;
			if (x2 < 0) x2 = -x2;
			else if (x2 > shape[0]-1) x2 = 2*shape[0]-x2-2;

			window[(z1+nz/2)*windownslice+(y1+ny/2)*nx+x1+nx/2] = frame[z2*framenslice+y2*shape[0]+x2];
		}}}

		return window;
	}
	float* NaiveOptimizer::get_correlation_window_cpu(float* frame, int x0, int y0, int z0, int dx, int dy, int dz)
	{
		int nz = window_shape[2]; int ny = window_shape[1]; int nx = window_shape[0];
		long long int windownslice = nx*ny;
		long long int windowsize = windownslice*nz;
		long long int framenslice = shape[0]*shape[1];
		float* window = (float*) malloc(windowsize*sizeof(*window));

		for (int z1 = -nz/2; z1 <= nz/2; z1++)
		{
			int z2 = z0+z1+dz;
			if (z2 < 0) z2 = -z2;
			else if (z2 > shape[2]-1) z2 = 2*shape[2]-z2-2;

		for (int y1 = -ny/2; y1 <= ny/2; y1++)
		{
			int y2 = y0+y1+dy;
			if (y2 < 0) y2 = -y2;
			else if (y2 > shape[1]-1) y2 = 2*shape[1]-y2-2;

		for (int x1 = -nx/2; x1 <= nx/2; x1++)
		{
			int x2 = x0+x1+dx;
			if (x2 < 0) x2 = -x2;
			else if (x2 > shape[0]-1) x2 = 2*shape[0]-x2-2;

			window[(z1+nz/2)*windownslice+(y1+ny/2)*nx+x1+nx/2] = frame[z2*framenslice+y2*shape[0]+x2];
		}}}

		return window;
	}
	float NaiveOptimizer::get_correlation_cpu(float* window0, float* window1)
	{
		float sum0 = 0.0; float sum1 = 0.0;
		long long int nslice = window_shape[0]*window_shape[1];
		long long int windowsize = nslice*window_shape[2];

		for (long long int idx = 0; idx < windowsize; idx++)
		{
			float val0 = window0[idx];
			float val1 = window1[idx];

			sum0 += val0;
			sum1 += val1;
		}

		float mean0 = sum0/windowsize;
		float mean1 = sum1/windowsize;
		float std0 = 0.0;
		float std1 = 0.0;
		float corr = 0.0;

		for (long long int idx = 0; idx < windowsize; idx++)
		{
			float val0 = window0[idx];
			float val1 = window1[idx];

			corr += (val0-mean0)*(val1-mean1);
			std0 += (val0-mean0)*(val0-mean0);
			std1 += (val1-mean1)*(val1-mean1);
		}

		std0 = std::sqrt(std0/windowsize);
		std1 = std::sqrt(std1/windowsize);
		corr = corr/(windowsize*std0*std1);

		return corr;
	}

	std::vector<float> NaiveOptimizer::prepare_stepsizes_(float min_stepsize, float max_stepsize, float scaling)
	{
		float stepsize = round(max_stepsize*1.e6f)/1.e6f;
		std::vector<float> stepsize_list;

		while(stepsize > min_stepsize)
		{
			stepsize_list.push_back(stepsize);
			stepsize *= scaling;
			stepsize = round(stepsize*1.e6f)/1.e6f;
		}

		stepsize_list.push_back(min_stepsize);

		return stepsize_list;
	}
}
