#ifndef HIGHPASSFILTER_H
#define HIGHPASSFILTER_H

#include <vector>
#include <stdexcept>
#include <cstdint>
#include <fftw3.h>
#include <omp.h>
#include "auxiliary.h"
#include "hdcommunication.h"
#include <chrono>
#include <thread>

/* Currently now parrallel dft compiled! */

namespace fourier_filtering
{
	fftw_complex* createKernel_Lowpass2DFull(double sigma, int shape[3], int padding, int planning_rigor)
	{
		int nx = shape[0]+2*padding; int ny = shape[1]+2*padding;
		long long int nslice = nx*ny;

		double* spatial_mask = (double*) fftw_malloc(nslice*sizeof(double));
		fftw_complex* fourier_mask = (fftw_complex*) fftw_malloc(sizeof(fftw_complex) * ((long long int) nx * (ny/2)+1));

		fftw_plan plan1 = fftw_plan_dft_r2c_2d(nx, ny, spatial_mask, fourier_mask, planning_rigor);         //forward plan mask
		double sum = 0.0;

		#pragma omp parallel for reduction(+: sum)
		for(long long int idx = 0; idx < nslice; idx++)
		{
			int x = idx/ny;
			int y = idx-x*ny;

			//double x1 = x-xc;
			//double y1 = y-yc;

			double x1 = std::min(x, nx-x);
			double y1 = std::min(y, ny-y);

			double sqdist = x1*x1+y1*y1;
			double val = exp(-(sqdist)/(2.*sigma*sigma));
			sum += val;

			spatial_mask[idx] = val;
		}

		//Normalize
		#pragma omp parallel for
		for(long long int idx = 0; idx < nslice; idx++)
			spatial_mask[idx] /= sum;

		fftw_execute(plan1);

		//fftw_free(spatial_mask);
		fftw_destroy_plan(plan1);

		return fourier_mask;
	}
	void pad2D_convert2rowmajor(float *imagestack, double *output, int shape[3], int padding, int z, std::string padding_type)
	{
		int outshape[2] = {shape[0]+2*padding, shape[1]+2*padding};
		long long int nslice_out = outshape[0]*outshape[1];
		long long int nslice = shape[0]*shape[1];

		#pragma omp parallel for
		for (long long int idx = 0; idx < nslice_out; idx++)
		{
			int x = idx/outshape[1];
			int y = idx-x*outshape[1];

			int xorig = x-padding;
			int yorig = y-padding;

			if (padding_type == "nearest")
			{
				if (xorig < 0) xorig = 0;
				else if (xorig >= shape[0]) xorig = shape[0]-1;
				if (yorig < 0) yorig = 0;
				else if (yorig >= shape[1]) yorig = shape[1]-1;
			}
			else
			{
				//"reflective"
				if (xorig < 0) xorig = abs(x-padding);
				else if (xorig >= shape[0]) xorig = shape[0]-x+shape[0]-1;
				if (yorig < 0) yorig = abs(y-padding);
				else if (yorig >= shape[1]) yorig = shape[1]-y+shape[1]-1;
			}

			long long int idxorig = (z*nslice) + (yorig*shape[0] + xorig);

			output[idx] = imagestack[idxorig];
		}
		return;
	}

    void dft_2Dhighpass(float *imagestack, float *output, int shape[3], double sigma, int planning_rigor)
    {
        int nx = shape[0], ny = shape[1];
        uint64_t n_slice = shape[0]*shape[1];
        double *mask = (double*) malloc(sizeof(double) * nx * ((ny/2)+1));

        //calculate the relative frequencies for each axis
        std::vector<double> y_freq = aux::linspace(0., 1., ((ny/2)+1));
        std::vector<double> x_freq = aux::linspace(0., 1., ceil(nx/2.));
        std::vector<double> x_freq_inv = aux::linspace(1., 0., ceil(nx/2.));
        x_freq.insert(x_freq.end(), x_freq_inv.begin(), x_freq_inv.end());
        if(x_freq.size() > (uint64_t) nx) x_freq.erase(x_freq.begin() + nx);

        //Provide a mask for transformation
        uint64_t idx_mask = 0;
        for (uint64_t x = 0; x < (uint64_t) nx; x++)
        {
            for(uint64_t y = 0; y < (uint64_t) ((ny/2)+1); y++)
            {
                mask[idx_mask] = sqrt(x_freq[x]*x_freq[x]+y_freq[y]*y_freq[y]);  //calculate radial spatial frequency
                mask[idx_mask] = 1-(exp(-(mask[idx_mask]*mask[idx_mask])/(2*sigma*sigma)));
                idx_mask++;
            }
        }

        //maintain brightness:
        mask[0] = 1.;
        mask[(uint64_t) (nx-1) * ((ny/2)+1)] = 1.;

        //Set up plans
        double *in1, *result;
        fftw_complex *out1;

        //Allocate
        in1  = (double*) malloc(sizeof(double) * nx * ny);
        out1 = (fftw_complex*) fftw_malloc(sizeof(fftw_complex) * nx * ((ny/2)+1));
        result = (double*) malloc(sizeof(double) * nx * ny);

        //Note: plans can not run in parallel threads. Would need to prepare a plan for every thread beforehand
        fftw_plan plan1 = fftw_plan_dft_r2c_2d(nx, ny, in1, out1, planning_rigor);         //forward plan
        fftw_plan plan3 = fftw_plan_dft_c2r_2d(nx, ny, out1, result, planning_rigor);      //backward plan

        for (uint64_t z = 0; z < (uint64_t) shape[2]; z++)
        {
            //fill in grayvalues
            //need to convert to row major format!!!!
            uint64_t inputidx = 0;
            for (uint64_t y = 0; y < (uint64_t) shape[1]; y++)
            {
                for (uint64_t x = 0; x < (uint64_t) shape[0]; x++)
                {
                    uint64_t idx = x*shape[1]+y;
                    in1[idx] = imagestack[inputidx+(z*n_slice)];
                    inputidx++;
                }
            }

            //Fourier transform
            fftw_execute(plan1);

            //Merge the DFTs
            double paganin_weight = 1.;
            for(uint64_t idx = 0; idx < (uint64_t) (nx * ((ny/2)+1)); idx++)
            {
                out1[idx][0] *= mask[idx];
                out1[idx][1] *= mask[idx];
            }

            fftw_execute(plan3);

            //Write output by converting back to column major format
            uint64_t idx_ifft = 0;
            for (uint64_t x = 0; x < (uint64_t) nx; x++)
            {
                for(uint64_t y = 0; y < (uint64_t) ny; y++)
                {
                    uint64_t idx_output = y*nx + x + (z*n_slice);
                    output[idx_output] = result[idx_ifft]/n_slice; //Output scales with number of elements
                    idx_ifft++;
                }
            }
        }

        free(mask);
        fftw_free(out1);
        free(in1);
        free(result);
        fftw_destroy_plan(plan1);
        fftw_destroy_plan(plan3);

        return;
    }
	void dft_lowpass2D(float *imagestack, float *output, int shape[3], double sigma, int planning_rigor)
	{
		int padding = (int) ceil(3.*sigma);

		int nx = shape[0]; int ny = shape[1];
		int nx_fft = nx+2*padding; int ny_fft = ny + 2*padding;
		long long int nslice = nx*ny;
		long long int nslice_fft = nx_fft*ny_fft;

		fftw_complex *fourier_mask = createKernel_Lowpass2DFull(sigma, shape, padding, planning_rigor);

		double *result = (double*) fftw_malloc(nslice_fft*sizeof(*result));
		fftw_complex *out1 = (fftw_complex*) fftw_malloc(sizeof(fftw_complex) * ((long long int) nx_fft * (ny_fft/2)+1));

		//Set up plans
		//////////////////////////////////////////////////////////////////////////////////
		//Note: plans can not run in parallel threads. Would need to prepare a plan for every thread beforehand
		fftw_plan plan1 = fftw_plan_dft_r2c_2d(nx_fft, ny_fft, result, out1, planning_rigor);         //forward plan
		fftw_plan plan2 = fftw_plan_dft_c2r_2d(nx_fft, ny_fft, out1, result, planning_rigor);      //backward plan
		//////////////////////////////////////////////////////////////////////////////////

		//Loop over stack
		for (int z = 0; z < shape[2]; z++)
		{
			pad2D_convert2rowmajor(imagestack, result, shape, padding, z, "nearest");
			fftw_execute(plan1);

			//Apply filter
			#pragma omp parallel for
			for(uint64_t idx = 0; idx < (uint64_t) (nx_fft * ((ny_fft/2)+1)); idx++)
			{
				//complex multiplication
				double realpart = out1[idx][0]*fourier_mask[idx][0] - out1[idx][1]*fourier_mask[idx][1];
				double imagpart = out1[idx][0]*fourier_mask[idx][1] + out1[idx][1]*fourier_mask[idx][0];

				out1[idx][0] = realpart;
				out1[idx][1] = imagpart;
			}

			fftw_execute(plan2);

			//Write output by converting back to column major format and shifting
			#pragma omp parallel for
			for (long long int idx = 0; idx < nslice; idx++)
			{
				int y = idx/shape[0];
				int x = idx-y*shape[0];
				uint64_t idx_ifft = (x+padding)*ny_fft + (y+padding);

				output[z*nslice + idx] = result[idx_ifft]/nslice_fft; //Output scales with number of elements
			}
		}

        fftw_destroy_plan(plan1);
        fftw_destroy_plan(plan2);

		//free(spatial_mask);
		fftw_free(fourier_mask);
		fftw_free(out1);

		return;
	}
}

#endif // HIGHPASSFILTER_H
