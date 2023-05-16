#include <iostream>
#include <cuda.h>
#include <omp.h>
#include <typeinfo>
#include <limits>
#include "optflow_gpu3d.h"
#include "../Derivatives/smoothnessterm_gpu3d.cuh"
#include "gpu_constants.cuh"

/*********************************************************************************************************************************************************
 * Location: Helmholtz-Zentrum fuer Material und Kuestenforschung, Max-Planck-Strasse 1, 21502 Geesthacht
 * Author: Stefan Bruns
 * Contact: bruns@nano.ku.dk
 *
 * License: TBA
 *********************************************************************************************************************************************************/

namespace optflow
{
	namespace gpu3d
	{
		__device__ __inline__ float interpolate_cubic(float &y0, float &y1, float &y2, float &y3, float &mu)
		{
			float mu2 = mu*mu;

			float a0 = y3-y2-y0+y1;
			float a1 = y0-y1-a0;
			float a2 = y2-y0;
			float a3 = y1;

			return a0*mu*mu2+a1*mu2+a2*mu+a3;
		}

		__global__ void gaussianfilter3D_x(optflow_type *input, optflow_type *output)
		{
			//acquire constants
			/////////////////////////////////////////////
			float sigma = gpu_const::filter_sigma_c;

			int nx = gpu_const::nx_c;
			int ny = gpu_const::ny_c;
			int nz = gpu_const::nz_c;

			idx_type nslice = nx*ny;
			idx_type nstack = nz*nslice;

			bool outofbounds = false;
			idx_type pos = (blockIdx.x*blockDim.x+threadIdx.x);
			if (pos >= nstack) {outofbounds = true; pos = threadIdx.x;}

			int z = pos/nslice;
			int y = (pos-z*nslice)/nx;
			int x = pos-z*nslice-y*nx;
			/////////////////////////////////////////////

			//Create Gaussian kernel
			///////////////////////////////////////////////////
			int fsize = (int) (3*sigma);
			float kernelsum = 0.0f;
			float valuesum = 0.0f;
			//////////////////////////////////////////////////

			if (sigma > 0.0f)
			{
				for(int xi=-fsize; xi<=fsize; xi++)
				{
					int x0 = x+xi;

					//reflective boundaries
					if (x0 < 0) x0 = -x0;
					else if (x0 >= nx) x0 = 2*nx-x0-2;

					float kernel_val = expf(-(xi*xi)/(sigma*sigma*2));
					kernelsum += kernel_val;

					__syncthreads();
					if (x0 < 0 || x0 >= nx) continue;
					valuesum += kernel_val*input[z*nslice+y*nx+x0];
				}
			}
			else
			{
				valuesum = input[pos];
				kernelsum = 1.f;
			}

			if(!outofbounds)
				output[pos] = valuesum/kernelsum;

			return;
		}
		__global__ void gaussianfilter3D_y(optflow_type *input, optflow_type *output)
		{
			//acquire constants
			/////////////////////////////////////////////
			float sigma = gpu_const::filter_sigma_c;

			int nx = gpu_const::nx_c;
			int ny = gpu_const::ny_c;
			int nz = gpu_const::nz_c;

			idx_type nslice = nx*ny;
			idx_type nstack = nz*nslice;

			bool outofbounds = false;
			idx_type pos = (blockIdx.x*blockDim.x+threadIdx.x);
			if (pos >= nstack) {outofbounds = true; pos = threadIdx.x;}

			int z = pos/nslice;
			int y = (pos-z*nslice)/nx;
			int x = pos-z*nslice-y*nx;
			/////////////////////////////////////////////

			//Create Gaussian kernel
			///////////////////////////////////////////////////
			int fsize = (int) (3*sigma);
			float kernelsum = 0.0f;
			float valuesum = 0.0f;
			//////////////////////////////////////////////////

			if(sigma > 0.0f){
				for(int yi=-fsize; yi<=fsize; yi++)
				{
					int y0 = y+yi;

					//reflective boundaries
					if (y0 < 0) y0 = -y0;
					else if (y0 >= ny) y0 = 2*ny-y0-2;

					float kernel_val = expf(-(yi*yi)/(sigma*sigma*2));
					kernelsum += kernel_val;

					__syncthreads();
					if (y0 < 0 || y0 >= ny) continue;
					valuesum += kernel_val*input[z*nslice+y0*nx+x];
				}
			}
			else
			{
				valuesum = input[pos];
				kernelsum = 1.f;
			}

			if(!outofbounds)
				output[pos] = valuesum/kernelsum;

			return;
		}
		__global__ void gaussianfilter3D_z(optflow_type *input, optflow_type *output)
		{
			//acquire constants
			/////////////////////////////////////////////
			float sigma = gpu_const::filter_sigma_c;

			int nx = gpu_const::nx_c;
			int ny = gpu_const::ny_c;
			int nz = gpu_const::nz_c;

			idx_type nslice = nx*ny;
			idx_type nstack = nz*nslice;

			bool outofbounds = false;
			idx_type pos = (blockIdx.x*blockDim.x+threadIdx.x);
			if (pos >= nstack) {outofbounds = true; pos = threadIdx.x;}

			int z = pos/nslice;
			int y = (pos-z*nslice)/nx;
			int x = pos-z*nslice-y*nx;
			/////////////////////////////////////////////

			//Create Gaussian kernel
			///////////////////////////////////////////////////
			int fsize = (int) (3*sigma);
			float kernelsum = 0.0f;
			float valuesum = 0.0f;
			//////////////////////////////////////////////////

			if (sigma > 0.0f){
				for(int zi=-fsize; zi<=fsize; zi++)
				{
					int z0 = z+zi;

					//reflective boundaries
					if (z0 < 0) z0 = -z0;
					else if (z0 >= nz) z0 = 2*nz-z0-2;

					float kernel_val = expf(-(zi*zi)/(sigma*sigma*2));
					kernelsum += kernel_val;

					__syncthreads();
					if (z0 < 0 || z0 >= nz) continue;
					valuesum += kernel_val*input[z0*nslice+y*nx+x];
				}
			}
			else{
				valuesum = input[pos];
				kernelsum = 1.f;
			}

			if(!outofbounds)
				output[pos] = valuesum/kernelsum;

			return;
		}

		__global__ void faridinterpolation3D(int dim, optflow_type *input, optflow_type *output)
		{
			//acquire constants
			/////////////////////////////////////////////
			int radius = gpu_const::farid_samples_c;

			int nx = gpu_const::nx_c;
			int ny = gpu_const::ny_c;
			int nz = gpu_const::nz_c;

			idx_type nslice = nx*ny;
			idx_type nstack = nz*nslice;

			bool outofbounds = false;
			idx_type pos = (blockIdx.x*blockDim.x+threadIdx.x);
			if (pos >= nstack) {outofbounds = true; pos = threadIdx.x;}

			int z = pos/nslice;
			int y = (pos-z*nslice)/nx;
			int x = pos-z*nslice-y*nx;
			/////////////////////////////////////////////

			optflow_type val0, val1, val2, val3, val4, val5, val6, val7, val8;
			float outvalue = 0.0f;

			//sum up the kernel
			///////////////////////////////////////////////////
			if (radius == 4)
			{
				if (dim == 0){
					int xn4 = abs(x-4); int xn3 = abs(x-3); int xn2 = abs(x-2); int xn1 = abs(x-1);
					int xp1 = x+1 < nx ? x+1 : 2*nx-(x+1)-2;
					int xp2 = x+2 < nx ? x+2 : 2*nx-(x+2)-2;
					int xp3 = x+3 < nx ? x+3 : 2*nx-(x+3)-2;
					int xp4 = x+4 < nx ? x+4 : 2*nx-(x+4)-2;

					__syncthreads();
					val0 = input[z*nslice+y*nx+xn4];
					val1 = input[z*nslice+y*nx+xn3];
					val2 = input[z*nslice+y*nx+xn2];
					val3 = input[z*nslice+y*nx+xn1];
					val4 = input[z*nslice+y*nx+x];
					val5 = input[z*nslice+y*nx+xp1];
					val6 = input[z*nslice+y*nx+xp2];
					val7 = input[z*nslice+y*nx+xp3];
					val8 = input[z*nslice+y*nx+xp4];
				}
				else if (dim == 1){
					int yn4 = abs(y-4); int yn3 = abs(y-3); int yn2 = abs(y-2); int yn1 = abs(y-1);
					int yp1 = y+1 < ny ? y+1 : 2*ny-(y+1)-2;
					int yp2 = y+2 < ny ? y+2 : 2*ny-(y+2)-2;
					int yp3 = y+3 < ny ? y+3 : 2*ny-(y+3)-2;
					int yp4 = y+4 < ny ? y+4 : 2*ny-(y+4)-2;

					__syncthreads();
					val0 = input[z*nslice+yn4*nx+x];
					val1 = input[z*nslice+yn3*nx+x];
					val2 = input[z*nslice+yn2*nx+x];
					val3 = input[z*nslice+yn1*nx+x];
					val4 = input[z*nslice+y*nx+x];
					val5 = input[z*nslice+yp1*nx+x];
					val6 = input[z*nslice+yp2*nx+x];
					val7 = input[z*nslice+yp3*nx+x];
					val8 = input[z*nslice+yp4*nx+x];
				}
				else{
					int zn4 = abs(z-4); int zn3 = abs(z-3); int zn2 = abs(z-2); int zn1 = abs(z-1);
					int zp1 = z+1 < nz ? z+1 : 2*nz-(z+1)-2;
					int zp2 = z+2 < nz ? z+2 : 2*nz-(z+2)-2;
					int zp3 = z+3 < nz ? z+3 : 2*nz-(z+3)-2;
					int zp4 = z+4 < nz ? z+4 : 2*nz-(z+4)-2;

					__syncthreads();
					val0 = input[zn4*nslice+y*nx+x];
					val1 = input[zn3*nslice+y*nx+x];
					val2 = input[zn2*nslice+y*nx+x];
					val3 = input[zn1*nslice+y*nx+x];
					val4 = input[z*nslice+y*nx+x];
					val5 = input[zp1*nslice+y*nx+x];
					val6 = input[zp2*nslice+y*nx+x];
					val7 = input[zp3*nslice+y*nx+x];
					val8 = input[zp4*nslice+y*nx+x];
				}

				outvalue = ((0.000721*(val0+val8) + 0.015486*(val1+val7)) + 0.090341*(val2+val6)) + 0.234494*(val3+val5) + 0.317916*val4;
			}
			else if (radius == 3)
			{
				if (dim == 0){
					int xn3 = abs(x-3); int xn2 = abs(x-2); int xn1 = abs(x-1);
					int xp1 = x+1 < nx ? x+1 : 2*nx-(x+1)-2;
					int xp2 = x+2 < nx ? x+2 : 2*nx-(x+2)-2;
					int xp3 = x+3 < nx ? x+3 : 2*nx-(x+3)-2;

					__syncthreads();
					val0 = input[z*nslice+y*nx+xn3];
					val1 = input[z*nslice+y*nx+xn2];
					val2 = input[z*nslice+y*nx+xn1];
					val3 = input[z*nslice+y*nx+x];
					val4 = input[z*nslice+y*nx+xp1];
					val5 = input[z*nslice+y*nx+xp2];
					val6 = input[z*nslice+y*nx+xp3];
				}
				else if (dim == 1){
					int yn3 = abs(y-3); int yn2 = abs(y-2); int yn1 = abs(y-1);
					int yp1 = y+1 < ny ? y+1 : 2*ny-(y+1)-2;
					int yp2 = y+2 < ny ? y+2 : 2*ny-(y+2)-2;
					int yp3 = y+3 < ny ? y+3 : 2*ny-(y+3)-2;

					__syncthreads();
					val0 = input[z*nslice+yn3*nx+x];
					val1 = input[z*nslice+yn2*nx+x];
					val2 = input[z*nslice+yn1*nx+x];
					val3 = input[z*nslice+y*nx+x];
					val4 = input[z*nslice+yp1*nx+x];
					val5 = input[z*nslice+yp2*nx+x];
					val6 = input[z*nslice+yp3*nx+x];
				}
				else{
					int zn3 = abs(z-3); int zn2 = abs(z-2); int zn1 = abs(z-1);
					int zp1 = z+1 < nz ? z+1 : 2*nz-(z+1)-2;
					int zp2 = z+2 < nz ? z+2 : 2*nz-(z+2)-2;
					int zp3 = z+3 < nz ? z+3 : 2*nz-(z+3)-2;

					__syncthreads();
					val0 = input[zn3*nslice+y*nx+x];
					val1 = input[zn2*nslice+y*nx+x];
					val2 = input[zn1*nslice+y*nx+x];
					val3 = input[z*nslice+y*nx+x];
					val4 = input[zp1*nslice+y*nx+x];
					val5 = input[zp2*nslice+y*nx+x];
					val6 = input[zp3*nslice+y*nx+x];
				}

				outvalue = (0.004711*(val0+val6)+0.069321*(val1+val5))+ 0.245410*(val2+val4) + 0.361117*val3;
			}
			else if (radius == 2)
			{
				if (dim == 0){
					int xn2 = abs(x-2); int xn1 = abs(x-1);
					int xp1 = x+1 < nx ? x+1 : 2*nx-(x+1)-2;
					int xp2 = x+2 < nx ? x+2 : 2*nx-(x+2)-2;

					__syncthreads();
					val0 = input[z*nslice+y*nx+xn2];
					val1 = input[z*nslice+y*nx+xn1];
					val2 = input[z*nslice+y*nx+x];
					val3 = input[z*nslice+y*nx+xp1];
					val4 = input[z*nslice+y*nx+xp2];
				}
				else if (dim == 1){
					int yn2 = abs(y-2); int yn1 = abs(y-1);
					int yp1 = y+1 < ny ? y+1 : 2*ny-(y+1)-2;
					int yp2 = y+2 < ny ? y+2 : 2*ny-(y+2)-2;

					__syncthreads();
					val0 = input[z*nslice+yn2*nx+x];
					val1 = input[z*nslice+yn1*nx+x];
					val2 = input[z*nslice+y*nx+x];
					val3 = input[z*nslice+yp1*nx+x];
					val4 = input[z*nslice+yp2*nx+x];
				}
				else{
					int zn2 = abs(z-2); int zn1 = abs(z-1);
					int zp1 = z+1 < nz ? z+1 : 2*nz-(z+1)-2;
					int zp2 = z+2 < nz ? z+2 : 2*nz-(z+2)-2;

					__syncthreads();
					val0 = input[zn2*nslice+y*nx+x];
					val1 = input[zn1*nslice+y*nx+x];
					val2 = input[z*nslice+y*nx+x];
					val3 = input[zp1*nslice+y*nx+x];
					val4 = input[zp2*nslice+y*nx+x];
				}
				outvalue = ((0.037659*(val0+val4) + 0.249153*(val1+val3)) + 0.426375*val2)/0.999999;
			}
			else if (radius == 0) outvalue = input[pos];
			else
			{
				if (dim == 0){
					int xn1 = abs(x-1);
					int xp1 = x+1 < nx ? x+1 : 2*nx-(x+1)-2;

					__syncthreads();
					val0 = input[z*nslice+y*nx+xn1];
					val1 = input[z*nslice+y*nx+x];
					val2 = input[z*nslice+y*nx+xp1];
				}
				else if (dim == 1){
					int yn1 = abs(y-1);
					int yp1 = y+1 < ny ? y+1 : 2*ny-(y+1)-2;

					__syncthreads();
					val0 = input[z*nslice+yn1*nx+x];
					val1 = input[z*nslice+y*nx+x];
					val2 = input[z*nslice+yp1*nx+x];
				}
				else{
					int zn1 = abs(z-1);
					int zp1 = z+1 < nz ? z+1 : 2*nz-(z+1)-2;

					__syncthreads();
					val0 = input[zn1*nslice+y*nx+x];
					val1 = input[z*nslice+y*nx+x];
					val2 = input[zp1*nslice+y*nx+x];
				}
				outvalue = 0.229879*(val0+val2) + 0.540242*val1;
			}

			if(!outofbounds)
				output[pos] = outvalue;

			return;
		}
		__global__ void faridinterpolation3D(int dim, optflow_type *input, optflow_type *output, int radius)
		{
			//acquire constants
			/////////////////////////////////////////////
			int nx = gpu_const::nx_c;
			int ny = gpu_const::ny_c;
			int nz = gpu_const::nz_c;

			idx_type nslice = nx*ny;
			idx_type nstack = nz*nslice;

			bool outofbounds = false;
			idx_type pos = (blockIdx.x*blockDim.x+threadIdx.x);
			if (pos >= nstack) {outofbounds = true; pos = threadIdx.x;}

			int z = pos/nslice;
			int y = (pos-z*nslice)/nx;
			int x = pos-z*nslice-y*nx;
			/////////////////////////////////////////////

			optflow_type val0, val1, val2, val3, val4, val5, val6, val7, val8;
			float outvalue = 0.0f;

			//sum up the kernel
			///////////////////////////////////////////////////
			if (radius == 4)
			{
				if (dim == 0){
					int xn4 = abs(x-4); int xn3 = abs(x-3); int xn2 = abs(x-2); int xn1 = abs(x-1);
					int xp1 = x+1 < nx ? x+1 : 2*nx-(x+1)-2;
					int xp2 = x+2 < nx ? x+2 : 2*nx-(x+2)-2;
					int xp3 = x+3 < nx ? x+3 : 2*nx-(x+3)-2;
					int xp4 = x+4 < nx ? x+4 : 2*nx-(x+4)-2;

					__syncthreads();
					val0 = input[z*nslice+y*nx+xn4];
					val1 = input[z*nslice+y*nx+xn3];
					val2 = input[z*nslice+y*nx+xn2];
					val3 = input[z*nslice+y*nx+xn1];
					val4 = input[z*nslice+y*nx+x];
					val5 = input[z*nslice+y*nx+xp1];
					val6 = input[z*nslice+y*nx+xp2];
					val7 = input[z*nslice+y*nx+xp3];
					val8 = input[z*nslice+y*nx+xp4];
				}
				else if (dim == 1){
					int yn4 = abs(y-4); int yn3 = abs(y-3); int yn2 = abs(y-2); int yn1 = abs(y-1);
					int yp1 = y+1 < ny ? y+1 : 2*ny-(y+1)-2;
					int yp2 = y+2 < ny ? y+2 : 2*ny-(y+2)-2;
					int yp3 = y+3 < ny ? y+3 : 2*ny-(y+3)-2;
					int yp4 = y+4 < ny ? y+4 : 2*ny-(y+4)-2;

					__syncthreads();
					val0 = input[z*nslice+yn4*nx+x];
					val1 = input[z*nslice+yn3*nx+x];
					val2 = input[z*nslice+yn2*nx+x];
					val3 = input[z*nslice+yn1*nx+x];
					val4 = input[z*nslice+y*nx+x];
					val5 = input[z*nslice+yp1*nx+x];
					val6 = input[z*nslice+yp2*nx+x];
					val7 = input[z*nslice+yp3*nx+x];
					val8 = input[z*nslice+yp4*nx+x];
				}
				else{
					int zn4 = abs(z-4); int zn3 = abs(z-3); int zn2 = abs(z-2); int zn1 = abs(z-1);
					int zp1 = z+1 < nz ? z+1 : 2*nz-(z+1)-2;
					int zp2 = z+2 < nz ? z+2 : 2*nz-(z+2)-2;
					int zp3 = z+3 < nz ? z+3 : 2*nz-(z+3)-2;
					int zp4 = z+4 < nz ? z+4 : 2*nz-(z+4)-2;

					__syncthreads();
					val0 = input[zn4*nslice+y*nx+x];
					val1 = input[zn3*nslice+y*nx+x];
					val2 = input[zn2*nslice+y*nx+x];
					val3 = input[zn1*nslice+y*nx+x];
					val4 = input[z*nslice+y*nx+x];
					val5 = input[zp1*nslice+y*nx+x];
					val6 = input[zp2*nslice+y*nx+x];
					val7 = input[zp3*nslice+y*nx+x];
					val8 = input[zp4*nslice+y*nx+x];
				}

				outvalue = ((0.000721*(val0+val8) + 0.015486*(val1+val7)) + 0.090341*(val2+val6)) + 0.234494*(val3+val5) + 0.317916*val4;
			}
			else if (radius == 3)
			{
				if (dim == 0){
					int xn3 = abs(x-3); int xn2 = abs(x-2); int xn1 = abs(x-1);
					int xp1 = x+1 < nx ? x+1 : 2*nx-(x+1)-2;
					int xp2 = x+2 < nx ? x+2 : 2*nx-(x+2)-2;
					int xp3 = x+3 < nx ? x+3 : 2*nx-(x+3)-2;

					__syncthreads();
					val0 = input[z*nslice+y*nx+xn3];
					val1 = input[z*nslice+y*nx+xn2];
					val2 = input[z*nslice+y*nx+xn1];
					val3 = input[z*nslice+y*nx+x];
					val4 = input[z*nslice+y*nx+xp1];
					val5 = input[z*nslice+y*nx+xp2];
					val6 = input[z*nslice+y*nx+xp3];
				}
				else if (dim == 1){
					int yn3 = abs(y-3); int yn2 = abs(y-2); int yn1 = abs(y-1);
					int yp1 = y+1 < ny ? y+1 : 2*ny-(y+1)-2;
					int yp2 = y+2 < ny ? y+2 : 2*ny-(y+2)-2;
					int yp3 = y+3 < ny ? y+3 : 2*ny-(y+3)-2;

					__syncthreads();
					val0 = input[z*nslice+yn3*nx+x];
					val1 = input[z*nslice+yn2*nx+x];
					val2 = input[z*nslice+yn1*nx+x];
					val3 = input[z*nslice+y*nx+x];
					val4 = input[z*nslice+yp1*nx+x];
					val5 = input[z*nslice+yp2*nx+x];
					val6 = input[z*nslice+yp3*nx+x];
				}
				else{
					int zn3 = abs(z-3); int zn2 = abs(z-2); int zn1 = abs(z-1);
					int zp1 = z+1 < nz ? z+1 : 2*nz-(z+1)-2;
					int zp2 = z+2 < nz ? z+2 : 2*nz-(z+2)-2;
					int zp3 = z+3 < nz ? z+3 : 2*nz-(z+3)-2;

					__syncthreads();
					val0 = input[zn3*nslice+y*nx+x];
					val1 = input[zn2*nslice+y*nx+x];
					val2 = input[zn1*nslice+y*nx+x];
					val3 = input[z*nslice+y*nx+x];
					val4 = input[zp1*nslice+y*nx+x];
					val5 = input[zp2*nslice+y*nx+x];
					val6 = input[zp3*nslice+y*nx+x];
				}

				outvalue = (0.004711*(val0+val6)+0.069321*(val1+val5))+ 0.245410*(val2+val4) + 0.361117*val3;
			}
			else if (radius == 2)
			{
				if (dim == 0){
					int xn2 = abs(x-2); int xn1 = abs(x-1);
					int xp1 = x+1 < nx ? x+1 : 2*nx-(x+1)-2;
					int xp2 = x+2 < nx ? x+2 : 2*nx-(x+2)-2;

					__syncthreads();
					val0 = input[z*nslice+y*nx+xn2];
					val1 = input[z*nslice+y*nx+xn1];
					val2 = input[z*nslice+y*nx+x];
					val3 = input[z*nslice+y*nx+xp1];
					val4 = input[z*nslice+y*nx+xp2];
				}
				else if (dim == 1){
					int yn2 = abs(y-2); int yn1 = abs(y-1);
					int yp1 = y+1 < ny ? y+1 : 2*ny-(y+1)-2;
					int yp2 = y+2 < ny ? y+2 : 2*ny-(y+2)-2;

					__syncthreads();
					val0 = input[z*nslice+yn2*nx+x];
					val1 = input[z*nslice+yn1*nx+x];
					val2 = input[z*nslice+y*nx+x];
					val3 = input[z*nslice+yp1*nx+x];
					val4 = input[z*nslice+yp2*nx+x];
				}
				else{
					int zn2 = abs(z-2); int zn1 = abs(z-1);
					int zp1 = z+1 < nz ? z+1 : 2*nz-(z+1)-2;
					int zp2 = z+2 < nz ? z+2 : 2*nz-(z+2)-2;

					__syncthreads();
					val0 = input[zn2*nslice+y*nx+x];
					val1 = input[zn1*nslice+y*nx+x];
					val2 = input[z*nslice+y*nx+x];
					val3 = input[zp1*nslice+y*nx+x];
					val4 = input[zp2*nslice+y*nx+x];
				}
				outvalue = ((0.037659*(val0+val4) + 0.249153*(val1+val3)) + 0.426375*val2)/0.999999;
			}
			else if (radius == 0) outvalue = input[pos];
			else
			{
				if (dim == 0){
					int xn1 = abs(x-1);
					int xp1 = x+1 < nx ? x+1 : 2*nx-(x+1)-2;

					__syncthreads();
					val0 = input[z*nslice+y*nx+xn1];
					val1 = input[z*nslice+y*nx+x];
					val2 = input[z*nslice+y*nx+xp1];
				}
				else if (dim == 1){
					int yn1 = abs(y-1);
					int yp1 = y+1 < ny ? y+1 : 2*ny-(y+1)-2;

					__syncthreads();
					val0 = input[z*nslice+yn1*nx+x];
					val1 = input[z*nslice+y*nx+x];
					val2 = input[z*nslice+yp1*nx+x];
				}
				else{
					int zn1 = abs(z-1);
					int zp1 = z+1 < nz ? z+1 : 2*nz-(z+1)-2;

					__syncthreads();
					val0 = input[zn1*nslice+y*nx+x];
					val1 = input[z*nslice+y*nx+x];
					val2 = input[zp1*nslice+y*nx+x];
				}
				outvalue = 0.229879*(val0+val2) + 0.540242*val1;
			}

			if(!outofbounds)
				output[pos] = outvalue;

			return;
		}

		__global__ void calculate_sorUpdate(int iter, img_type *frame0, img_type *warped1, optflow_type *phi, optflow_type *psi, optflow_type *u, optflow_type *du, optflow_type *confidencemap)
		{
			//acquire constants
			/////////////////////////////////////////////
			int nx = gpu_const::nx_c;
			int ny = gpu_const::ny_c;
			int nz = gpu_const::nz_c;

			mathtype_solver epsilon_psi_squared = gpu_const::epsilon_psi_squared_c;
			mathtype_solver hx = gpu_const::hx_c;
			mathtype_solver hy = gpu_const::hy_c;
			mathtype_solver hz = gpu_const::hz_c;
			mathtype_solver alpha = gpu_const::alpha_c;
			mathtype_solver omega = gpu_const::omega_c;
			mathtype_solver max_step = gpu_const::max_sorupdate;

			bool precalculated_psi = gpu_const::precalculated_psi_c;
			bool decoupled_smoothness = gpu_const::decoupled_smoothness_c;
			int slip_depth = gpu_const::slip_depth_c;

			float minIntensity = gpu_const::lowerIntensityCutoff_c;
			float maxIntensity = gpu_const::upperIntensityCutoff_c;

			mathtype_solver alphax = alpha/(hx*hx);
			mathtype_solver alphay = alpha/(hy*hy);
			mathtype_solver alphaz = alpha/(hz*hz);

			int spatiotemporalderivative_id = gpu_const::spatiotemporalderivative_id_c;
			bool use_confidencemap = gpu_const::use_confidencemap_c;

			idx_type nslice = nx*ny;
			idx_type nstack = nz*nslice;
			idx_type nstack2 = 2*nstack;

			//Adjust for even/odd updates in 2D
			///////////////////////////////////
			bool outofbounds = false;
			idx_type pos = (blockIdx.x*blockDim.x+threadIdx.x);
			pos *= 2;

			int z = pos/nslice;
			int y = (pos-z*nslice)/nx;
			int x = pos-z*nslice-y*nx;

			//checkerboard switching
			if((iter%2) == 0)
			{
					 if ((nx%2) == 0 && (y%2) != 0 && (z%2) == 0) {x++; pos++;}
				else if ((nx%2) == 0 && (y%2) == 0 && (z%2) != 0) {x++; pos++;}
				else if ((nx%2) != 0 && (nslice%2) == 0 && (z%2) != 0) {x++; pos++;}
			}
			else
			{
				if((nx%2) != 0 && (z%2) == 0) {x++; pos++;}
				else if((nslice%2) != 0 && (z%2) != 0) {x++; pos++;}
				else if((nx%2) == 0 && (y%2) == 0 && (z%2) == 0){x++; pos++;}
				else if((nx%2) == 0 && (y%2) != 0 && (z%2) != 0){x++; pos++;}
			}

			if (pos < nstack && x >= nx) {x = 0; y++; pos = z*nslice+y*nx+x;}

			if (pos >= nstack)
			{
				outofbounds = true; pos = threadIdx.x;
				z = pos/nslice;
				y = (pos-z*nslice)/nx;
				x = pos-z*nslice-y*nx;
			}

			///////////////////////////////////

			mathtype_solver confidence = 1.0f;
			mathtype_solver psi0 = 0.0f;
			mathtype_solver normalizer_x1 = 0.125f/hx;
			mathtype_solver normalizer_y1 = 0.125f/hy;
			mathtype_solver normalizer_z1 = 0.125f/hz;
			mathtype_solver normalizer_x2 = 0.25f/hx;
			mathtype_solver normalizer_y2 = 0.25f/hy;
			mathtype_solver normalizer_z2 = 0.25f/hz;
			/////////////////////////////////////////////

			//Define the neighbourhood
			/////////////////////////////////////////////
			int zp = z+1;
			int zn = z-1;
			int yp = y+1;
			int yn = y-1;
			int xp = x+1;
			int xn = x-1;

			mathtype_solver xp_active = 1.0f;
			mathtype_solver xn_active = 1.0f;
			mathtype_solver yp_active = 1.0f;
			mathtype_solver yn_active = 1.0f;
			mathtype_solver zp_active = 1.0f;
			mathtype_solver zn_active = 1.0f;

			bool boundary_voxel = false;
			if (xp == nx) {xp_active = 0.0f; xp = x; boundary_voxel = true;}
			else if (xn < 0) {xn_active = 0.0f; xn = x; boundary_voxel = true;}
			if (yp == ny) {yp_active = 0.0f; yp = y; boundary_voxel = true;}
			else if (yn < 0) {yn_active = 0.0f; yn = y; boundary_voxel = true;}
			if (zp == nz) {zp_active = 0.0f; zp = z; boundary_voxel = true;}
			else if (zn < 0) {zn_active = 0.0f; zn = z; boundary_voxel = true;}

			idx_type npos0 = z*nslice + y*nx + xp;
			idx_type npos1 = z*nslice + y*nx + xn;
			idx_type npos2 = z*nslice + yp*nx + x;
			idx_type npos3 = z*nslice + yn*nx + x;
			idx_type npos4 = zp*nslice + y*nx + x;
			idx_type npos5 = zn*nslice + y*nx + x;

			mathtype_solver phi_neighbour[18] = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
			mathtype_solver du_neighbour[6]  = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
			mathtype_solver dv_neighbour[6]  = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
			mathtype_solver dw_neighbour[6]  = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
			/////////////////////////////////////////////

			//Switch to reflective boundary conditions
			/////////////////////////////////////////////
			zp = z+1;
			zn = z-1;
			yp = y+1;
			yn = y-1;
			xp = x+1;
			xn = x-1;

			if (zp == nz) zp -= 2;
			else if (zn == -1) zn = 1;
			if (yp == ny) yp -= 2;
			else if (yn == -1) yn = 1;
			if (xp == nx) xp -= 2;
			else if (xn == -1) xn = 1;
			/////////////////////////////////////////////

			/////////////////////////////////////////////
			__syncthreads();
			mathtype_solver phi0 = phi[pos];
			mathtype_solver u0 = u[pos];
			mathtype_solver v0 = u[pos+nstack];
			mathtype_solver w0 = u[pos+nstack2];
			mathtype_solver du0 = du[pos];
			mathtype_solver dv0 = du[pos+nstack];
			mathtype_solver dw0 = du[pos+nstack2];
			mathtype_solver frame0_val = frame0[pos];

			if (use_confidencemap) confidence = confidencemap[pos];
			if (precalculated_psi) psi0 = psi[pos]; //since we are now calculating the accurate tensor, this could be eliminated leaving just the 4 components
			/////////////////////////////////////////////

			//Read in neighbours with 0-boundaries
			/////////////////////////////////////////////
			phi_neighbour[0] = xp_active*0.5f*(phi[npos0] + phi0);
			du_neighbour[0]  = u[npos0] + du[npos0] - u0;
			dv_neighbour[0]  = u[npos0 + nstack] + du[npos0 + nstack] - v0;
			dw_neighbour[0]  = u[npos0 + nstack2] + du[npos0 + nstack2] - w0;

			phi_neighbour[1] = xn_active*0.5f*(phi[npos1] + phi0);
			du_neighbour[1]  = u[npos1] + du[npos1] - u0;
			dv_neighbour[1]  = u[npos1 + nstack] + du[npos1 + nstack] - v0;
			dw_neighbour[1]  = u[npos1 + nstack2] + du[npos1 + nstack2] - w0;

			phi_neighbour[2] = yp_active*0.5f*(phi[npos2] + phi0);
			du_neighbour[2]  = u[npos2] + du[npos2] - u0;
			dv_neighbour[2]  = u[npos2 + nstack] + du[npos2 + nstack] - v0;
			dw_neighbour[2]  = u[npos2 + nstack2] + du[npos2 + nstack2] - w0;

			phi_neighbour[3] = yn_active*0.5f*(phi[npos3] + phi0);
			du_neighbour[3]  = u[npos3] + du[npos3] - u0;
			dv_neighbour[3]  = u[npos3 + nstack] + du[npos3 + nstack] - v0;
			dw_neighbour[3]  = u[npos3 + nstack2] + du[npos3 + nstack2] - w0;

			phi_neighbour[4] = zp_active*0.5f*(phi[npos4] + phi0);
			du_neighbour[4]  = u[npos4] + du[npos4] - u0;
			dv_neighbour[4]  = u[npos4 + nstack] + du[npos4 + nstack] - v0;
			dw_neighbour[4]  = u[npos4 + nstack2] + du[npos4 + nstack2] - w0;

			phi_neighbour[5] = zn_active*0.5f*(phi[npos5] + phi0);
			du_neighbour[5]  = u[npos5] + du[npos5] - u0;
			dv_neighbour[5]  = u[npos5 + nstack] + du[npos5 + nstack] - v0;
			dw_neighbour[5]  = u[npos5 + nstack2] + du[npos5 + nstack2] - w0;

			if(decoupled_smoothness)
			{
				mathtype_solver phi1 = phi[nstack+pos];
				mathtype_solver phi2 = phi[nstack2+pos];
				phi_neighbour[6] = xp_active*0.5f*(phi[nstack+npos0] + phi1);
				phi_neighbour[7] = xn_active*0.5f*(phi[nstack+npos1] + phi1);
				phi_neighbour[8] = yp_active*0.5f*(phi[nstack+npos2] + phi1);
				phi_neighbour[9] = yn_active*0.5f*(phi[nstack+npos3] + phi1);
				phi_neighbour[10]= zp_active*0.5f*(phi[nstack+npos4] + phi1);
				phi_neighbour[11]= zn_active*0.5f*(phi[nstack+npos5] + phi1);
				phi_neighbour[12] = xp_active*0.5f*(phi[nstack2+npos0] + phi2);
				phi_neighbour[13] = xn_active*0.5f*(phi[nstack2+npos1] + phi2);
				phi_neighbour[14] = yp_active*0.5f*(phi[nstack2+npos2] + phi2);
				phi_neighbour[15] = yn_active*0.5f*(phi[nstack2+npos3] + phi2);
				phi_neighbour[16]= zp_active*0.5f*(phi[nstack2+npos4] + phi2);
				phi_neighbour[17]= zn_active*0.5f*(phi[nstack2+npos5] + phi2);
			}
			/////////////////////////////////////////////

			mathtype_solver Idx, Idy, Idz, Idt;

			//Calculate spatiotemporal derivatives on the fly
			/////////////////////////////////////////////
			if (spatiotemporalderivative_id < 0){
				Idx = psi[pos+nstack];
				Idy = psi[pos+(2*nstack)];
				Idz = psi[pos+(3*nstack)];
				Idt = psi[pos+(4*nstack)];
			}
			else if (spatiotemporalderivative_id == 3){
				//Fourth Order Finite Difference
				//////////////////////////////////////////////////////////////////////////////////////////

				int yp2 = y+2; int yn2 = y-2; int xp2 = x+2; int xn2 = x-2; int zp2 = z+2; int zn2 = z-2;
				if (yp2 >= ny) yp2 = 2*ny-yp2-2; if (yn2 < 0) yn2 = -yn2;
				if (xp2 >= nx) xp2 = 2*nx-xp2-2; if (xn2 < 0) xn2 = -xn2;
				if (zp2 >= nz) zp2 = 2*nz-zp2-2; if (zn2 < 0) zn2 = -zn2;

				__syncthreads();
				mathtype_solver val_xn2_a = frame0[z*nslice+y*nx + xn2];
				mathtype_solver val_xn_a = frame0[z*nslice+y*nx + xn];
				//mathtype_solver val0a    = frame0[pos];
				mathtype_solver val_xp_a = frame0[z*nslice+y*nx + xp];
				mathtype_solver val_xp2_a = frame0[z*nslice+y*nx + xp2];
				mathtype_solver val_yn2_a = frame0[z*nslice+yn2*nx + x];
				mathtype_solver val_yn_a = frame0[z*nslice+yn*nx + x];
				mathtype_solver val_yp_a = frame0[z*nslice+yp*nx + x];
				mathtype_solver val_yp2_a = frame0[z*nslice+yp2*nx + x];
				mathtype_solver val_zn2_a = frame0[zn2*nslice+y*nx + x];
				mathtype_solver val_zn_a = frame0[zn*nslice+y*nx + x];
				mathtype_solver val_zp_a = frame0[zp*nslice+y*nx + x];
				mathtype_solver val_zp2_a = frame0[zp2*nslice+y*nx + x];

				mathtype_solver val_xn2_b = warped1[z*nslice+y*nx + xn2];
				mathtype_solver val_xn_b = warped1[z*nslice+y*nx + xn];
				mathtype_solver val0b    = warped1[pos];
				mathtype_solver val_xp_b = warped1[z*nslice+y*nx + xp];
				mathtype_solver val_xp2_b = warped1[z*nslice+y*nx + xp2];
				mathtype_solver val_yn2_b = warped1[z*nslice+yn2*nx + x];
				mathtype_solver val_yn_b = warped1[z*nslice+yn*nx + x];
				mathtype_solver val_yp_b = warped1[z*nslice+yp*nx + x];
				mathtype_solver val_yp2_b = warped1[z*nslice+yp2*nx + x];
				mathtype_solver val_zn2_b = warped1[zn2*nslice+y*nx + x];
				mathtype_solver val_zn_b = warped1[zn*nslice+y*nx + x];
				mathtype_solver val_zp_b = warped1[zp*nslice+y*nx + x];
				mathtype_solver val_zp2_b = warped1[zp2*nslice+y*nx + x];

				Idx = normalizer_x1/6.f*((val_xn2_a-8.f*val_xn_a+8.f*val_xp_a-val_xp2_a)+(val_xn2_b-8.f*val_xn_b+8.f*val_xp_b-val_xp2_b));
				Idy = normalizer_y1/6.f*((val_yn2_a-8.f*val_yn_a+8.f*val_yp_a-val_yp2_a)+(val_yn2_b-8.f*val_yn_b+8.f*val_yp_b-val_yp2_b));
				Idz = normalizer_z1/6.f*((val_zn2_a-8.f*val_zn_a+8.f*val_zp_a-val_zp2_a)+(val_zn2_b-8.f*val_zn_b+8.f*val_zp_b-val_zp2_b));
				Idt = val0b-frame0_val;
				//////////////////////////////////////////////////////////////////////////////////////////
			}
			else if (spatiotemporalderivative_id == 4){
				Idx = 0.0f; Idy = 0.0f; Idz = 0.0f;
				//Farid 3: average of frame1 and frame2 using Farid kernel, central difference, dt: forward difference
				//////////////////////////////////////////////////////////////////////////////////////////
				float farid[4] = {0.0f, 0.12412487720f, 0.05281651765f, 0.02247401886f};
				for(int r = -1; r <= 1; r++){
					int z2 = z+r;
					if (z2 == nz) z2 -= 2; if (z2 < 0) z2 = 1;
					int n_edges_r = abs(r);
					for (int q = -1; q <= 1; q++){
						int y2 = y+q;
						if (y2 == ny) y2 -= 2; if (y2 < 0) y2 = 1;
						int n_edges_q = n_edges_r + abs(q);
						for(int p = -1; p <= 1; p++){
							int x2 = x+p;
							if (x2 == nx) x2 -= 2; if (x2 < 0) x2 = 1;
							int n_edges = n_edges_q + abs(p);
							if (n_edges == 0) continue;

							__syncthreads();
							mathtype_solver val_a =  frame0[z2*nslice+y2*nx+x2];
							mathtype_solver val_b = warped1[z2*nslice+y2*nx+x2];

							float this_weight = 2.f*farid[n_edges]; //times 2 to compensate the wrong normalizer

							Idx += (normalizer_x2*p*this_weight)*(val_a+val_b);
							Idy += (normalizer_y2*q*this_weight)*(val_a+val_b);
							Idz += (normalizer_z2*r*this_weight)*(val_a+val_b);
						}
					}
				}
				__syncthreads();
				mathtype_solver val0b    = warped1[pos];
				Idt = val0b-frame0_val;
				//////////////////////////////////////////////////////////////////////////////////////////
			}
			else if (spatiotemporalderivative_id == 5){
				Idx = 0.0f; Idy = 0.0f; Idz = 0.0f;
				//Farid 5: average of frame1 and frame2 using Farid kernel, central difference, dt: forward difference
				//////////////////////////////////////////////////////////////////////////////////////////
				float farid[27] = {0.0f, 0.0503013134f,  0.01992556825f,0.0f, 0.02939366736f, 0.01164354291f,
						0.0f, 0.004442796111f,0.001759899082f,0.0f, 0.02939366549f,  0.01164354291f,0.0f, 0.01717624255f,  0.006803925615f,
						0.0f, 0.002596156206f, 0.001028400264f,
						0.0f, 0.004442796111f, 0.001759899198f,0.0f, 0.002596156206f, 0.001028400264f,0.0f, 0.0003924040357f, 0.0001554407354f
				};
				int this_radius = 2;
				for(int r = -this_radius; r <= this_radius; r++){
					int z2 = z+r;
					if (z2 >= nz) z2 = 2*nz-z2-2; if (z2 < 0) z2 = -z2;
					int absr = abs(r);
					mathtype_solver sign_z = r < 0 ? -normalizer_z2 : normalizer_z2;
					for (int q = -this_radius; q <= this_radius; q++){
						int y2 = y+q;
						if (y2 >= ny) y2 = 2*ny-y2-2; if (y2 < 0) y2 = -y2;
						int absq = abs(q);
						mathtype_solver sign_y = q < 0 ? -normalizer_y2 : normalizer_y2;
						for(int p = -this_radius; p <= this_radius; p++){
							int x2 = x+p;
							if (x2 >= nx) x2 = 2*nx-x2-2; if (x2 < 0) x2 = -x2;
							if (p == 0 && q == 0 && r == 0) continue;
							int absp = abs(p);
							mathtype_solver sign_x = p < 0 ? -normalizer_x2 : normalizer_x2;
							__syncthreads();
							mathtype_solver val_a =  frame0[z2*nslice+y2*nx+x2];
							mathtype_solver val_b = warped1[z2*nslice+y2*nx+x2];

							int idx_x = absr*9 + absq*3 + absp;
							int idx_y = absr*9 + absp*3 + absq;
							int idx_z = absq*9 + absp*3 + absr;

							float this_weight_x = 2.f*sign_x*farid[idx_x];//times 2 to compensate the wrong normalizer
							float this_weight_y = 2.f*sign_y*farid[idx_y];
							float this_weight_z = 2.f*sign_z*farid[idx_z];

							Idx += this_weight_x*(val_a+val_b);
							Idy += this_weight_y*(val_a+val_b);
							Idz += this_weight_z*(val_a+val_b);
						}
					}
				}
				__syncthreads();
				mathtype_solver val0b    = warped1[pos];
				Idt = val0b-frame0_val;
				//////////////////////////////////////////////////////////////////////////////////////////
			}
			else if (spatiotemporalderivative_id == 6){
				Idx = 0.0f; Idy = 0.0f; Idz = 0.0f;
				//Farid 7: average of frame1 and frame2 using Farid kernel, central difference, dt: forward difference
				//////////////////////////////////////////////////////////////////////////////////////////
				int this_radius = 3;
				float farid[64] = {0.0f, 0.02518012933f, 0.01634971797f, 0.002439625794f,0.0f, 0.01711205766f, 0.01111103687f, 0.001657935209f,0.0f, 0.004833645653f,0.003138536355f,0.0004683172156f,
					0.0f, 0.0003284906852f,0.0002132924128f,3.182646469e-05f,0.0f,0.01711205766f,0.01111103687f,0.001657935092f,0.0f,0.01162911206f,0.00755090313f,0.001126709278f,
					0.0f,0.003284876933f,0.002132904949f,0.0003182617365f,0.0f,0.0002232376137f,0.0001449505071f,2.162881356e-05f,0.0f,0.004833645653f,0.003138536355f,0.0004683171865f,
					0.0f,0.003284876933f,0.002132904716f,0.0003182617365f,0.0f,0.0009278797079f,0.000602481945f,8.989944035e-05f,0.0f,6.305796705e-05f,4.094419273e-05f,6.109494279e-06f,
					0.0f,0.0003284907143f,0.0002132924274f,3.182646105e-05f,0.0f,0.0002232376137f,0.0001449505071f,2.162881356e-05f,0.0f,6.305796705e-05f,4.094419273e-05f,6.109494279e-06f,
					0.0f,4.285368959e-06f,2.782534466e-06f,4.151963537e-07f,};
				for(int r = -this_radius; r <= this_radius; r++){
					int z2 = z+r;
					if (z2 >= nz) z2 = 2*nz-z2-2; if (z2 < 0) z2 = -z2;
					int absr = abs(r);
					mathtype_solver sign_z = r < 0 ? -normalizer_z2 : normalizer_z2;
					for (int q = -this_radius; q <= this_radius; q++){
						int y2 = y+q;
						if (y2 >= ny) y2 = 2*ny-y2-2; if (y2 < 0) y2 = -y2;
						int absq = abs(q);
						mathtype_solver sign_y = q < 0 ? -normalizer_y2 : normalizer_y2;
						for(int p = -this_radius; p <= this_radius; p++){
							int x2 = x+p;
							if (x2 >= nx) x2 = 2*nx-x2-2; if (x2 < 0) x2 = -x2;
							if (p == 0 && q == 0 && r == 0) continue;
							int absp = abs(p);
							mathtype_solver sign_x = p < 0 ? -normalizer_x2 : normalizer_x2;
							__syncthreads();
							mathtype_solver val_a =  frame0[z2*nslice+y2*nx+x2];
							mathtype_solver val_b = warped1[z2*nslice+y2*nx+x2];

							int idx_x = absr*16 + absq*4 + absp;
							int idx_y = absr*16 + absp*4 + absq;
							int idx_z = absq*16 + absp*4 + absr;

							float this_weight_x = 2.f*sign_x*farid[idx_x];//times 2 to compensate the wrong normalizer
							float this_weight_y = 2.f*sign_y*farid[idx_y];
							float this_weight_z = 2.f*sign_z*farid[idx_z];

							Idx += this_weight_x*(val_a+val_b);
							Idy += this_weight_y*(val_a+val_b);
							Idz += this_weight_z*(val_a+val_b);
						}
					}
				}
				__syncthreads();
				mathtype_solver val0b    = warped1[pos];
				Idt = val0b-frame0_val;
				//////////////////////////////////////////////////////////////////////////////////////////
			}
			else if (spatiotemporalderivative_id == 7){
				Idx = 0.0f; Idy = 0.0f; Idz = 0.0f;
				//Farid 9: average of frame1 and frame2 using Farid kernel, central difference, dt: forward difference
				//////////////////////////////////////////////////////////////////////////////////////////
				int this_radius = 4;
				float farid[125] = {0.0f,0.01454688795f,0.01200102083f,0.003556370735f,0.0003091749386f,0.0f,0.01072974596f,0.008851921186f,0.00262316945f,0.0002280466142f,
					0.0f,0.004133734852f,0.003410285106f,0.001010600477f,8.785707905e-05f, 0.0f,0.000708593172f,0.0005845814594f,0.0001732342935f,1.506021363e-05f,
					0.0f,3.299081072e-05f,2.721705096e-05f,8.065474503e-06f,7.01176134e-07f, 0.0f,0.01072974596f,0.008851921186f,0.00262316945f,0.0002280465997f,
					0.0f,0.00791423209f,0.006529153325f,0.001934842789f,0.0001682065777f, 0.0f,0.003049031831f,0.002515417291f,0.0007454162696f,6.480314914e-05f,
					0.0f,0.0005226564244f,0.0004311857338f,0.0001277771516f,1.110837366e-05f,  0.0f,2.433393274e-05f,2.00752238e-05f,5.949072147e-06f,5.171856401e-07f,
					0.0f,0.004133734852f,0.003410285106f,0.001010600594f,8.785708633e-05f, 0.0f,0.003049031831f,0.002515417291f,0.0007454162696f,6.480314914e-05f,
					0.0f,0.001174667967f,0.0009690879378f,0.0002871785546f,2.496601883e-05f, 0.0f,0.0002013582707f,0.0001661183342f,4.922733933e-05f,4.279604582e-06f,
					0.0f,9.374874935e-06f,7.734167411e-06f,2.291935516e-06f,1.992506071e-07f, 0.0f,0.000708593172f,0.0005845814594f,0.0001732342935f,1.506021363e-05f,
					0.0f,0.0005226564244f,0.0004311857338f,0.0001277771516f,1.110837366e-05f, 0.0f,0.0002013582707f,0.0001661183342f,4.922733933e-05f,4.279604582e-06f,
					0.0f,3.451626617e-05f,2.847553696e-05f,8.438411896e-06f,7.335977443e-07f, 0.0f,1.60701461e-06f,1.325769176e-06f,3.928771548e-07f,3.415497574e-08f,
					0.0f,3.299081072e-05f,2.721705096e-05f,8.065474503e-06f,7.011761909e-07f, 0.0f,2.433393456e-05f,2.00752238e-05f,5.949072147e-06f,5.171856969e-07f,
					0.0f,9.374874935e-06f,7.734167411e-06f,2.291935516e-06f,1.992506071e-07f,0.0f,1.607014724e-06f,1.32576929e-06f,3.928771264e-07f,3.415497929e-08f,
					0.0f,7.481968112e-08f,6.172540878e-08f,1.829164553e-08f,1.590193643e-09f};
				for(int r = -this_radius; r <= this_radius; r++){
					int z2 = z+r;
					if (z2 >= nz) z2 = 2*nz-z2-2; if (z2 < 0) z2 = -z2;
					int absr = abs(r);
					mathtype_solver sign_z = r < 0 ? -normalizer_z2 : normalizer_z2;
					for (int q = -this_radius; q <= this_radius; q++){
						int y2 = y+q;
						if (y2 >= ny) y2 = 2*ny-y2-2; if (y2 < 0) y2 = -y2;
						int absq = abs(q);
						mathtype_solver sign_y = q < 0 ? -normalizer_y2 : normalizer_y2;
						for(int p = -this_radius; p <= this_radius; p++){
							int x2 = x+p;
							if (x2 >= nx) x2 = 2*nx-x2-2; if (x2 < 0) x2 = -x2;
							if (p == 0 && q == 0 && r == 0) continue;
							int absp = abs(p);
							mathtype_solver sign_x = p < 0 ? -normalizer_x2 : normalizer_x2;
							__syncthreads();
							mathtype_solver val_a =  frame0[z2*nslice+y2*nx+x2];
							mathtype_solver val_b = warped1[z2*nslice+y2*nx+x2];

							int idx_x = absr*25 + absq*5 + absp;
							int idx_y = absr*25 + absp*5 + absq;
							int idx_z = absq*25 + absp*5 + absr;

							float this_weight_x = 2.f*sign_x*farid[idx_x];//times 2 to compensate the wrong normalizer
							float this_weight_y = 2.f*sign_y*farid[idx_y];
							float this_weight_z = 2.f*sign_z*farid[idx_z];

							Idx += this_weight_x*(val_a+val_b);
							Idy += this_weight_y*(val_a+val_b);
							Idz += this_weight_z*(val_a+val_b);
						}
					}
				}
				__syncthreads();
				mathtype_solver val0b    = warped1[pos];
				Idt = val0b-frame0_val;
				//////////////////////////////////////////////////////////////////////////////////////////
			}
			else if (spatiotemporalderivative_id == 2){
				//Ershov: average of frame1 and frame2, central difference, dt: forward difference
				//////////////////////////////////////////////////////////////////////////////////////////
				mathtype_solver val_xn_a = frame0[z*nslice+y*nx + xn];
				mathtype_solver val_xp_a = frame0[z*nslice+y*nx + xp];
				mathtype_solver val_yn_a = frame0[z*nslice+yn*nx + x];
				mathtype_solver val_yp_a = frame0[z*nslice+yp*nx + x];
				mathtype_solver val_zn_a = frame0[zn*nslice+y*nx + x];
				mathtype_solver val_zp_a = frame0[zp*nslice+y*nx + x];

				mathtype_solver val_xn_b = warped1[z*nslice+y*nx + xn];
				mathtype_solver val0b    = warped1[pos];
				mathtype_solver val_xp_b = warped1[z*nslice+y*nx + xp];
				mathtype_solver val_yn_b = warped1[z*nslice+yn*nx + x];
				mathtype_solver val_yp_b = warped1[z*nslice+yp*nx + x];
				mathtype_solver val_zn_b = warped1[zn*nslice+y*nx + x];
				mathtype_solver val_zp_b = warped1[zp*nslice+y*nx + x];

				Idx = normalizer_x2*((val_xp_a-val_xn_a)+(val_xp_b-val_xn_b));
				Idy = normalizer_y2*((val_yp_a-val_yn_a)+(val_yp_b-val_yn_b));
				Idz = normalizer_z2*((val_zp_a-val_zn_a)+(val_zp_b-val_zn_b));
				Idt = val0b-frame0_val;
				//////////////////////////////////////////////////////////////////////////////////////////
			}
			else if (spatiotemporalderivative_id == 1){
				//Horn-Schunck: average of frame1 and frame2, dx-kernel := [-1,1; -1,1], dt: local average
				//////////////////////////////////////////////////////////////////////////////////////////
				mathtype_solver val100a = frame0[z*nslice+y*nx + xp];
				mathtype_solver val010a = frame0[z*nslice+yp*nx + x];
				mathtype_solver val110a = frame0[z*nslice+yp*nx + xp];
				mathtype_solver val001a = frame0[zp*nslice+y*nx + x];
				mathtype_solver val101a = frame0[zp*nslice+y*nx + xp];
				mathtype_solver val011a = frame0[zp*nslice+yp*nx + x];
				mathtype_solver val111a = frame0[zp*nslice+yp*nx + xp];

				mathtype_solver val000b = warped1[z*nslice+y*nx + x];
				mathtype_solver val100b = warped1[z*nslice+y*nx + xp];
				mathtype_solver val010b = warped1[z*nslice+yp*nx + x];
				mathtype_solver val110b = warped1[z*nslice+yp*nx + xp];
				mathtype_solver val001b = warped1[zp*nslice+y*nx + x];
				mathtype_solver val101b = warped1[zp*nslice+y*nx + xp];
				mathtype_solver val011b = warped1[zp*nslice+yp*nx + x];
				mathtype_solver val111b = warped1[zp*nslice+yp*nx + xp];

				Idx = normalizer_x1*((-frame0_val + val100a - val010a + val110a) + (-val001a + val101a - val011a + val111a)
								   + (-val000b + val100b - val010b + val110b) + (-val001b + val101b - val011b + val111b));
				Idy = normalizer_y1*((-frame0_val - val100a + val010a + val110a) + (-val001a - val101a + val011a + val111a)
								   + (-val000b - val100b + val010b + val110b) + (-val001b - val101b + val011b + val111b));
				Idz = normalizer_z1*((-frame0_val - val100a + val001a + val101a) + (-val010a - val110a + val011a + val111a)
								   + (-val000b - val100b + val001b + val101b) + (-val010b - val110b + val011b + val111b));
				Idt = 0.125f*((val000b+val100b+val010b+val110b)+(val001b+val101b+val011b+val111b)-(frame0_val+val100a+val010a+val110a)-(val001a+val101a+val011a+val111a));
				//////////////////////////////////////////////////////////////////////////////////////////
			}
			/////////////////////////////////////////////

			//Intensity constancy:
			/////////////////////////////////////////////
			mathtype_solver J11 = Idx*Idx;
			mathtype_solver J22 = Idy*Idy;
			mathtype_solver J33 = Idz*Idz;
			mathtype_solver J12 = Idx * Idy;
			mathtype_solver J13 = Idx * Idz;
			mathtype_solver J23 = Idy * Idz;
			mathtype_solver J14 = Idx * Idt;
			mathtype_solver J24 = Idy * Idt;
			mathtype_solver J34 = Idz * Idt;
			/////////////////////////////////////////////

			//Calculating data term on the fly doesn't hurt much and saves memory
			//(doesn't work for local global approach)
			////////////////////////////////////////////////////////////////
			if(!precalculated_psi)
			{
				//assuming inner_iterations = 1 we could do psi0 = Idt;
				psi0 = Idt+Idx*du0+Idy*dv0+Idz*dw0;
				psi0 *= psi0;
			}
			psi0 = 0.5f/sqrtf(psi0+epsilon_psi_squared);

			if(use_confidencemap) psi0 *= max(0.0f, min(1.0f, confidence));

			//deactivate data term for background and out of bounds:
			if (frame0_val < minIntensity || frame0_val > maxIntensity) psi0 = 0.0f;
			if (slip_depth > 0 && (x < slip_depth || x >= nx-slip_depth || y < slip_depth || y >= ny-slip_depth || z < slip_depth || z >= nz-slip_depth)) psi0 = 0.0f; //avoid objects getting pinned to the boundary
			//we should also deactivate the data term in voxels that move out of bounds
			if(x+u0+du0 < 0.0f || x+u0+du0 > nx-1.f || y+v0+dv0 < 0.0f || y+v0+dv0 > ny-1 || z+w0+dw0 < 0.0f || z+w0+dw0 > nz-1) psi0 = 0.0f;

			////////////////////////////////////////////////////////////////

			//Calculate SOR update
			/////////////////////////////////////////////
			mathtype_solver sumH = alphax*(phi_neighbour[0]+phi_neighbour[1]) + alphay*(phi_neighbour[2]+phi_neighbour[3]) + alphaz*(phi_neighbour[4]+phi_neighbour[5]);
			mathtype_solver sumU = alphax*(phi_neighbour[0]*du_neighbour[0] + phi_neighbour[1]*du_neighbour[1])
								 + alphay*(phi_neighbour[2]*du_neighbour[2] + phi_neighbour[3]*du_neighbour[3])
								 + alphaz*(phi_neighbour[4]*du_neighbour[4] + phi_neighbour[5]*du_neighbour[5]);

			mathtype_solver sumH2 = sumH;
			mathtype_solver sumH3 = sumH;
			mathtype_solver sumV, sumW;

			if(!decoupled_smoothness){
				sumV = alphax*(phi_neighbour[0]*dv_neighbour[0] + phi_neighbour[1]*dv_neighbour[1])
					 + alphay*(phi_neighbour[2]*dv_neighbour[2] + phi_neighbour[3]*dv_neighbour[3])
					 + alphaz*(phi_neighbour[4]*dv_neighbour[4] + phi_neighbour[5]*dv_neighbour[5]);
				sumW = alphax*(phi_neighbour[0]*dw_neighbour[0] + phi_neighbour[1]*dw_neighbour[1])
					 + alphay*(phi_neighbour[2]*dw_neighbour[2] + phi_neighbour[3]*dw_neighbour[3])
					 + alphaz*(phi_neighbour[4]*dw_neighbour[4] + phi_neighbour[5]*dw_neighbour[5]);
			}
			else
			{
				sumV = alphax*(phi_neighbour[6]*dv_neighbour[0]  + phi_neighbour[7]*dv_neighbour[1])
					 + alphay*(phi_neighbour[8]*dv_neighbour[2]  + phi_neighbour[9]*dv_neighbour[3])
					 + alphaz*(phi_neighbour[10]*dv_neighbour[4] + phi_neighbour[11]*dv_neighbour[5]);
				sumW = alphax*(phi_neighbour[12]*dw_neighbour[0] + phi_neighbour[13]*dw_neighbour[1])
					 + alphay*(phi_neighbour[14]*dw_neighbour[2] + phi_neighbour[15]*dw_neighbour[3])
					 + alphaz*(phi_neighbour[16]*dw_neighbour[4] + phi_neighbour[17]*dw_neighbour[5]);
				sumH2 = alphax*(phi_neighbour[6]+phi_neighbour[7])   + alphay*(phi_neighbour[8]+phi_neighbour[9])   + alphaz*(phi_neighbour[10]+phi_neighbour[11]);
				sumH3 = alphax*(phi_neighbour[12]+phi_neighbour[13]) + alphay*(phi_neighbour[14]+phi_neighbour[15]) + alphaz*(phi_neighbour[16]+phi_neighbour[17]);
			}

			mathtype_solver next_du, next_dv, next_dw;

			//SOR-step unless Dirichlet boundary conditions
			////////////////////////////////////////////////////////
			if (boundary_voxel)
			{
				if (   (x == 0 && gpu_const::fixedDirichletBoundary_c[2] == 1) || (x == nx-1 && gpu_const::fixedDirichletBoundary_c[3] == 1)
					|| (y == 0 && gpu_const::fixedDirichletBoundary_c[0] == 1) || (y == ny-1 && gpu_const::fixedDirichletBoundary_c[1] == 1)
					|| (z == 0 && gpu_const::fixedDirichletBoundary_c[4] == 1) || (z == nz-1 && gpu_const::fixedDirichletBoundary_c[5] == 1))
				{
					next_du = 0.0f;
					next_dv = 0.0f;
					next_dw = 0.0f;
				}
				else
				{
					if ((x == 0 && gpu_const::zeroDirichletBoundary_c[2] == 1) || (x == nx-1 && gpu_const::zeroDirichletBoundary_c[3] == 1))
						next_du = 0.0f; //boundary condition set
					else
						next_du = (1.f-omega)*du0 + omega*(psi0 *(-J14 -(J12*dv0)     -(J13*dw0)) + sumU)/((psi0*J11) + sumH);

					if ((y == 0 && gpu_const::zeroDirichletBoundary_c[0] == 1) || (y == ny-1 && gpu_const::zeroDirichletBoundary_c[1] == 1))
						next_dv = 0.0f;
					else
						next_dv = (1.f-omega)*dv0 + omega*(psi0 *(-J24 -(J12*next_du) -(J23*dw0)) + sumV)/((psi0*J22) + sumH2);

					if ((z == 0 && gpu_const::zeroDirichletBoundary_c[4] == 1) || (z == nz-1 && gpu_const::zeroDirichletBoundary_c[5] == 1))
						next_dw = 0.0f;
					else
						next_dw = (1.f-omega)*dw0 + omega*(psi0 *(-J34 -(J13*next_du) -(J23*next_dv)) + sumW)/((psi0*J33) + sumH3);
				}
			}
			else
			{
				next_du = (1.f-omega)*du0 + omega*(psi0 *(-J14 -(J12*dv0)     -(J13*dw0)) + sumU)/((psi0*J11) + sumH);
				next_du = min(max_step, max(-max_step,next_du));
				next_dv = (1.f-omega)*dv0 + omega*(psi0 *(-J24 -(J12*next_du) -(J23*dw0)) + sumV)/((psi0*J22) + sumH2);
				next_dv = min(max_step, max(-max_step,next_dv));
				next_dw = (1.f-omega)*dw0 + omega*(psi0 *(-J34 -(J13*next_du) -(J23*next_dv)) + sumW)/((psi0*J33) + sumH3);
				next_dw = min(max_step, max(-max_step,next_dw));
			}

			if (gpu_const::protect_overlap_c)
			{
				//extend the Dirichlet boundary inwards for mosaic processing
				int half_overlap = gpu_const::overlap_c/2;

				if (    (x < half_overlap && gpu_const::fixedDirichletBoundary_c[2] == 1) || (x >= nx-1-half_overlap && gpu_const::fixedDirichletBoundary_c[3] == 1)
					 || (y < half_overlap && gpu_const::fixedDirichletBoundary_c[0] == 1) || (y >= ny-1-half_overlap && gpu_const::fixedDirichletBoundary_c[1] == 1)
					 || (z < half_overlap && gpu_const::fixedDirichletBoundary_c[4] == 1) || (z >= nz-1-half_overlap && gpu_const::fixedDirichletBoundary_c[5] == 1))
				{
					next_du = 0.0f;
					next_dv = 0.0f;
					next_dw = 0.0f;
				}
			}
			////////////////////////////////////////////////////////

			/////////////////////////////////////////////
			__syncthreads();
			if(!outofbounds)
			{
				du[pos] = next_du;
				du[pos+nstack] = next_dv;
				du[pos+nstack2] = next_dw;
			}
			/////////////////////////////////////////////

			return;
		}

		__global__ void addsolution_warpFrame1_z(bool rewarp, optflow_type *intermediate_warp, img_type *frame0, img_type *frame1, optflow_type *u, optflow_type *du)
		{
			//No need to allocate additional memory. We can use the phi array.

			//acquire constants and position
			/////////////////////////////////////////////
			int nx = gpu_const::nx_c;
			int ny = gpu_const::ny_c;
			int nz = gpu_const::nz_c;
			if (nz == 1) return;

			//int outOfBounds_id = gpu_const::outOfBounds_id_c; <- only replace mode allowed here
			int interpolation_id = gpu_const::warpInterpolation_id_c;

			idx_type nslice = nx*ny;
			idx_type nstack = nz*nslice;
			idx_type nstack2 = 2*nstack;

			bool outofbounds = false;
			idx_type pos = (blockIdx.x*blockDim.x+threadIdx.x);
			if (pos >= nstack) {pos = threadIdx.x; outofbounds=true;}

			int z = pos/nslice;
			int y = (pos-z*nslice)/nx;
			int x = pos-z*nslice-y*nx;
			/////////////////////////////////////////////

			/////////////////////////////////////////////
			__syncthreads();
			mathtype_solver w0 = u[pos+nstack2];
			mathtype_solver dw0 = du[pos+nstack2];

			w0 += dw0;
			float z0 = z + w0;

			if(rewarp) z0 = z+dw0;

			img_type replace_val = 0.0f;
			bool moved_out = false;

			//out of bounds?
			if (z0 < 0 || z0 > (nz-1)) {
				moved_out = true;
				replace_val = frame0[pos];
				z0 = z;
			}

			int zf = floor(z0);
			int zc = ceil(z0);

			float wz = z0-zf;
			float value = 0.0f;
			/////////////////////////////////////////////

			/////////////////////////////////////////////
			if (interpolation_id == 1) //cubic interpolation
			{
				//extrapolate with zero-gradient
				int zf2 = max(0, zf-1);
				int zc2 = min(zc+1, nz-1);

				__syncthreads();
				img_type P000 = frame1[zf2*nslice+y*nx + x];
				img_type P001 = frame1[zf *nslice+y*nx + x];
				img_type P002 = frame1[zc *nslice+y*nx + x];
				img_type P003 = frame1[zc2*nslice+y*nx + x];

				value = interpolate_cubic(P000,P001,P002,P003,wz);
			}
			else //linear interpolation
			{
				__syncthreads();
				img_type P000 = frame1[zf*nslice+y*nx + x];
				img_type P001 = frame1[zc*nslice+y*nx + x];

				value = (P001-P000)*wz+P000;
			}

			if(moved_out) value = replace_val;
			/////////////////////////////////////////////

			/////////////////////////////////////////////
			__syncthreads();
			if(!outofbounds)
			{
				intermediate_warp[pos] = value;
				u[pos+nstack2] = w0;
				du[pos+nstack2] = 0.0f;
			}
			/////////////////////////////////////////////

			return;
		}
		__global__ void addsolution_warpFrame1_xy(bool rewarp, img_type *warped1, img_type *frame0, optflow_type *frame1, optflow_type *u, optflow_type *du, optflow_type *confidence){
			//acquire constants
			/////////////////////////////////////////////
			int nx = gpu_const::nx_c;
			int ny = gpu_const::ny_c;
			int nz = gpu_const::nz_c;

			int outOfBounds_id = gpu_const::outOfBounds_id_c;
			int interpolation_id = gpu_const::warpInterpolation_id_c;
			bool use_confidencemap = gpu_const::use_confidencemap_c;

			idx_type nslice = nx*ny;
			idx_type nstack = nz*nslice;

			bool outofbounds = false;
			idx_type pos = (blockIdx.x*blockDim.x+threadIdx.x);
			if (pos >= nstack) {outofbounds = true; pos = threadIdx.x;}

			int z = pos/nslice;
			int y = (pos-z*nslice)/nx;
			int x = pos-z*nslice-y*nx;
			/////////////////////////////////////////////

			/////////////////////////////////////////////
			__syncthreads();
			mathtype_solver u0  = u[pos];
			mathtype_solver v0  = u[pos+nstack];
			mathtype_solver w0 = 0.0f; //should already be warped
			mathtype_solver du0 = du[pos];
			mathtype_solver dv0 = du[pos+nstack];

			if(nz > 1)
				w0 = u[pos+2*nstack]; //for out of bounds checking

			u0 += du0;
			v0 += dv0;

			float x0 = x + u0;
			float y0 = y + v0;
			float z0 = z + w0;

			if(rewarp)
			{
				x0 = x + du0;
				y0 = y + dv0;
				z0 = z;
			}

			//out of bounds?
			////////////////////////
			img_type replace_val = 0.0f;
			bool moved_out = false;

			if (y0 < 0 || x0 < 0 || z0 < 0 || x0 > (nx-1) || y0 > (ny-1) || z0 > (nz-1))
			{
				moved_out = true;
				if (outOfBounds_id == 0) replace_val = frame0[pos];
				else replace_val = gpu_const::nanf_c;

				if (use_confidencemap) confidence[pos] = 0.0f;

				x0 = x; y0 = y; //z0 = z;
			}
			////////////////////////

			int xf = floor(x0);
			int xc = ceil(x0);
			int yf = floor(y0);
			int yc = ceil(y0);

			//extrapolate with zero-gradient
			int xf2 = max(0, xf-1);
			int xc2 = min(xc+1, nx-1);
			int yf2 = max(0, yf-1);
			int yc2 = min(yc+1, ny-1);

			float wx = x0-xf;
			float wy = y0-yf;

			img_type value = 0.0f;

			__syncthreads();
			/////////////////////////////////////////////
			img_type P11 = frame1[z*nslice+ yf*nx + xf];
			img_type P21 = frame1[z*nslice+ yf*nx + xc];
			img_type P12 = frame1[z*nslice+ yc*nx + xf];
			img_type P22 = frame1[z*nslice+ yc*nx + xc];

			if (interpolation_id == 1)
			{
				img_type P10 = frame1[z*nslice + yf2*nx + xf];
				img_type P20 = frame1[z*nslice + yf2*nx + xc];
				img_type P01 = frame1[z*nslice + yf*nx  + xf2];
				img_type P31 = frame1[z*nslice + yf*nx  + xc2];
				img_type P02 = frame1[z*nslice + yc*nx  + xf2];
				img_type P32 = frame1[z*nslice + yc*nx  + xc2];
				img_type P13 = frame1[z*nslice + yc2*nx + xf];
				img_type P23 = frame1[z*nslice + yc2*nx + xc];

				float gtu = gpu3d::interpolate_cubic(P01,P11,P21,P31,wx);
				float gbu = gpu3d::interpolate_cubic(P02,P12,P22,P32,wx);

				float glv = gpu3d::interpolate_cubic(P10,P11,P12,P13,wy);
				float grv = gpu3d::interpolate_cubic(P20,P21,P22,P23,wy);

				float sigma_lr = (1.f-wx)*glv + wx*grv;
				float sigma_bt = (1.f-wy)*gtu + wy*gbu;
				float corr_lrbt = P11*(1.f-wy)*(1.f-wx) + P12*wy*(1.f-wx) + P21*(1.f-wy)*wx + P22*wx*wy;

				value = sigma_lr+sigma_bt-corr_lrbt;
			}
			else
			{
				float glv = (P12-P11)*wy+P11; //left
				float grv = (P22-P21)*wy+P21; //right
				float gtu = (P21-P11)*wx+P11; //top
				float gbu = (P22-P12)*wx+P12; //bottom

				float sigma_lr = (1.f-wx)*glv + wx*grv;
				float sigma_bt = (1.f-wy)*gtu + wy*gbu;
				float corr_lrbt = P11*(1.f-wy)*(1.f-wx) + P12*wy*(1.f-wx) + P21*(1.f-wy)*wx + P22*wx*wy;

				value = sigma_lr+sigma_bt-corr_lrbt;
			}

			if(moved_out) value = replace_val;

			__syncthreads();
			////////////////////////////
			if(!outofbounds)
			{
				warped1[pos] = value;
				u[pos] = u0;
				u[pos+nstack] = v0;
				du[pos] = 0.0f;
				du[pos+nstack] = 0.0f;
			}
			////////////////////////////

			return;
		}
		__global__ void addsolution(optflow_type *u, optflow_type *du)
		{
			//acquire constants
			/////////////////////////////////////////////
			int nx = gpu_const::nx_c;
			int ny = gpu_const::ny_c;
			int nz = gpu_const::nz_c;

			idx_type nslice = nx*ny;
			idx_type nstack = nz*nslice;

			idx_type pos = (blockIdx.x*blockDim.x+threadIdx.x);
			if (pos >= nstack) {pos = threadIdx.x;}
			/////////////////////////////////////////////

			/////////////////////////////////////////////
			__syncthreads();
			mathtype_solver u0  = u[pos];
			mathtype_solver v0  = u[pos+nstack];
			mathtype_solver w0  = u[pos+(2*nstack)];
			mathtype_solver du0 = du[pos];
			mathtype_solver dv0 = du[pos+nstack];
			mathtype_solver dw0  = du[pos+(2*nstack)];

			u0 += du0;
			v0 += dv0;
			w0 += dw0;

			u[pos] = u0;
			u[pos+nstack] = v0;
			u[pos+(2*nstack)] = w0;
			du[pos] = 0.0f;
			du[pos+nstack] = 0.0f;
			du[pos+(2*nstack)] = 0.0f;
			////////////////////////////

			return;
		}
		__global__ void update_dataterm(img_type *frame0, img_type *warped1, optflow_type *du, optflow_type *psi)
		{
			//acquire constants and position
			/////////////////////////////////////////////
			int nx = gpu_const::nx_c;
			int ny = gpu_const::ny_c;
			int nz = gpu_const::nz_c;

			int spatiotemporalderivative_id = gpu_const::spatiotemporalderivative_id_c;
			mathtype_solver hx = gpu_const::hx_c;
			mathtype_solver hy = gpu_const::hy_c;
			mathtype_solver hz = gpu_const::hz_c;

			mathtype_solver normalizer_x1 = 0.125f/hx;
			mathtype_solver normalizer_y1 = 0.125f/hy;
			mathtype_solver normalizer_z1 = 0.125f/hz;
			mathtype_solver normalizer_x2 = 0.25f/hx;
			mathtype_solver normalizer_y2 = 0.25f/hy;
			mathtype_solver normalizer_z2 = 0.25f/hz;

			idx_type nslice = nx*ny;
			idx_type nstack = nz*nslice;

			idx_type pos = (blockIdx.x*blockDim.x+threadIdx.x);
			if (pos >= nstack) pos = threadIdx.x;

			int z = pos/nslice;
			int y = (pos-z*nslice)/nx;
			int x = pos-z*nslice-y*nx;

			int zp = z+1; int zn = z-1;
			int yp = y+1; int yn = y-1;
			int xp = x+1; int xn = x-1;

			//Reflective boundary conditions
			if (zp == nz) zp -= 2; if (zn < 0) zn = 1;
			if (yp == ny) yp -= 2; if (yn < 0) yn = 1;
			if (xp == nx) xp -= 2; if (xn < 0) xn = 1;

			__syncthreads();
			/////////////////////////////////////////////

			float Idx, Idy, Idz, Idt;

			mathtype_solver du0 = du[pos];
			mathtype_solver dv0 = du[pos+nstack];
			mathtype_solver dw0 = du[pos+(2*nstack)];
			mathtype_solver frame0_val = frame0[pos];

			//Precalculate spatiotemporal derivatives for local-global
			/////////////////////////////////////////////
			if (abs(spatiotemporalderivative_id) == 3){
				//Fourth Order Finite Difference
				//////////////////////////////////////////////////////////////////////////////////////////

				int yp2 = y+2; int yn2 = y-2; int xp2 = x+2; int xn2 = x-2; int zp2 = z+2; int zn2 = z-2;
				if (yp2 >= ny) yp2 = 2*ny-yp2-2; if (yn2 < 0) yn2 = -yn2;
				if (xp2 >= nx) xp2 = 2*nx-xp2-2; if (xn2 < 0) xn2 = -xn2;
				if (zp2 >= nz) zp2 = 2*nz-zp2-2; if (zn2 < 0) zn2 = -zn2;

				__syncthreads();
				mathtype_solver val_xn2_a = frame0[z*nslice+y*nx + xn2];
				mathtype_solver val_xn_a = frame0[z*nslice+y*nx + xn];
				//mathtype_solver val0a    = frame0[pos];
				mathtype_solver val_xp_a = frame0[z*nslice+y*nx + xp];
				mathtype_solver val_xp2_a = frame0[z*nslice+y*nx + xp2];
				mathtype_solver val_yn2_a = frame0[z*nslice+yn2*nx + x];
				mathtype_solver val_yn_a = frame0[z*nslice+yn*nx + x];
				mathtype_solver val_yp_a = frame0[z*nslice+yp*nx + x];
				mathtype_solver val_yp2_a = frame0[z*nslice+yp2*nx + x];
				mathtype_solver val_zn2_a = frame0[zn2*nslice+y*nx + x];
				mathtype_solver val_zn_a = frame0[zn*nslice+y*nx + x];
				mathtype_solver val_zp_a = frame0[zp*nslice+y*nx + x];
				mathtype_solver val_zp2_a = frame0[zp2*nslice+y*nx + x];

				mathtype_solver val_xn2_b = warped1[z*nslice+y*nx + xn2];
				mathtype_solver val_xn_b = warped1[z*nslice+y*nx + xn];
				mathtype_solver val0b    = warped1[pos];
				mathtype_solver val_xp_b = warped1[z*nslice+y*nx + xp];
				mathtype_solver val_xp2_b = warped1[z*nslice+y*nx + xp2];
				mathtype_solver val_yn2_b = warped1[z*nslice+yn2*nx + x];
				mathtype_solver val_yn_b = warped1[z*nslice+yn*nx + x];
				mathtype_solver val_yp_b = warped1[z*nslice+yp*nx + x];
				mathtype_solver val_yp2_b = warped1[z*nslice+yp2*nx + x];
				mathtype_solver val_zn2_b = warped1[zn2*nslice+y*nx + x];
				mathtype_solver val_zn_b = warped1[zn*nslice+y*nx + x];
				mathtype_solver val_zp_b = warped1[zp*nslice+y*nx + x];
				mathtype_solver val_zp2_b = warped1[zp2*nslice+y*nx + x];

				Idx = normalizer_x1/6.f*((val_xn2_a-8.f*val_xn_a+8.f*val_xp_a-val_xp2_a)+(val_xn2_b-8.f*val_xn_b+8.f*val_xp_b-val_xp2_b));
				Idy = normalizer_y1/6.f*((val_yn2_a-8.f*val_yn_a+8.f*val_yp_a-val_yp2_a)+(val_yn2_b-8.f*val_yn_b+8.f*val_yp_b-val_yp2_b));
				Idz = normalizer_z1/6.f*((val_zn2_a-8.f*val_zn_a+8.f*val_zp_a-val_zp2_a)+(val_zn2_b-8.f*val_zn_b+8.f*val_zp_b-val_zp2_b));
				Idt = val0b-frame0_val;
				//////////////////////////////////////////////////////////////////////////////////////////
			}
			else if (abs(spatiotemporalderivative_id) == 4){
				Idx = 0.0f; Idy = 0.0f; Idz = 0.0f;
				//Farid 3: average of frame1 and frame2 using Farid kernel, central difference, dt: forward difference
				//////////////////////////////////////////////////////////////////////////////////////////
				float farid[4] = {0.0f, 0.12412487720f, 0.05281651765f, 0.02247401886f};
				for(int r = -1; r <= 1; r++){
					int z2 = z+r;
					if (z2 == nz) z2 -= 2; if (z2 < 0) z2 = 1;
					int n_edges_r = abs(r);
					for (int q = -1; q <= 1; q++){
						int y2 = y+q;
						if (y2 == ny) y2 -= 2; if (y2 < 0) y2 = 1;
						int n_edges_q = n_edges_r + abs(q);
						for(int p = -1; p <= 1; p++){
							int x2 = x+p;
							if (x2 == nx) x2 -= 2; if (x2 < 0) x2 = 1;
							int n_edges = n_edges_q + abs(p);
							if (n_edges == 0) continue;

							__syncthreads();
							mathtype_solver val_a =  frame0[z2*nslice+y2*nx+x2];
							mathtype_solver val_b = warped1[z2*nslice+y2*nx+x2];

							float this_weight = 2.f*farid[n_edges]; //times 2 to compensate the wrong normalizer

							Idx += (normalizer_x2*p*this_weight)*(val_a+val_b);
							Idy += (normalizer_y2*q*this_weight)*(val_a+val_b);
							Idz += (normalizer_z2*r*this_weight)*(val_a+val_b);
						}
					}
				}
				__syncthreads();
				mathtype_solver val0b    = warped1[pos];
				Idt = val0b-frame0_val;
				//////////////////////////////////////////////////////////////////////////////////////////
			}
			else if (abs(spatiotemporalderivative_id) == 5){
				Idx = 0.0f; Idy = 0.0f; Idz = 0.0f;
				//Farid 5: average of frame1 and frame2 using Farid kernel, central difference, dt: forward difference
				//////////////////////////////////////////////////////////////////////////////////////////
				float farid[27] = {0.0f, 0.0503013134f,  0.01992556825f,0.0f, 0.02939366736f, 0.01164354291f,
						0.0f, 0.004442796111f,0.001759899082f,0.0f, 0.02939366549f,  0.01164354291f,0.0f, 0.01717624255f,  0.006803925615f,
						0.0f, 0.002596156206f, 0.001028400264f,
						0.0f, 0.004442796111f, 0.001759899198f,0.0f, 0.002596156206f, 0.001028400264f,0.0f, 0.0003924040357f, 0.0001554407354f
				};
				int this_radius = 2;
				for(int r = -this_radius; r <= this_radius; r++){
					int z2 = z+r;
					if (z2 >= nz) z2 = 2*nz-z2-2; if (z2 < 0) z2 = -z2;
					int absr = abs(r);
					mathtype_solver sign_z = r < 0 ? -normalizer_z2 : normalizer_z2;
					for (int q = -this_radius; q <= this_radius; q++){
						int y2 = y+q;
						if (y2 >= ny) y2 = 2*ny-y2-2; if (y2 < 0) y2 = -y2;
						int absq = abs(q);
						mathtype_solver sign_y = q < 0 ? -normalizer_y2 : normalizer_y2;
						for(int p = -this_radius; p <= this_radius; p++){
							int x2 = x+p;
							if (x2 >= nx) x2 = 2*nx-x2-2; if (x2 < 0) x2 = -x2;
							if (p == 0 && q == 0 && r == 0) continue;
							int absp = abs(p);
							mathtype_solver sign_x = p < 0 ? -normalizer_x2 : normalizer_x2;
							__syncthreads();
							mathtype_solver val_a =  frame0[z2*nslice+y2*nx+x2];
							mathtype_solver val_b = warped1[z2*nslice+y2*nx+x2];

							int idx_x = absr*9 + absq*3 + absp;
							int idx_y = absr*9 + absp*3 + absq;
							int idx_z = absq*9 + absp*3 + absr;

							float this_weight_x = 2.f*sign_x*farid[idx_x];//times 2 to compensate the wrong normalizer
							float this_weight_y = 2.f*sign_y*farid[idx_y];
							float this_weight_z = 2.f*sign_z*farid[idx_z];

							Idx += this_weight_x*(val_a+val_b);
							Idy += this_weight_y*(val_a+val_b);
							Idz += this_weight_z*(val_a+val_b);
						}
					}
				}
				__syncthreads();
				mathtype_solver val0b    = warped1[pos];
				Idt = val0b-frame0_val;
				//////////////////////////////////////////////////////////////////////////////////////////
			}
			else if (abs(spatiotemporalderivative_id) == 6){
				Idx = 0.0f; Idy = 0.0f; Idz = 0.0f;
				//Farid 7: average of frame1 and frame2 using Farid kernel, central difference, dt: forward difference
				//////////////////////////////////////////////////////////////////////////////////////////
				int this_radius = 3;
				float farid[64] = {0.0f, 0.02518012933f, 0.01634971797f, 0.002439625794f,0.0f, 0.01711205766f, 0.01111103687f, 0.001657935209f,0.0f, 0.004833645653f,0.003138536355f,0.0004683172156f,
					0.0f, 0.0003284906852f,0.0002132924128f,3.182646469e-05f,0.0f,0.01711205766f,0.01111103687f,0.001657935092f,0.0f,0.01162911206f,0.00755090313f,0.001126709278f,
					0.0f,0.003284876933f,0.002132904949f,0.0003182617365f,0.0f,0.0002232376137f,0.0001449505071f,2.162881356e-05f,0.0f,0.004833645653f,0.003138536355f,0.0004683171865f,
					0.0f,0.003284876933f,0.002132904716f,0.0003182617365f,0.0f,0.0009278797079f,0.000602481945f,8.989944035e-05f,0.0f,6.305796705e-05f,4.094419273e-05f,6.109494279e-06f,
					0.0f,0.0003284907143f,0.0002132924274f,3.182646105e-05f,0.0f,0.0002232376137f,0.0001449505071f,2.162881356e-05f,0.0f,6.305796705e-05f,4.094419273e-05f,6.109494279e-06f,
					0.0f,4.285368959e-06f,2.782534466e-06f,4.151963537e-07f,};
				for(int r = -this_radius; r <= this_radius; r++){
					int z2 = z+r;
					if (z2 >= nz) z2 = 2*nz-z2-2; if (z2 < 0) z2 = -z2;
					int absr = abs(r);
					mathtype_solver sign_z = r < 0 ? -normalizer_z2 : normalizer_z2;
					for (int q = -this_radius; q <= this_radius; q++){
						int y2 = y+q;
						if (y2 >= ny) y2 = 2*ny-y2-2; if (y2 < 0) y2 = -y2;
						int absq = abs(q);
						mathtype_solver sign_y = q < 0 ? -normalizer_y2 : normalizer_y2;
						for(int p = -this_radius; p <= this_radius; p++){
							int x2 = x+p;
							if (x2 >= nx) x2 = 2*nx-x2-2; if (x2 < 0) x2 = -x2;
							if (p == 0 && q == 0 && r == 0) continue;
							int absp = abs(p);
							mathtype_solver sign_x = p < 0 ? -normalizer_x2 : normalizer_x2;
							__syncthreads();
							mathtype_solver val_a =  frame0[z2*nslice+y2*nx+x2];
							mathtype_solver val_b = warped1[z2*nslice+y2*nx+x2];

							int idx_x = absr*16 + absq*4 + absp;
							int idx_y = absr*16 + absp*4 + absq;
							int idx_z = absq*16 + absp*4 + absr;

							float this_weight_x = 2.f*sign_x*farid[idx_x];//times 2 to compensate the wrong normalizer
							float this_weight_y = 2.f*sign_y*farid[idx_y];
							float this_weight_z = 2.f*sign_z*farid[idx_z];

							Idx += this_weight_x*(val_a+val_b);
							Idy += this_weight_y*(val_a+val_b);
							Idz += this_weight_z*(val_a+val_b);
						}
					}
				}
				__syncthreads();
				mathtype_solver val0b    = warped1[pos];
				Idt = val0b-frame0_val;
				//////////////////////////////////////////////////////////////////////////////////////////
			}
			else if (abs(spatiotemporalderivative_id) == 7){
				Idx = 0.0f; Idy = 0.0f; Idz = 0.0f;
				//Farid 9: average of frame1 and frame2 using Farid kernel, central difference, dt: forward difference
				//////////////////////////////////////////////////////////////////////////////////////////
				int this_radius = 4;
				float farid[125] = {0.0f,0.01454688795f,0.01200102083f,0.003556370735f,0.0003091749386f,0.0f,0.01072974596f,0.008851921186f,0.00262316945f,0.0002280466142f,
					0.0f,0.004133734852f,0.003410285106f,0.001010600477f,8.785707905e-05f, 0.0f,0.000708593172f,0.0005845814594f,0.0001732342935f,1.506021363e-05f,
					0.0f,3.299081072e-05f,2.721705096e-05f,8.065474503e-06f,7.01176134e-07f, 0.0f,0.01072974596f,0.008851921186f,0.00262316945f,0.0002280465997f,
					0.0f,0.00791423209f,0.006529153325f,0.001934842789f,0.0001682065777f, 0.0f,0.003049031831f,0.002515417291f,0.0007454162696f,6.480314914e-05f,
					0.0f,0.0005226564244f,0.0004311857338f,0.0001277771516f,1.110837366e-05f,  0.0f,2.433393274e-05f,2.00752238e-05f,5.949072147e-06f,5.171856401e-07f,
					0.0f,0.004133734852f,0.003410285106f,0.001010600594f,8.785708633e-05f, 0.0f,0.003049031831f,0.002515417291f,0.0007454162696f,6.480314914e-05f,
					0.0f,0.001174667967f,0.0009690879378f,0.0002871785546f,2.496601883e-05f, 0.0f,0.0002013582707f,0.0001661183342f,4.922733933e-05f,4.279604582e-06f,
					0.0f,9.374874935e-06f,7.734167411e-06f,2.291935516e-06f,1.992506071e-07f, 0.0f,0.000708593172f,0.0005845814594f,0.0001732342935f,1.506021363e-05f,
					0.0f,0.0005226564244f,0.0004311857338f,0.0001277771516f,1.110837366e-05f, 0.0f,0.0002013582707f,0.0001661183342f,4.922733933e-05f,4.279604582e-06f,
					0.0f,3.451626617e-05f,2.847553696e-05f,8.438411896e-06f,7.335977443e-07f, 0.0f,1.60701461e-06f,1.325769176e-06f,3.928771548e-07f,3.415497574e-08f,
					0.0f,3.299081072e-05f,2.721705096e-05f,8.065474503e-06f,7.011761909e-07f, 0.0f,2.433393456e-05f,2.00752238e-05f,5.949072147e-06f,5.171856969e-07f,
					0.0f,9.374874935e-06f,7.734167411e-06f,2.291935516e-06f,1.992506071e-07f,0.0f,1.607014724e-06f,1.32576929e-06f,3.928771264e-07f,3.415497929e-08f,
					0.0f,7.481968112e-08f,6.172540878e-08f,1.829164553e-08f,1.590193643e-09f};
				for(int r = -this_radius; r <= this_radius; r++){
					int z2 = z+r;
					if (z2 >= nz) z2 = 2*nz-z2-2; if (z2 < 0) z2 = -z2;
					int absr = abs(r);
					mathtype_solver sign_z = r < 0 ? -normalizer_z2 : normalizer_z2;
					for (int q = -this_radius; q <= this_radius; q++){
						int y2 = y+q;
						if (y2 >= ny) y2 = 2*ny-y2-2; if (y2 < 0) y2 = -y2;
						int absq = abs(q);
						mathtype_solver sign_y = q < 0 ? -normalizer_y2 : normalizer_y2;
						for(int p = -this_radius; p <= this_radius; p++){
							int x2 = x+p;
							if (x2 >= nx) x2 = 2*nx-x2-2; if (x2 < 0) x2 = -x2;
							if (p == 0 && q == 0 && r == 0) continue;
							int absp = abs(p);
							mathtype_solver sign_x = p < 0 ? -normalizer_x2 : normalizer_x2;
							__syncthreads();
							mathtype_solver val_a =  frame0[z2*nslice+y2*nx+x2];
							mathtype_solver val_b = warped1[z2*nslice+y2*nx+x2];

							int idx_x = absr*25 + absq*5 + absp;
							int idx_y = absr*25 + absp*5 + absq;
							int idx_z = absq*25 + absp*5 + absr;

							float this_weight_x = 2.f*sign_x*farid[idx_x];//times 2 to compensate the wrong normalizer
							float this_weight_y = 2.f*sign_y*farid[idx_y];
							float this_weight_z = 2.f*sign_z*farid[idx_z];

							Idx += this_weight_x*(val_a+val_b);
							Idy += this_weight_y*(val_a+val_b);
							Idz += this_weight_z*(val_a+val_b);
						}
					}
				}
				__syncthreads();
				mathtype_solver val0b    = warped1[pos];
				Idt = val0b-frame0_val;
				//////////////////////////////////////////////////////////////////////////////////////////
			}
			else if (abs(spatiotemporalderivative_id) == 2){
				//Ershov: average of frame1 and frame2, central difference, dt: forward difference
				//////////////////////////////////////////////////////////////////////////////////////////
				mathtype_solver val_xn_a = frame0[z*nslice+y*nx + xn];
				mathtype_solver val_xp_a = frame0[z*nslice+y*nx + xp];
				mathtype_solver val_yn_a = frame0[z*nslice+yn*nx + x];
				mathtype_solver val_yp_a = frame0[z*nslice+yp*nx + x];
				mathtype_solver val_zn_a = frame0[zn*nslice+y*nx + x];
				mathtype_solver val_zp_a = frame0[zp*nslice+y*nx + x];

				mathtype_solver val_xn_b = warped1[z*nslice+y*nx + xn];
				mathtype_solver val0b    = warped1[pos];
				mathtype_solver val_xp_b = warped1[z*nslice+y*nx + xp];
				mathtype_solver val_yn_b = warped1[z*nslice+yn*nx + x];
				mathtype_solver val_yp_b = warped1[z*nslice+yp*nx + x];
				mathtype_solver val_zn_b = warped1[zn*nslice+y*nx + x];
				mathtype_solver val_zp_b = warped1[zp*nslice+y*nx + x];

				Idx = normalizer_x2*((val_xp_a-val_xn_a)+(val_xp_b-val_xn_b));
				Idy = normalizer_y2*((val_yp_a-val_yn_a)+(val_yp_b-val_yn_b));
				Idz = normalizer_z2*((val_zp_a-val_zn_a)+(val_zp_b-val_zn_b));
				Idt = val0b-frame0_val;
				//////////////////////////////////////////////////////////////////////////////////////////
			}
			else if (abs(spatiotemporalderivative_id) == 1){
				//Horn-Schunck: average of frame1 and frame2, dx-kernel := [-1,1; -1,1], dt: local average
				//////////////////////////////////////////////////////////////////////////////////////////
				mathtype_solver val100a = frame0[z*nslice+y*nx + xp];
				mathtype_solver val010a = frame0[z*nslice+yp*nx + x];
				mathtype_solver val110a = frame0[z*nslice+yp*nx + xp];
				mathtype_solver val001a = frame0[zp*nslice+y*nx + x];
				mathtype_solver val101a = frame0[zp*nslice+y*nx + xp];
				mathtype_solver val011a = frame0[zp*nslice+yp*nx + x];
				mathtype_solver val111a = frame0[zp*nslice+yp*nx + xp];

				mathtype_solver val000b = warped1[z*nslice+y*nx + x];
				mathtype_solver val100b = warped1[z*nslice+y*nx + xp];
				mathtype_solver val010b = warped1[z*nslice+yp*nx + x];
				mathtype_solver val110b = warped1[z*nslice+yp*nx + xp];
				mathtype_solver val001b = warped1[zp*nslice+y*nx + x];
				mathtype_solver val101b = warped1[zp*nslice+y*nx + xp];
				mathtype_solver val011b = warped1[zp*nslice+yp*nx + x];
				mathtype_solver val111b = warped1[zp*nslice+yp*nx + xp];

				Idx = normalizer_x1*((-frame0_val + val100a - val010a + val110a) + (-val001a + val101a - val011a + val111a)
								   + (-val000b + val100b - val010b + val110b) + (-val001b + val101b - val011b + val111b));
				Idy = normalizer_y1*((-frame0_val - val100a + val010a + val110a) + (-val001a - val101a + val011a + val111a)
								   + (-val000b - val100b + val010b + val110b) + (-val001b - val101b + val011b + val111b));
				Idz = normalizer_z1*((-frame0_val - val100a + val001a + val101a) + (-val010a - val110a + val011a + val111a)
								   + (-val000b - val100b + val001b + val101b) + (-val010b - val110b + val011b + val111b));
				Idt = 0.125f*((val000b+val100b+val010b+val110b)+(val001b+val101b+val011b+val111b)-(frame0_val+val100a+val010a+val110a)-(val001a+val101a+val011a+val111a));
				//////////////////////////////////////////////////////////////////////////////////////////
			}
			/////////////////////////////////////////////

			mathtype_solver psi0 = Idt+Idx*du0+Idy*dv0+Idz*dw0;
			psi0 *= psi0;

			__syncthreads();
			psi[pos] = psi0;

			if (spatiotemporalderivative_id < 0)
			{
				psi[nstack+pos] = Idx;
				psi[(2*nstack)+pos] = Idy;
				psi[(3*nstack)+pos] = Idz;
				psi[(4*nstack)+pos] = Idt;
			}

			return;
		}

		__global__ void zeroinitialize(optflow_type *u, optflow_type *du, optflow_type *confidence)
		{
			int ndim = gpu_const::ndim_c;
			idx_type nstack = gpu_const::nstack_c;
			bool use_confidencemap = gpu_const::use_confidencemap_c;

			idx_type pos = (blockIdx.x*blockDim.x+threadIdx.x);
			if (pos >= nstack) pos = threadIdx.x;
			__syncthreads();

			if(ndim > 2)
			{
				u[pos] = 0.0f;
				u[pos+nstack] = 0.0f;
				u[pos+2*nstack] = 0.0f;
				du[pos] = 0.0f;
				du[pos+nstack] = 0.0f;
				du[pos+2*nstack] = 0.0f;
			}
			else
			{
				u[pos] = 0.0f;
				u[pos+nstack] = 0.0f;
				du[pos] = 0.0f;
				du[pos+nstack] = 0.0f;
			}
			if (use_confidencemap)
			{
				confidence[pos] = 1.0f;
			}

			return;
		}
		__global__ void reset_du(optflow_type *du)
		{
			int ndim = gpu_const::ndim_c;
			idx_type nstack = gpu_const::nstack_c;

			idx_type pos = (blockIdx.x*blockDim.x+threadIdx.x);
			if (pos >= nstack) pos = threadIdx.x;
			__syncthreads();

			if(ndim > 2)
			{
				du[pos] = 0.0f;
				du[pos+nstack] = 0.0f;
				du[pos+2*nstack] = 0.0f;
			}
			else
			{
				du[pos] = 0.0f;
				du[pos+nstack] = 0.0f;
			}
			return;
		}
	}

	int OptFlow_GPU3D::configure_device(int maxshape[3], ProtocolParameters *params){

		deviceID = params->gpu.deviceID;
		cudaSetDevice(deviceID);

		idx_type ndim = 3;
		bool use_confidencemap = params->confidence.use_confidencemap;
		idx_type nstack = maxshape[0]*maxshape[1];
		nstack *= maxshape[2];

		mathtype_solver epsilon_phi_squared = params->smoothness.epsilon_phi;
		epsilon_phi_squared *= epsilon_phi_squared;
		mathtype_solver epsilon_psi_squared = params->solver.epsilon_psi;
		epsilon_psi_squared *= epsilon_psi_squared;
		float nanf = std::numeric_limits<float>::quiet_NaN();

		int outOfBounds_id = 0;
		int warp_interpolation_id = 0;
		int spatiotemporalderivative_id = 0;

			 if (params->warp.outOfBounds_mode == "replace") outOfBounds_id = 0;
		else if (params->warp.outOfBounds_mode == "NaN") outOfBounds_id = 1;
		else std::cout << "Warning! Unknown outOfBounds_mode!" << std::endl;

		     if (params->warp.interpolation_mode == "linear") warp_interpolation_id = 0;
		else if (params->warp.interpolation_mode == "cubic")  warp_interpolation_id = 1;
		else std::cout << "Warning! Unknow warp interpolation mode!" << std::endl;

			 if (params->solver.spatiotemporalDerivative_type == "HornSchunck") spatiotemporalderivative_id = 1;
		else if (params->solver.spatiotemporalDerivative_type == "Ershov") spatiotemporalderivative_id = 2;
		else if (params->solver.spatiotemporalDerivative_type == "centraldifference") spatiotemporalderivative_id = 2;
		else if (params->solver.spatiotemporalDerivative_type == "Barron") spatiotemporalderivative_id = 3;
		else if (params->solver.spatiotemporalDerivative_type == "Farid3") spatiotemporalderivative_id = 4;
		else if (params->solver.spatiotemporalDerivative_type == "Farid5") spatiotemporalderivative_id = 5;
		else if (params->solver.spatiotemporalDerivative_type == "Farid7") spatiotemporalderivative_id = 6;
		else if (params->solver.spatiotemporalDerivative_type == "Farid9") spatiotemporalderivative_id = 7;
		else {std::cout << "Warning! Unknown spatiotemporal derivative type!" << std::endl;}

		//identify precalculated spatiotemporal derivatives:
		if (params->solver.precalculate_derivatives) spatiotemporalderivative_id *= -1;

		bool anisotropic_smoothness = params->smoothness.anisotropic_smoothness;
		bool decoupled_smoothness = params->smoothness.decoupled_smoothness;
		bool adaptive_smoothness = params->smoothness.adaptive_smoothness;
		bool complementary_smoothness = params->smoothness.complementary_smoothness;

		int slip_depth = params->confidence.slip_depth;

		//check memory requirements
		////////////////////////////////////////////////////
		size_t free_byte, total_byte ;
		cudaMemGetInfo( &free_byte, &total_byte ) ;

		int n_optflow = 7;
		int n_img = 2;

		double free_db = (double)free_byte ;
		double expected_usage = 7.*nstack *sizeof(optflow_type);
		expected_usage += 2.*nstack *sizeof(img_type);
		if(params->confidence.use_confidencemap) {expected_usage += nstack *sizeof(optflow_type); n_optflow++;}
		if (params->solver.precalculate_derivatives) {expected_usage += (5*nstack)*sizeof(optflow_type); n_optflow+=5;}
		else if(params->solver.precalculate_psi) {expected_usage += nstack *sizeof(optflow_type); n_optflow++;}
		if(params->warp.rewarp_frame1 == false) {expected_usage += nstack *sizeof(img_type); n_img++;}
		if(params->smoothness.decoupled_smoothness) {expected_usage += (2*nstack) *sizeof(optflow_type); n_optflow+=2;}
		if(params->smoothness.adaptive_smoothness)  {expected_usage += (2*nstack) *sizeof(optflow_type); n_optflow+=2;}
		if(params->solver.precalculate_derivatives && params->special.localglobal_dataterm && !params->smoothness.decoupled_smoothness) {expected_usage += (4*nstack) *sizeof(optflow_type); n_optflow+=4;}
		else if (params->solver.precalculate_derivatives && params->special.localglobal_dataterm && params->smoothness.decoupled_smoothness) {expected_usage += (2*nstack) *sizeof(optflow_type); n_optflow+=2;}

		if (params->mosaicing.mosaic_decomposition && params->mosaicing.max_nstack == -1)
		{
			params->mosaicing.max_nstack = (free_db-(params->mosaicing.memory_buffer*1024*1024))/(n_optflow*sizeof(optflow_type)+n_img*sizeof(img_type));

			if (nstack > params->mosaicing.max_nstack)
			{
				//set nstack to available memory
				expected_usage = expected_usage/nstack*params->mosaicing.max_nstack;
				nstack = params->mosaicing.max_nstack;
				std::cout << "\033[1;31mmax allowed nstack: " << nstack << "\033[0m" << std::endl;
				std::cout << "\033[1;32mGPU memory: " << round(expected_usage/(1024.*1024.)) << " MB out of " << round(free_db/(1024.*1024.)) << " MB\033[0m" << std::endl;
			}
			else
			{
				//deactivate
				params->mosaicing.mosaic_decomposition = false;
				std::cout << "\033[1;32mGPU memory: " << round(expected_usage/(1024.*1024.)) << " MB out of " << round(free_db/(1024.*1024.)) << " MB\033[0m" << std::endl;
			}
		}
		else
		{
			if (expected_usage > free_db){std::cout << "\033[1;31mError! Expected to run out of GPU memory! " <<
				round(expected_usage/(1024.*1024.)) << " MB out of " << round(free_db/(1024.*1024.)) << " MB required\033[0m" << std::endl;
				return 2;
			}
			else std::cout << "\033[1;32mGPU memory: " << round(expected_usage/(1024.*1024.)) << " MB out of " << round(free_db/(1024.*1024.)) << " MB\033[0m" << std::endl;
		}
		////////////////////////////////////////////////////

		if (params->mosaicing.mosaic_decomposition && params->mosaicing.sequential_approximation == false && params->gpu.n_gpus == 1)
			params->warp.rewarp_frame1 = true; //no reason to keep frame1 in GPU memory (with single GPU)


		//allocate memory and set constant memory
		////////////////////////////////////////////////////
		(optflow_type*) cudaMalloc((void**)&u, (ndim*nstack)*sizeof(*u));
		(optflow_type*) cudaMalloc((void**)&du, (ndim*nstack)*sizeof(*du));
		if(params->solver.precalculate_derivatives && params->special.localglobal_dataterm) cudaMalloc((void**)&phi, (5*nstack)*sizeof(*phi));
		else if(params->smoothness.decoupled_smoothness) (optflow_type*) cudaMalloc((void**)&phi, (3*nstack)*sizeof(*phi));
		else (optflow_type*) cudaMalloc((void**)&phi, nstack*sizeof(*phi));
		//confidence map or background mask:
		if(params->confidence.use_confidencemap) (optflow_type*) cudaMalloc((void**)&confidence, nstack*sizeof(*confidence));
		else (optflow_type*) cudaMalloc((void**)&confidence, 0);
		if (params->solver.precalculate_derivatives) (optflow_type*) cudaMalloc((void**)&psi, (5*nstack)*sizeof(*psi));
		else if(params->solver.precalculate_psi) (optflow_type*) cudaMalloc((void**)&psi, nstack*sizeof(*psi));
		else (optflow_type*) cudaMalloc((void**)&psi, 0);
		if(params->smoothness.adaptive_smoothness) (optflow_type*) cudaMalloc((void**)&adaptivity, (2*nstack)*sizeof(*adaptivity));
		else (optflow_type*) cudaMalloc((void**)&adaptivity, 0);

		//using an extra copy to warp from source (rewarp would save a copy)
		(img_type*) cudaMalloc((void**)&warped1, nstack*sizeof(*warped1));
		(img_type*) cudaMalloc((void**)&dev_frame0, nstack*sizeof(*dev_frame0));
		if (params->warp.rewarp_frame1 == false) (img_type*) cudaMalloc((void**)&dev_frame1, nstack*sizeof(*dev_frame1));
		else (img_type*) cudaMalloc((void**)&dev_frame1, 0);

		cudaMemcpyToSymbol(gpu_const::ndim_c, &ndim, sizeof(gpu_const::ndim_c));
		cudaMemcpyToSymbol(gpu_const::nstack_c, &nstack, sizeof(gpu_const::nstack_c));
		cudaMemcpyToSymbol(gpu_const::use_confidencemap_c, &use_confidencemap, sizeof(gpu_const::use_confidencemap_c));
		cudaMemcpyToSymbol(gpu_const::epsilon_phi_squared_c, &epsilon_phi_squared, sizeof(gpu_const::epsilon_phi_squared_c));
		cudaMemcpyToSymbol(gpu_const::epsilon_psi_squared_c, &epsilon_psi_squared, sizeof(gpu_const::epsilon_psi_squared_c));
		cudaMemcpyToSymbol(gpu_const::outOfBounds_id_c, &outOfBounds_id, sizeof(gpu_const::outOfBounds_id_c));
		cudaMemcpyToSymbol(gpu_const::warpInterpolation_id_c, &warp_interpolation_id, sizeof(gpu_const::warpInterpolation_id_c));
		cudaMemcpyToSymbol(gpu_const::spatiotemporalderivative_id_c, &spatiotemporalderivative_id, sizeof(gpu_const::spatiotemporalderivative_id_c));
		cudaMemcpyToSymbol(gpu_const::nanf_c, &nanf, sizeof(gpu_const::nanf_c));
		cudaMemcpyToSymbol(gpu_const::anisotropic_smoothness_c, &anisotropic_smoothness, sizeof(gpu_const::anisotropic_smoothness_c));
		cudaMemcpyToSymbol(gpu_const::decoupled_smoothness_c, &decoupled_smoothness, sizeof(gpu_const::decoupled_smoothness_c));
		cudaMemcpyToSymbol(gpu_const::adaptive_smoothness_c, &adaptive_smoothness, sizeof(gpu_const::adaptive_smoothness_c));
		cudaMemcpyToSymbol(gpu_const::complementary_smoothness_c, &complementary_smoothness, sizeof(gpu_const::complementary_smoothness_c));
		cudaMemcpyToSymbol(gpu_const::slip_depth_c, &slip_depth, sizeof(gpu_const::slip_depth_c));
		cudaDeviceSynchronize();
		////////////////////////////////////////////////////

		//Initialize arrays
		////////////////////////////////////////////////////
		int threadsPerBlock(params->gpu.threadsPerBlock);
		int blocksPerGrid = (nstack + threadsPerBlock - 1) / (threadsPerBlock);

		gpu3d::zeroinitialize<<<blocksPerGrid,threadsPerBlock>>>(u,du,confidence);
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
	void OptFlow_GPU3D::free_device(){
		cudaSetDevice(deviceID);

		cudaFree(u);
		cudaFree(du);
		cudaFree(phi);
		cudaFree(psi);
		cudaFree(confidence);
		cudaFree(dev_frame0);
		cudaFree(dev_frame1);
		cudaFree(warped1);
		cudaFree(adaptivity);
	}

	void OptFlow_GPU3D::run_outeriterations(int level, img_type *frame0, img_type *frame1, int shape[3], ProtocolParameters *params, bool resumed_state, bool frames_set)
	{
		cudaSetDevice(deviceID);

		//Set constant memory
		////////////////////////////////////////////////////////////////////////////////////////
		int nx = shape[0]; int ny = shape[1]; int nz = shape[2];
		idx_type nslice = shape[0]*shape[1];
		idx_type nstack = shape[2]*nslice;
		idx_type n_odd = nstack/2;
		idx_type n_even = nstack-n_odd;

		int threadsPerBlock(params->gpu.threadsPerBlock);
		int blocksPerGrid = (nstack + threadsPerBlock - 1) / (threadsPerBlock);
		int blocksPerGrid2 = (n_even + threadsPerBlock -1) / (threadsPerBlock); //iterate over every second voxel

		idx_type asize1 = nstack*sizeof(*dev_frame0);

		mathtype_solver hx = params->scaling.hx;
		mathtype_solver hy = params->scaling.hy;
		mathtype_solver hz = params->scaling.hy;
		mathtype_solver alpha = params->alpha;
		mathtype_solver omega = params->solver.sor_omega;
		bool precalculate_psi = params->solver.precalculate_psi;
		float localglobal_sigma_data = params->special.localglobal_sigma_data;
		bool rewarp = params->warp.rewarp_frame1;
		bool use_confidencemap = params->confidence.use_confidencemap;
		bool protect_overlap = params->mosaicing.protect_overlap;
		int overlap = params->mosaicing.overlap;

		if (params->pyramid.alpha_scaling)
			alpha = alpha/pow(params->pyramid.scaling_factor, level);

		int smoothness_id = 0;

			 if (params->solver.flowDerivative_type == "Barron") smoothness_id = 0;
		else if (params->solver.flowDerivative_type == "centraldifference") smoothness_id = 1; //Ershov style
		else if (params->solver.flowDerivative_type == "forwarddifference") smoothness_id = 2; //Liu style
		else if (params->solver.flowDerivative_type == "Farid3") smoothness_id = 3; //3x3x3 Farid kernel
		else if (params->solver.flowDerivative_type == "Farid5") smoothness_id = 4; //5x5x5 Farid kernel
		else if (params->solver.flowDerivative_type == "Farid7") smoothness_id = 5; //7x7x7 Farid kernel
		else if (params->solver.flowDerivative_type == "Farid9") smoothness_id = 6; //9x9x9 Farid kernel
		else std::cout << "Warning! Unknown flowDerivative_type!" << std::endl;

		if (!resumed_state)
		{
			cudaMemcpyToSymbol(gpu_const::nx_c, &nx, sizeof(gpu_const::nx_c));
			cudaMemcpyToSymbol(gpu_const::ny_c, &ny, sizeof(gpu_const::ny_c));
			cudaMemcpyToSymbol(gpu_const::nz_c, &nz, sizeof(gpu_const::nz_c));
			cudaMemcpyToSymbol(gpu_const::nstack_c, &nstack, sizeof(gpu_const::nstack_c));
			cudaMemcpyToSymbol(gpu_const::hx_c, &hx, sizeof(gpu_const::hx_c));
			cudaMemcpyToSymbol(gpu_const::hy_c, &hy, sizeof(gpu_const::hy_c));
			cudaMemcpyToSymbol(gpu_const::hz_c, &hz, sizeof(gpu_const::hz_c));
			cudaMemcpyToSymbol(gpu_const::alpha_c, &alpha, sizeof(gpu_const::alpha_c));
			cudaMemcpyToSymbol(gpu_const::omega_c, &omega, sizeof(gpu_const::omega_c));
			cudaMemcpyToSymbol(gpu_const::zeroDirichletBoundary_c, &params->constraint.zeroDirichletBoundary,  6*sizeof(int), 0);
			cudaMemcpyToSymbol(gpu_const::fixedDirichletBoundary_c, &params->constraint.fixedDirichletBoundary,  6*sizeof(int), 0);
			cudaMemcpyToSymbol(gpu_const::lowerIntensityCutoff_c, &(params->constraint.intensityRange[0]), sizeof(gpu_const::lowerIntensityCutoff_c));
			cudaMemcpyToSymbol(gpu_const::upperIntensityCutoff_c, &(params->constraint.intensityRange[1]), sizeof(gpu_const::upperIntensityCutoff_c));
			cudaMemcpyToSymbol(gpu_const::use_confidencemap_c, &use_confidencemap, sizeof(gpu_const::use_confidencemap_c));
			cudaMemcpyToSymbol(gpu_const::precalculated_psi_c, &precalculate_psi, sizeof(gpu_const::precalculated_psi_c));
			cudaMemcpyToSymbol(gpu_const::filter_sigma_c, &localglobal_sigma_data, sizeof(gpu_const::filter_sigma_c));
			cudaMemcpyToSymbol(gpu_const::protect_overlap_c, &protect_overlap, sizeof(gpu_const::protect_overlap_c));
			cudaMemcpyToSymbol(gpu_const::overlap_c, &overlap, sizeof(gpu_const::overlap_c));

			if(!frames_set)
			{
				cudaMemcpy(dev_frame0, frame0, asize1, cudaMemcpyHostToDevice);
				if(!rewarp) cudaMemcpy(dev_frame1, frame1, asize1, cudaMemcpyHostToDevice);
				else cudaMemcpy(warped1, frame1, asize1, cudaMemcpyHostToDevice);
				cudaDeviceSynchronize();
			}

			std::string error_string = (std::string) cudaGetErrorString(cudaGetLastError());
			if (error_string != "no error")
			{
				std::cout << "Device Variable Copying: " << error_string << std::endl;
				return;
			}
			////////////////////////////////////////////////////////////////////////////////////////

			//initial warp for frame 1
			if(!rewarp){
				gpu3d::addsolution_warpFrame1_z<<<blocksPerGrid,threadsPerBlock>>>(false, phi, dev_frame0, dev_frame1, u, du);
				cudaDeviceSynchronize();
				gpu3d::addsolution_warpFrame1_xy<<<blocksPerGrid,threadsPerBlock>>>(false, warped1, dev_frame0, phi, u, du, confidence);
				cudaDeviceSynchronize();
			}
			else{
				gpu3d::addsolution_warpFrame1_z<<<blocksPerGrid,threadsPerBlock>>>(false, phi, dev_frame0, warped1, u, du);
				cudaDeviceSynchronize();
				gpu3d::addsolution_warpFrame1_xy<<<blocksPerGrid,threadsPerBlock>>>(false, warped1, dev_frame0, phi, u, du, confidence);
				cudaDeviceSynchronize();
			}
		}

		for (int i_outer = 0; i_outer < params->solver.outerIterations; i_outer++)
		{
			std::cout << "level " << level << " (" << nx << "," << ny << "," << nz << "): " << (i_outer+1) << " \r";
			std::cout.flush();

			for (int i_inner = 0; i_inner < params->solver.innerIterations; i_inner++)
			{
				if (params->special.localglobal_dataterm)
				{
					gpu3d::update_dataterm<<<blocksPerGrid,threadsPerBlock>>>(dev_frame0, warped1, du, phi);
					cudaDeviceSynchronize();

					//could be reduced to 4 in the next version without storing psi0
					int maxoffset = params->solver.precalculate_derivatives ? 5 : 1; //needs to be always 5 or it's not proper local-global since the Tensor needs to be blurred
					if(maxoffset == 1) std::cout << "maxoffset needs to be 5!" << std::endl;

					if (params->special.localglobal_mode == "Farid")
					{
						int farid_radius = (int) std::max(0.f, std::min(4.f, params->special.localglobal_sigma_data));

						for (long long int offset = 0; offset < maxoffset*nstack; offset += nstack)
						{
						gpu3d::faridinterpolation3D<<<blocksPerGrid,threadsPerBlock>>>(0, phi+offset, psi+offset, farid_radius);
						cudaDeviceSynchronize();
						gpu3d::faridinterpolation3D<<<blocksPerGrid,threadsPerBlock>>>(1, psi+offset, phi+offset, farid_radius);
						cudaDeviceSynchronize();
						gpu3d::faridinterpolation3D<<<blocksPerGrid,threadsPerBlock>>>(2, phi+offset, psi+offset, farid_radius);
						cudaDeviceSynchronize();
						}
					}
					else
					{
						for (long long int offset = 0; offset < maxoffset*nstack; offset += nstack)
						{
						gpu3d::gaussianfilter3D_x<<<blocksPerGrid,threadsPerBlock>>>(phi+offset, psi+offset);
						cudaDeviceSynchronize();
						gpu3d::gaussianfilter3D_y<<<blocksPerGrid,threadsPerBlock>>>(psi+offset, phi+offset);
						cudaDeviceSynchronize();
						gpu3d::gaussianfilter3D_z<<<blocksPerGrid,threadsPerBlock>>>(phi+offset, psi+offset);
						cudaDeviceSynchronize();
						}
					}
				}
				else if(precalculate_psi)
				{
					gpu3d::update_dataterm<<<blocksPerGrid,threadsPerBlock>>>(dev_frame0, warped1, du, psi);
					//cudaDeviceSynchronize();
				}

				//Calculate the smoothness term
				//////////////////////////////////////////////////////////////////////////////
				if      (smoothness_id == 0) gpu3d::update_smoothnessterm_Barron<<<blocksPerGrid,threadsPerBlock>>>(u, du,  phi, adaptivity);
				else if (smoothness_id == 1) gpu3d::update_smoothnessterm_centralDiff<<<blocksPerGrid,threadsPerBlock>>>(u, du,  phi, adaptivity);
				else if (smoothness_id == 2) gpu3d::update_smoothnessterm_forwardDiff<<<blocksPerGrid,threadsPerBlock>>>(u, du,  phi, adaptivity);
				else if (smoothness_id == 3) gpu3d::update_smoothnessterm_Farid3<<<blocksPerGrid,threadsPerBlock>>>(u, du,  phi, adaptivity);
				else if (smoothness_id == 4) gpu3d::update_smoothnessterm_Farid5<<<blocksPerGrid,threadsPerBlock>>>(u, du,  phi, adaptivity);
				else if (smoothness_id == 5) gpu3d::update_smoothnessterm_Farid7<<<blocksPerGrid,threadsPerBlock>>>(u, du,  phi, adaptivity);
				else if (smoothness_id == 6) gpu3d::update_smoothnessterm_Farid9<<<blocksPerGrid,threadsPerBlock>>>(u, du,  phi, adaptivity);
				cudaDeviceSynchronize();
				//////////////////////////////////////////////////////////////////////////////

				//SOR-Updates with psi calculated on the fly
				//////////////////////////////////////////////////////////////////////////////
				//switching between even and odd
				for (int i_sor = 0; i_sor < 2*params->solver.sorIterations; i_sor++)
				{
					//if(omega > 1.0) std::cout << "SOR still unstable! Is it the testset without gradient? Try flipping forward sweep and backward sweep" << std::endl;

					//reset on first sor is optional... deprecated for now, previous result is a better guess
					gpu3d::calculate_sorUpdate<<<blocksPerGrid2,threadsPerBlock>>>(i_sor, dev_frame0, warped1, phi, psi, u, du, confidence);
					//cudaDeviceSynchronize();
				}
				//////////////////////////////////////////////////////////////////////////////
			}

			if(!rewarp){
				gpu3d::addsolution_warpFrame1_z<<<blocksPerGrid,threadsPerBlock>>>(false, phi, dev_frame0, dev_frame1, u, du);
				cudaDeviceSynchronize();
				gpu3d::addsolution_warpFrame1_xy<<<blocksPerGrid,threadsPerBlock>>>(false, warped1, dev_frame0, phi, u, du, confidence);
				cudaDeviceSynchronize();
			}
			else{
				gpu3d::addsolution_warpFrame1_z<<<blocksPerGrid,threadsPerBlock>>>(true, phi, dev_frame0, warped1, u, du);
				cudaDeviceSynchronize();
				gpu3d::addsolution_warpFrame1_xy<<<blocksPerGrid,threadsPerBlock>>>(true, warped1, dev_frame0, phi, u, du, confidence);
				cudaDeviceSynchronize();
			}
		}

		return;
	}
	void OptFlow_GPU3D::run_singleiteration(int level, img_type *frame0, img_type *frame1, int shape[3], ProtocolParameters *params, bool frames_set)
	{
		cudaSetDevice(deviceID);

		//Set constant memory
		////////////////////////////////////////////////////////////////////////////////////////
		int nx = shape[0]; int ny = shape[1]; int nz = shape[2];
		idx_type nslice = shape[0]*shape[1];
		idx_type nstack = shape[2]*nslice;
		idx_type n_odd = nstack/2;
		idx_type n_even = nstack-n_odd;

		int threadsPerBlock(params->gpu.threadsPerBlock);
		int blocksPerGrid = (nstack + threadsPerBlock - 1) / (threadsPerBlock);
		int blocksPerGrid2 = (n_even + threadsPerBlock -1) / (threadsPerBlock); //iterate over every second voxel

		idx_type asize1 = nstack*sizeof(*dev_frame0);

		mathtype_solver hx = params->scaling.hx;
		mathtype_solver hy = params->scaling.hy;
		mathtype_solver hz = params->scaling.hy;
		mathtype_solver alpha = params->alpha;
		mathtype_solver omega = params->solver.sor_omega;
		bool precalculate_psi = params->solver.precalculate_psi;
		float localglobal_sigma_data = params->special.localglobal_sigma_data;
		bool use_confidencemap = params->confidence.use_confidencemap;
		bool protect_overlap = params->mosaicing.protect_overlap;
		int overlap = params->mosaicing.overlap;

		if (params->pyramid.alpha_scaling)
			alpha = alpha/pow(params->pyramid.scaling_factor, level);

		int smoothness_id = 0;

			 if (params->solver.flowDerivative_type == "Barron") smoothness_id = 0;
		else if (params->solver.flowDerivative_type == "centraldifference") smoothness_id = 1; //Ershov style
		else if (params->solver.flowDerivative_type == "forwarddifference") smoothness_id = 2; //Liu style
		else if (params->solver.flowDerivative_type == "Farid3") smoothness_id = 3; //3x3x3 Farid kernel
		else if (params->solver.flowDerivative_type == "Farid5") smoothness_id = 4; //5x5x5 Farid kernel
		else if (params->solver.flowDerivative_type == "Farid7") smoothness_id = 5; //7x7x7 Farid kernel
		else if (params->solver.flowDerivative_type == "Farid9") smoothness_id = 6; //9x9x9 Farid kernel
		else std::cout << "Warning! Unknown flowDerivative_type!" << std::endl;

		cudaMemcpyToSymbol(gpu_const::nx_c, &nx, sizeof(gpu_const::nx_c));
		cudaMemcpyToSymbol(gpu_const::ny_c, &ny, sizeof(gpu_const::ny_c));
		cudaMemcpyToSymbol(gpu_const::nz_c, &nz, sizeof(gpu_const::nz_c));
		cudaMemcpyToSymbol(gpu_const::nstack_c, &nstack, sizeof(gpu_const::nstack_c));
		cudaMemcpyToSymbol(gpu_const::hx_c, &hx, sizeof(gpu_const::hx_c));
		cudaMemcpyToSymbol(gpu_const::hy_c, &hy, sizeof(gpu_const::hy_c));
		cudaMemcpyToSymbol(gpu_const::hz_c, &hz, sizeof(gpu_const::hz_c));
		cudaMemcpyToSymbol(gpu_const::alpha_c, &alpha, sizeof(gpu_const::alpha_c));
		cudaMemcpyToSymbol(gpu_const::omega_c, &omega, sizeof(gpu_const::omega_c));
		cudaMemcpyToSymbol(gpu_const::zeroDirichletBoundary_c, &params->constraint.zeroDirichletBoundary,  6*sizeof(int), 0);
		cudaMemcpyToSymbol(gpu_const::fixedDirichletBoundary_c, &params->constraint.fixedDirichletBoundary,  6*sizeof(int), 0);
		cudaMemcpyToSymbol(gpu_const::lowerIntensityCutoff_c, &(params->constraint.intensityRange[0]), sizeof(gpu_const::lowerIntensityCutoff_c));
		cudaMemcpyToSymbol(gpu_const::upperIntensityCutoff_c, &(params->constraint.intensityRange[1]), sizeof(gpu_const::upperIntensityCutoff_c));
		cudaMemcpyToSymbol(gpu_const::use_confidencemap_c, &use_confidencemap, sizeof(gpu_const::use_confidencemap_c));
		cudaMemcpyToSymbol(gpu_const::precalculated_psi_c, &precalculate_psi, sizeof(gpu_const::precalculated_psi_c));
		cudaMemcpyToSymbol(gpu_const::filter_sigma_c, &localglobal_sigma_data, sizeof(gpu_const::filter_sigma_c));
		cudaMemcpyToSymbol(gpu_const::protect_overlap_c, &protect_overlap, sizeof(gpu_const::protect_overlap_c));
		cudaMemcpyToSymbol(gpu_const::overlap_c, &overlap, sizeof(gpu_const::overlap_c));

		if(!frames_set)
		{
			cudaMemcpy(dev_frame0, frame0, asize1, cudaMemcpyHostToDevice);
			cudaMemcpy(warped1, frame1, asize1, cudaMemcpyHostToDevice);
		}
		cudaDeviceSynchronize();

		std::string error_string = (std::string) cudaGetErrorString(cudaGetLastError());
		if (error_string != "no error")
		{
			std::cout << "Device Variable Copying: " << error_string << std::endl;
			return;
		}
		////////////////////////////////////////////////////////////////////////////////////////

		//initial warp for frame 1
		gpu3d::addsolution_warpFrame1_z<<<blocksPerGrid,threadsPerBlock>>>(false, phi, dev_frame0, warped1, u, du);
		cudaDeviceSynchronize();
		gpu3d::addsolution_warpFrame1_xy<<<blocksPerGrid,threadsPerBlock>>>(false, warped1, dev_frame0, phi, u, du, confidence);
		cudaDeviceSynchronize();

		for (int i_inner = 0; i_inner < params->solver.innerIterations; i_inner++)
		{
			if (params->special.localglobal_dataterm)
			{
				gpu3d::update_dataterm<<<blocksPerGrid,threadsPerBlock>>>(dev_frame0, warped1, du, phi);
				cudaDeviceSynchronize();

				//could be reduced to 4 in the next version without storing psi0
				int maxoffset = params->solver.precalculate_derivatives ? 5 : 1; //needs to be always 5 or it's not proper local-global since the Tensor needs to be blurred

				if (params->special.localglobal_mode == "Farid")
				{
					int farid_radius = (int) std::max(0.f, std::min(4.f, params->special.localglobal_sigma_data));
					if(farid_radius == 4) std::cout << "Farid radius 4 might be buggy. Needs checking!" << std::endl;

					for (long long int offset = 0; offset < maxoffset*nstack; offset += nstack)
					{
					gpu3d::faridinterpolation3D<<<blocksPerGrid,threadsPerBlock>>>(0, phi+offset, psi+offset, farid_radius);
					cudaDeviceSynchronize();
					gpu3d::faridinterpolation3D<<<blocksPerGrid,threadsPerBlock>>>(1, psi+offset, phi+offset, farid_radius);
					cudaDeviceSynchronize();
					gpu3d::faridinterpolation3D<<<blocksPerGrid,threadsPerBlock>>>(2, phi+offset, psi+offset, farid_radius);
					cudaDeviceSynchronize();
					}
				}
				else
				{
					for (long long int offset = 0; offset < maxoffset*nstack; offset += nstack)
					{
					gpu3d::gaussianfilter3D_x<<<blocksPerGrid,threadsPerBlock>>>(phi+offset, psi+offset);
					cudaDeviceSynchronize();
					gpu3d::gaussianfilter3D_y<<<blocksPerGrid,threadsPerBlock>>>(psi+offset, phi+offset);
					cudaDeviceSynchronize();
					gpu3d::gaussianfilter3D_z<<<blocksPerGrid,threadsPerBlock>>>(phi+offset, psi+offset);
					cudaDeviceSynchronize();
					}
				}
			}
			else if(precalculate_psi)
			{
				gpu3d::update_dataterm<<<blocksPerGrid,threadsPerBlock>>>(dev_frame0, warped1, du, psi);
				//cudaDeviceSynchronize();
			}

			//Calculate the smoothness term
			//////////////////////////////////////////////////////////////////////////////
			if      (smoothness_id == 0) gpu3d::update_smoothnessterm_Barron<<<blocksPerGrid,threadsPerBlock>>>(u, du,  phi, adaptivity);
			else if (smoothness_id == 1) gpu3d::update_smoothnessterm_centralDiff<<<blocksPerGrid,threadsPerBlock>>>(u, du,  phi, adaptivity);
			else if (smoothness_id == 2) gpu3d::update_smoothnessterm_forwardDiff<<<blocksPerGrid,threadsPerBlock>>>(u, du,  phi, adaptivity);
			else if (smoothness_id == 3) gpu3d::update_smoothnessterm_Farid3<<<blocksPerGrid,threadsPerBlock>>>(u, du,  phi, adaptivity);
			else if (smoothness_id == 4) gpu3d::update_smoothnessterm_Farid5<<<blocksPerGrid,threadsPerBlock>>>(u, du,  phi, adaptivity);
			else if (smoothness_id == 5) gpu3d::update_smoothnessterm_Farid7<<<blocksPerGrid,threadsPerBlock>>>(u, du,  phi, adaptivity);
			else if (smoothness_id == 6) gpu3d::update_smoothnessterm_Farid9<<<blocksPerGrid,threadsPerBlock>>>(u, du,  phi, adaptivity);
			cudaDeviceSynchronize();

			//////////////////////////////////////////////////////////////////////////////

			//SOR-Updates with psi calculated on the fly
			//////////////////////////////////////////////////////////////////////////////
			//switching between even and odd
			for (int i_sor = 0; i_sor < 2*params->solver.sorIterations; i_sor++)
			{
				//if(omega > 1.0) std::cout << "SOR still unstable! Is it the testset without gradient? Try flipping forward sweep and backward sweep" << std::endl;

				//reset on first sor is optional... deprecated for now, previous result is a better guess
				gpu3d::calculate_sorUpdate<<<blocksPerGrid2,threadsPerBlock>>>(i_sor, dev_frame0, warped1, phi, psi, u, du, confidence);
				//cudaDeviceSynchronize();
			}
			//////////////////////////////////////////////////////////////////////////////
		}

		gpu3d::addsolution<<<blocksPerGrid,threadsPerBlock>>>(u, du);
		cudaDeviceSynchronize();

		return;
	}

	void OptFlow_GPU3D::set_frames(float* frame0, float *frame1, int shape[3], std::vector<int> &boundaries, bool rewarp)
	{
		cudaSetDevice(deviceID);

		int nx = shape[0]; int ny = shape[1];
		long long int nslice_full = nx*ny;

		int nx_target = (boundaries[3]-boundaries[0]);
		int ny_target = (boundaries[4]-boundaries[1]);
		int nz_target = (boundaries[5]-boundaries[2]);
		idx_type nslice_target = nx_target*ny_target;
		idx_type nstack_target = nz_target*nslice_target;

		if (nx_target == nx && ny_target == ny)
		{
			//preferable case
			idx_type offset = nslice_full*boundaries[2];
			idx_type asize1 = nstack_target*sizeof(*dev_frame0);

			cudaMemcpyAsync(dev_frame0, frame0 + offset, asize1, cudaMemcpyHostToDevice);
			if(!rewarp) cudaMemcpyAsync(dev_frame1, frame1 + offset, asize1, cudaMemcpyHostToDevice);
			else		cudaMemcpyAsync(warped1, frame1 + offset, asize1, cudaMemcpyHostToDevice);
		}
		else if (nx_target == nx)
		{
			//need to loop over slices
			idx_type asize1 = nslice_target*sizeof(*dev_frame0);

			for (int z = 0; z < nz_target; z++)
			{
				idx_type offset_target = z*nslice_target;
				idx_type offset_source = (z+boundaries[2])*nslice_full+boundaries[1]*nx;

				cudaMemcpyAsync(dev_frame0+offset_target, frame0+offset_source, asize1, cudaMemcpyHostToDevice);
				if(!rewarp) cudaMemcpyAsync(dev_frame1+offset_target, frame1+offset_source, asize1, cudaMemcpyHostToDevice);
				else cudaMemcpyAsync(warped1+offset_target, frame1+offset_source, asize1, cudaMemcpyHostToDevice);
			}
		}
		else
		{
			//worst case
			idx_type asize1 = nx_target*sizeof(*dev_frame0);

			for (int z = 0; z < nz_target; z++)
			{
				for (int y = 0; y < ny_target; y++)
				{
					idx_type offset_target = y*nx_target + z*nslice_target;
					idx_type offset_source = (z+boundaries[2])*nslice_full+(y+boundaries[1])*nx+boundaries[0];

					cudaMemcpyAsync(dev_frame0+offset_target, frame0+offset_source, asize1, cudaMemcpyHostToDevice);
					if(!rewarp) cudaMemcpyAsync(dev_frame1+offset_target, frame1+offset_source, asize1, cudaMemcpyHostToDevice);
					else cudaMemcpyAsync(warped1+offset_target, frame1+offset_source, asize1, cudaMemcpyHostToDevice);
				}
			}
		}
		cudaDeviceSynchronize();
		return;
	}
	void OptFlow_GPU3D::set_flowvector(float* in_vector, int shape[3])
	{
		idx_type nslice = shape[0]*shape[1];
		idx_type nstack = nslice*shape[2];
		idx_type asize1 = 3*nstack*sizeof(*u);
		optflow_type *u_tmp;

		if(typeid(float) != typeid(optflow_type))
		{
			u_tmp = (optflow_type*) malloc(3*nstack*sizeof(*u_tmp));

			#pragma omp parallel for
			for (idx_type pos = 0; pos < nstack; pos++)
			{
				u_tmp[pos] = in_vector[pos];
				u_tmp[pos+nstack] = in_vector[pos+nstack];
				u_tmp[pos+2*nstack] = in_vector[pos+2*nstack];
			}
		}
		else
			u_tmp = in_vector;

		cudaSetDevice(deviceID);
		cudaMemcpy(u, u_tmp, asize1, cudaMemcpyHostToDevice);
		cudaDeviceSynchronize();

		return;
	}
	void OptFlow_GPU3D::set_flowvector(float* in_vector, int shape[3], std::vector<int> &boundaries)
	{
		cudaSetDevice(deviceID);

		int nx = shape[0]; int ny = shape[1]; int nz = shape[2];
		long long int nslice_full = nx*ny;
		long long int nstack_full = nz*nslice_full;

		int nx_target = (boundaries[3]-boundaries[0]);
		int ny_target = (boundaries[4]-boundaries[1]);
		int nz_target = (boundaries[5]-boundaries[2]);
		idx_type nslice_target = nx_target*ny_target;
		idx_type nstack_target = nz_target*nslice_target;


		if (nx_target == nx && ny_target == ny)
		{
			//preferable case
			idx_type offset = nslice_full*boundaries[2];
			idx_type asize1 = nstack_target*sizeof(*u);

			cudaMemcpyAsync(u, in_vector + offset, asize1, cudaMemcpyHostToDevice);
			cudaMemcpyAsync(u+nstack_target, in_vector + offset + nstack_full, asize1, cudaMemcpyHostToDevice);
			cudaMemcpyAsync(u+(2*nstack_target), in_vector + offset + (2*nstack_full), asize1, cudaMemcpyHostToDevice);
		}
		else if (nx_target == nx)
		{
			//need to loop over slices
			idx_type asize1 = nslice_target*sizeof(*u);

			for (int z = 0; z < nz_target; z++)
			{
				idx_type offset_target = z*nslice_target;
				idx_type offset_source = (z+boundaries[2])*nslice_full+boundaries[1]*nx;

				cudaMemcpyAsync(u+offset_target, in_vector+offset_source, asize1, cudaMemcpyHostToDevice);
				cudaMemcpyAsync(u+offset_target+nstack_target, in_vector+offset_source+nstack_full, asize1, cudaMemcpyHostToDevice);
				cudaMemcpyAsync(u+offset_target+(2*nstack_target), in_vector+offset_source+(2*nstack_full), asize1, cudaMemcpyHostToDevice);
			}
		}
		else
		{
			//worst case
			idx_type asize1 = nx_target*sizeof(*u);

			for (int z = 0; z < nz_target; z++)
			{
				for (int y = 0; y < ny_target; y++)
				{
					idx_type offset_target = y*nx_target + z*nslice_target;
					idx_type offset_source = (z+boundaries[2])*nslice_full+(y+boundaries[1])*nx+boundaries[0];

					cudaMemcpyAsync(u+offset_target, in_vector+offset_source, asize1, cudaMemcpyHostToDevice);
					cudaMemcpyAsync(u+offset_target+nstack_target, in_vector+offset_source+nstack_full, asize1, cudaMemcpyHostToDevice);
					cudaMemcpyAsync(u+offset_target+(2*nstack_target), in_vector+offset_source+(2*nstack_full), asize1, cudaMemcpyHostToDevice);
				}
			}
		}
		cudaDeviceSynchronize();
		return;
	}
	void OptFlow_GPU3D::set_confidencemap(float* confidencemap, int shape[3])
	{
		idx_type nslice = shape[0]*shape[1];
		idx_type nstack = nslice*shape[2];
		idx_type asize1 = nstack*sizeof(*confidence);
		optflow_type *c_tmp;

		if(typeid(float) != typeid(optflow_type))
		{
			c_tmp = (optflow_type*) malloc(nstack*sizeof(*c_tmp));

			#pragma omp parallel for
			for (idx_type pos = 0; pos < nstack; pos++)
			{
				c_tmp[pos] = confidencemap[pos];
			}
		}
		else
			c_tmp = confidencemap;

		cudaSetDevice(deviceID);
		cudaMemcpy(confidence, c_tmp, asize1, cudaMemcpyHostToDevice);
		cudaDeviceSynchronize();

		return;
	}
	void OptFlow_GPU3D::set_confidencemap(float* confidencemap, int shape[3], std::vector<int> &boundaries)
	{
		cudaSetDevice(deviceID);

		int nx = shape[0]; int ny = shape[1];
		long long int nslice_full = nx*ny;

		int nx_target = (boundaries[3]-boundaries[0]);
		int ny_target = (boundaries[4]-boundaries[1]);
		int nz_target = (boundaries[5]-boundaries[2]);
		idx_type nslice_target = nx_target*ny_target;
		idx_type nstack_target = nz_target*nslice_target;

		if (nx_target == nx && ny_target == ny)
		{
			//preferable case
			idx_type offset = nslice_full*boundaries[2];
			idx_type asize1 = nstack_target*sizeof(*confidence);

			cudaMemcpy(confidence, confidencemap + offset, asize1, cudaMemcpyHostToDevice);
		}
		else if (nx_target == nx)
		{
			//need to loop over slices
			idx_type asize1 = nslice_target*sizeof(*confidence);

			for (int z = 0; z < nz_target; z++)
			{
				idx_type offset_target = z*nslice_target;
				idx_type offset_source = (z+boundaries[2])*nslice_full+boundaries[1]*nx;

				cudaMemcpyAsync(confidence+offset_target, confidencemap+offset_source, asize1, cudaMemcpyHostToDevice);
			}
		}
		else
		{
			//worst case
			idx_type asize1 = nx_target*sizeof(*confidence);

			for (int z = 0; z < nz_target; z++)
			{
				for (int y = 0; y < ny_target; y++)
				{
					idx_type offset_target = y*nx_target + z*nslice_target;
					idx_type offset_source = (z+boundaries[2])*nslice_full+(y+boundaries[1])*nx+boundaries[0];

					cudaMemcpyAsync(confidence+offset_target, confidencemap+offset_source, asize1, cudaMemcpyHostToDevice);
				}
			}
		}
		cudaDeviceSynchronize();
		return;
	}
	void OptFlow_GPU3D::set_adaptivitymap(float* adaptivitymap, int shape[3])
	{
		idx_type nslice = shape[0]*shape[1];
		idx_type nstack = nslice*shape[2];
		idx_type asize1 = (2*nstack)*sizeof(*adaptivity);
		optflow_type *c_tmp;

		if(typeid(float) != typeid(optflow_type))
		{
			c_tmp = (optflow_type*) malloc((2*nstack)*sizeof(*c_tmp));

			#pragma omp parallel for
			for (idx_type pos = 0; pos < (2*nstack); pos++)
			{
				c_tmp[pos] = adaptivitymap[pos];
			}
		}
		else
			c_tmp = adaptivitymap;

		cudaSetDevice(deviceID);
		cudaMemcpy(adaptivity, c_tmp, asize1, cudaMemcpyHostToDevice);
		cudaDeviceSynchronize();

		return;
	}
	void OptFlow_GPU3D::set_adaptivitymap(float* adaptivitymap, int shape[3], std::vector<int> &boundaries)
	{
		cudaSetDevice(deviceID);

		int nx = shape[0]; int ny = shape[1];
		long long int nslice_full = nx*ny;

		int nx_target = (boundaries[3]-boundaries[0]);
		int ny_target = (boundaries[4]-boundaries[1]);
		int nz_target = (boundaries[5]-boundaries[2]);
		idx_type nslice_target = nx_target*ny_target;
		idx_type nstack_target = nz_target*nslice_target;

		if (nx_target == nx && ny_target == ny)
		{
			//preferable case
			idx_type offset = nslice_full*boundaries[2];
			idx_type asize1 = nstack_target*sizeof(*adaptivity);

			cudaMemcpy(adaptivity, adaptivitymap + offset, asize1, cudaMemcpyHostToDevice);
		}
		else if (nx_target == nx)
		{
			//need to loop over slices
			idx_type asize1 = nslice_target*sizeof(*adaptivity);

			for (int z = 0; z < nz_target; z++)
			{
				idx_type offset_target = z*nslice_target;
				idx_type offset_source = (z+boundaries[2])*nslice_full+boundaries[1]*nx;

				cudaMemcpyAsync(adaptivity+offset_target, adaptivitymap+offset_source, asize1, cudaMemcpyHostToDevice);
			}
		}
		else
		{
			//worst case
			idx_type asize1 = nx_target*sizeof(*adaptivity);

			for (int z = 0; z < nz_target; z++)
			{
				for (int y = 0; y < ny_target; y++)
				{
					idx_type offset_target = y*nx_target + z*nslice_target;
					idx_type offset_source = (z+boundaries[2])*nslice_full+(y+boundaries[1])*nx+boundaries[0];

					cudaMemcpyAsync(adaptivity+offset_target, adaptivitymap+offset_source, asize1, cudaMemcpyHostToDevice);
				}
			}
		}
		cudaDeviceSynchronize();
		return;
	}
	void OptFlow_GPU3D::get_resultcopy(float* out_vector, int shape[3])
	{
		cudaSetDevice(deviceID);

		idx_type nslice = shape[0]*shape[1];
		idx_type nstack = nslice*shape[2];
		idx_type asize1 = (3*nstack)*sizeof(*u);

		if(typeid(float) != typeid(optflow_type))
		{
			optflow_type *u_tmp = (optflow_type*) malloc(3*nstack*sizeof(*u_tmp));

			cudaMemcpy(u_tmp,u, asize1, cudaMemcpyDeviceToHost);

			#pragma omp parallel for
			for (idx_type pos = 0; pos < nstack; pos++)
			{
				out_vector[pos] = u_tmp[pos];
				out_vector[pos+nstack] = u_tmp[pos+nstack];
				out_vector[pos+2*nstack] = u_tmp[pos+2*nstack];
			}
		}
		else
		{
			//asize1/=3;
			cudaMemcpy(out_vector,u, asize1, cudaMemcpyDeviceToHost);
		}

		cudaDeviceSynchronize();
		return;
	}
	void OptFlow_GPU3D::get_resultcopy(float* out_vector, int shape[3], std::vector<int> &boundaries)
	{
		cudaSetDevice(deviceID);

		int nx = shape[0]; int ny = shape[1]; int nz = shape[2];
		long long int nslice_full = nx*ny;
		long long int nstack_full = nz*nslice_full;

		int nx_source = (boundaries[3]-boundaries[0]);
		int ny_source = (boundaries[4]-boundaries[1]);
		int nz_source = (boundaries[5]-boundaries[2]);
		idx_type nslice_source = nx_source*ny_source;
		idx_type nstack_source = nz_source*nslice_source;

		if (nx_source == nx && ny_source == ny)
		{
			//preferable case
			idx_type offset = nslice_full*boundaries[2];
			idx_type asize1 = nstack_source*sizeof(*u);

			cudaMemcpyAsync(out_vector + offset, u, asize1, cudaMemcpyDeviceToHost);
			cudaMemcpyAsync(out_vector + offset + nstack_full, u+nstack_source, asize1, cudaMemcpyDeviceToHost);
			cudaMemcpyAsync(out_vector + offset + (2*nstack_full), u+(2*nstack_source), asize1, cudaMemcpyDeviceToHost);
		}
		else if (nx_source == nx)
		{
			//need to loop over slices
			idx_type asize1 = nslice_source*sizeof(*u);

			for (int z = 0; z < nz_source; z++)
			{
				idx_type offset_source = z*nslice_source;
				idx_type offset_target = (z+boundaries[2])*nslice_full+boundaries[1]*nx;

				cudaMemcpyAsync(out_vector+offset_target, u+offset_source, asize1, cudaMemcpyDeviceToHost);
				cudaMemcpyAsync(out_vector+offset_target+nstack_full, u+offset_source+nstack_source,  asize1, cudaMemcpyDeviceToHost);
				cudaMemcpyAsync(out_vector+offset_target+(2*nstack_full), u+offset_source+(2*nstack_source), asize1, cudaMemcpyDeviceToHost);
			}
		}
		else
		{
			//worst case
			idx_type asize1 = nx_source*sizeof(*u);

			for (int z = 0; z < nz_source; z++)
			{
				for (int y = 0; y < ny_source; y++)
				{
					idx_type offset_source = y*nx_source + z*nslice_source;
					idx_type offset_target = (z+boundaries[2])*nslice_full+(y+boundaries[1])*nx+boundaries[0];

					cudaMemcpyAsync(out_vector+offset_target, u+offset_source, asize1, cudaMemcpyDeviceToHost);
					cudaMemcpyAsync(out_vector+offset_target+nstack_full, u+offset_source+nstack_source,  asize1, cudaMemcpyDeviceToHost);
					cudaMemcpyAsync(out_vector+offset_target+(2*nstack_full), u+offset_source+(2*nstack_source),  asize1, cudaMemcpyDeviceToHost);
				}
			}
		}

		cudaDeviceSynchronize();
		return;
	}
}
