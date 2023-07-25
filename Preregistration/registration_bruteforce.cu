#include <iostream>
#include <cuda.h>
#include <omp.h>
#include <typeinfo>
#include <limits>
#include "registration_bruteforce.h"

/*********************************************************************************************************************************************************
 * Location: Helmholtz-Zentrum fuer Material und Kuestenforschung, Max-Planck-Strasse 1, 21502 Geesthacht
 * Author: Stefan Bruns
 * Contact: bruns@nano.ku.dk
 *
 * License: TBA
 *********************************************************************************************************************************************************/

namespace registrate
{
	namespace gpu_const
	{
		__constant__ int nx_c, ny_c, nz_c;
		__constant__ int interpolation_order_c = 1; //1 = linear, 2 = cubic

		__constant__ float rotationmatrix_c[9] = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
		__constant__ float rotcenter_c[3] = {0.0f, 0.0f, 0.0f};
	}

	namespace gpu_solve
	{
		//Device Code for Interpolation
		////////////////////////////////////////////////////////
		__device__ __inline__ float interpolate_cubic(float &y0, float &y1, float &y2, float &y3, float &mu)
		{
			float mu2 = mu*mu;

			float a0 = y3-y2-y0+y1;
			float a1 = y0-y1-a0;
			float a2 = y2-y0;
			float a3 = y1;

			return a0*mu*mu2+a1*mu2+a2*mu+a3;
		}
		__inline__ __device__ float cubicinterpolation(float *image, float &x, float &y, float &z)
		{
			int nx = gpu_const::nx_c;
			int ny = gpu_const::ny_c;
			int nz = gpu_const::nz_c;
			long long int nslice = nx*ny;

			int xf = (int) x; int xc = ceil(x);
			int yf = (int) y; int yc = ceil(y);
			int zf = (int) z; int zc = ceil(z);

			float wx = x-xf;
			float wy = y-yf;
			float wz = z-zf;

			float val;

			if (zf != zc && yf != yc && xf != xc)
			{
				//extrapolate with zero-gradient
				int xf2 = max(0, xf-1);
				int xc2 = min(xc+1, nx-1);
				int yf2 = max(0, yf-1);
				int yc2 = min(yc+1, ny-1);
				int zf2 = max(0, zf-1);
				int zc2 = min(zc+1, nz-1);

				float P100 = image[zf2*nslice+yf2*nx + xf];
				float P200 = image[zf2*nslice+yf2*nx + xc];
				float P101 = image[zf*nslice+yf2*nx + xf];
				float P201 = image[zf*nslice+yf2*nx + xc];
				float P102 = image[zc*nslice+yf2*nx + xf];
				float P202 = image[zc*nslice+yf2*nx + xc];
				float P103 = image[zc2*nslice+yf2*nx + xf];
				float P203 = image[zc2*nslice+yf2*nx + xc];

				float P10 = interpolate_cubic(P100, P101, P102, P103, wz);
				float P20 = interpolate_cubic(P200, P201, P202, P203, wz);

				float P010 = image[zf2*nslice+yf*nx + xf2];
				float P110 = image[zf2*nslice+yf*nx + xf];
				float P210 = image[zf2*nslice+yf*nx + xc];
				float P310 = image[zf2*nslice+yf*nx + xc2];
				float P011 = image[zf*nslice+yf*nx + xf2];
				float P111 = image[zf*nslice+yf*nx + xf];
				float P211 = image[zf*nslice+yf*nx + xc];
				float P311 = image[zf*nslice+yf*nx + xc2];
				float P012 = image[zc*nslice+yf*nx + xf2];
				float P112 = image[zc*nslice+yf*nx + xf];
				float P212 = image[zc*nslice+yf*nx + xc];
				float P312 = image[zc*nslice+yf*nx + xc2];
				float P013 = image[zc2*nslice+yf*nx + xf2];
				float P113 = image[zc2*nslice+yf*nx + xf];
				float P213 = image[zc2*nslice+yf*nx + xc];
				float P313 = image[zc2*nslice+yf*nx + xc2];

				float P01 = interpolate_cubic(P010, P011, P012, P013, wz);
				float P11 = interpolate_cubic(P110, P111, P112, P113, wz);
				float P21 = interpolate_cubic(P210, P211, P212, P213, wz);
				float P31 = interpolate_cubic(P310, P311, P312, P313, wz);

				float P020 = image[zf2*nslice+yc*nx + xf2];
				float P120 = image[zf2*nslice+yc*nx + xf];
				float P220 = image[zf2*nslice+yc*nx + xc];
				float P320 = image[zf2*nslice+yc*nx + xc2];
				float P021 = image[zf*nslice+yc*nx + xf2];
				float P121 = image[zf*nslice+yc*nx + xf];
				float P221 = image[zf*nslice+yc*nx + xc];
				float P321 = image[zf*nslice+yc*nx + xc2];
				float P022 = image[zc*nslice+yc*nx + xf2];
				float P122 = image[zc*nslice+yc*nx + xf];
				float P222 = image[zc*nslice+yc*nx + xc];
				float P322 = image[zc*nslice+yc*nx + xc2];
				float P023 = image[zc2*nslice+yc*nx + xf2];
				float P123 = image[zc2*nslice+yc*nx + xf];
				float P223 = image[zc2*nslice+yc*nx + xc];
				float P323 = image[zc2*nslice+yc*nx + xc2];

				float P02 = interpolate_cubic(P020, P021, P022, P023, wz);
				float P12 = interpolate_cubic(P120, P121, P122, P123, wz);
				float P22 = interpolate_cubic(P220, P221, P222, P223, wz);
				float P32 = interpolate_cubic(P320, P321, P322, P323, wz);

				float P130 = image[zf2*nslice+yc2*nx + xf];
				float P230 = image[zf2*nslice+yc2*nx + xc];
				float P131 = image[zf*nslice+yc2*nx + xf];
				float P231 = image[zf*nslice+yc2*nx + xc];
				float P132 = image[zc*nslice+yc2*nx + xf];
				float P232 = image[zc*nslice+yc2*nx + xc];
				float P133 = image[zc2*nslice+yc2*nx + xf];
				float P233 = image[zc2*nslice+yc2*nx + xc];

				float P13 = interpolate_cubic(P130, P131, P132, P133, wz);
				float P23 = interpolate_cubic(P230, P231, P232, P233, wz);

				float gtu = interpolate_cubic(P01,P11,P21,P31,wx);
				float gbu = interpolate_cubic(P02,P12,P22,P32,wx);

				float glv = interpolate_cubic(P10,P11,P12,P13,wy);
				float grv = interpolate_cubic(P20,P21,P22,P23,wy);

				float sigma_lr = (1.-wx)*glv + wx*grv;
				float sigma_bt = (1.-wy)*gtu + wy*gbu;
				float corr_lrbt = P11*(1.-wy)*(1.-wx) + P12*wy*(1.-wx) + P21*(1.-wy)*wx + P22*wx*wy;

				val = sigma_lr+sigma_bt-corr_lrbt;
			}
			else if (zf != zc && yf != yc)
			{
				//extrapolate with zero-gradient
				int zf2 = max(0, zf-1);
				int zc2 = min(zc+1, nz-1);
				int yf2 = max(0, yf-1);
				int yc2 = min(yc+1, ny-1);

				float P10 = image[zf*nslice+yf2*nx + xf];
				float P20 = image[zc*nslice+yf2*nx + xf];

				float P01 = image[zf2*nslice+yf*nx + xf];
				float P11 = image[zf *nslice+yf*nx + xf];
				float P21 = image[zc *nslice+yf*nx + xf];
				float P31 = image[zc2*nslice+yf*nx + xf];

				float P02 = image[zf2*nslice+yc*nx + xf];
				float P12 = image[zf *nslice+yc*nx + xf];
				float P22 = image[zc *nslice+yc*nx + xf];
				float P32 = image[zc2*nslice+yc*nx + xf];

				float P13 = image[zf*nslice+yc2*nx + xf];
				float P23 = image[zc*nslice+yc2*nx + xf];

				float gtu = interpolate_cubic(P01,P11,P21,P31,wz);
				float gbu = interpolate_cubic(P02,P12,P22,P32,wz);

				float glv = interpolate_cubic(P10,P11,P12,P13,wy);
				float grv = interpolate_cubic(P20,P21,P22,P23,wy);

				float sigma_lr = (1.-wz)*glv + wz*grv;
				float sigma_bt = (1.-wy)*gtu + wy*gbu;
				float corr_lrbt = P11*(1.-wy)*(1.-wz) + P12*wy*(1.-wz) + P21*(1.-wy)*wx + P22*wz*wy;

				val = sigma_lr+sigma_bt-corr_lrbt;
			}
			else if (zf != zc && xf != xc)
			{
				//extrapolate with zero-gradient
				int zf2 = max(0, zf-1);
				int zc2 = min(zc+1, nz-1);
				int xf2 = max(0, xf-1);
				int xc2 = min(xc+1, nx-1);

				float P10 = image[zf*nslice+yf*nx + xf2];
				float P20 = image[zc*nslice+yf*nx + xf2];

				float P01 = image[zf2*nslice+yf*nx + xf];
				float P11 = image[zf *nslice+yf*nx + xf];
				float P21 = image[zc *nslice+yf*nx + xf];
				float P31 = image[zc2*nslice+yf*nx + xf];

				float P02 = image[zf2*nslice+yf*nx + xc];
				float P12 = image[zf *nslice+yf*nx + xc];
				float P22 = image[zc *nslice+yf*nx + xc];
				float P32 = image[zc2*nslice+yf*nx + xc];

				float P13 = image[zf*nslice+yf*nx + xc2];
				float P23 = image[zc*nslice+yf*nx + xc2];

				float gtu = interpolate_cubic(P01,P11,P21,P31,wz);
				float gbu = interpolate_cubic(P02,P12,P22,P32,wz);

				float glv = interpolate_cubic(P10,P11,P12,P13,wy);
				float grv = interpolate_cubic(P20,P21,P22,P23,wy);

				float sigma_lr = (1.-wz)*glv + wz*grv;
				float sigma_bt = (1.-wx)*gtu + wx*gbu;
				float corr_lrbt = P11*(1.-wx)*(1.-wz) + P12*wx*(1.-wz) + P21*(1.-wx)*wx + P22*wz*wx;

				val = sigma_lr+sigma_bt-corr_lrbt;
			}
			else if (zf != zc)
			{
				int zf2 = max(0, zf-1);
				int zc2 = min(zc+1, nz-1);

				float P0 = image[zf2*nslice+yf*nx + xf];
				float P1 = image[zf *nslice+yf*nx + xf];
				float P2 = image[zc *nslice+yf*nx + xf];
				float P3 = image[zc2*nslice+yf*nx + xf];

				val = interpolate_cubic(P0,P1,P2,P3,wz);
			}
			else if (yf != yc && xf != xc)
			{
				//extrapolate with zero-gradient
				int xf2 = max(0, xf-1);
				int xc2 = min(xc+1, nx-1);
				int yf2 = max(0, yf-1);
				int yc2 = min(yc+1, ny-1);

				float P10 = image[zf*nslice+yf2*nx + xf];
				float P20 = image[zf*nslice+yf2*nx + xc];

				float P01 = image[zf*nslice+yf*nx + xf2];
				float P11 = image[zf*nslice+yf*nx + xf];
				float P21 = image[zf*nslice+yf*nx + xc];
				float P31 = image[zf*nslice+yf*nx + xc2];

				float P02 = image[zf*nslice+yc*nx + xf2];
				float P12 = image[zf*nslice+yc*nx + xf];
				float P22 = image[zf*nslice+yc*nx + xc];
				float P32 = image[zf*nslice+yc*nx + xc2];

				float P13 = image[zf*nslice+yc2*nx + xf];
				float P23 = image[zf*nslice+yc2*nx + xc];

				float gtu = interpolate_cubic(P01,P11,P21,P31,wx);
				float gbu = interpolate_cubic(P02,P12,P22,P32,wx);

				float glv = interpolate_cubic(P10,P11,P12,P13,wy);
				float grv = interpolate_cubic(P20,P21,P22,P23,wy);

				float sigma_lr = (1.-wx)*glv + wx*grv;
				float sigma_bt = (1.-wy)*gtu + wy*gbu;
				float corr_lrbt = P11*(1.-wy)*(1.-wx) + P12*wy*(1.-wx) + P21*(1.-wy)*wx + P22*wx*wy;

				val = sigma_lr+sigma_bt-corr_lrbt;
			}
			else if (xf != xc)
			{
				int xf2 = max(0, xf-1);
				int xc2 = min(xc+1, nx-1);

				float P0 = image[zf *nslice+yf*nx + xf2];
				float P1 = image[zf *nslice+yf*nx + xf];
				float P2 = image[zf *nslice+yf*nx + xc];
				float P3 = image[zf *nslice+yf*nx + xc2];

				val = interpolate_cubic(P0,P1,P2,P3,wx);
			}
			else if (yf != yc)
			{
				int yf2 = max(0, yf-1);
				int yc2 = min(yc+1, ny-1);

				float P0 = image[zf *nslice+yf2*nx + xf];
				float P1 = image[zf *nslice+yf*nx + xf];
				float P2 = image[zf *nslice+yc*nx + xf];
				float P3 = image[zf *nslice+yc2*nx + xf];

				val = interpolate_cubic(P0,P1,P2,P3,wy);
			}
			else val = image[zf*nslice+yf*nx + xf];

			return val;
		}
		__inline__ __device__ float linearinterpolation(float *image, float &x, float &y, float &z)
		{
			int nx = gpu_const::nx_c;
			int ny = gpu_const::ny_c;
			long long int nslice = nx*ny;

			int xf = (int) x; int xc = ceil(x);
			int yf = (int) y; int yc = ceil(y);
			int zf = (int) z; int zc = ceil(z);

			float wx = x-xf;
			float wy = y-yf;
			float wz = z-zf;

			float val, vala, valb, valc, vald;

			if (zf != zc && yf != yc && xf != xc)
			{
				float val000 = image[zf*nslice+yf*nx+xf];
				float val001 = image[zf*nslice+yf*nx+xc];
				float val010 = image[zf*nslice+yc*nx+xf];
				float val100 = image[zc*nslice+yf*nx+xf];
				float val011 = image[zf*nslice+yc*nx+xc];
				float val110 = image[zc*nslice+yc*nx+xf];
				float val101 = image[zc*nslice+yf*nx+xc];
				float val111 = image[zc*nslice+yc*nx+xc];

				vala = (1.f-wz)*val000 + wz*val001;
				valb = (1.f-wz)*val100 + wz*val101;
				valc = (1.f-wz)*val010 + wz*val011;
				vald = (1.f-wz)*val110 + wz*val111;

				vala = (1.f-wx)*vala + wx*valb;
				valb = (1.f-wx)*valc + wx*vald;

				val = (1.f-wy)*vala + wy*valb;
			}
			else if (zf != zc && yf != yc)
			{
				float val000 = image[zf*nslice+yf*nx+xf];
				float val001 = image[zf*nslice+yf*nx+xc];
				float val010 = image[zf*nslice+yc*nx+xf];
				float val011 = image[zf*nslice+yc*nx+xc];

				vala = (1.f-wz)*val000 + wz*val001;
				valb = (1.f-wz)*val010 + wz*val011;

				val = (1.f-wy)*vala + wy*valb;
			}
			else if (zf != zc && xf != xc)
			{
				float val000 = image[zf*nslice+yf*nx+xf];
				float val001 = image[zf*nslice+yf*nx+xc];
				float val100 = image[zc*nslice+yf*nx+xf];
				float val101 = image[zc*nslice+yf*nx+xc];

				vala = (1.f-wz)*val000 + wz*val001;
				valb = (1.f-wz)*val100 + wz*val101;

				val = (1.f-wx)*vala + wx*valb;
			}
			else if (zf != zc)
			{
				vala = image[zf*nslice+yf*nx+xf];
				valb = image[zc*nslice+yf*nx+xf];

				val = (1.f-wz)*vala + wz*valb;
			}
			else if (xf != xc && yf != yc)
			{
				vala = image[zf*nslice+yf*nx+xf];
				valb = image[zf*nslice+yf*nx+xc];
				valc = image[zf*nslice+yc*nx+xf];
				vald = image[zf*nslice+yc*nx+xc];

				vala = (1.f-wx)*vala + wx*valb;
				valb = (1.f-wx)*valc + wx*vald;

				val = (1.f-wy)*vala + wy*valb;
			}
			else if (xf != xc)
			{
				vala = image[zf*nslice+yf*nx+xf];
				valb = image[zf*nslice+yf*nx+xc];

				val = (1.f-wx)*vala + wx*valb;
			}
			else if (yf != yc)
			{
				vala = image[zf*nslice+yf*nx+xf];
				valb = image[zf*nslice+yc*nx+xf];

				val = (1.f-wy)*vala + wy*valb;
			}
			else
				val = image[zf*nslice+yf*nx+xf];

			return val;
		}
		////////////////////////////////////////////////////////

		//Device Code for Parallel reduction
		////////////////////////////////////////////////////////
		__inline__ __device__ float warpReduceSum(float val) {
		  for (int offset = warpSize/2; offset > 0; offset /= 2)
			val += __shfl_down_sync(0xFFFFFFFF, val, offset, warpSize);
		  return val;
		}
		__inline__ __device__ float blockReduceSum(float val)
		{
			static __shared__ float shared[32];
			int lane = threadIdx.x % warpSize;
			int wid = threadIdx.x / warpSize;

			val = warpReduceSum(val);

		  if (lane==0) shared[wid]=val;

		  __syncthreads();

		  val = (threadIdx.x < blockDim.x / warpSize) ? shared[lane] : 0;

		  if (wid==0) val = warpReduceSum(val);

		  return val;
		}
		__global__ void reduceThread(float *frame0, float *gridreduce, long long int nstack)
		{
			//acquire constants
			/////////////////////////////////////////////
			bool outofbounds = false;
			long long int idx = (blockIdx.x*blockDim.x+threadIdx.x);
			if (idx >= nstack) {outofbounds = true; idx = threadIdx.x;}
			int tid = threadIdx.x;
			__syncthreads();
			/////////////////////////////////////////////

			float val = frame0[idx];
			if (outofbounds) val = 0.0f;
			__syncthreads();

			val = blockReduceSum(val);

			if(tid == 0) gridreduce[blockIdx.x] = val;
			__syncthreads();

			return;
		}
		__global__ void reduceThread_5Layers(float *frame0, float *gridreduce, long long int nstack, long long int nblocks)
		{
			//acquire constants
			/////////////////////////////////////////////
			bool outofbounds = false;
			long long int idx = (blockIdx.x*blockDim.x+threadIdx.x);
			if (idx >= nstack) {outofbounds = true; idx = threadIdx.x;}
			int tid = threadIdx.x;
			__syncthreads();
			/////////////////////////////////////////////

			float val0 = frame0[idx];
			float val1 = frame0[idx+nstack];
			float val2 = frame0[idx+2*nstack];
			float val3 = frame0[idx+3*nstack];
			float val4 = frame0[idx+4*nstack];

			if (outofbounds){
				val0 = val1 = val2 = val3 = val4 = 0.0f;
			}
			__syncthreads();

			val0 = blockReduceSum(val0);
			val1 = blockReduceSum(val1);
			val2 = blockReduceSum(val2);
			val3 = blockReduceSum(val3);
			val4 = blockReduceSum(val4);

			if(tid == 0)
			{
				gridreduce[blockIdx.x] = val0;
				gridreduce[blockIdx.x+nblocks] = val1;
				gridreduce[blockIdx.x+2*nblocks] = val2;
				gridreduce[blockIdx.x+3*nblocks] = val3;
				gridreduce[blockIdx.x+4*nblocks] = val4;
			}
			__syncthreads();

			return;
		}
		__global__ void reduceThread_6Layers(float *frame0, float *gridreduce, long long int nstack, long long int nblocks)
		{
			//acquire constants
			/////////////////////////////////////////////
			bool outofbounds = false;
			long long int idx = (blockIdx.x*blockDim.x+threadIdx.x);
			if (idx >= nstack) {outofbounds = true; idx = threadIdx.x;}
			int tid = threadIdx.x;
			__syncthreads();
			/////////////////////////////////////////////

			float val0 = frame0[idx];
			float val1 = frame0[idx+nstack];
			float val2 = frame0[idx+2*nstack];
			float val3 = frame0[idx+3*nstack];
			float val4 = frame0[idx+4*nstack];
			float val5 = frame0[idx+5*nstack];

			if (outofbounds){
				val0 = val1 = val2 = val3 = val4 = val5 = 0.0f;
			}
			__syncthreads();

			val0 = blockReduceSum(val0);
			val1 = blockReduceSum(val1);
			val2 = blockReduceSum(val2);
			val3 = blockReduceSum(val3);
			val4 = blockReduceSum(val4);
			val5 = blockReduceSum(val5);

			if(tid == 0)
			{
				gridreduce[blockIdx.x] = val0;
				gridreduce[blockIdx.x+nblocks] = val1;
				gridreduce[blockIdx.x+2*nblocks] = val2;
				gridreduce[blockIdx.x+3*nblocks] = val3;
				gridreduce[blockIdx.x+4*nblocks] = val4;
				gridreduce[blockIdx.x+5*nblocks] = val5;
			}
			__syncthreads();

			return;
		}
		////////////////////////////////////////////////////////

		//Device Code for Correlation
		////////////////////////////////////////////////////////
		__global__ void correlate_translate_reduce(float dx, float dy, float dz, float *frame0, float *frame1, float *gridreduce, long long int nstack, long long int nblocks)
		{
			//acquire constants
			/////////////////////////////////////////////
			bool outofbounds = false;
			long long int idx0 = (blockIdx.x*blockDim.x+threadIdx.x);
			if (idx0 >= nstack) {outofbounds = true; idx0 = threadIdx.x;}
			int tid = threadIdx.x;

			int nx = gpu_const::nx_c;
			int ny = gpu_const::ny_c;
			int nz = gpu_const::nz_c;
			int interpolation_order = gpu_const::interpolation_order_c;
			long long int nslice = nx*ny;

			int z0 = idx0/nslice;
			int y0 = (idx0-z0*nslice)/nx;
			int x0 = idx0-z0*nslice-y0*nx;

			float z1 = z0-dz;
			float y1 = y0-dy;
			float x1 = x0-dx;

			int z1f = (int) z1; int z1c = ceil(z1);
			int y1f = (int) y1; int y1c = ceil(y1);
			int x1f = (int) x1; int x1c = ceil(x1);

			float nvalid = 1.f;

			if(z1f < 0 || y1f < 0 || x1f < 0 || z1c >= nz || y1c >= ny || x1c >= nx)
			{
				z1 = z1f = z1c = z0; y1 = y1f = y1c = y0; x1 = x1f = x1c = x0;
				outofbounds = true;
			}
			if (outofbounds)
				nvalid = 0.0f;

			__syncthreads();
			/////////////////////////////////////////////

			float val0 = frame0[idx0];
			float val1;

			if (interpolation_order == 1) //Linear interpolation
				val1 = linearinterpolation(frame1, x1, y1, z1);
			else //Cubic interpolation
				val1 = cubicinterpolation(frame1, x1, y1, z1);

			if (outofbounds) {val0 = val1 = 0.0f;}
			__syncthreads();

			float sum0 = blockReduceSum(val0);
			float sum1 = blockReduceSum(val1);
			float covar = blockReduceSum(val0*val1);
			float sqsum0 = blockReduceSum(val0*val0);
			float sqsum1 = blockReduceSum(val1*val1);
			nvalid = blockReduceSum(nvalid);

			if(tid == 0)
			{
				gridreduce[blockIdx.x] = sum0;
				gridreduce[nblocks+blockIdx.x] = sum1;
				gridreduce[2*nblocks+blockIdx.x] = covar;
				gridreduce[3*nblocks+blockIdx.x] = sqsum0;
				gridreduce[4*nblocks+blockIdx.x] = sqsum1;
				gridreduce[5*nblocks+blockIdx.x] = nvalid;
			}
			__syncthreads();

			return;
		}
		__global__ void correlate_translate_rotate_reduce(float dx, float dy, float dz, float *frame0, float *frame1, float *gridreduce, long long int nstack, long long int nblocks)
		{
			//acquire constants
			/////////////////////////////////////////////
			int nx = gpu_const::nx_c;
			int ny = gpu_const::ny_c;
			int nz = gpu_const::nz_c;
			long long int nslice = nx*ny;

			float a11 = gpu_const::rotationmatrix_c[0]; float a12 = gpu_const::rotationmatrix_c[1]; float a13 = gpu_const::rotationmatrix_c[2];
			float a21 = gpu_const::rotationmatrix_c[3]; float a22 = gpu_const::rotationmatrix_c[4]; float a23 = gpu_const::rotationmatrix_c[5];
			float a31 = gpu_const::rotationmatrix_c[6]; float a32 = gpu_const::rotationmatrix_c[7]; float a33 = gpu_const::rotationmatrix_c[8];
			float rotcenter0 = gpu_const::rotcenter_c[0]; float rotcenter1 = gpu_const::rotcenter_c[0]; float rotcenter2 = gpu_const::rotcenter_c[0];

			int interpolation_order = gpu_const::interpolation_order_c;

			bool outofbounds = false;
			long long int idx0 = (blockIdx.x*blockDim.x+threadIdx.x);
			if (idx0 >= nstack) {outofbounds = true; idx0 = threadIdx.x;}
			int tid = threadIdx.x;

			int z0 = idx0/nslice;
			int y0 = (idx0-z0*nslice)/nx;
			int x0 = idx0-z0*nslice-y0*nx;

			float x1 = a11*(x0-rotcenter0) + a12*(y0-rotcenter1) + a13*(z0-rotcenter2) + rotcenter0 - dx;
			float y1 = a21*(x0-rotcenter0) + a22*(y0-rotcenter1) + a23*(z0-rotcenter2) + rotcenter1 - dy;
			float z1 = a31*(x0-rotcenter0) + a32*(y0-rotcenter1) + a33*(z0-rotcenter2) + rotcenter2 - dz;

			int z1f = (int) z1; int z1c = ceil(z1);
			int y1f = (int) y1; int y1c = ceil(y1);
			int x1f = (int) x1; int x1c = ceil(x1);

			float nvalid = 1.f;

			if(z1f < 0 || y1f < 0 || x1f < 0 || z1c >= nz || y1c >= ny || x1c >= nx)
			{
				z1 = z1f = z1c = z0; y1 = y1f = y1c = y0; x1 = x1f = x1c = x0;
				outofbounds = true;
			}
			if (outofbounds)
				nvalid = 0.0f;

			__syncthreads();
			/////////////////////////////////////////////

			float val1;
			float val0= frame0[idx0];

			if (interpolation_order == 1) //Linear interpolation
				val1 = linearinterpolation(frame1, x1, y1, z1);
			else //Cubic interpolation
				val1 = cubicinterpolation(frame1, x1, y1, z1);

			if (outofbounds) {val0 = val1 = 0.0f;}
			__syncthreads();

			float sum0 = blockReduceSum(val0);
			float sum1 = blockReduceSum(val1);
			float covar = blockReduceSum(val0*val1);
			float sqsum0 = blockReduceSum(val0*val0);
			float sqsum1 = blockReduceSum(val1*val1);
			nvalid = blockReduceSum(nvalid);

			if(tid == 0)
			{
				gridreduce[blockIdx.x] = sum0;
				gridreduce[nblocks+blockIdx.x] = sum1;
				gridreduce[2*nblocks+blockIdx.x] = covar;
				gridreduce[3*nblocks+blockIdx.x] = sqsum0;
				gridreduce[4*nblocks+blockIdx.x] = sqsum1;
				gridreduce[5*nblocks+blockIdx.x] = nvalid;
			}
			__syncthreads();

			return;
		}
		__global__ void correlate_translate_rotate_reduce(float dx, float dy, float dz, float *frame0, float *frame1, float *mask, float *gridreduce, long long int nstack, long long int nblocks)
		{
			//acquire constants
			/////////////////////////////////////////////
			int nx = gpu_const::nx_c;
			int ny = gpu_const::ny_c;
			int nz = gpu_const::nz_c;
			long long int nslice = nx*ny;

			float a11 = gpu_const::rotationmatrix_c[0]; float a12 = gpu_const::rotationmatrix_c[1]; float a13 = gpu_const::rotationmatrix_c[2];
			float a21 = gpu_const::rotationmatrix_c[3]; float a22 = gpu_const::rotationmatrix_c[4]; float a23 = gpu_const::rotationmatrix_c[5];
			float a31 = gpu_const::rotationmatrix_c[6]; float a32 = gpu_const::rotationmatrix_c[7]; float a33 = gpu_const::rotationmatrix_c[8];
			float rotcenter0 = gpu_const::rotcenter_c[0]; float rotcenter1 = gpu_const::rotcenter_c[0]; float rotcenter2 = gpu_const::rotcenter_c[0];

			int interpolation_order = gpu_const::interpolation_order_c;

			bool outofbounds = false;
			long long int idx0 = (blockIdx.x*blockDim.x+threadIdx.x);
			if (idx0 >= nstack) {outofbounds = true; idx0 = threadIdx.x;}
			int tid = threadIdx.x;

			/////
			__syncthreads();
			float maskval = mask[idx0];
			if (maskval == 0.0f) outofbounds = true;
			/////

			int z0 = idx0/nslice;
			int y0 = (idx0-z0*nslice)/nx;
			int x0 = idx0-z0*nslice-y0*nx;

			float x1 = a11*(x0-rotcenter0) + a12*(y0-rotcenter1) + a13*(z0-rotcenter2) + rotcenter0 - dx;
			float y1 = a21*(x0-rotcenter0) + a22*(y0-rotcenter1) + a23*(z0-rotcenter2) + rotcenter1 - dy;
			float z1 = a31*(x0-rotcenter0) + a32*(y0-rotcenter1) + a33*(z0-rotcenter2) + rotcenter2 - dz;

			int z1f = (int) z1; int z1c = ceil(z1);
			int y1f = (int) y1; int y1c = ceil(y1);
			int x1f = (int) x1; int x1c = ceil(x1);

			float nvalid = 1.f;

			if(z1f < 0 || y1f < 0 || x1f < 0 || z1c >= nz || y1c >= ny || x1c >= nx)
			{
				z1 = z1f = z1c = z0; y1 = y1f = y1c = y0; x1 = x1f = x1c = x0;
				outofbounds = true;
			}
			if (outofbounds)
				nvalid = 0.0f;

			__syncthreads();
			/////////////////////////////////////////////

			float val0 = frame0[idx0];
			float val1;

			if (interpolation_order == 1) //Linear interpolation
				val1 = linearinterpolation(frame1, x1, y1, z1);
			else //Cubic interpolation
				val1 = cubicinterpolation(frame1, x1, y1, z1);

			if (outofbounds) {val0 = val1 = 0.0f;}
			__syncthreads();

			float sum0 = blockReduceSum(val0);
			float sum1 = blockReduceSum(val1);
			float covar = blockReduceSum(val0*val1);
			float sqsum0 = blockReduceSum(val0*val0);
			float sqsum1 = blockReduceSum(val1*val1);
			nvalid = blockReduceSum(nvalid);

			if(tid == 0)
			{
				gridreduce[blockIdx.x] = sum0;
				gridreduce[nblocks+blockIdx.x] = sum1;
				gridreduce[2*nblocks+blockIdx.x] = covar;
				gridreduce[3*nblocks+blockIdx.x] = sqsum0;
				gridreduce[4*nblocks+blockIdx.x] = sqsum1;
				gridreduce[5*nblocks+blockIdx.x] = nvalid;
			}
			__syncthreads();

			return;
		}
		////////////////////////////////////////////////////////
	}

	int BruteForce::configure_device(int maxshape[3], int deviceID_, bool use_mask, int interpolation_order)
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
		if (use_mask) expected_usage += nstack*sizeof(float);

		if (expected_usage > free_db){std::cout << "\033[1;31mError! Expected to run out of GPU memory!\033[0m" << std::endl;return 2;}
		////////////////////////////////////////////////////

		//allocate memory and set constant memory
		////////////////////////////////////////////////////
		(float*) cudaMalloc((void**)&devframe0, nstack*sizeof(*devframe0));
		(float*) cudaMalloc((void**)&devframe1, nstack*sizeof(*devframe1));
		(float*) cudaMalloc((void**)&gridreduce0, 6*blocksPerGrid*sizeof(*gridreduce0));
		(float*) cudaMalloc((void**)&gridreduce1, 6*blocksPerGrid*sizeof(*gridreduce1));

		if (use_mask) (float*) cudaMalloc((void**)&devmask, nstack*sizeof(*devmask));
		else (float*) cudaMalloc((void**)&devmask, 1*sizeof(*devmask));

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
	void BruteForce::free_device()
	{
		cudaSetDevice(deviceID);

		cudaFree(devframe0);
		cudaFree(devframe1);
		cudaFree(gridreduce0);
		cudaFree(gridreduce1);
		cudaFree(devmask);
	}

	void BruteForce::set_frames(float* frame0, float *frame1, int shape[3])
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
	void BruteForce::set_frames(float* frame0, float *frame1, float *mask, int shape[3])
	{
		cudaSetDevice(deviceID);

		int nx = shape[0]; int ny = shape[1]; int nz = shape[2];
		long long int nslice = shape[0]*shape[1];
		long long int nstack = shape[2]*nslice;
		long long int asize = nstack*sizeof(*devframe0);

		cudaMemcpyAsync(devframe0, frame0, asize, cudaMemcpyHostToDevice);
		cudaMemcpyAsync(devframe1, frame1, asize, cudaMemcpyHostToDevice);
		cudaMemcpyAsync(devmask, mask, asize, cudaMemcpyHostToDevice);

		cudaMemcpyToSymbol(gpu_const::nx_c, &nx, sizeof(gpu_const::nx_c));
		cudaMemcpyToSymbol(gpu_const::ny_c, &ny, sizeof(gpu_const::ny_c));
		cudaMemcpyToSymbol(gpu_const::nz_c, &nz, sizeof(gpu_const::nz_c));
		cudaDeviceSynchronize();

		return;
	}

	float BruteForce::execute_correlation(float dx, float dy, float dz, int shape[3])
	{
		cudaSetDevice(deviceID);

		long long int nslice = shape[0]*shape[1];
		long long int nstack = shape[2]*nslice;
		long long int blocksPerGrid = (nstack + threadsPerBlock - 1) / (threadsPerBlock);
		long long int blocksPerGrid2;

		gpu_solve::correlate_translate_reduce<<<blocksPerGrid,threadsPerBlock>>>(dx, dy, dz, devframe0, devframe1, gridreduce0, nstack, blocksPerGrid);
		cudaDeviceSynchronize();
		while(blocksPerGrid > threadsPerBlock)
		{
			blocksPerGrid2 = (blocksPerGrid + threadsPerBlock - 1) / (threadsPerBlock);
			gpu_solve::reduceThread_6Layers<<<blocksPerGrid2,threadsPerBlock>>>(gridreduce0,gridreduce1, blocksPerGrid, blocksPerGrid2);
			cudaDeviceSynchronize();

			blocksPerGrid = blocksPerGrid2;
			std::swap(gridreduce0, gridreduce1);
		}
		gpu_solve::reduceThread_6Layers<<<1,threadsPerBlock>>>(gridreduce0,gridreduce1, blocksPerGrid, 1);
		cudaDeviceSynchronize();

		blocksPerGrid = 1;
		float sum[6];
		cudaMemcpy(sum, gridreduce1, 6*sizeof(*gridreduce1), cudaMemcpyDeviceToHost);

		float mean0 = sum[0];
		float mean1 = sum[1];
		float covar = sum[2];
		float sqmean0 = sum[3];
		float sqmean1 = sum[4];
		float nvalid = sum[5];

		mean0 /= nvalid;
		mean1 /= nvalid;
		covar /= nvalid;
		sqmean0 /= nvalid;
		sqmean1 /= nvalid;

		float result = (covar-mean0*mean1)/(sqrt(sqmean0-mean0*mean0)*sqrt(sqmean1-mean1*mean1));

		return result;
	}
	float BruteForce::execute_correlation(float jaw, float pitch, float roll, float dx, float dy, float dz, int shape[3], float rotcenter[3], bool use_mask)
	{
		cudaSetDevice(deviceID);

		long long int nslice = shape[0]*shape[1];
		long long int nstack = shape[2]*nslice;
		long long int blocksPerGrid = (nstack + threadsPerBlock - 1) / (threadsPerBlock);
		long long int blocksPerGrid2;

		//prepare rotation
		///////////////////////////////////////////////////////////////////////////////
		float rotationmatrix[9] =  {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
		prepare_rotation_coefficients(rotationmatrix, jaw, roll, pitch);
		cudaMemcpyToSymbol(gpu_const::rotationmatrix_c, rotationmatrix,  9*sizeof(float), 0);
		cudaMemcpyToSymbol(gpu_const::rotcenter_c, rotcenter,  3*sizeof(float), 0);
		cudaDeviceSynchronize();
		///////////////////////////////////////////////////////////////////////////////

		if(!use_mask) gpu_solve::correlate_translate_rotate_reduce<<<blocksPerGrid,threadsPerBlock>>>(dx, dy, dz, devframe0, devframe1, gridreduce0, nstack, blocksPerGrid);
		else gpu_solve::correlate_translate_rotate_reduce<<<blocksPerGrid,threadsPerBlock>>>(dx, dy, dz, devframe0, devframe1, devmask, gridreduce0, nstack, blocksPerGrid);
		cudaDeviceSynchronize();

		while(blocksPerGrid > threadsPerBlock)
		{
			blocksPerGrid2 = (blocksPerGrid + threadsPerBlock - 1) / (threadsPerBlock);
			gpu_solve::reduceThread_6Layers<<<blocksPerGrid2,threadsPerBlock>>>(gridreduce0,gridreduce1, blocksPerGrid, blocksPerGrid2);
			cudaDeviceSynchronize();

			blocksPerGrid = blocksPerGrid2;
			std::swap(gridreduce0, gridreduce1);
		}
		gpu_solve::reduceThread_6Layers<<<1,threadsPerBlock>>>(gridreduce0,gridreduce1, blocksPerGrid, 1);
		cudaDeviceSynchronize();

		float sum[6];
		cudaMemcpy(sum, gridreduce1, 6*sizeof(*gridreduce1), cudaMemcpyDeviceToHost);
		cudaDeviceSynchronize();

		float mean0 = sum[0];
		float mean1 = sum[1];
		float covar = sum[2];
		float sqmean0 = sum[3];
		float sqmean1 = sum[4];
		float nvalid = sum[5];

		mean0 /= nvalid;
		mean1 /= nvalid;
		covar /= nvalid;
		sqmean0 /= nvalid;
		sqmean1 /= nvalid;

		float result = (covar-mean0*mean1)/(sqrt(sqmean0-mean0*mean0)*sqrt(sqmean1-mean1*mean1));

		return result;
	}

	float BruteForce::next_gradientascentstep_translation(float result[3], float h, float gamma, int shape[3], float out_step[3])
	{
		//get gradient
		float dxp = execute_correlation(result[0]+h, result[1], result[2], shape);
		float dxn = execute_correlation(result[0]-h, result[1], result[2], shape);
		float dyp = execute_correlation(result[0], result[1]+h, result[2], shape);
		float dyn = execute_correlation(result[0], result[1]-h, result[2], shape);
		float dzp = execute_correlation(result[0], result[1], result[2]+h, shape);
		float dzn = execute_correlation(result[0], result[1], result[2]-h, shape);

		float dx = (dxp-dxn)/(2.f*h);
		float dy = (dyp-dyn)/(2.f*h);
		float dz = (dzp-dzn)/(2.f*h);
		float length = sqrtf(dx*dx+dy*dy+dz*dz);
		dx /= length;
		dy /= length;
		dz /= length;

		out_step[0] = gamma*dx;
		out_step[1] = gamma*dy;
		out_step[2] = gamma*dz;

		float next_corr = execute_correlation(result[0]+gamma*dx, result[1]+gamma*dy, result[2]+gamma*dz, shape);

		return next_corr;
	}
	float BruteForce::next_gradientascentstep(float result[6], float h_trans, float h_rot, float gamma, int dofflag[6],
			int shape[3], float out_step[6], float rotcenter[3], bool use_mask)
	{
		//get gradient
		float dxp = 0; float dxn = 0; float dyp = 0; float dyn = 0; float dzp = 0; float dzn = 0.0f;
		float djp = 0; float djn = 0; float drp = 0; float drn = 0; float dpp = 0; float dpn = 0.0f;

		if (dofflag[0] != 0){
		dxp = execute_correlation(result[3], result[4], result[5], result[0]+h_trans, result[1], result[2], shape, rotcenter, use_mask);
		dxn = execute_correlation(result[3], result[4], result[5], result[0]-h_trans, result[1], result[2], shape, rotcenter, use_mask);}
		if (dofflag[1] != 0){
		dyp = execute_correlation(result[3], result[4], result[5], result[0], result[1]+h_trans, result[2], shape, rotcenter, use_mask);
		dyn = execute_correlation(result[3], result[4], result[5], result[0], result[1]-h_trans, result[2], shape, rotcenter, use_mask);}
		if (dofflag[2] != 0){
		dzp = execute_correlation(result[3], result[4], result[5], result[0], result[1], result[2]+h_trans, shape, rotcenter, use_mask);
		dzn = execute_correlation(result[3], result[4], result[5], result[0], result[1], result[2]-h_trans, shape, rotcenter, use_mask);}
		if (dofflag[3] != 0){
		djp = execute_correlation(result[3]+h_rot, result[4], result[5], result[0], result[1], result[2], shape, rotcenter, use_mask);
		djn = execute_correlation(result[3]-h_rot, result[4], result[5], result[0], result[1], result[2], shape, rotcenter, use_mask);}
		if (dofflag[4] != 0){
		drp = execute_correlation(result[3], result[4]+h_rot, result[5], result[0], result[1], result[2], shape, rotcenter, use_mask);
		drn = execute_correlation(result[3], result[4]-h_rot, result[5], result[0], result[1], result[2], shape, rotcenter, use_mask);}
		if (dofflag[5] != 0){
		dpp = execute_correlation(result[3], result[4], result[5]+h_rot, result[0], result[1], result[2], shape, rotcenter, use_mask);
		dpn = execute_correlation(result[3], result[4], result[5]-h_rot, result[0], result[1], result[2], shape, rotcenter, use_mask);}

		float dx = (dxp-dxn)/(2.f*h_trans);
		float dy = (dyp-dyn)/(2.f*h_trans);
		float dz = (dzp-dzn)/(2.f*h_trans);
		float dj = (djp-djn)/(2.f*h_rot);
		float dr = (drp-drn)/(2.f*h_rot);
		float dp = (dpp-dpn)/(2.f*h_rot);
		float length = sqrtf(dx*dx+dy*dy+dz*dz+dr*dr+dj*dj+dp*dp);
		dx /= length;
		dy /= length;
		dz /= length;
		dr /= length;
		dj /= length;
		dp /= length;

		out_step[0] = gamma*dx;
		out_step[1] = gamma*dy;
		out_step[2] = gamma*dz;
		out_step[3] = gamma*dj;
		out_step[4] = gamma*dr;
		out_step[5] = gamma*dp;

		float next_corr = execute_correlation(result[3]+gamma*dj, result[4]+gamma*dr, result[5]+gamma*dp,result[0]+gamma*dx, result[1]+gamma*dy, result[2]+gamma*dz,
				shape, rotcenter, use_mask);

		return next_corr;
	}
	float BruteForce::ascent_translation_singledimension(int dim, float best_corr, float stepsize, float result[3], int shape[3], int max_extensions)
	{
		float increment[3] = {0.0, 0.0, 0.0};
		increment[dim] += stepsize;
		bool change = false;
		int n_extensions = 0;
		int n_forward = 0;

		float pcorr = execute_correlation(result[0]+increment[0], result[1]+increment[1], result[2]+increment[2], shape);
		while(pcorr > best_corr || n_extensions < max_extensions)
		{
			change = true;
			if (pcorr > best_corr)
			{
				best_corr = pcorr;
				result[dim] += (n_extensions+1)*stepsize;
				n_extensions = 0;
				n_forward++;
			}
			else
				n_extensions++;
			float pcorr = execute_correlation(result[0]+(n_extensions+1)*increment[0], result[1]+(n_extensions+1)*increment[1], result[2]+(n_extensions+1)*increment[2], shape);
		}

		if (!change || n_forward < max_extensions)
		{
			n_extensions = n_forward;
			float ncorr = execute_correlation(result[0]-(n_extensions+1)*increment[0], result[1]-(n_extensions+1)*increment[1], result[2]-(n_extensions+1)*increment[2], shape);
			while(ncorr > best_corr || n_extensions < max_extensions)
			{
				if (ncorr > best_corr)
				{
					best_corr = ncorr;
					result[dim] -= (n_extensions+1)*stepsize;
					n_extensions = 0;
				}
				else
					n_extensions++;
				float ncorr = execute_correlation(result[0]-(n_extensions+1)*increment[0], result[1]-(n_extensions+1)*increment[1], result[2]-(n_extensions+1)*increment[2], shape);
			}
		}
		return best_corr;
	}
	float BruteForce::ascent_singledimension(int dim, float best_corr, float stepsize, float result[3], int shape[3], int max_extensions, float rotcenter[3], bool use_mask)
	{
		float increment[6] = {0.0, 0.0, 0.0, 0.0, 0.0, 0.0};
		increment[dim] += stepsize;
		bool change = false;
		int n_extensions = 0;
		int n_forward = 0;

		float pcorr = execute_correlation(result[3]+increment[3], result[4]+increment[4], result[5]+increment[5],
				                          result[0]+increment[0], result[1]+increment[1], result[2]+increment[2], shape, rotcenter, use_mask);
		while(pcorr > best_corr || n_extensions < max_extensions)
		{
			change = true;
			if (pcorr > best_corr)
			{
				best_corr = pcorr;
				result[dim] += (n_extensions+1)*stepsize;
				n_extensions = 0;
				n_forward++;
			}
			else
				n_extensions++;
			float pcorr = execute_correlation(result[3]+(n_extensions+1)*increment[3], result[4]+(n_extensions+1)*increment[4], result[5]+(n_extensions+1)*increment[5],
					                          result[0]+(n_extensions+1)*increment[0], result[1]+(n_extensions+1)*increment[1], result[2]+(n_extensions+1)*increment[2], shape, rotcenter, use_mask);
		}

		if (!change || n_forward < max_extensions)
		{
			n_extensions = n_forward;
			float ncorr = execute_correlation(result[3]-(n_extensions+1)*increment[3], result[4]-(n_extensions+1)*increment[4], result[5]-(n_extensions+1)*increment[5],
					                          result[0]-(n_extensions+1)*increment[0], result[1]-(n_extensions+1)*increment[1], result[2]-(n_extensions+1)*increment[2], shape, rotcenter, use_mask);
			while(ncorr > best_corr || n_extensions < max_extensions)
			{
				if (ncorr > best_corr)
				{
					best_corr = ncorr;
					result[dim] -= (n_extensions+1)*stepsize;
					n_extensions = 0;
				}
				else
					n_extensions++;
				float ncorr = execute_correlation(result[3]-(n_extensions+1)*increment[3], result[4]-(n_extensions+1)*increment[4], result[5]-(n_extensions+1)*increment[5],
						                          result[0]-(n_extensions+1)*increment[0], result[1]-(n_extensions+1)*increment[1], result[2]-(n_extensions+1)*increment[2], shape, rotcenter, use_mask);
			}
		}
		return best_corr;
	}

	void BruteForce::prepare_rotation_coefficients(float *out_coefficients, float jaw, float roll, float pitch)
	{
		//Preprare rotation coefficients
		float phi = jaw*0.01745329252f;
		float theta = pitch*0.01745329252f;
		float psi = roll*0.01745329252f;

		float costheta = cos(theta);
		float sintheta = sin(theta);
		float cospsi = cos(psi);
		float sinpsi = sin(psi);
		float sinpsisintheta = sinpsi*sintheta;
		float cospsisintheta = cospsi*sintheta;
		float cosphi = cos(phi);
		float sinphi = sin(phi);

		//pitch-roll-yaw convention
		float a11 = costheta*cosphi;
		float a12 = costheta*sinphi;
		float a13 = -sintheta;
		float a21 = sinpsisintheta*cosphi-cospsi*sinphi;
		float a22 = sinpsisintheta*sinphi+cospsi*cosphi;
		float a23 = costheta*sinpsi;
		float a31 = cospsisintheta*cosphi+sinpsi*sinphi;
		float a32 = cospsisintheta*sinphi-sinpsi*cosphi;
		float a33 = costheta*cospsi;

		out_coefficients[0] = a11; out_coefficients[1] = a12; out_coefficients[2] = a13;
		out_coefficients[3] = a21; out_coefficients[4] = a22; out_coefficients[5] = a23;
		out_coefficients[6] = a31; out_coefficients[7] = a32; out_coefficients[8] = a33;

		return;
	}
}
