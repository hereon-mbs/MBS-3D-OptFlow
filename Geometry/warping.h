#ifndef WARPING_H
#define WARPING_H

#include <iostream>
#include <string.h>
#include <cstdint>
#include <math.h>
#include "../protocol_parameters.h"

namespace warp
{
	typedef float img_type;
	typedef float optflow_type;
	typedef long long int idx_type;

	float interpolate_cubic(float y0, float y1, float y2, float y3, float mu)
	{
		float mu2 = mu*mu;

		float a0 = y3-y2-y0+y1;
		float a1 = y0-y1-a0;
		float a2 = y2-y0;
		float a3 = y1;

		return a0*mu*mu2+a1*mu2+a2*mu+a3;
	}

	void warpFrame1_z(img_type *output, img_type *frame0, img_type *frame1, optflow_type *u, int shape[3], optflow::ProtocolParameters *params)
	{
		//No need to allocate additional memory. We can use the phi array.
		//////////////////////////////////////////////////////////////////
		int outOfBounds_id = 0;
		int interpolation_id = 0;

		if (params->warp.outOfBounds_mode == "replace") outOfBounds_id = 0;
		else if (params->warp.outOfBounds_mode == "NaN") outOfBounds_id = 1;
		else if (params->warp.outOfBounds_mode == "zero") outOfBounds_id = 2;
		else std::cout << "Warning! Unknown outOfBounds_mode!" << std::endl;

		if (params->warp.interpolation_mode == "cubic") interpolation_id = 1;
		else if (params->warp.interpolation_mode == "linear") interpolation_id = 0;
		else std::cout << "Warning! Unknow warp interpolation mode!" << std::endl;

		int nx = shape[0];
		int ny = shape[1];
		int nz = shape[2];
		//////////////////////////////////////////////////////////////////

		idx_type nslice = nx*ny;
		idx_type nstack = nz*nslice;

		#pragma omp parallel for
		for (idx_type pos = 0; pos < nstack; pos++)
		{
			/////////////////////////////////////////////
			float w0 = u[pos+2*nstack];

			int z = pos/nslice;
			int y = (pos-z*nslice)/nx;
			int x = pos-z*nslice-y*nx;

			float z0 = z + w0;

			img_type replace_val = 0.0f;
			bool moved_out = false;

			//out of bounds?
			if (z0 < 0 || z0 > (nz-1)) {
				moved_out = true;
				replace_val = frame0[pos];
				z0 = z;

				if (outOfBounds_id == 2)
					replace_val = 0.0;
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
				int zf2 = std::max(0, zf-1);
				int zc2 = std::min(zc+1, nz-1);

				img_type P000 = frame1[zf2*nslice+y*nx + x];
				img_type P001 = frame1[zf *nslice+y*nx + x];
				img_type P002 = frame1[zc *nslice+y*nx + x];
				img_type P003 = frame1[zc2*nslice+y*nx + x];

				value = interpolate_cubic(P000,P001,P002,P003,wz);
			}
			else //linear interpolation
			{
				img_type P000 = frame1[zf*nslice+y*nx + x];
				img_type P001 = frame1[zc*nslice+y*nx + x];

				value = (P001-P000)*wz+P000;
			}

			if (moved_out) value = replace_val;
			/////////////////////////////////////////////

			/////////////////////////////////////////////
			output[pos] = value;
			/////////////////////////////////////////////
		}
		return;
	}
	void warpFrame1_xy(img_type *warped1, img_type *frame0, img_type *frame1, optflow_type *u, int shape[3], optflow::ProtocolParameters *params)
	{
		//////////////////////////////////////////////////////////////////
		int outOfBounds_id = 0;
		int interpolation_id = 0;

		if (params->warp.outOfBounds_mode == "replace") outOfBounds_id = 0;
		else if (params->warp.outOfBounds_mode == "NaN") outOfBounds_id = 1;
		else if (params->warp.outOfBounds_mode == "zero") outOfBounds_id = 2;
		else std::cout << "Warning! Unknown outOfBounds_mode!" << std::endl;

		if (params->warp.interpolation_mode == "cubic") interpolation_id = 1;
		else if (params->warp.interpolation_mode == "linear") interpolation_id = 0;
		else std::cout << "Warning! Unknow warp interpolation mode!" << std::endl;

		int nx = shape[0];
		int ny = shape[1];
		int nz = shape[2];
		//////////////////////////////////////////////////////////////////

		idx_type nslice = nx*ny;
		idx_type nstack = nz*nslice;

		#pragma omp parallel for
		for (idx_type pos = 0; pos < nstack; pos++)
		{
			float u0 = u[pos];
			float v0 = u[pos+nstack];
			float w0 = 0.0f;
			if (nz > 1) w0 = u[pos+2*nstack];

			int z = pos/nslice;
			int y = (pos-z*nslice)/nx;
			int x = pos-z*nslice-y*nx;

			float x0 = x + u0;
			float y0 = y + v0;
			float z0 = z + w0;

			int xf = floor(x0);
			int xc = ceil(x0);
			int yf = floor(y0);
			int yc = ceil(y0);

			float wx = x0-xf;
			float wy = y0-yf;

			float value = 0.0;

			if (y0 < 0.0f || x0 < 0.0f || x0 > (nx-1) || y0 > (ny-1) || z0 < 0.0f || z0 > (nz-1))
			{
				if (outOfBounds_id == 0)
					value = frame0[pos];
				else if (outOfBounds_id == 2)
					value = 0.0;
				else
					value = std::numeric_limits<float>::quiet_NaN();
			}
			else if (interpolation_id == 1)//cubic interpolation
			{
				//extrapolate with zero-gradient
				int xf2 = std::max(0, xf-1);
				int xc2 = std::min(xc+1, shape[0]-1);
				int yf2 = std::max(0, yf-1);
				int yc2 = std::min(yc+1, shape[1]-1);

				float P10 = frame1[z*nslice+yf2*nx + xf];
				float P20 = frame1[z*nslice+yf2*nx + xc];

				float P01 = frame1[z*nslice+yf*nx + xf2];
				float P11 = frame1[z*nslice+yf*nx + xf];
				float P21 = frame1[z*nslice+yf*nx + xc];
				float P31 = frame1[z*nslice+yf*nx + xc2];

				float P02 = frame1[z*nslice+yc*nx + xf2];
				float P12 = frame1[z*nslice+yc*nx + xf];
				float P22 = frame1[z*nslice+yc*nx + xc];
				float P32 = frame1[z*nslice+yc*nx + xc2];

				float P13 = frame1[z*nslice+yc2*nx + xf];
				float P23 = frame1[z*nslice+yc2*nx + xc];

				float gtu = interpolate_cubic(P01,P11,P21,P31,wx);
				float gbu = interpolate_cubic(P02,P12,P22,P32,wx);

				float glv = interpolate_cubic(P10,P11,P12,P13,wy);
				float grv = interpolate_cubic(P20,P21,P22,P23,wy);

				float sigma_lr = (1.-wx)*glv + wx*grv;
				float sigma_bt = (1.-wy)*gtu + wy*gbu;
				float corr_lrbt = P11*(1.-wy)*(1.-wx) + P12*wy*(1.-wx) + P21*(1.-wy)*wx + P22*wx*wy;

				value = sigma_lr+sigma_bt-corr_lrbt;
			}
			else //linear_interpolation
			{
				float P00 = frame1[z*nslice+yf*nx + xf];
				float P10 = frame1[z*nslice+yf*nx + xc];
				float P01 = frame1[z*nslice+yc*nx + xf];
				float P11 = frame1[z*nslice+yc*nx + xc];

				float glv = (P01-P00)*wy+P00; //left
				float grv = (P11-P10)*wy+P10; //right
				float gtu = (P10-P00)*wx+P00; //top
				float gbu = (P11-P01)*wx+P01; //bottom

				float sigma_lr = (1.-wx)*glv + wx*grv;
				float sigma_bt = (1.-wy)*gtu + wy*gbu;
				float corr_lrbt = P00*(1.-wy)*(1.-wx) + P01*wy*(1.-wx) + P10*(1.-wy)*wx + P11*wx*wy;

				value = sigma_lr+sigma_bt-corr_lrbt;
			}

			warped1[pos] = value;
		}

		return;
	}
	img_type* warpFrame1_xyz(img_type *frame0, img_type *frame1, optflow_type *u, int shape[3], optflow::ProtocolParameters *params)
	{
		//frame0 is only passed for replacing out of bounds values


		long long int nslice = shape[0]*shape[1];
		long long int nstack = shape[2]*nslice;

		img_type* intermediate = (img_type*) malloc(nstack*sizeof(*intermediate));

		if(shape[2] == 1)
		{
			warpFrame1_xy(intermediate, frame0, frame1, u, shape, params);
			std::swap(frame1, intermediate);
		}
		else
		{
			warpFrame1_z(intermediate, frame0, frame1, u, shape, params);
			warpFrame1_xy(frame1, frame0, intermediate, u, shape, params);
		}

		free(intermediate);

		return frame1;
	}
	void warpFrame1_xyz(img_type *output, img_type *frame0, img_type *frame1, optflow_type *u, int shape[3], optflow::ProtocolParameters *params)
	{
		//pass a sufficiently sized array such that frame1 can be preserved
		//frame0 is only passed for replacing out of bounds values

		long long int nslice = shape[0]*shape[1];
		long long int nstack = shape[2]*nslice;

		img_type* intermediate = (img_type*) malloc(nstack*sizeof(*intermediate));

		if(shape[2] == 1)
		{
			warpFrame1_xy(intermediate, frame0, frame1, u, shape, params);
			std::swap(output, intermediate);
		}
		else
		{
			warpFrame1_z(intermediate, frame0, frame1, u, shape, params);
			warpFrame1_xy(output, frame0, intermediate, u, shape, params);
		}

		free(intermediate);

		return;
	}

	void warpVector1_xyz(img_type *vector0, img_type *vector1, optflow_type *displacement, int shape[3], optflow::ProtocolParameters *params)
	{
		//Warps the backward result on the forward result for consistency checking

		long long int nslice = shape[0]*shape[1];
		long long int nstack = shape[2]*nslice;

		img_type* intermediate = (img_type*) malloc(nstack*sizeof(*intermediate));
		img_type* warped = (img_type*) malloc(nstack*sizeof(*warped));

		if(shape[2] == 1)
		{
			warpFrame1_xy(intermediate, vector0, vector1, displacement, shape, params);
			warpFrame1_xy(warped, &vector0[nstack], &vector1[nstack], displacement, shape, params);

			#pragma omp parallel for
			for (long long int pos = 0; pos < nstack; pos++)
			{
				vector1[pos] = intermediate[pos];
				vector1[nstack+pos] = warped[pos];
			}
		}
		else
		{
			warpFrame1_z(intermediate, vector0, vector1, displacement, shape, params);
			warpFrame1_xy(warped, vector0, intermediate, displacement, shape, params);

			#pragma omp parallel for
			for (long long int pos = 0; pos < nstack; pos++) vector1[pos] = warped[pos];

			warpFrame1_z(intermediate, &vector0[nstack], &vector1[nstack], displacement, shape, params);
			warpFrame1_xy(warped, &vector0[nstack], intermediate, displacement, shape, params);

			#pragma omp parallel for
			for (long long int pos = 0; pos < nstack; pos++) vector1[nstack+pos] = warped[pos];

			warpFrame1_z(intermediate, &vector0[2*nstack], &vector1[2*nstack], displacement, shape, params);
			warpFrame1_xy(warped, &vector0[2*nstack], intermediate, displacement, shape, params);

			#pragma omp parallel for
			for (long long int pos = 0; pos < nstack; pos++) vector1[2*nstack+pos] = warped[pos];
		}

		free(intermediate);

		return;
	}

	img_type* warpMaskForward_xyz(img_type *mask, optflow_type *displacement, int shape[3], float cutoff = 0.1f)
	{
		//warps the mask to the next timestep

		//////////////////////////////////////////////////////////////////
		int outOfBounds_id = 2; //always using zero replacement
		int interpolation_id = 0; //always linear interpolation

		int nx = shape[0];
		int ny = shape[1];
		int nz = shape[2];
		//////////////////////////////////////////////////////////////////

		idx_type nslice = nx*ny;
		idx_type nstack = nz*nslice;

		img_type *output = (img_type*) calloc(nstack,sizeof(*output));


		//scale to max 1
		#pragma omp parallel for
		for (long long int pos = 0; pos < nstack; pos++)
			mask[pos] = std::min(1.0f,mask[pos]);

		for (idx_type pos = 0; pos < nstack; pos++)
		{
			img_type mask_val = mask[pos];

			if (mask_val > 0.0)
			{
				float u0 = displacement[pos];
				float v0 = displacement[pos+nstack];
				float w0 = 0.0f;
				if (nz > 1) w0 = displacement[pos+2*nstack];

				int z = pos/nslice;
				int y = (pos-z*nslice)/nx;
				int x = pos-z*nslice-y*nx;

				float x0 = x + u0;
				float y0 = y + v0;
				float z0 = z + w0;

				int xf = floor(x0);
				int xc = ceil(x0);
				int yf = floor(y0);
				int yc = ceil(y0);
				int zf = floor(z0);
				int zc = ceil(z0);

				float wx = x0-xf;
				float wy = y0-yf;
				float wz = z0-zf;

				if (xf >= 0 && xf < nx && yf >= 0 && yf < ny && zf >= 0 && zf < nz) output[zf*nslice+yf*nx + xf] += (1.f-wx)*(1.f-wy)*(1.f-wz);
				if (xf >= 0 && xf < nx && yc >= 0 && yc < ny && zf >= 0 && zf < nz) output[zf*nslice+yc*nx + xf] += (1.f-wx)*wy*(1.f-wz);
				if (xc >= 0 && xc < nx && yf >= 0 && yf < ny && zf >= 0 && zf < nz) output[zf*nslice+yf*nx + xc] += wx*(1.f-wy)*(1.f-wz);
				if (xc >= 0 && xc < nx && yc >= 0 && yc < ny && zf >= 0 && zf < nz) output[zf*nslice+yc*nx + xc] += wx*wy*(1.f-wz);
				if (xf >= 0 && xf < nx && yf >= 0 && yf < ny && zf >= 0 && zf < nz) output[zc*nslice+yf*nx + xf] += (1.f-wx)*(1.f-wy)*wz;
				if (xf >= 0 && xf < nx && yc >= 0 && yc < ny && zf >= 0 && zf < nz) output[zc*nslice+yc*nx + xf] += (1.f-wx)*wy*wz;
				if (xc >= 0 && xc < nx && yf >= 0 && yf < ny && zf >= 0 && zf < nz) output[zc*nslice+yf*nx + xc] += wx*(1.f-wy)*wz;
				if (xc >= 0 && xc < nx && yc >= 0 && yc < ny && zf >= 0 && zf < nz) output[zc*nslice+yc*nx + xc] += wx*wy*wz;
			}
		}

		#pragma omp parallel for
		for (long long int pos = 0; pos < nstack; pos++)
		{
			if (output[pos] < cutoff) output[pos] = 0.0f;
			else output[pos] = 1.0f;
		}

		//and fade out the values
		//#pragma omp parallel for
		//for (long long int pos = 0; pos < nstack; pos++)
		//	output[pos] *= mask[pos];

		return output;
	}
}

#endif // WARPING_H
