#include <iostream>
#include <algorithm>
#include <omp.h>
#include <math.h>
#include "filtering.h"
#include "histogram.h"
#include "eig3.h"
#include "auxiliary.h"

namespace derive
{
	typedef float imgtype;

	imgtype* firstDerivative_fourthOrder(int dim, imgtype *imgstack, int shape[3])
	{
		int nx = shape[0]; int ny = shape[1]; int nz = shape[2];
		long long int nslice = nx*ny;
		long long int nstack = nslice*nz;
		long long int asize1 = nstack;

		imgtype* output = (imgtype*) malloc(asize1*sizeof(*output));

		float normalizer = 1.f/12.f;

		#pragma omp parallel for
		for (long long int pos = 0; pos < nstack; pos++)
		{
			int z = pos/nslice;
			int y = (pos-z*nslice)/nx;
			int x = pos-z*nslice-y*nx;

			if (dim == 0)
			{
				//x-derivative

				//Reflective boundary conditions (mirrored on first/last value)
				int xp = x+1; int xn = x-1;
				int xp2 = x+2; int xn2 = x-2;
				if (xp == nx) xp -= 2; if (xn < 0) xn = 1;
				if (xp2 >= nx) xp2 -= 2*nx-xp2-2; if (xn2 < 0) xn2 = -xn2;

				imgtype val_xn2 = imgstack[z*nslice + y*nx + xn2];
				imgtype val_xn  = imgstack[z*nslice + y*nx + xn ];
				imgtype val_xp  = imgstack[z*nslice + y*nx + xp ];
				imgtype val_xp2 = imgstack[z*nslice + y*nx + xp2];

				imgtype dx = normalizer*(-val_xn2 +8*val_xn -8*val_xp +val_xp2);
				output[pos] = dx;
			}
			else if (dim == 1)
			{
				//y-derivative
				int yp = y+1; int yn = y-1;
				int yp2 = y+2; int yn2 = y-2;
				if (yp == ny) yp -= 2; if (yn < 0) yn = 1;
				if (yp2 >= ny) yp2 -= 2*ny-yp2-2; if (yn2 < 0) yn2 = -yn2;

				imgtype val_yn2 = imgstack[z*nslice + yn2*nx + x];
				imgtype val_yn  = imgstack[z*nslice + yn*nx + x ];
				imgtype val_yp  = imgstack[z*nslice + yp*nx + x ];
				imgtype val_yp2 = imgstack[z*nslice + yp2*nx + x];

				imgtype dy = normalizer*(-val_yn2 +8*val_yn -8*val_yp +val_yp2);
				output[pos] = dy;
			}
			else
			{
				//z-derivative
				int zp = z+1; int zn = z-1;
				int zp2 = z+2; int zn2 = z-2;
				if (zp == nz) zp -= 2; if (zn < 0) zn = 1;
				if (zp2 >= nz) zp2 -= 2*nz-zp2-2; if (zn2 < 0) zn2 = -zn2;

				imgtype val_zn2 = imgstack[zn2*nslice + y*nx + x];
				imgtype val_zn  = imgstack[zn*nslice  + y*nx + x];
				imgtype val_zp  = imgstack[zp*nslice  + y*nx + x];
				imgtype val_zp2 = imgstack[zp2*nslice + y*nx + x];

				imgtype dz = normalizer*(-val_zn2 +8*val_zn -8*val_zp +val_zp2);
				output[pos] = dz;
			}
		}

		return output;
	}
	imgtype* firstDerivative_fourthOrder(imgtype *imgstack, int shape[3])
	{
		int nx = shape[0]; int ny = shape[1]; int nz = shape[2];
		long long int nslice = nx*ny;
		long long int nstack = nslice*nz;
		long long int asize1 = 3*nstack;
		if (nz <= 1)  asize1 = 2*nstack;

		imgtype* output = (imgtype*) malloc(asize1*sizeof(*output));

		float normalizer = 1.f/12.f;

		#pragma omp parallel for
		for (long long int pos = 0; pos < nstack; pos++)
		{
			int z = pos/nslice;
			int y = (pos-z*nslice)/nx;
			int x = pos-z*nslice-y*nx;

			int zp = z+1; int zn = z-1;
			int yp = y+1; int yn = y-1;
			int xp = x+1; int xn = x-1;
			int zp2 = z+2; int zn2 = z-2;
			int yp2 = y+2; int yn2 = y-2;
			int xp2 = x+2; int xn2 = x-2;

			//Reflective boundary conditions (mirrored on first/last value)
			if (zp == nz) zp -= 2; if (zn < 0) zn = 1;
			if (yp == ny) yp -= 2; if (yn < 0) yn = 1;
			if (xp == nx) xp -= 2; if (xn < 0) xn = 1;
			if (zp2 >= nz) zp2 -= 2*nz-zp2-2; if (zn2 < 0) zn2 = -zn2;
			if (yp2 >= ny) yp2 -= 2*ny-yp2-2; if (yn2 < 0) yn2 = -yn2;
			if (xp2 >= nx) xp2 -= 2*nx-xp2-2; if (xn2 < 0) xn2 = -xn2;

			//x-derivative
			imgtype val_xn2 = imgstack[z*nslice + y*nx + xn2];
			imgtype val_xn  = imgstack[z*nslice + y*nx + xn ];
			imgtype val_xp  = imgstack[z*nslice + y*nx + xp ];
			imgtype val_xp2 = imgstack[z*nslice + y*nx + xp2];

			//y-derivative
			imgtype val_yn2 = imgstack[z*nslice + yn2*nx + x];
			imgtype val_yn  = imgstack[z*nslice + yn*nx + x ];
			imgtype val_yp  = imgstack[z*nslice + yp*nx + x ];
			imgtype val_yp2 = imgstack[z*nslice + yp2*nx + x];

			imgtype dx = normalizer*(-val_xn2 +8*val_xn -8*val_xp +val_xp2);
			imgtype dy = normalizer*(-val_yn2 +8*val_yn -8*val_yp +val_yp2);

			output[pos] = dx;
			output[nstack+pos] = dy;

			//z-derivative
			if (nz > 1)
			{
				imgtype val_zn2 = imgstack[zn2*nslice + y*nx + x];
				imgtype val_zn  = imgstack[zn*nslice  + y*nx + x];
				imgtype val_zp  = imgstack[zp*nslice  + y*nx + x];
				imgtype val_zp2 = imgstack[zp2*nslice + y*nx + x];

				imgtype dz = normalizer*(-val_zn2 +8*val_zn -8*val_zp +val_zp2);
				output[2*nstack + pos] = dz;
			}
		}

		return output;
	}
	imgtype* firstDerivativeMagnitude_fourthOrder(imgtype *imgstack, int shape[3])
	{
		int nx = shape[0]; int ny = shape[1]; int nz = shape[2];
		long long int nslice = nx*ny;
		long long int nstack = nslice*nz;
		long long int asize1 = 3*nstack;
		if (nz <= 1)  asize1 = 2*nstack;

		imgtype* output = (imgtype*) malloc(asize1*sizeof(*output));

		float normalizer = 1.f/12.f;

		#pragma omp parallel for
		for (long long int pos = 0; pos < nstack; pos++)
		{
			int z = pos/nslice;
			int y = (pos-z*nslice)/nx;
			int x = pos-z*nslice-y*nx;

			int zp = z+1; int zn = z-1;
			int yp = y+1; int yn = y-1;
			int xp = x+1; int xn = x-1;
			int zp2 = z+2; int zn2 = z-2;
			int yp2 = y+2; int yn2 = y-2;
			int xp2 = x+2; int xn2 = x-2;

			//Reflective boundary conditions (mirrored on first/last value)
			if (zp == nz) zp -= 2; if (zn < 0) zn = 1;
			if (yp == ny) yp -= 2; if (yn < 0) yn = 1;
			if (xp == nx) xp -= 2; if (xn < 0) xn = 1;
			if (zp2 >= nz) zp2 -= 2*nz-zp2-2; if (zn2 < 0) zn2 = -zn2;
			if (yp2 >= ny) yp2 -= 2*ny-yp2-2; if (yn2 < 0) yn2 = -yn2;
			if (xp2 >= nx) xp2 -= 2*nx-xp2-2; if (xn2 < 0) xn2 = -xn2;

			//x-derivative
			imgtype val_xn2 = imgstack[z*nslice + y*nx + xn2];
			imgtype val_xn  = imgstack[z*nslice + y*nx + xn ];
			imgtype val_xp  = imgstack[z*nslice + y*nx + xp ];
			imgtype val_xp2 = imgstack[z*nslice + y*nx + xp2];

			//y-derivative
			imgtype val_yn2 = imgstack[z*nslice + yn2*nx + x];
			imgtype val_yn  = imgstack[z*nslice + yn*nx + x ];
			imgtype val_yp  = imgstack[z*nslice + yp*nx + x ];
			imgtype val_yp2 = imgstack[z*nslice + yp2*nx + x];

			imgtype dx = normalizer*(-val_xn2 +8*val_xn -8*val_xp +val_xp2);
			imgtype dy = normalizer*(-val_yn2 +8*val_yn -8*val_yp +val_yp2);
			imgtype dz = 0.0f;

			//z-derivative
			if (nz > 1)
			{
				imgtype val_zn2 = imgstack[zn2*nslice + y*nx + x];
				imgtype val_zn  = imgstack[zn*nslice  + y*nx + x];
				imgtype val_zp  = imgstack[zp*nslice  + y*nx + x];
				imgtype val_zp2 = imgstack[zp2*nslice + y*nx + x];

				dz = normalizer*(-val_zn2 +8*val_zn -8*val_zp +val_zp2);
			}

			output[pos] = sqrtf(dx*dx+dy*dy+dz*dz);
		}

		return output;
	}
	void add_gradientmask(imgtype *confidencemap, imgtype *frame0, imgtype *frame1, int shape[3], float p_used, float sigma_blur, int n_histobins)
	{
		int nx = shape[0]; int ny = shape[1]; int nz = shape[2];
		long long int nslice = nx*ny;
		long long int nstack = nz*nslice;

		imgtype *spatial_gradient = firstDerivativeMagnitude_fourthOrder(frame0, shape);
		imgtype *temporal_gradient = (imgtype*) malloc(nstack*sizeof(*temporal_gradient));

		//temporal derivative
		#pragma omp parallel for
		for (long long int pos = 0; pos < nstack; pos++)
			temporal_gradient[pos] = fabs(frame0[pos]-frame1[pos]);

		//apply gaussian filter
		if (sigma_blur > 0.0f)
		{
			filter::apply_3DGaussianFilter(spatial_gradient, shape, sigma_blur);
			filter::apply_3DGaussianFilter(temporal_gradient, shape, sigma_blur);
		}

		//Get effective histograms
		histo::Histogram histo;
		std::vector<double> histobins1, histobins2, histoedges1, histoedges2;
		histo.calculateeffectivehistogram(spatial_gradient, shape, n_histobins, histobins1, histoedges1);
		histo.calculateeffectivehistogram(temporal_gradient, shape, n_histobins, histobins2, histoedges2);
		histobins1 = histo.normalize(histobins1, "area");
		histobins2 = histo.normalize(histobins2,"area");

		double p_cum1 = 0.0; double p_cum2 = 0.0;
		int cutoff1 = n_histobins-1;; int cutoff2 = n_histobins-1;;

		//find cutoff bins
		for (int i = 0; i < n_histobins; i++)
		{
			p_cum1 += histobins1[i];
			if (p_cum1 > (1.-p_used))
			{
				cutoff1 = i;
				break;
			}
		}
		for (int i = 0; i < n_histobins; i++)
		{
			p_cum2 += histobins2[i];
			if (p_cum2 > (1.-p_used))
			{
				cutoff2 = i;
				break;
			}
		}

		cutoff1 = std::max(cutoff1, 1);
		cutoff2 = std::max(cutoff2, 1);

		//apply cutoff to confidencemap
		#pragma omp parallel for
		for (long long int pos = 0; pos < nstack; pos++)
		{
			imgtype val1 = spatial_gradient[pos];
			imgtype val2 = temporal_gradient[pos];

			if (val1 < histoedges1[cutoff1])// && val2 < histoedges2[cutoff2])
				confidencemap[pos] = 0.0f;
		}

		free(spatial_gradient);
		free(temporal_gradient);

		return;
	}
	void add_gradientweightedconfidence(imgtype *confidencemap, imgtype *frame0, int shape[3], float sigma_blur)
	{
		int nx = shape[0]; int ny = shape[1]; int nz = shape[2];
		long long int nslice = nx*ny;
		long long int nstack = nz*nslice;

		imgtype *spatial_gradient = firstDerivativeMagnitude_fourthOrder(frame0, shape);

		//apply gaussian filter
		if (sigma_blur > 0.0f)
			filter::apply_3DGaussianFilter(spatial_gradient, shape, sigma_blur);

		aux::normalizeframe_histogram(spatial_gradient, shape);

		//apply cutoff to confidencemap
		#pragma omp parallel for
		for (long long int pos = 0; pos < nstack; pos++)
			confidencemap[pos] = spatial_gradient[pos];

		free(spatial_gradient);

		return;
	}
	void add_intensity_and_gradientweightedconfidence(imgtype *confidencemap, imgtype *frame0, int shape[3], float sigma_blur)
	{
		int nx = shape[0]; int ny = shape[1]; int nz = shape[2];
		long long int nslice = nx*ny;
		long long int nstack = nz*nslice;

		imgtype *spatial_gradient = firstDerivativeMagnitude_fourthOrder(frame0, shape);

		//apply gaussian filter
		if (sigma_blur > 0.0f)
			filter::apply_3DGaussianFilter(spatial_gradient, shape, sigma_blur);

		aux::normalizeframe_histogram(spatial_gradient, shape);

		//apply cutoff to confidencemap
		#pragma omp parallel for
		for (long long int pos = 0; pos < nstack; pos++)
			confidencemap[pos] = frame0[pos]*(spatial_gradient[pos]);

		free(spatial_gradient);

		return;
	}

	imgtype* calculate_edgeorientation(imgtype *imgstack, int shape[3], std::string mode, float sigma)
	{
		int nx = shape[0]; int ny = shape[1]; int nz = shape[2];
		long long int nslice = nx*ny;
		long long int nstack = nslice*nz;
		long long int asize1 = 2*nstack;
		int ndims = 3;
		if (nz <= 1)  {asize1 = nstack; ndims = 2;}

		imgtype* output = (imgtype*) malloc(asize1*sizeof(*output));
		imgtype* derivatives = firstDerivative_fourthOrder(imgstack, shape);

		if (mode == "structure_tensor")
		{
			//select largest eigenvector (pointing across structures)
			if (ndims == 2)
			{
				//set up structure tensor
				#pragma omp parallel for
				for(uint64_t pos = 0; pos < nstack; pos++)
				{
					imgtype Idx = derivatives[pos];
					imgtype Idy = derivatives[pos+nstack];
					output[pos] = Idx*Idy;

					derivatives[pos] = Idx*Idx;
					derivatives[pos+nstack] = Idy*Idy;
				}

				//and apply blur
				filter::apply_3DGaussianFilter2Vector(derivatives, shape, sigma, ndims);
				filter::apply_3DGaussianFilter2Vector(output, shape, sigma, 1);

				#pragma omp parallel for
				for(uint64_t pos = 0; pos < nstack; pos++)
				{
					float dx2 = derivatives[pos];
					float dy2 = derivatives[pos+nstack];
					float dxy = output[pos];

					float T = dx2+dy2;
					float D = dx2*dy2-dxy*dxy;

					//1st eigenvalue
					float L1 = T/2.f+sqrtf(std::max(1e-10f,(T*T)/4.f-D));

					float x = 0.0f;
					float y = 1.0f;

					//1st eigenvector
						 if (dxy != 0){x = (L1-dy2); y = dxy;}
					else if (dx2 != 0){x = 1.0f; y = 0.0f;}

					float length = sqrtf(x*x+y*y);

					//store angle of vector basis rotation
					output[pos] = -asin(y/length);
				}
			}
			else
			{
				imgtype* temp = (imgtype*) malloc(nstack*sizeof(*temp));

				//set up structure tensor
				#pragma omp parallel for
				for(uint64_t pos = 0; pos < nstack; pos++)
				{
					imgtype Idx = derivatives[pos];
					imgtype Idy = derivatives[pos+nstack];
					imgtype Idz = derivatives[pos+2*nstack];
					output[pos] = Idx*Idy;
					output[pos+nstack] = Idx*Idz;
					temp[pos] = Idy*Idz;

					derivatives[pos] = Idx*Idx;
					derivatives[pos+nstack] = Idy*Idy;
					derivatives[pos+2*nstack] = Idz*Idz;
				}

				//and apply blur
				filter::apply_3DGaussianFilter2Vector(derivatives, shape, sigma, ndims);
				filter::apply_3DGaussianFilter2Vector(output, shape, sigma, 2);
				filter::apply_3DGaussianFilter(temp, shape, sigma);

				#pragma omp parallel for
				for(uint64_t pos = 0; pos < nstack; pos++)
				{
					float dx2 = derivatives[pos];
					float dy2 = derivatives[pos+nstack];
					float dz2 = derivatives[pos+nstack];
					float dxy = output[pos];
					float dxz = output[pos+nstack];
					float dyz = temp[pos];

					double S[3][3] ={
							{dx2,dxy,dxz},
							{dxy,dy2,dyz},
							{dxz,dxz,dz2}};

					double V[3][3] = {0};
					double D[3] = {0};

					eigen_decomposition(S, V, D);

					//select by largest eigenvalue
					int i = 0;
					if(D[1] > D[i]) i = 1;
					if(D[2] > D[i]) i = 2;

					float length = 1.;
					if (fabs(V[0][i]) > 1.e-9f && fabs(V[1][i]) > 1.e-9f)
					{
						length = sqrtf(V[0][i]*V[0][i]+V[1][i]*V[1][i]);
						output[pos] = -asin(V[1][i]/length);
					}else output[pos] = 0.0f;

					if (fabs(V[0][i]) > 1.e-9f && fabs(V[2][i]) > 1.e-9f)
					{
						length = sqrtf(V[0][i]*V[0][i]+V[2][i]*V[2][i]);
						output[pos+nstack] = asin(V[2][i]/length);
					}else output[pos+nstack] = 0.0f;
				}
			}
		}
		else if (mode == "gradient")
		{
			filter::apply_3DGaussianFilter2Vector(derivatives, shape, sigma, ndims);

			#pragma omp parallel for
			for(uint64_t pos = 0; pos < nstack; pos++)
			{
				float dx = derivatives[pos];
				float dy = derivatives[pos+nstack];

				float length = 1.;

				if (fabs(dx) > 1.e-9f && fabs(dy) > 1.e-9f)
				{
					//yaw
					length = sqrtf(dx*dx+dy*dy);
					output[pos] = -asin(dy/length);//*180./M_PI;
				}
				else output[pos] = 0.0f;

				if(ndims == 3)
				{
					float dz = derivatives[pos+2*nstack];
					length = 1.0f;

					if (fabs(dx) > 1.e-9f && fabs(dz) > 1.e-9f)
					{
						//pitch
						length = sqrtf(dx*dx+dz*dz);
						output[pos+nstack] = asin(dz/length);//needs to be flipped compared to yaw because z-axis is upside down
					}
					else
						output[pos+nstack] = 0.0f;
				}
			}
		}

		return output;
	}
}
