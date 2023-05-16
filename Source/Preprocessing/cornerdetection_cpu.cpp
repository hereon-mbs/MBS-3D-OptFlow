#include <iostream>
#include <string.h>
#include <cstdint>
#include <omp.h>
#include <math.h>

#include "cornerdetection_cpu.h"
#include "../Geometry/filtering.h"
#include "../Geometry/derivatives.h"

namespace lk
{
	img_type* NobleCornerDetector::detectcorners(img_type *img, int shape[3], float sigma_gauss_integration, int radius_suppression, int kill_boundaries,
			bool box_maxima, float min_fraction)
	{
		int difference_id = 0; //0: Sobel, 1: central_diff,2: reweighted sobel
		int corner_measure = 1; //Harris, Noble

		float epsilon = 1e-9;

		int nx = shape[0]; int ny = shape[1]; int nz = shape[2];
		long long int nslice = nx*ny;
		long long int nstack = nslice*nz;

		float *Ixx = (float*) malloc(nstack*sizeof(*Ixx));
		float *Iyy = (float*) malloc(nstack*sizeof(*Iyy));
		float *Izz = (float*) malloc(nstack*sizeof(*Izz));
		float *Ixy = (float*) malloc(nstack*sizeof(*Ixy));
		float *Ixz = (float*) malloc(nstack*sizeof(*Ixz));
		float *Iyz = (float*) malloc(nstack*sizeof(*Iyz));

		//Derivatives
		get_structuretensor_components(img, Ixx, Iyy, Izz, Ixy, Ixz, Iyz, shape, difference_id);

		//Gaussian
		filter::apply_3DGaussianFilter(Ixx, shape, sigma_gauss_integration);
		filter::apply_3DGaussianFilter(Ixy, shape, sigma_gauss_integration);
		filter::apply_3DGaussianFilter(Ixz, shape, sigma_gauss_integration);
		filter::apply_3DGaussianFilter(Iyy, shape, sigma_gauss_integration);
		filter::apply_3DGaussianFilter(Iyz, shape, sigma_gauss_integration);
		filter::apply_3DGaussianFilter(Izz, shape, sigma_gauss_integration);

		//Corner Measure
		float mc_ev = 0.0f;

		#pragma omp parallel for reduction(+: mc_ev)
		for (long long int pos = 0; pos < nstack; pos++)
		{
			float xx = Ixx[pos];
			float xy = Ixy[pos];
			float xz = Ixz[pos];
			float yy = Iyy[pos];
			float yz = Iyz[pos];
			float zz = Izz[pos];

			float det_A = (xx*yy*zz) + 2.*(xy*yz*xz) - (xz*xz*yy) - (yz*yz*xx) - (xy*xy*zz);
			//float det_A = (xx*yy)-(xy*xy);
			float tr_A = xx+yy+zz;
			float m_corner;

			if (corner_measure == 0) m_corner = det_A-kappa*(tr_A*tr_A);
			else m_corner = 2.f*det_A/(tr_A + epsilon);

			Ixx[pos] = m_corner;
			mc_ev += m_corner;
		}
		mc_ev /= nstack;

		//Non-maximum suppression
		if (radius_suppression != 0)
		{
			int fsize = radius_suppression;

			//Could dilate on Iyy but in a small window brute force should be more efficient
			#pragma omp parallel for
			for (long long int pos = 0; pos < nstack; pos++){

				int z = pos/nslice;
				int y = (pos-z*nslice)/nx;
				int x = (pos-z*nslice-y*nx);

				float val0 = Ixx[pos];

				for (int r = -fsize; r <= fsize; r++)
				{
					if (z+r < 0 || z+r >= nz) continue;

					for (int q = -fsize; q <= fsize; q++)
					{
						if (y+q < 0 || y+q >= ny) continue;

						for (int p = -fsize; p <= fsize; p++)
						{
							if (x+p < 0 || x+p >= nx) continue;
							if (p == 0 && q == 0 && r == 0) continue;
							if (!box_maxima && p*p+q*q+r*r > fsize*fsize) continue;

							if (Ixx[(z+r)*nslice+(y+q)*nx + (x+p)] > val0){
								val0 = 0.0f; break;}
						}

						if(val0 == 0.0f) break;
					}

					if(val0 == 0.0f) break;
				}

				Iyy[pos] = val0;
			}

			std::swap(Iyy, Ixx);
		}

		if (kill_boundaries > 0)
		{
			int kb = kill_boundaries;

			#pragma omp parallel for
			for (long long int pos = 0; pos < nstack; pos++){
				int z = pos/nslice;
				int y = (pos-z*nslice)/nx;
				int x = (pos-z*nslice-y*nx);

				if (x < kb || x >= nx-kb || y < kb || y >= ny-kb || z < kb || z >= nz-kb)
					Ixx[pos] = 0.0f;
			}
		}

		float valid = 0.0f;
		#pragma omp parallel for reduction(+: valid)
		for (long long int pos = 0; pos < nstack; pos++)
		{
			if (Ixx[pos] < mc_ev*min_fraction) Ixx[pos] = 0.0f;
			else
				valid++;
		}
		std::cout << "valid: " << valid << " (" << valid/nstack*100.  << "%)" << std::endl;

		free(Iyy); free(Izz);
		free(Ixy); free(Iyz); free(Ixz);

		return Ixx;
	}
	void NobleCornerDetector::get_structuretensor_components(img_type *input, float *Ixx, float *Iyy, float *Izz, float *Ixy, float *Ixz, float *Iyz, int shape[3], int difference_id)
	{
		//The tensor is defined over a patch, i.e. it still needs to be filtered by a box or a Gaussian filter

		int nx = shape[0];
		int ny = shape[1];
		int nz = shape[2];
		long long int nslice = shape[0]*shape[1];
		long long int nstack = nslice*shape[2];

		#pragma omp parallel for
		for(int y = 0; y < ny; y++)
		{
			int yn = y-1; int yp = y+1;
			if(yn < 0) yn = 1; if(yp == ny) yp = ny-2;

			for (int z = 0; z < nz; z++)
			{
				long long int znslice = z* nslice;

				int zn = z-1; int zp = z+1;
				if(zn < 0) zn = 1; if(zp == nz) zp = nz-2;

				for(int x = 0; x < nx; x++)
				{
					long long int pos = znslice + y*nx + x;

					int xn = x-1; int xp = x+1;
					if(xn < 0) xn = 1; if(xp == nx) xp = nx-2;

					float dx, dy, dz;

					if (difference_id == 0)
					{
						//sobel-feldmann
						float val001 = input[z*nslice + yn*nx + xn];
						float val101 = input[z*nslice + yn*nx + x ];
						float val201 = input[z*nslice + yn*nx + xp];
						float val011 = input[z*nslice +  y*nx + xn];
						float val211 = input[z*nslice +  y*nx + xp];
						float val021 = input[z*nslice + yp*nx + xn];
						float val121 = input[z*nslice + yp*nx + x ];
						float val221 = input[z*nslice + yp*nx + xp];

						if (nz > 1)
						{
							float val000 = input[zn*nslice + yn*nx + xn];
							float val100 = input[zn*nslice + yn*nx + x ];
							float val200 = input[zn*nslice + yn*nx + xp];
							float val010 = input[zn*nslice +  y*nx + xn];
							float val110 = input[zn*nslice +  y*nx + x ];
							float val210 = input[zn*nslice +  y*nx + xp];
							float val020 = input[zn*nslice + yp*nx + xn];
							float val120 = input[zn*nslice + yp*nx + x ];
							float val220 = input[zn*nslice + yp*nx + xp];

							float val002 = input[zp*nslice + yn*nx + xn];
							float val102 = input[zp*nslice + yn*nx + x ];
							float val202 = input[zp*nslice + yn*nx + xp];
							float val012 = input[zp*nslice +  y*nx + xn];
							float val112 = input[zp*nslice +  y*nx + x ];
							float val212 = input[zp*nslice +  y*nx + xp];
							float val022 = input[zp*nslice + yp*nx + xn];
							float val122 = input[zp*nslice + yp*nx + x ];
							float val222 = input[zp*nslice + yp*nx + xp];

							dx = (val200+val220+val202+val222) - (val000+val020+val002+val022) + 2.f*(val210+val212+val201+val221)
							   - 2.f*(val010+val012+val001+val021) + 4.f*(val211-val011);

							dy = (val020+val022+val220+val222) - (val000+val002+val200+val202) + 2.f*(val021+val221+val120+val122)
								- 2.f*(val001+val201+val100+val102) + 4.f*(val121-val101);

							dz = (val002+val022+val202+val222) - (val000+val020+val200+val220) + 2.f*(val012+val212+val102+val122)
							   - 2.f*(val010+val210+val100+val120) + 4.f*(val112-val110);
						}
						else
						{
							dx = (-val001 + val201) + 2.f*(-val011 + val211) + (-val021 + val221);
							dy = (-val001 - val201) + 2.f*(-val101 + val121) + ( val021 + val221);
							dz = 0.0f;
						}
					}
					else if (difference_id == 1)
					{
						//central_difference
						float val01 = input[z*nslice +  y*nx + xn];
						float val21 = input[z*nslice +  y*nx + xp];
						float val10 = input[z*nslice +  yn*nx + x];
						float val12 = input[z*nslice +  yp*nx + x];

						dx = 0.5f*(val21-val01);
						dy = 0.5f*(val12-val10);
						dz = 0.0f;

						if (nz > 1)
						{
							float val110 = input[zn*nslice +  y*nx + x];
							float val112 = input[zp*nslice +  y*nx + x];

							dz = 0.5f*(val112-val110);
						}
					}
					else if (difference_id == 2)
					{
						//less weight on center than sobel
						float val001 = input[z*nslice + yn*nx + xn];
						float val101 = input[z*nslice + yn*nx + x ];
						float val201 = input[z*nslice + yn*nx + xp];
						float val011 = input[z*nslice +  y*nx + xn];
						float val211 = input[z*nslice +  y*nx + xp];
						float val021 = input[z*nslice + yp*nx + xn];
						float val121 = input[z*nslice + yp*nx + x ];
						float val221 = input[z*nslice + yp*nx + xp];

						if (nz > 1)
						{
							float val000 = input[zn*nslice + yn*nx + xn];
							float val100 = input[zn*nslice + yn*nx + x ];
							float val200 = input[zn*nslice + yn*nx + xp];
							float val010 = input[zn*nslice +  y*nx + xn];
							float val110 = input[zn*nslice +  y*nx + x ];
							float val210 = input[zn*nslice +  y*nx + xp];
							float val020 = input[zn*nslice + yp*nx + xn];
							float val120 = input[zn*nslice + yp*nx + x ];
							float val220 = input[zn*nslice + yp*nx + xp];

							float val002 = input[zp*nslice + yn*nx + xn];
							float val102 = input[zp*nslice + yn*nx + x ];
							float val202 = input[zp*nslice + yn*nx + xp];
							float val012 = input[zp*nslice +  y*nx + xn];
							float val112 = input[zp*nslice +  y*nx + x ];
							float val212 = input[zp*nslice +  y*nx + xp];
							float val022 = input[zp*nslice + yp*nx + xn];
							float val122 = input[zp*nslice + yp*nx + x ];
							float val222 = input[zp*nslice + yp*nx + xp];

							dx = (val200+val220+val202+val222) - (val000+val020+val002+val022) + (val210+val212+val201+val221)
							   - (val010+val012+val001+val021) + (val211-val011);

							dy = (val020+val022+val220+val222) - (val000+val002+val200+val202) + (val021+val221+val120+val122)
								- (val001+val201+val100+val102) + (val121-val101);

							dz = (val002+val022+val202+val222) - (val000+val020+val200+val220) + (val012+val212+val102+val122)
							   - (val010+val210+val100+val120) + (val112-val110);
						}
						else
						{
							dx = (-val001 + val201) + (-val011 + val211) + (-val021 + val221);
							dy = (-val001 - val201) + (-val101 + val121) + ( val021 + val221);
							dz = 0.0f;
						}
					}

					Ixx[pos] = dx*dx;
					Ixy[pos] = dx*dy;
					Ixz[pos] = dx*dz;
					Iyy[pos] = dy*dy;
					Iyz[pos] = dy*dz;
					Izz[pos] = dz*dz;
				}
			}
		}

		return;
	}

	void HistogramsOfOrientedGradients::_add2feature(float* feature_image, float* Ix, float* Iy, float* Iz, int shape[3], long long int idx, long long int pos)
	{
		int nx = shape[0]; int ny = shape[1]; int nz = shape[2];
		long long int nslice = nx*ny;

		int z0 = idx/nslice;
		int y0 = (idx-z0*nslice)/nx;
		int x0 = idx-z0*nslice-y0*nx;

		for(int r = -radius_neighbourhood; r <= radius_neighbourhood; r++){
			int z2 = (z0+r >= 0) ? (z0+r < nz ? z0+r : 2*nz-(z0+r)-2) : z0-r;
			for(int q = -radius_neighbourhood; q <= radius_neighbourhood; q++){
				int y2 = (y0+q >= 0) ? (y0+q < ny ? y0+q : 2*ny-(y0+q)-2) : y0-q;
				for(int p = -radius_neighbourhood; p <= radius_neighbourhood; p++){
					int x2 = (x0+p >= 0) ? (x0+p < nx ? x0+p : 2*nx-(x0+p)-2) : x0-p;

					long long int idx2 = z2*nslice + y2*nx + x2;

					float radius = Ix[idx2];
					float phi = Iy[idx2];
					float theta = Iz[idx2];

					//split into 4 bins
					float weight_c0 = phi-floor(phi); float weight_f0 = 1.f-weight_c0;
					float weight_c1 = theta-floor(theta); float weight_f1 = 1.f-weight_c1;

					float val_ff = weight_f0*weight_f1*radius;
					float val_cf = weight_c0*weight_f1*radius;
					float val_fc = weight_f0*weight_c1*radius;
					float val_cc = weight_c0*weight_c1*radius;

					int p_ff = floor(phi)*angular_bins_inclination + floor(theta);
					int p_cf = ceil(phi)*angular_bins_inclination + floor(theta);
					int p_fc = floor(phi)*angular_bins_inclination + ceil(theta);
					int p_cc = ceil(phi)*angular_bins_inclination + ceil(theta);

					feature_image[pos+p_ff] += val_ff;
					feature_image[pos+p_cf] += val_cf;
					feature_image[pos+p_fc] += val_fc;
					feature_image[pos+p_cc] += val_cc;
				}}}
		return;
	}

	float* HistogramsOfOrientedGradients::create_HOG_descriptorimage(float* image, float* feature_locations, int shape[3])
	{
		int nx = shape[0]; int ny = shape[1]; int nz = shape[2];
		long long int nslice = nx*ny;
		long long int nstack = nz*nslice;

		int* feature_index = (int*) calloc(nstack, sizeof(*feature_index));

		//calculate expected feature size and required nstack to fit it
		///////////////////////////////////////////////////////////////
		long long int nfeatures = angular_bins_azimuth*angular_bins_inclination; //2 angles in spherical coordinates
		nfeatures *= neighbour_type; //the amount of neighbours

		long long int npossible = nstack/nfeatures;
		long long int nlocations = 0;
		std::cout << "can fit " << npossible << " locations with " << nfeatures << " features each" << std::endl;

		for (long long int pos = 0; pos < nstack; pos++)
		{
			//Let's give the feature_locations an index
			if (feature_locations[pos] > 0.0f){
				nlocations++;
				feature_index[pos] = nlocations;
			}
		}

		if (nlocations > npossible)
		{
			std::cout << "Warning! Found " << nlocations << " feature locations. Continue?" << std::endl;
			std::cin.get();
		}

		float* output = (float*) calloc(nlocations*nfeatures, sizeof(*output));
		///////////////////////////////////////////////////////////////

		//calculate the smoothed derivatives and convert to
		//spherical coordinates
		///////////////////////////////////////////////////////////////
		std::vector<float> gkernel = filter::create_gaussiankernel(presmoothing);
		float* tmp = filter::apply_1Dconvolution(0, image, shape, gkernel);
		float* Ix = derive::firstDerivative_fourthOrder(0, tmp, shape); free(tmp);
		tmp = filter::apply_1Dconvolution(1, image, shape, gkernel);
		float* Iy = derive::firstDerivative_fourthOrder(1, tmp, shape); free(tmp);
		tmp = filter::apply_1Dconvolution(2, image, shape, gkernel);
		float* Iz = derive::firstDerivative_fourthOrder(2, tmp, shape); free(tmp);

		double minval = 1e9; double maxval = -1e9;
		#pragma omp parallel for reduction(min: minval), reduction(max: maxval)
		for(long long int idx = 0; idx < nstack; idx++)
		{
			double dx = Ix[idx]; double dy = Iy[idx]; double dz = Iz[idx];

			double radius = sqrt(dx*dx+dy*dy+dz*dz);
			double theta = atan(sqrt(dx*dx+dy*dy)/dz); //inclination (shifted and normalized to 0 to 1 range)
			double phi; //azimuth (shifted and normalized to 0 to 1 range)

			if (dx > 0.0) phi = atan(dy/dx);
			else if (dx < 0.0) phi = atan(dy/dx)+ 3.14159265359;
			else phi = 1.57079632679;

			theta =(theta+1.57079632679)*57.2958/180.;
			phi = (phi+1.57079632679)*57.2958/360.;

			//now assign a weighted bin
			theta = theta*angular_bins_inclination - 0.5;
			phi = phi*angular_bins_azimuth - 0.5;

			theta = std::max(0.0, std::min(theta, angular_bins_inclination-1.));
			phi = std::max(0.0, std::min(phi, angular_bins_azimuth-1.));

			maxval = std::max(theta, maxval);
			minval = std::min(theta, minval);

			Ix[idx] = radius;
			Iy[idx] = phi;
			Iz[idx] = theta;
		}
		///////////////////////////////////////////////////////////////
		std::cout << minval << " " << maxval << std::endl;

		//calculate the HOG descriptor and write in a concatenated way (feature_index-1)
		///////////////////////////////////////////////////////////////
		#pragma omp parallel for
		for(long long int idx = 0; idx < nstack; idx++)
		{
			if (feature_index[idx] != 0)
			{
				int z0 = idx/nslice;
				int y0 = (idx-z0*nslice)/nx;
				int x0 = idx-z0*nslice-y0*nx;

				//central position
				long long int pos = (feature_index[idx]-1)*nfeatures;
				_add2feature(output, Ix, Iy, Iz, shape, idx, pos);

				for (int i = 1; i < neighbour_type; i++)
				{
					pos += angular_bins_inclination*angular_bins_azimuth; //shifted to next histogram

					int x1 = x0; int y1 = y0; int z1 = z0;

					if (i == 1 || i == 7  || i == 9  || i == 11  || i == 13 || i == 19 || i == 21 || i == 23 || i == 25) x1 = x1+1 < nx ? x1+1 : x1-1;
					if (i == 2 || i == 8  || i ==10  || i == 12  || i == 14 || i == 20 || i == 22 || i == 24 || i == 26) x1 = x1-1 >= 0 ? x1-1 : x1+1;
					if (i == 3 || i == 7  || i == 8  || i == 15  || i == 17 || i == 19 || i == 20 || i == 23 || i == 24) y1 = y1+1 < ny ? y1+1 : y1-1;
					if (i == 4 || i == 9  || i ==10  || i == 16  || i == 18 || i == 21 || i == 22 || i == 25 || i == 26) y1 = y1-1 >= 0 ? y1-1 : y1+1;
					if (i == 5 || i ==11  || i ==12  || i == 15  || i == 16 || i == 19 || i == 20 || i == 21 || i == 22) z1 = z1+1 < nz ? z1+1 : z1-1;
					if (i == 6 || i ==13  || i ==14  || i == 17  || i == 18 || i == 23 || i == 24 || i == 25 || i == 26) z1 = z1-1 >= 0 ? z1-1 : z1+1;

					long long int idx1 = z1*nslice + y1*nx + x1;

					_add2feature(output, Ix, Iy, Iz, shape, idx1, pos);
				}
			}
		}
		///////////////////////////////////////////////////////////////

		free(Ix); free(Iy); free(Iz);
		return output; //now e.g. brute force through a window of allowed shifts and compare features with im2 to get a translation
	}
}
