#ifndef ANALYSIS_H
#define ANALYSIS_H

#include <iostream>
#include <vector>
#include <string.h>
#include "Geometry/histogram.h"
#include "Geometry/hdcommunication.h"
#include "Geometry/auxiliary.h"
#include "Geometry/SymmetricEigensolver3x3.h"

/*********************************************************************************************************************************************************
 * Location: Helmholtz-Zentrum fuer Material und Kuestenforschung, Max-Planck-Strasse 1, 21502 Geesthacht
 * Author: Stefan Bruns
 * Contact: bruns@nano.ku.dk
 *
 * License: TBA
 *
 * Preprocessing:
 * 		normalization: "none"
 * 		               "simple" := rescale to 0-1 interval (same scaling for both frames)
 * 		               "histogram": rescale 99.9% of intensities to 0-1 interval
 * 		               "histogram_independent": perform the histogram scaling (independently on the 2 frames)
 *
 * 		prefilter:     "gaussian"       := 3D Gaussian filtering with specified sigma on frame0 and frame1
 * 					   "median"         := 3D median with spheric kernel where prefilter_sigma is the floating point radius
 * 					   "median_simple"  := 3D median boxfilter where prefilter_sigma is the integer radius
 *
 *********************************************************************************************************************************************************/

namespace anal
{
	typedef float img_type;
	typedef float optflow_type;
	typedef long long int idx_type;

	int n_histobins = 256;

	std::vector<float> get_qualitymeasures(img_type *frame0, img_type *frame1, optflow_type *result, int shape[3], img_type *mask, bool mask_active)
	{
		int nx = shape[0]; int ny = shape[1]; int nz = shape[2];
		idx_type nslice = nx*ny;
		idx_type nstack = nz*nslice;

		int64_t n_roi = 0; //voxels that do not move in or out of the frame
		int64_t n_full = 0; //all but the masked voxels
		uint8_t *roi_mask = (uint8_t*) calloc(nstack, sizeof(*roi_mask));

		//////////////////////////////////
		double mean0_full = 0.0f; double mean1_full = 0.0f;
		double std0_full = 0.0f; double std1_full = 0.0f;
		double corr_full = 0.0f;
		double mssd_full = 0.0f; //mean sum of squared distances
		double mi_full = 0.0f; //mutual information

		double mean0_roi = 0.0f; double mean1_roi = 0.0f;
		double std0_roi = 0.0f; double std1_roi = 0.0f;
		double corr_roi = 0.0f;
		double mssd_roi = 0.0f; //mean sum of squared distances
		double mi_roi = 0.0f; //mutual information
		//////////////////////////////////

		#pragma omp parallel for reduction(+: mean0_full, mean1_full, std0_full, std1_full, mssd_full, n_roi, n_full, mean0_roi, mean1_roi, std0_roi, std1_roi, mssd_roi)
		for (idx_type pos = 0; pos < nstack; pos++)
		{
			if (!mask_active || mask[pos] > 0.0f)
			{
				img_type val0 = frame0[pos];
				img_type val1 = frame1[pos];

				//Measures for the whole frame
				///////////////////////////////////
				mean0_full += val0;
				mean1_full += val1;
				std0_full += val0*val0;
				std1_full += val1*val1;

				mssd_full += (val0-val1)*(val0-val1);
				n_full++;
				///////////////////////////////////

				//Measures for voxels that do not move out of the frame
				///////////////////////////////////
				int z = pos/nslice;
				int y = (pos-z*nslice)/nx;
				int x = pos-z*nslice-y*nx;

				float x0 = x + result[pos];
				float y0 = y + result[pos+nstack];
				float z0 = z;
				if (nz > 1) z0 += result[pos+2*nstack];

				if (x0 >= 0.0f && x0 <= nx-1 && y0 >= 0.0f && y0 <= ny-1 && z0 >= 0.0f && z0 <= nz-1)
				{
					roi_mask[pos] = 1;
					n_roi++;

					mean0_roi += val0;
					mean1_roi += val1;
					std0_roi += val0*val0;
					std1_roi += val1*val1;

					mssd_roi += (val0-val1)*(val0-val1);
				}
				///////////////////////////////////
			}
		}

		mean0_full /= n_full;
		mean1_full /= n_full;
		mssd_full /= n_full;
		mean0_roi /= n_roi;
		mean1_roi /= n_roi;
		mssd_roi /= n_roi;

		std0_full = std::sqrt(std0_full/n_full-mean0_full*mean0_full);
		std1_full = std::sqrt(std1_full/n_full-mean1_full*mean1_full);
		std0_roi = std::sqrt(std0_roi/n_roi-mean0_roi*mean0_roi);
		std1_roi = std::sqrt(std1_roi/n_roi-mean1_roi*mean1_roi);

		#pragma omp parallel for reduction(+: corr_full, corr_roi)
		for (idx_type pos = 0; pos < nstack; pos++)
		{
			if (!mask_active || mask[pos] > 0.0f)
			{
				corr_full += (frame0[pos]-mean0_full)*(frame1[pos]-mean1_full);

				if(roi_mask[pos] == 1)
					corr_roi += (frame0[pos]-mean0_roi)*(frame1[pos]-mean1_roi);
			}
		}
		corr_full /= n_full*(std0_full*std1_full);
		corr_roi /= n_roi*(std0_roi*std1_roi);

		/*//Calculate 2D-Histogram for mutual information
		//////////////////////////////////
		histo::Histogram histo;
		int histoshape[2] = {n_histobins, n_histobins};

		std::vector<double> histobins_f0, histobins_f1, histoedges_f0, histoedges_f1;
		histo.calculateeffectivehistogram(frame0, shape, n_histobins, histobins_f0, histoedges_f0);
		histo.calculateeffectivehistogram(frame1, shape, n_histobins, histobins_f1, histoedges_f1);
		histobins_f0 = histo.normalize(histobins_f0, "area");
		histobins_f1 = histo.normalize(histobins_f1, "area");
		float *histo2D = histo.calculate2DHistogram(frame0, frame1, shape, histoedges_f0, histoedges_f0, histoshape);

		for (int j = 0; j < n_histobins; j++)
		{
			double pj = histobins_f1[j];

			for(int i = 0; i < n_histobins; i++)
			{
				double pi = histobins_f0[i];
				double pij = histo2D[j*n_histobins+i];

				if(pi != 0.0f && pj != 0.0f && pij != 0.0f)
					mi_full += pij*log2(pij/(pi*pj));
			}
		}

		free(histo2D);
		histo.calculatehistogram(frame0, roi_mask, shape, n_histobins, histoedges_f0[0], histoedges_f0[histoedges_f0.size()-1], histobins_f0, histoedges_f0);
		histo.calculatehistogram(frame1, roi_mask, shape, n_histobins, histoedges_f1[0], histoedges_f1[histoedges_f1.size()-1], histobins_f1, histoedges_f1);
		histo2D = histo.calculate2DHistogram(frame0, frame1, roi_mask, shape, histoedges_f0, histoedges_f0, histoshape);
		histobins_f0 = histo.normalize(histobins_f0, "area");
		histobins_f1 = histo.normalize(histobins_f1, "area");

		for (int j = 0; j < n_histobins; j++)
		{
			double pj = histobins_f1[j];

			for(int i = 0; i < n_histobins; i++)
			{
				double pi = histobins_f0[i];
				double pij = histo2D[j*n_histobins+i];

				if(pi != 0.0f && pj != 0.0f && pij != 0.0f)
					mi_roi += pij*log2(pij/(pi*pj));
			}
		}
		//////////////////////////////////*/

		float rel_valid = ((float) n_roi)/n_full;

		//std::cout << "rel_valid: " << rel_valid << std::endl;
		//std::cout << "cross-corr: " << corr_full << " / " << corr_roi << std::endl;
		//std::cout << "mean ssd: " << mssd_full << " / "  << mssd_roi << std::endl;
		//std::cout << "mut info: " << mi_full << " / "  << mi_roi << std::endl;

		std::vector<float> output = {(float) rel_valid,(float) corr_roi, (float) mssd_roi};

		return output;
	}
	void get_autocorrelation(float *img, int shape[3])
	{
		int nx = shape[0];
		int ny = shape[1];
		int nz = shape[2];
		idx_type nslice = nx*ny;
		idx_type nstack = nz*nslice;

		float mean = 0.0f;
		float stdev = 0.0f;
		float corr = 0.0f;

		#pragma omp parallel for reduction(+: mean, stdev)
		for (idx_type pos = 0; pos < nstack; pos++)
		{
			float val = img[pos];
			mean += val;
			stdev += val*val;
		}

		mean /= nstack;
		stdev = std::sqrt(stdev/nstack-mean*mean);

		float counter = 0.0f;

		#pragma omp parallel for reduction(+: corr, counter)
		for (idx_type pos = 0; pos < nstack; pos++)
		{
			int z = pos/nslice;
			int y = (pos-z*nslice)/nx;
			int x = pos-z*nslice-y*nx;

			if (x > 0)    {corr += (img[pos]-mean)*(img[pos-1]-mean); counter++;}
			if (x < nx-1) {corr += (img[pos]-mean)*(img[pos+1]-mean); counter++;}
			if (y > 0)    {corr += (img[pos]-mean)*(img[pos-nx]-mean); counter++;}
			if (y < ny-1) {corr += (img[pos]-mean)*(img[pos+nx]-mean); counter++;}
			if (nz > 1)
			{
				if (z > 0)    {corr += (img[pos]-mean)*(img[pos-nslice]-mean); counter++;}
				if (z < nz-1) {corr += (img[pos]-mean)*(img[pos+nslice]-mean); counter++;}
			}
		}
		corr /= counter*(stdev*stdev);

		std::cout << "autocorr: " << corr << std::endl;

		return;
	}

	img_type* plot_fissures(img_type *vectorfield, img_type *frame0, img_type *frame1, img_type *mask, int shape[3], std::string finitedifference_type)
	{
		//deprecated and replaced with volumetric strain
		int nx = shape[0]; int ny = shape[1]; int nz = shape[2];
		long long int nslice = nx*ny;
		long long int nstack = nz*nslice;

		img_type *output = (img_type*) calloc(nstack,sizeof(*output));

		float normalizer = 1.f/2.f;

		if (finitedifference_type == "centraldifference" || finitedifference_type == "secondorder")
			normalizer = 1.f/2.f;
		else if (finitedifference_type == "Barron" || finitedifference_type == "fourthorder")
			normalizer = 1.f/12.f;
		else
			std::cout << "Warning! unknown finite difference for plot_divergence. Using central difference." << std::endl;

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

			img_type divergence;

			if (finitedifference_type == "Barron" || finitedifference_type == "fourthorder")
			{
				//x-derivative of ux displacement
				img_type val_xn2 = vectorfield[z*nslice + y*nx + xn2];
				img_type val_xn  = vectorfield[z*nslice + y*nx + xn ];
				img_type val_xp  = vectorfield[z*nslice + y*nx + xp ];
				img_type val_xp2 = vectorfield[z*nslice + y*nx + xp2];

				//y-derivative of uy displacement
				img_type val_yn2 = vectorfield[nstack + z*nslice + yn2*nx + x];
				img_type val_yn  = vectorfield[nstack + z*nslice + yn*nx + x ];
				img_type val_yp  = vectorfield[nstack + z*nslice + yp*nx + x ];
				img_type val_yp2 = vectorfield[nstack + z*nslice + yp2*nx + x];

				img_type dx = normalizer*(val_xn2 -8*val_xn + 8*val_xp - val_xp2);
				img_type dy = normalizer*(val_yn2 -8*val_yn + 8*val_yp - val_yp2);
				img_type dz = 0.0f;

				if (shape[2] > 1)
				{
					//z-derivative of uz displacement
					img_type val_zn2 = vectorfield[2*nstack + zn2*nslice + y*nx + x];
					img_type val_zn  = vectorfield[2*nstack + zn*nslice  + y*nx + x ];
					img_type val_zp  = vectorfield[2*nstack + zp*nslice  + y*nx + x ];
					img_type val_zp2 = vectorfield[2*nstack + zp2*nslice + y*nx + x];

					dz = normalizer*(val_zn2 -8*val_zn + 8*val_zp - val_zp2);
				}

				divergence = dx+dy+dz;
			}
			else
			{
				//x-derivative of ux displacement
				img_type val_xn  = vectorfield[z*nslice + y*nx + xn ];
				img_type val_xp  = vectorfield[z*nslice + y*nx + xp ];

				//y-derivative of uy displacement
				img_type val_yn  = vectorfield[nstack + z*nslice + yn*nx + x ];
				img_type val_yp  = vectorfield[nstack + z*nslice + yp*nx + x ];

				img_type dx = normalizer*(-val_xn + val_xp);
				img_type dy = normalizer*(-val_yn + val_yp);
				img_type dz = 0.0f;

				if (shape[2] > 1)
				{
					//z-derivative of uz displacement
					img_type val_zn  = vectorfield[2*nstack + zn*nslice  + y*nx + x ];
					img_type val_zp  = vectorfield[2*nstack + zp*nslice  + y*nx + x ];

					dz = normalizer*(-val_zn + val_zp);
				}

				divergence = dx+dy+dz;
			}

			/*img_type maskmin = 1000000000000;

			//check if neighbour is not in the mask
			maskmin = std::min(mask[z*nslice + yn*nx + xn], maskmin);
			maskmin = std::min(mask[z*nslice + yn*nx + x], maskmin);
			maskmin = std::min(mask[z*nslice + yn*nx + xp], maskmin);
			maskmin = std::min(mask[z*nslice + y*nx + xn], maskmin);
			maskmin = std::min(mask[z*nslice + y*nx + x ], maskmin);
			maskmin = std::min(mask[z*nslice + y*nx + xp], maskmin);
			maskmin = std::min(mask[z*nslice + yp*nx + xn], maskmin);
			maskmin = std::min(mask[z*nslice + yp*nx + x ], maskmin);
			maskmin = std::min(mask[z*nslice + yp*nx + xp], maskmin);
			if (shape[2] > 1)
			{
				maskmin = std::min(mask[zp*nslice + yn*nx + xn], maskmin);
				maskmin = std::min(mask[zp*nslice + yn*nx + x], maskmin);
				maskmin = std::min(mask[zp*nslice + yn*nx + xp], maskmin);
				maskmin = std::min(mask[zp*nslice + y*nx + xn], maskmin);
				maskmin = std::min(mask[zp*nslice + y*nx + x ], maskmin);
				maskmin = std::min(mask[zp*nslice + y*nx + xp], maskmin);
				maskmin = std::min(mask[zp*nslice + yp*nx + xn], maskmin);
				maskmin = std::min(mask[zp*nslice + yp*nx + x ], maskmin);
				maskmin = std::min(mask[zp*nslice + yp*nx + xp], maskmin);

				maskmin = std::min(mask[zn*nslice + yn*nx + xn], maskmin);
				maskmin = std::min(mask[zn*nslice + yn*nx + x], maskmin);
				maskmin = std::min(mask[zn*nslice + yn*nx + xp], maskmin);
				maskmin = std::min(mask[zn*nslice + y*nx + xn], maskmin);
				maskmin = std::min(mask[zn*nslice + y*nx + x ], maskmin);
				maskmin = std::min(mask[zn*nslice + y*nx + xp], maskmin);
				maskmin = std::min(mask[zn*nslice + yp*nx + xn], maskmin);
				maskmin = std::min(mask[zn*nslice + yp*nx + x ], maskmin);
				maskmin = std::min(mask[zn*nslice + yp*nx + xp], maskmin);
			}*/

			//if (maskmin <= 0.0f) divergence = 0.0f;
			//if (divergence < 0.0f) divergence = 0.0f;
			/*if (frame1[pos] >= frame0[pos]) divergence = 0.0f;

			divergence = std::max(0.0f, divergence)*(frame0[pos]-frame1[pos])/(frame0[pos]+1e-9f);*/

			output[pos] = divergence;
		}

		/*img_type *output2 = (img_type*) calloc(nstack,sizeof(*output2));
		hdcom::HdCommunication hdcom;
		for (int theta = 0; theta < 180; theta += 10)
		{
			filter::apply_2DanisotropicGaussian_spatialdomain(output, shape, 5.f, 0.5f, theta, output2);
			hdcom.Save2DTifImage_32bit(output2, shape, "/home/stefan/Documents/WBB/Debug/TestSequence/angle/", "test"+aux::zfill_int2string(theta, 3), 0);
		}*/

		/*filter::apply_3DGaussianFilter(output, shape, 2.f);
		float *der = derive::firstDerivative_fourthOrder(output, shape);
		float *dx2 = derive::firstDerivative_fourthOrder(der, shape);
		float *dy2 = derive::firstDerivative_fourthOrder(der+nstack, shape);*/

		return output;
	}
	img_type* calc_from_green_strain(img_type* vectorfield, int shape[3], std::string outval, std::string finitedifference_type="fourthorder")
	{
		int nx = shape[0]; int ny = shape[1]; int nz = shape[2];
		long long int nslice = nx*ny;
		long long int nstack = nz*nslice;

		img_type *output;

		if (outval != "extensional strain" && outval != "max principal strain vector"  && outval != "min principal strain vector"
				&& outval != "straintensor"  && outval != "intermediate principal strain vector")
			output = (img_type*) calloc(nstack,sizeof(*output));
		else if (outval != "straintensor") output = (img_type*) calloc(3*nstack,sizeof(*output));
		else output = (img_type*) calloc(6*nstack,sizeof(*output));

		img_type normalizer = 1.f/2.f;

		if (finitedifference_type == "centraldifference" || finitedifference_type == "secondorder")
			normalizer = 1.f/2.f;
		else if (finitedifference_type == "Barron" || finitedifference_type == "fourthorder")
			normalizer = 1.f/12.f;
		else
			std::cout << "Warning! unknown finite difference for plot_divergence. Using central difference." << std::endl;

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

			img_type strain = 0.0f;
			img_type dxx, dxy, dxz, dyx, dyy, dyz, dzx, dzy, dzz;

			if (finitedifference_type == "Barron" || finitedifference_type == "fourthorder")
			{
				//x-derivative of ux displacement
				img_type val_xn2 = vectorfield[z*nslice + y*nx + xn2];
				img_type val_xn  = vectorfield[z*nslice + y*nx + xn ];
				img_type val_xp  = vectorfield[z*nslice + y*nx + xp ];
				img_type val_xp2 = vectorfield[z*nslice + y*nx + xp2];

				dxx = normalizer*(val_xn2 -8*val_xn + 8*val_xp - val_xp2);

				//y-derivative of ux displacement
				img_type val_yn2 = vectorfield[z*nslice + yn2*nx + x];
				img_type val_yn  = vectorfield[z*nslice + yn*nx + x ];
				img_type val_yp  = vectorfield[z*nslice + yp*nx + x ];
				img_type val_yp2 = vectorfield[z*nslice + yp2*nx + x];

				dxy = normalizer*(val_yn2 -8*val_yn + 8*val_yp - val_yp2);

				//z-derivative of ux displacement
				img_type val_zn2 = vectorfield[zn2*nslice + y*nx + x];
				img_type val_zn  = vectorfield[zn*nslice  + y*nx + x ];
				img_type val_zp  = vectorfield[zp*nslice  + y*nx + x ];
				img_type val_zp2 = vectorfield[zp2*nslice + y*nx + x];

				dxz = normalizer*(val_zn2 -8*val_zn + 8*val_zp - val_zp2);

				//x-derivative of uy displacement
				val_xn2 = vectorfield[nstack + z*nslice + y*nx + xn2];
				val_xn  = vectorfield[nstack + z*nslice + y*nx + xn ];
				val_xp  = vectorfield[nstack + z*nslice + y*nx + xp ];
				val_xp2 = vectorfield[nstack + z*nslice + y*nx + xp2];

				dyx = normalizer*(val_xn2 -8*val_xn + 8*val_xp - val_xp2);

				//y-derivative of uy displacement
				val_yn2 = vectorfield[nstack + z*nslice + yn2*nx + x];
				val_yn  = vectorfield[nstack + z*nslice + yn*nx + x ];
				val_yp  = vectorfield[nstack + z*nslice + yp*nx + x ];
				val_yp2 = vectorfield[nstack + z*nslice + yp2*nx + x];

				dyy = normalizer*(val_yn2 -8*val_yn + 8*val_yp - val_yp2);

				//z-derivative of uy displacement
				val_zn2 = vectorfield[nstack + zn2*nslice + y*nx + x];
				val_zn  = vectorfield[nstack + zn*nslice  + y*nx + x ];
				val_zp  = vectorfield[nstack + zp*nslice  + y*nx + x ];
				val_zp2 = vectorfield[nstack + zp2*nslice + y*nx + x];

				dyz = normalizer*(val_zn2 -8*val_zn + 8*val_zp - val_zp2);

				//x-derivative of uz displacement
				val_xn2 = vectorfield[2*nstack + z*nslice + y*nx + xn2];
				val_xn  = vectorfield[2*nstack + z*nslice + y*nx + xn ];
				val_xp  = vectorfield[2*nstack + z*nslice + y*nx + xp ];
				val_xp2 = vectorfield[2*nstack + z*nslice + y*nx + xp2];

				dzx = normalizer*(val_xn2 -8*val_xn + 8*val_xp - val_xp2);

				//y-derivative of uz displacement
				val_yn2 = vectorfield[2*nstack + z*nslice + yn2*nx + x];
				val_yn  = vectorfield[2*nstack + z*nslice + yn*nx + x ];
				val_yp  = vectorfield[2*nstack + z*nslice + yp*nx + x ];
				val_yp2 = vectorfield[2*nstack + z*nslice + yp2*nx + x];

				dzy = normalizer*(val_yn2 -8*val_yn + 8*val_yp - val_yp2);

				//z-derivative of uz displacement
				val_zn2 = vectorfield[2*nstack + zn2*nslice + y*nx + x];
				val_zn  = vectorfield[2*nstack + zn*nslice  + y*nx + x ];
				val_zp  = vectorfield[2*nstack + zp*nslice  + y*nx + x ];
				val_zp2 = vectorfield[2*nstack + zp2*nslice + y*nx + x];

				dzz = normalizer*(val_zn2 -8*val_zn + 8*val_zp - val_zp2);
			}
			else
			{
				//x-derivative of ux displacement
				img_type val_xn  = vectorfield[z*nslice + y*nx + xn ];
				img_type val_xp  = vectorfield[z*nslice + y*nx + xp ];

				dxx = normalizer*(-val_xn + val_xp);

				//y-derivative of ux displacement
				img_type val_yn  = vectorfield[z*nslice + yn*nx + x ];
				img_type val_yp  = vectorfield[z*nslice + yp*nx + x ];

				dxy = normalizer*(-val_yn + val_yp);

				//z-derivative of ux displacement
				img_type val_zn  = vectorfield[zn*nslice  + y*nx + x ];
				img_type val_zp  = vectorfield[zp*nslice  + y*nx + x ];

				dxz = normalizer*(-val_zn + val_zp);

				//x-derivative of uy displacement
				val_xn  = vectorfield[nstack + z*nslice + y*nx + xn ];
				val_xp  = vectorfield[nstack + z*nslice + y*nx + xp ];

				dyx = normalizer*(-val_xn + val_xp);

				//y-derivative of uy displacement
				val_yn  = vectorfield[nstack + z*nslice + yn*nx + x ];
				val_yp  = vectorfield[nstack + z*nslice + yp*nx + x ];

				dyy = normalizer*(-val_yn + val_yp);

				//z-derivative of uy displacement
				val_zn  = vectorfield[nstack + zn*nslice  + y*nx + x ];
				val_zp  = vectorfield[nstack + zp*nslice  + y*nx + x ];

				dyz = normalizer*(-val_zn + val_zp);

				//x-derivative of uz displacement
				val_xn  = vectorfield[2*nstack + z*nslice + y*nx + xn ];
				val_xp  = vectorfield[2*nstack + z*nslice + y*nx + xp ];

				dzx = normalizer*(-val_xn + val_xp);

				//y-derivative of uy displacement
				val_zn  = vectorfield[2*nstack + z*nslice + yn*nx + x ];
				val_zp  = vectorfield[2*nstack + z*nslice + yp*nx + x ];

				dzy = normalizer*(-val_yn + val_yp);

				//z-derivative of uz displacement
				val_zn  = vectorfield[2*nstack + zn*nslice  + y*nx + x ];
				val_zp  = vectorfield[2*nstack + zp*nslice  + y*nx + x ];

				dzz = normalizer*(-val_zn + val_zp);
			}

			//Green-Lagrange strain tensor
			img_type Exx = dxx + 0.5f*(dxx*dxx+dyx*dyx+dzx*dzx);
			img_type Eyy = dyy + 0.5f*(dxy*dxy+dyy*dyy+dzy*dzy);
			img_type Ezz = dzz + 0.5f*(dxz*dxz+dyz*dyz+dzz*dzz);
			img_type Exy = 0.5f*(dxy+dyx)+0.5f*(dxx*dxy+dyx*dyy+dzx*dzy);
			img_type Exz = 0.5f*(dxz+dzx)+0.5f*(dxx*dxz+dyx*dyz+dzx*dzz);
			img_type Eyz = 0.5f*(dyz+dzz)+0.5f*(dxy*dxz+dyy*dyz+dzy*dzz);

			if (outval == "maxshear")
			{
				//strain = 1.f/3.f*sqrt(2.f*(dxx-dyy)*(dxx-dyy) + 2.f*(dxx-dzz)*(dxx-dzz) + 2.f*(dyy-Ezz)*(dyy-dzz)
				//		+ 12.f*dxy*dxy + 12.f*dxz*dxz + 12.f*dyz*dyz);
				strain = 1.f/3.f*sqrt(2.f*(Exx-Eyy)*(Exx-Eyy) + 2.f*(Exx-Ezz)*(Exx-Ezz) + 2.f*(Eyy-Ezz)*(Eyy-Ezz)
						+ 12.f*Exy*Exy + 12.f*Exz*Exz + 12.f*Eyz*Eyz);
			}
			else if (outval == "volstrain")
			{
				//adding identity gives the deformation gradient F
				dxx += 1.f;
				dyy += 1.f;
				dzz += 1.f;

				//volumetric strain calculated as det(F)-1
				strain = dxx*dyy*dzz + dxy*dyz*dxz + dxz*dxy*dyz - dxz*dyy*dxz - dyz*dyz*dxx - dzz*dxy*dxy-1;
				//strain = Exx*Eyy*Ezz + Exy*Eyz*Exz + Exz*Exy*Eyz - Exz*Eyy*Exz - Eyz*Eyz*Exx - Ezz*Exy*Exy-1;
			}
			else if (outval == "divergence")
			{
				strain = Exx+Eyy+Ezz;
			}
			else if (outval == "extensional strain")
			{
				strain = Exx;
				output[pos+nstack] = Eyy;
				output[pos+2*nstack] = Ezz;
			}
			else if (outval == "max principal strain vector")
			{
				std::vector<std::vector<float>> eigenvectors = aux::eigenvalues_and_eigenvectors_3x3symmetric(Exx,Eyy,Ezz,Exy,Exz,Eyz);
				strain = eigenvectors[0][0]*eigenvectors[1][0];
				output[pos+nstack] = eigenvectors[0][0]*eigenvectors[1][1];
				output[pos+2*nstack] = eigenvectors[0][0]*eigenvectors[1][2];
			}
			else if (outval == "intermediate principal strain vector")
			{
				std::vector<std::vector<float>> eigenvectors = aux::eigenvalues_and_eigenvectors_3x3symmetric(Exx,Eyy,Ezz,Exy,Exz,Eyz);
				strain = eigenvectors[0][1]*eigenvectors[2][0];
				output[pos+nstack] = eigenvectors[0][1]*eigenvectors[2][1];
				output[pos+2*nstack] = eigenvectors[0][1]*eigenvectors[2][2];
			}
			else if (outval == "min principal strain vector")
			{
				std::vector<std::vector<float>> eigenvectors = aux::eigenvalues_and_eigenvectors_3x3symmetric(Exx,Eyy,Ezz,Exy,Exz,Eyz);
				strain = eigenvectors[0][2]*eigenvectors[3][0];
				output[pos+nstack] = eigenvectors[0][2]*eigenvectors[3][1];
				output[pos+2*nstack] = eigenvectors[0][2]*eigenvectors[3][2];
			}
			else if (outval == "maximumshear")
			{
				std::vector<std::vector<float>> eigenvectors = aux::eigenvalues_and_eigenvectors_3x3symmetric(Exx,Eyy,Ezz,Exy,Exz,Eyz);
				strain = std::max(std::max(eigenvectors[0][0],eigenvectors[0][1]),eigenvectors[0][2])-
						std::min(std::min(eigenvectors[0][0],eigenvectors[0][1]),eigenvectors[0][2]);
			}
			else if (outval == "vonMisesStrain")
			{
				//http://www.feacluster.com/CalculiX/cgx_2.11/doc/cgx/node194.html
				strain = 2./(3.*sqrt(2.))*sqrt((Exx-Eyy)*(Exx-Eyy)+(Exx-Ezz)*(Exx-Ezz)+(Eyy-Ezz)*(Eyy-Ezz)+6.*Exy*Exy+6.*Exz*Exz+6.*Eyz*Eyz);
			}
			else if (outval == "straintensor")
			{
				strain = Exx;
				output[pos+nstack] = Eyy;
				output[pos+2*nstack] = Ezz;
				output[pos+3*nstack] = Exy;
				output[pos+4*nstack] = Exz;
				output[pos+5*nstack] = Eyz;
			}
			else
				std::cout << "unknown outval" << std::endl;

			output[pos] = strain;
		}

		return output;
	}
	std::vector<float> measure_pin_displacement(img_type pin_value, img_type *vectorfield, img_type *mask, int shape[3])
	{
		long long int nslice = shape[0]*shape[1];
		long long int nstack = shape[2]* nslice;

		float dx = 0.0f;
		float dy = 0.0f;
		float dz = 0.0f;
		long long int npin = 0;

		#pragma omp parallel for reduction(+: npin,dx,dy,dz)
		for (long long int idx = 0; idx < nstack; idx++)
		{
			if(mask[idx] == pin_value)
			{
				dx += vectorfield[idx];
				dy += vectorfield[nstack+idx];
				dz += vectorfield[2*nstack+idx];
				npin++;
			}
		}

		std::vector<float> output = {dx/npin, dy/npin, dz/npin};
		return output;
	}
	img_type* measure_confidence(img_type *result0, img_type *warped_result1, int shape[3], std::string confidence_mode, float beta)
	{
		float this_eps = 0.00000001f;

		int nx = shape[0]; int ny = shape[1]; int nz = shape[2];
		long long int nslice = nx*ny;
		long long int nstack = nz*nslice;

		int confidence_id = 0;
		if (confidence_mode == "Ershov") confidence_id = 0; //take beta times the inverse of the displacement distance
		else if (confidence_mode == "threshold") confidence_id = 1; //threshold the distance with beta similar to Lei and Yang: Optical Flow Estimation on Coarse-to-Fine Region-Trees using Discrete Optimization
		else if (confidence_mode == "exponential") confidence_id = 2; //use exponential decay controlled by beta
		else if (confidence_mode == "gaussian") confidence_id = 3; //use gaussian decay controlled by beta

		img_type *output = (img_type*) calloc(nstack,sizeof(*output));

		#pragma omp parallel for
		for (long long int pos = 0; pos < nstack; pos++)
		{
			img_type this_c = 0.0f;

			int z = pos/nslice;
			int y = (pos-z*nslice)/nx;
			int x = pos-z*nslice-y*nx;

			optflow_type u0 = result0[pos];
			optflow_type v0 = result0[nstack+pos];
			optflow_type w0 = 0.0f;
			if (nz > 1) w0 = result0[2*nstack+pos];

			optflow_type u1 = warped_result1[pos];
			optflow_type v1 = warped_result1[nstack+pos];
			optflow_type w1 = 0.0f;
			if (nz > 1) w1 = warped_result1[2*nstack+pos];

			float x0 = x + u0;
			float y0 = y + v0;
			float z0 = z + w0;

			if (y0 < 0.0f || x0 < 0.0f || x0 > (nx-1) || y0 > (ny-1) || z0 < 0.0f || z0 > (nz-1))
			{
				//Out of bounds = zero consistency
			}
			else
			{
				this_c = sqrtf(std::max(this_eps,(u0+u1)*(u0+u1)+(v0+v1)*(v0+v1)+(w0+w1)*(w0+w1)));

				if (confidence_id == 0) this_c = std::min(1.f, 1.f/(beta*this_c));
				else if (confidence_id == 1){
					if (this_c > beta) this_c = 0.0f;
					else this_c = 1.0f;
				}
				else if (confidence_id == 2){
					this_c = expf(-beta*this_c);
				}
				else if (confidence_id == 3){
					this_c = expf(-beta*this_c*this_c);
				}
			}

			output[pos] = this_c;
		}

		return output;
	}
	std::vector<float> average_vectormagnitude_masked(img_type *result, float *mask, int shape[3])
	{
		long long int nslice = shape[0]*shape[1];
		long long int nstack = shape[2]*nslice;

		float sum = 0.0f;
		float sumx = 0.0f; float sumy = 0.0f; float sumz = 0.0f; float sumxy = 0.0f; float sumyz = 0.0f;
		float meanx = 0.0f; float meany = 0.0f; float meanz = 0.0f;
		float counter = 0.0f;

		#pragma omp parallel for reduction(+: sum, counter, sumx, sumy, sumz, sumxy, sumyz, meanx, meany, meanz)
		for (long long int idx = 0; idx < nstack; idx++)
		{
			if (mask[idx] != 0)
			{
				float x = result[idx];
				float y = result[idx+nstack];
				float z = result[idx+2*nstack];

				float magnitude = sqrtf(x*x+y*y+z*z);
				sum += magnitude;
				sumx += fabs(x);
				sumy += fabs(y);
				sumz += fabs(z);
				sumxy += sqrtf(x*x+y*y);
				sumyz += sqrtf(y*y+z*z);
				meanx += x;
				meany += y;
				meanz += z;
				counter++;
			}
		}

		if (counter > 0)
		{
			sum /= counter;
			sumxy /= counter;
			sumyz /= counter;
			sumx /= counter;
			sumy /= counter;
			sumz /= counter;
			meanx /= counter;
			meany /= counter;
			meanz /= counter;
		}

		std::vector<float> output = {sum, sumxy, sumyz, sumx, sumy, sumz, meanx, meany, meanz};

		return output;
	}
}

#endif //ANALYSIS_H
