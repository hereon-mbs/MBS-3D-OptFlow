#ifndef CORNERDETECTION_H
#define CORNERDETECTION_H

#include <iostream>
#include <string.h>
#include <cstdint>
#include <omp.h>
#include "../Geometry/filtering.h"

namespace lk
{
	typedef float img_type;
	typedef long long int idx_type;

	class NobleCornerDetector
	{
	public:
		float kappa = 0.04;

		//detect corner points with Noble Corner Detector:
		//
		//sigma_gauss_integration := we need to correlate the structure tensor with a Gaussian kernel or lmbda_min is always 0
		//radius_suppression := when > 0 we only keep maxima in a sphere of this radius
		//kill_boundaries := boundary voxels might not be very good for tracking. This gives the required distance to the boundary
		//box_maxima := use box non-maxima suppression instead of spherical
		//min_fraction := according to Brox2011: "Large Displacement Optical Flow: Descriptor Matching in Variational Motion Estimation"
		//                we only want to keep eigenvalues that are at least 1/8th of the average

		img_type* detectcorners(img_type *img, int shape[3], float sigma_gauss = 0.5, int radius_suppression = 2, int kill_boundaries = 1, bool box_maxima = false,
				float min_fraction = 0.125);

		std::vector<std::vector<int>> corners2coordinatelist(img_type *supportimg, int shape[3])
		{
			int nx = shape[0]; int ny = shape[1]; int nz = shape[2];
			long long int nslice = nx*ny;
			long long int nstack = nz*nslice;

			std::vector<std::vector<int>> support_points;

			for (long long int idx = 0; idx < nstack; idx++)
			{
				if (supportimg[idx] != 0)
				{
					int z = idx/nslice;
					int y = (idx-z*nslice)/nx;
					int x = idx-z*nslice-y*nx;
					support_points.push_back({x,y,z});
				}
			}

			return support_points;
		}

	private:
		void get_structuretensor_components(img_type *input, float *Ixx, float *Iyy, float *Izz, float *Ixy, float *Ixz, float *Iyz, int shape[3], int difference_id);
	};

	class HistogramsOfOrientedGradients
	{
	public:

		//Reference: Brox2011 "Large Displacement Optical Flow: Descriptor Matching in Variational Motion Estimation"
		//
		//We'll use the HOG approach for some quick and dirty matching (potentially PatchMatch)
		//Since we'll need 2D-Histograms our descriptor will be quite large and should only be applied sparsely.
		//It would be best for a future GPU implementation if the descriptor would fit into the original image space
		//
		int radius_neighbourhood = 3; //3 is equivalent to Brox who uses a 7x7 neighbourhood
		int angular_bins_azimuth = 8; //Brox uses 15 but this would mean 15x15 for 3D
		int angular_bins_inclination = 5;
		int neighbour_spacing = 4; //27 neighbours spaced in this distance
		int neighbour_type = 19; //collect features from 7,19 or 27 locations
		float presmoothing = 0.8f; //Gaussian smoothing in derivative direction

		bool use_sign = false; //true does not neglect the sign when calculating gradient orientation
		bool spherical_neighbourhood = true; //Brox uses a box-type neighbourhood but who doesn't like spheres instead?

		float* create_HOG_descriptorimage(float* image, float* feature_locations, int shape[3]);
	private:
		void _add2feature(float* feature_image, float* Ix, float* Iy, float* Iz, int shape[3], long long int idx, long long int pos);
	};
}

#endif //CORNERDETECTION_H
