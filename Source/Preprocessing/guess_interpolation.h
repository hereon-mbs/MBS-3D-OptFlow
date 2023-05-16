#ifndef GUESS_INTERPOLATION_H
#define GUESS_INTERPOLATION_H

#include <iostream>
#include <string.h>
#include <cstdint>
#include <vector>
#include <omp.h>
#include <map>

#include "../Geometry/filtering.h"

namespace guess_interpolate
{
	std::vector<int> __get_n_nearest(int i, int N, std::vector<std::vector<int>> &coordinates)
	{
		std::vector<int> output(N, -1);

		//just plain linear search
		std::vector<std::pair<int,float>> sqdistances;

		int x0 = coordinates[i][0];
		int y0 = coordinates[i][1];
		int z0 = coordinates[i][2];

		for (long long int p = 0; p < coordinates.size(); p++)
		{
			if (p == i) continue;

			int x1 = coordinates[p][0];
			int y1 = coordinates[p][1];
			int z1 = coordinates[p][2];

			sqdistances.push_back({p,(x0-x1)*(x0-x1)+(y0-y1)*(y0-y1)+(z0-z1)*(z0-z1)});
		}

		std::sort(sqdistances.begin(), sqdistances.end(),[](const std::pair<int,float>& a, const std::pair<int,float>& b) {return a.second < b.second;});

		for (int idx = 0; idx < output.size(); idx++)
			output[idx] = sqdistances[idx].first;

		return output;
	}


	float* sparseresult2image(std::vector<std::vector<float>> &sparse_result, std::vector<std::vector<int>> &coordinates, int n_nearest_median, float sigma_gauss, int shape[3])
	{
		// -We select the median of n nearest neighbour for each support point
		// -Next we dilate until the whole image is flushed to get the nearest neighbour result
		// -Finally we apply a Gaussian filter to smoothen the transition

		int nx = shape[0]; int ny = shape[1]; int nz = shape[2];
		long long int nslice = shape[0]*shape[1];
		long long int nstack = shape[2]*nslice;

		float* displacements = (float*) calloc(3*nstack, sizeof(*displacements));
		uint8_t* labeled = (uint8_t*) calloc(nstack, sizeof(*labeled));

		//Find the median of the nearest neighbours
		///////////////////////////////////////////////////////////////////////////////////
		std::cout << "calculating median of initial sparse guess...";
		std::cout.flush();

		#pragma omp parallel for
		for (int i = 0; i < coordinates.size(); i++)
		{
			int x = coordinates[i][0];
			int y = coordinates[i][1];
			int z = coordinates[i][2];
			long long int idx = z*nslice+y*nx+x;

			std::vector<int> n_nearest = __get_n_nearest(i, n_nearest_median, coordinates);
			std::vector<float> dx_median, dy_median, dz_median;

			for (int p = 0; p < n_nearest.size(); p++)
			{
				dx_median.push_back(sparse_result[n_nearest[p]][0]);
				dy_median.push_back(sparse_result[n_nearest[p]][1]);
				dz_median.push_back(sparse_result[n_nearest[p]][2]);

				std::sort(dx_median.begin(), dx_median.end());
				std::sort(dy_median.begin(), dy_median.end());
				std::sort(dz_median.begin(), dz_median.end());

				float dx = dx_median[n_nearest.size()/2];
				float dy = dy_median[n_nearest.size()/2];
				float dz = dz_median[n_nearest.size()/2];

				displacements[idx] = dx;
				displacements[idx+nstack] = dy;
				displacements[idx+2*nstack] = dz;
				labeled[idx] = 1;
			}
		}

		std::cout << "done" << std::endl;
		///////////////////////////////////////////////////////////////////////////////////

		//Fill in nearest neighbour
		///////////////////////////////////////////////////////////////////////////////////
		std::cout << "dilating sparse guess...";
		std::cout.flush();

		long long int change = 1;
		while(change > 0)
		{
			change = 0;
			uint8_t* next_labels = (uint8_t*) calloc(nstack, sizeof(next_labels));

			#pragma omp parallel for reduction(+: change)
			for (long long int idx = 0; idx < nstack; idx++)
			{
				uint8_t this_val = labeled[idx];

				if (this_val == 0)
				{
					int z = idx/nslice;
					int y = (idx-z*nslice)/nx;
					int x = idx-z*nslice-y*nx;

					if (x-1 >= 0 && labeled[idx-1] != 0){
						change++; next_labels[idx] = 1;
						displacements[idx] = displacements[idx-1];
						displacements[idx+nstack] = displacements[idx-1+nstack];
						displacements[idx+2*nstack] = displacements[idx-1+2*nstack];}
					else if (y-1 >= 0 && labeled[idx-nx] != 0){
						change++; next_labels[idx] = 1;
						displacements[idx] = displacements[idx-nx];
						displacements[idx+nstack] = displacements[idx-nx+nstack];
						displacements[idx+2*nstack] = displacements[idx-nx+2*nstack];}
					else if (z-1 >= 0 && labeled[idx-nslice] != 0){
						change++; next_labels[idx] = 1;
						displacements[idx] = displacements[idx-nslice];
						displacements[idx+nstack] = displacements[idx-nslice+nstack];
						displacements[idx+2*nstack] = displacements[idx-nslice+2*nstack];}
					else if (x+1 < nx && labeled[idx+1] != 0){
						change++; next_labels[idx] = 1;
						displacements[idx] = displacements[idx+1];
						displacements[idx+nstack] = displacements[idx+1+nstack];
						displacements[idx+2*nstack] = displacements[idx+1+2*nstack];}
					else if (y+1 < ny && labeled[idx+nx] != 0){
						change++; next_labels[idx] = 1;
						displacements[idx] = displacements[idx+nx];
						displacements[idx+nstack] = displacements[idx+nx+nstack];
						displacements[idx+2*nstack] = displacements[idx+nx+2*nstack];}
					else if (z+1 < nz && labeled[idx+nslice] != 0){
						change++; next_labels[idx] = 1;
						displacements[idx] = displacements[idx+nslice];
						displacements[idx+nstack] = displacements[idx+nslice+nstack];
						displacements[idx+2*nstack] = displacements[idx+nslice+2*nstack];}
				}
				else
				{
					next_labels[idx] = 1;
				}
			}

			std::swap(labeled, next_labels);
			free(next_labels);
		}

		std::cout << "done" << std::endl;
		///////////////////////////////////////////////////////////////////////////////////

		//Smoothen
		///////////////////////////////////////////////////////////////////////////////////
		std::cout << "smoothing sparse guess...";
		std::cout.flush();
		filter::apply_3DGaussianFilter2Vector(displacements,shape, sigma_gauss, 3);
		std::cout << std::endl;
		///////////////////////////////////////////////////////////////////////////////////

		return displacements;
	}

}

#endif
