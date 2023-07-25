#ifndef SAMPLE_TRANSFORMATION_H
#define SAMPLE_TRANSFORMATION_H

#include <iostream>
#include <string.h>
#include <cstdint>
#include <vector>
#include "registration_bruteforce.h"
#include "Geometry/hdcommunication.h"
#include "Geometry/auxiliary.h"

namespace sampling
{
	void export_projections(float *frame0, float *frame1, int shape[3], std::string outpath)
	{
		float *xy_proj0 = aux::project_average(2, frame0, shape);
		float *xy_proj1 = aux::project_average(2, frame1, shape);
		float *xz_proj0 = aux::project_average(1, frame0, shape);
		float *xz_proj1 = aux::project_average(1, frame1, shape);
		float *yz_proj0 = aux::project_average(0, frame0, shape);
		float *yz_proj1 = aux::project_average(0, frame1, shape);

		hdcom::HdCommunication hdcom;

		hdcom.Save2DTifImage_32bit(xy_proj0, shape, outpath, "xy_proj0", 0);
		hdcom.Save2DTifImage_32bit(xy_proj1, shape, outpath, "xy_proj1", 0);
		int tmpshape[2] = {shape[0], shape[2]};
		hdcom.Save2DTifImage_32bit(xz_proj0, tmpshape, outpath, "xz_proj0", 0);
		hdcom.Save2DTifImage_32bit(xz_proj1, tmpshape, outpath, "xz_proj1", 0);
		tmpshape[0] = shape[1];
		hdcom.Save2DTifImage_32bit(yz_proj0, tmpshape, outpath, "yz_proj0", 0);
		hdcom.Save2DTifImage_32bit(yz_proj1, tmpshape, outpath, "yz_proj1", 0);

		free(xy_proj0); free(xy_proj1);
		free(xz_proj0); free(xz_proj1);
		free(yz_proj0); free(yz_proj1);

		return;
	}
	void export_central_reslice(std::string direction, float *image, int shape[3], std::string outpath, std::string name)
	{
		int nx = shape[0]; int ny = shape[1]; int nz = shape[2];
		long long int nslice = shape[0]*shape[1];

		hdcom::HdCommunication hdcom;

		if (direction == "xz")
		{
			long long int nslice_out = nx*nz;
			float *output = (float*) calloc(nslice_out, sizeof(*output));

			#pragma omp parallel for
			for (long long int idx = 0; idx < nslice_out; idx++)
			{
				int z = idx/nx;
				int x = idx-nx*z;
				int y = ny/2;

				output[idx] = image[z*nslice+y*nx+x];
			}

			int outshape[2] = {nx, nz};
			hdcom.Save2DTifImage_32bit(output, outshape, outpath, name, 0);
		}
		if (direction == "yz")
		{
			long long int nslice_out = ny*nz;
			float *output = (float*) calloc(nslice_out, sizeof(*output));

			#pragma omp parallel for
			for (long long int idx = 0; idx < nslice_out; idx++)
			{
				int z = idx/ny;
				int x = nx/2;
				int y = idx-ny*z;

				output[idx] = image[z*nslice+y*nx+x];
			}

			int outshape[2] = {ny, nz};
			hdcom.Save2DTifImage_32bit(output, outshape, outpath, name, 0);
		}
		return;
	}
}

#endif //SAMPLE_TRANSFORMATION_H
