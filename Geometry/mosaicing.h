#ifndef MOSAICING_H
#define MOSAICING_H

#include <iostream>
#include <string.h>
#include <cstdint>
#include "../protocol_parameters.h"

namespace mosaic
{
	typedef float optflow_type;

	std::vector<std::vector<int>> get_mosaic_coordinates(int shape[3], optflow::ProtocolParameters *params, int maxshape[3])
	{
		//returns a list of coordinates for constructing an overlapping mosaic
		//and remembers the biggest patch in maxshape

		int nx = shape[0]; int ny = shape[1]; int nz = shape[2];
		long long int stack_nslice = nx*ny;
		long long int stack_nstack = nz*stack_nslice;

		long long int patch_nstack = 0;
		long long int max_nstack = params->mosaicing.max_nstack;
		int overlap = params->mosaicing.overlap;

		//select cutting scheme
		//////////////////////////////////////////////////////////
		int xyz_cuts[3] = {0,0,0};

		long long int last_nstack = stack_nstack;
		int active_nx = nx; int active_ny = ny; int active_nz = nz;

		int n_patches = 1;

		while(last_nstack > max_nstack || n_patches < params->gpu.n_gpus)
		{
			int cut_dimension = 0;
			if (active_ny > active_nx) cut_dimension = 1;
			if (active_nz > active_nx && active_nz > active_ny) cut_dimension = 2;

			//preferential cuts have preference
			if (params->mosaicing.preferential_cut_dimension == 2 && active_nz/2 > overlap/2) cut_dimension = 2;
			else if (params->mosaicing.preferential_cut_dimension == 1 && active_ny/2 > overlap/2) cut_dimension = 1;
			else if (params->mosaicing.preferential_cut_dimension == 0 && active_nx/2 > overlap/2) cut_dimension = 0;

			if (active_nx/2 <= overlap/2 && active_ny/2 <= overlap/2 && active_nz/2 <= overlap/2)
			{
				std::cout << "Warning! Overlap too big for further decomposition. Trying last configuration!" << std::endl;
				break;
			}

			xyz_cuts[cut_dimension]++;

			if (params->gpu.n_gpus == 3 && n_patches == 1)
			{
				if (active_nx/3 > overlap/2 || active_ny/2 > overlap/2 || active_nz/3 > overlap/2)
					xyz_cuts[cut_dimension]++;
			}

			//last patch should be the largest not considering overlap
			active_nx = nx-nx/(xyz_cuts[0]+1)*xyz_cuts[0];
			active_ny = ny-ny/(xyz_cuts[1]+1)*xyz_cuts[1];
			active_nz = nz-nz/(xyz_cuts[2]+1)*xyz_cuts[2];

			//add the overlap for estimating the stack size to be calculated
			int total_nx = active_nx + std::min(2, xyz_cuts[0])*overlap/2;
			int total_ny = active_ny + std::min(2, xyz_cuts[1])*overlap/2;
			int total_nz = active_nz + std::min(2, xyz_cuts[2])*overlap/2;

			long long int last_nslice = total_nx*total_ny;
			last_nstack = last_nslice*total_nz;

			n_patches = (xyz_cuts[0]+1)*(xyz_cuts[1]+1)*(xyz_cuts[2]+1);
		}

		//std::cout << "cuts: " << xyz_cuts[0] << " " << xyz_cuts[1] << " " << xyz_cuts[2] << std::endl;
		//////////////////////////////////////////////////////////

		//set the cut locations
		//////////////////////////////////////////////////////////
		std::vector<std::vector<int>> mosaic;
		long long patch_nslice = 0;

		for (int zi = 0; zi <= xyz_cuts[2]; zi++)
		{
			int z0 = nz/(xyz_cuts[2]+1)*zi;
			int z1 = nz/(xyz_cuts[2]+1)*(zi+1);

			if (zi == xyz_cuts[2]) z1 = nz;

			if (z0-overlap/2 >= 0) z0 -= overlap/2;
			if (z1+overlap/2 < nz) z1 += overlap/2;

			for (int yi = 0; yi <= xyz_cuts[1]; yi++)
			{
				int y0 = ny/(xyz_cuts[1]+1)*yi;
				int y1 = ny/(xyz_cuts[1]+1)*(yi+1);

				if (yi == xyz_cuts[1]) y1 = ny;

				if (y0-overlap/2 >= 0) y0 -= overlap/2;
				if (y1+overlap/2 < ny) y1 += overlap/2;

				for (int xi = 0; xi <= xyz_cuts[0]; xi++)
				{
					int x0 = nx/(xyz_cuts[0]+1)*xi;
					int x1 = nx/(xyz_cuts[0]+1)*(xi+1);

					if (xi == xyz_cuts[0]) x1 = nx;

					if (x0-overlap/2 >= 0) x0 -= overlap/2;
					if (x1+overlap/2 < nx) x1 += overlap/2;

					std::vector<int> coordinates = {x0, y0, z0, x1, y1, z1};
					mosaic.push_back(coordinates);

					//remember the biggest patch for allocating GPU-memory
					patch_nslice = (x1-x0)*(y1-y0);
					long long int this_nstack = (z1-z0)*patch_nslice;
					if (this_nstack > patch_nstack)
					{
						patch_nstack = this_nstack;
						maxshape[0] = (x1-x0);
						maxshape[1] = (y1-y0);
						maxshape[2] = (z1-z0);
					}
				}
			}
		}
		//////////////////////////////////////////////////////////

		return mosaic;
	}
	void set_patchvalues(std::vector<int> &mosaic_coordinates, optflow_type *patch, optflow_type *input, int stack_shape[3], int vectordims, int patch_offset)
	{
		int x0 = mosaic_coordinates[0]; int x1 = mosaic_coordinates[3]; int active_nx = x1-x0;
		int y0 = mosaic_coordinates[1]; int y1 = mosaic_coordinates[4]; int active_ny = y1-y0;
		int z0 = mosaic_coordinates[2]; int z1 = mosaic_coordinates[5]; int active_nz = z1-z0;
		int stack_nx = stack_shape[0];

		long long int active_nslice = active_nx*active_ny;
		long long int active_nstack = active_nslice*active_nz;
		long long int stack_nslice = stack_shape[0]*stack_shape[1];
		long long int stack_nstack = stack_shape[2]*stack_nslice;

		for (int dim = 0; dim < vectordims; dim++)
		{
			#pragma omp parallel for
			for(long long int patch_pos = 0; patch_pos < active_nstack; patch_pos++)
			{
				int z = patch_pos/active_nslice;
				int y = (patch_pos-z*active_nslice)/active_nx;
				int x = patch_pos-z*active_nslice-y*active_nx;

				long long int stack_pos = (z+z0)*stack_nslice + (y+y0)*stack_nx + (x+x0);

				patch[(dim+patch_offset)*active_nstack + patch_pos] = input[dim*stack_nstack + stack_pos];
			}
		}

		return;
	}
	void insert_patchresult(std::vector<int> &mosaic_coordinates, optflow_type *patch_result, optflow_type *result, int stack_shape[3], int vectordims)
	{
		int x0 = mosaic_coordinates[0]; int x1 = mosaic_coordinates[3]; int active_nx = x1-x0;
		int y0 = mosaic_coordinates[1]; int y1 = mosaic_coordinates[4]; int active_ny = y1-y0;
		int z0 = mosaic_coordinates[2]; int z1 = mosaic_coordinates[5]; int active_nz = z1-z0;
		int stack_nx = stack_shape[0];

		long long int active_nslice = active_nx*active_ny;
		long long int active_nstack = active_nslice*active_nz;
		long long int stack_nslice = stack_shape[0]*stack_shape[1];
		long long int stack_nstack = stack_shape[2]*stack_nslice;

		for (int dim = 0; dim < vectordims; dim++)
		{
			#pragma omp parallel for
			for(long long int patch_pos = 0; patch_pos < active_nstack; patch_pos++)
			{
				int z = patch_pos/active_nslice;
				int y = (patch_pos-z*active_nslice)/active_nx;
				int x = patch_pos-z*active_nslice-y*active_nx;

				long long int stack_pos = (z+z0)*stack_nslice + (y+y0)*stack_nx + (x+x0);

				result[dim*stack_nstack + stack_pos] = patch_result[dim*active_nstack + patch_pos];
			}
		}
	}

	/*Transposing axis reduces load for copying
	*********************************************************/
	void reorder_axis_bylength(float *&image, int shape[3], int new_order[3], int &last_dimension)
	{
		int axis0 = 0;
		int axis1 = 1;
		int axis2 = 2;

		//sort axis by length
		if (shape[1] < shape[0]) {axis0 = 1; axis1 = 0;}
		if (shape[2] > 1)
		{
			if (shape[2] < shape[axis0]) {axis2 = axis1; axis1 = axis0; axis0 = 2;}
			if (shape[axis2] < shape[axis1]) {int tmp = axis1; axis1 = axis2; axis2 = tmp;}
		}

		//preferential_cut should be the last dimension when reordering
		if (shape[2] > 1 && last_dimension == axis0) {int tmp = axis0; axis0 = axis1; axis1 = axis2; axis2 = tmp;}
		else if (shape[2] > 1 && last_dimension == axis1) {int tmp = axis1; axis1 = axis2; axis2 = tmp;}
		else if (shape[2] == 1 && last_dimension == axis0) {int tmp = axis0; axis0 = axis1; axis1 = tmp;}

		new_order[0] = axis0;
		new_order[1] = axis1;
		new_order[2] = axis2;

		if (new_order[0] == 0 && new_order[1] == 1 && new_order[2] == 2)
			return;

		long long int nslice = shape[0]*shape[1];
		long long int nstack = nslice*shape[2];
		long long int new_nslice = shape[axis0]*shape[axis1];

		float *output = (float*) malloc(nstack*sizeof(*output));

		#pragma omp parallel for
		for (long long int pos = 0; pos < nstack; pos++)
		{
			int z0 = pos/nslice;
			int y0 = (pos-z0*nslice)/shape[0];
			int x0 = pos-z0*nslice-y0*shape[0];

			int z1 = z0;
			int y1 = y0;
			int x1 = x0;

			if (axis0 == 1) x1 = y0;
			else if (axis0 == 2) x1 = z0;
			if (axis1 == 0) y1 = x0;
			else if (axis1 == 2) y1 = z0;
			if (axis2 == 0) z1 = x0;
			else if (axis2 == 1) z1 = y0;

			long long int outpos = z1*new_nslice+y1*shape[axis0]+x1;

			output[outpos] = image[pos];
		}

		std::swap(image, output);
		//free(output);

		return;
	}
	void reorder_axis_byorder(float *&image, int shape[3], int order[3])
	{
		if (order[0] == 0 && order[1] == 1 && order[2] == 2)
			return;

		int axis0 = order[0];
		int axis1 = order[1];
		int axis2 = order[2];

		long long int nslice = shape[0]*shape[1];
		long long int nstack = nslice*shape[2];
		long long int new_nslice = shape[axis0]*shape[axis1];

		float *output = (float*) malloc(nstack*sizeof(*output));

		#pragma omp parallel for
		for (long long int pos = 0; pos < nstack; pos++)
		{
			int z0 = pos/nslice;
			int y0 = (pos-z0*nslice)/shape[0];
			int x0 = pos-z0*nslice-y0*shape[0];

			int z1 = z0;
			int y1 = y0;
			int x1 = x0;

			if (axis0 == 1) x1 = y0;
			else if (axis0 == 2) x1 = z0;
			if (axis1 == 0) y1 = x0;
			else if (axis1 == 2) y1 = z0;
			if (axis2 == 0) z1 = x0;
			else if (axis2 == 1) z1 = y0;

			long long int outpos = z1*new_nslice+y1*shape[axis0]+x1;

			output[outpos] = image[pos];
		}

		std::swap(image, output);
		//free(output);

		return;
	}
	void reorder_vector_byorder(float *&vectorimage, int shape[3], int order[3], int ndims)
	{
		if (order[0] == 0 && order[1] == 1 && order[2] == 2)
			return;

		int axis0 = order[0];
		int axis1 = order[1];
		int axis2 = order[2];

		long long int nslice = shape[0]*shape[1];
		long long int nstack = nslice*shape[2];
		long long int new_nslice = shape[axis0]*shape[axis1];

		float *output = (float*) malloc((ndims*nstack)*sizeof(*output));

		#pragma omp parallel for
		for (long long int pos = 0; pos < nstack; pos++)
		{
			int z0 = pos/nslice;
			int y0 = (pos-z0*nslice)/shape[0];
			int x0 = pos-z0*nslice-y0*shape[0];

			int z1 = z0;
			int y1 = y0;
			int x1 = x0;

			if (axis0 == 1) x1 = y0;
			else if (axis0 == 2) x1 = z0;
			if (axis1 == 0) y1 = x0;
			else if (axis1 == 2) y1 = z0;
			if (axis2 == 0) z1 = x0;
			else if (axis2 == 1) z1 = y0;

			long long int outpos = z1*new_nslice+y1*shape[axis0]+x1;

			for (int dim = 0; dim < ndims; dim++)
				output[order[dim]*nstack+outpos] = vectorimage[dim*nstack+pos];
		}

		std::swap(vectorimage, output);
		//free(output);

		return;
	}
}

#endif //MOSAICING_H
