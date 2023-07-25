#include <iostream>
#include <vector>
#include <algorithm>
#include <omp.h>

namespace aux
{
    /*String-Manipulation
    *********************************************************/
    std::string zfill_int2string(int inint, const unsigned int &zfill)
    {
        std::string outstring = std::to_string(inint);
        while(outstring.length() < zfill)
            outstring = "0" + outstring;
        return outstring;
    }

    /*Numpy-like
    *********************************************************/
    std::vector<float> linspace(float startval, float endval, uint64_t bins)
    {
        std::vector<float> linspaced(bins);
        float delta = (endval-startval)/(bins-1);
        for(uint64_t i = 0; i < (bins-1); i++)
        {
            linspaced[i] = startval + delta * i;
        }
        linspaced[bins-1] = endval;
        return linspaced;
    }
    std::vector<double> linspace(double startval, double endval, uint64_t bins)
	{
		std::vector<double> linspaced(bins);
		double delta = (endval-startval)/(bins-1);
		for(uint64_t i = 0; i < (bins-1); i++)
		{
			linspaced[i] = startval + delta * i;
		}
		linspaced[bins-1] = endval;
		return linspaced;
	}

    /*Reduction to active voxels
     *********************************************************/
    float* reduce2activevoxels(float *image, float *mask, int shape[3], long long int &nactive_out)
    {
    	int nx = shape[0];
    	long long int nslice = nx*shape[1];
    	long long int nstack = shape[2]*nslice;
    	long long int nactive = 0;

		#pragma omp parallel for reduction(+: nactive)
    	for (long long int idx = 0; idx < nstack; idx++)
    		if(mask[idx] != 0) nactive++;

    	float* output = (float*) calloc(2*nactive, sizeof(*output));

    	long long int pos = 0;
    	for (long long int idx = 0; idx < nstack; idx++)
    	{
    		if(mask[idx] != 0)
    		{
    			output[pos] = (float) idx;
    			output[nactive+pos] = image[idx];
    			pos++;
    		}
    	}

    	nactive_out = nactive;
    	return output;
    }

    /*Projections
    *********************************************************/
    float* project_maximum(int axis, float* imagestack, int shape[3])
	{
		float *output;

		int nx = shape[0]; int ny = shape[1]; int nz = shape[2];
		long long int nslice = shape[0]*shape[1];
		long long int nstack = shape[2]*nslice;

		long long int n_output;

		if (axis == 2)
		{
			n_output = nx*ny;
			output = (float*) calloc(n_output, sizeof(*output));

			#pragma omp parallel for
			for (long long int idx = 0; idx < n_output; idx++)
				output[idx] = -1e9f;

			#pragma omp parallel for
			for (long long int idx = 0; idx < nstack; idx++)
			{
				int z = idx/nslice;
				int y = (idx-z*nslice)/nx;
				int x = idx-z*nslice-y*nx;

				long long int idx_out = y*nx+x;

				float val = imagestack[idx];
				if (val > output[idx_out]) output[idx_out] = val;
			}
		}
		else if (axis == 1)
		{
			n_output = nx*nz;
			output = (float*) calloc(n_output, sizeof(*output));

			#pragma omp parallel for
			for (long long int idx = 0; idx < n_output; idx++)
				output[idx] = -1e9f;

			#pragma omp parallel for
			for (long long int idx = 0; idx < nstack; idx++)
			{
				int z = idx/nslice;
				int y = (idx-z*nslice)/nx;
				int x = idx-z*nslice-y*nx;

				long long int idx_out = z*nx+x;

				float val = imagestack[idx];
				if (val > output[idx_out]) output[idx_out] = val;
			}
		}
		else
		{
			n_output = ny*nz;
			output = (float*) calloc(n_output, sizeof(*output));

			#pragma omp parallel for
			for (long long int idx = 0; idx < n_output; idx++)
				output[idx] = -1e9f;

			#pragma omp parallel for
			for (long long int idx = 0; idx < nstack; idx++)
			{
				int z = idx/nslice;
				int y = (idx-z*nslice)/nx;
				int x = idx-z*nslice-y*nx;

				long long int idx_out = z*ny+y;

				float val = imagestack[idx];
				if (val > output[idx_out]) output[idx_out] = val;
			}
		}

		return output;
	}
    float *project_average(int axis, float* imagestack, int shape[3])
	{
		float *output;

		int nx = shape[0]; int ny = shape[1]; int nz = shape[2];
		long long int nslice = shape[0]*shape[1];
		long long int nstack = shape[2]*nslice;

		long long int n_output;

		if (axis == 2)
		{
			n_output = nx*ny;
			output = (float*) calloc(n_output, sizeof(*output));

			#pragma omp parallel for
			for (long long int idx = 0; idx < nstack; idx++)
			{
				int z = idx/nslice;
				int y = (idx-z*nslice)/nx;
				int x = idx-z*nslice-y*nx;

				long long int idx_out = y*nx+x;

				output[idx_out]+=imagestack[idx];
			}

			#pragma omp parallel for
			for (long long int idx = 0; idx < n_output; idx++)
				output[idx] /= nz;
		}
		else if (axis == 1)
		{
			n_output = nx*nz;
			output = (float*) calloc(n_output, sizeof(*output));

			#pragma omp parallel for
			for (long long int idx = 0; idx < nstack; idx++)
			{
				int z = idx/nslice;
				int y = (idx-z*nslice)/nx;
				int x = idx-z*nslice-y*nx;

				long long int idx_out = z*nx+x;

				output[idx_out]+=imagestack[idx];
			}

			#pragma omp parallel for
			for (long long int idx = 0; idx < n_output; idx++)
				output[idx] /= ny;
		}
		else
		{
			n_output = ny*nz;
			output = (float*) calloc(n_output, sizeof(*output));

			#pragma omp parallel for
			for (long long int idx = 0; idx < nstack; idx++)
			{
				int z = idx/nslice;
				int y = (idx-z*nslice)/nx;
				int x = idx-z*nslice-y*nx;

				long long int idx_out = z*ny+y;

				output[idx_out]+=imagestack[idx];
			}

			#pragma omp parallel for
			for (long long int idx = 0; idx < n_output; idx++)
				output[idx] /= nx;
		}

		return output;
	}
    float *project_masked_average(int axis, float* imagestack, float *mask, int shape[3], bool mask2D = true)
    {
    	float *output, *counter;

    	int nx = shape[0]; int ny = shape[1]; int nz = shape[2];
    	long long int nslice = shape[0]*shape[1];
    	long long int nstack = shape[2]*nslice;

    	long long int n_output;

    	if (axis == 2)
    	{
    		n_output = nx*ny;
    		output = (float*) calloc(n_output, sizeof(*output));
    		counter = (float*) calloc(n_output, sizeof(*counter));

			#pragma omp parallel for
    		for (long long int idx = 0; idx < nstack; idx++)
    		{
    			int z = idx/nslice;
    			int y = (idx-z*nslice)/nx;
    			int x = idx-z*nslice-y*nx;

    			long long int idx_mask = mask2D ? y*nx+x : idx;
    			long long int idx_out = y*nx+x;

    			if (mask[idx_mask] != 0)
    			{
    				counter[idx_out]++;
    				output[idx_out]+=imagestack[idx];
    			}
    		}
    	}
    	else if (axis == 1)
		{
			n_output = nx*nz;
			output = (float*) calloc(n_output, sizeof(*output));
			counter = (float*) calloc(n_output, sizeof(*counter));

			#pragma omp parallel for
			for (long long int idx = 0; idx < nstack; idx++)
			{
				int z = idx/nslice;
				int y = (idx-z*nslice)/nx;
				int x = idx-z*nslice-y*nx;

				long long int idx_mask = mask2D ? y*nx+x : idx;
				long long int idx_out = z*nx+x;

				if (mask[idx_mask] != 0)
				{
					counter[idx_out]++;
					output[idx_out]+=imagestack[idx];
				}
			}
		}
    	else
		{
			n_output = ny*nz;
			output = (float*) calloc(n_output, sizeof(*output));
			counter = (float*) calloc(n_output, sizeof(*counter));

			#pragma omp parallel for
			for (long long int idx = 0; idx < nstack; idx++)
			{
				int z = idx/nslice;
				int y = (idx-z*nslice)/nx;
				int x = idx-z*nslice-y*nx;

				long long int idx_mask = mask2D ? y*nx+x : idx;
				long long int idx_out = z*ny+y;

				if (mask[idx_mask] != 0)
				{
					counter[idx_out]++;
					output[idx_out]+=imagestack[idx];
				}
			}
		}

		#pragma omp parallel for
    	for (long long int idx = 0; idx < n_output; idx++)
    		if(counter[idx] > 0) output[idx] /= counter[idx];

    	free(counter);
    	return output;
    }
    /*
     ************************************************************/
}
