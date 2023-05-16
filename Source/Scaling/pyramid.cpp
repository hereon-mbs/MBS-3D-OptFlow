#include "pyramid.h"
#include <math.h>
#include "../Geometry/filtering.h"

namespace pyramid
{
	ImagePyramid::ImagePyramid(optflow::ProtocolParameters *params, int shape[3], bool allocate_memory)
	{
		float scaling_factor = params->pyramid.scaling_factor;
		std::string scaling_mode = params->pyramid.scaling_mode;

		//slow down downsampling at this level
		int n_liu = log(0.25)/log(scaling_factor);

		int min_edge = params->pyramid.min_edge;

		//Zero level
		pyramid_shapes.push_back(shape);

		for (int level = 1; level < params->pyramid.nLevels; level++)
		{
			int* active_shape = (int*) malloc(3*sizeof(*active_shape));

			/////////////////////////////////////////////////////////////////////////////////////////////
			float resampling_factor = pow(scaling_factor,level);

			if (scaling_mode == "Liu" && level > n_liu) //slow down rescaling
				resampling_factor *= ((float) shape[0])/((float) pyramid_shapes[level-1][0]);
			/////////////////////////////////////////////////////////////////////////////////////////////

			//make sure we are reducing the size
			/////////////////////////////////////////////////////////////////////////////////////////////
			active_shape[0] = std::min((int) std::max((float) min_edge, shape[0]*resampling_factor), pyramid_shapes[level-1][0]);
			active_shape[1] = std::min((int) std::max((float) min_edge, shape[1]*resampling_factor), pyramid_shapes[level-1][1]);
			if (shape[2] != 1)
				active_shape[2] = std::min((int) std::max((float) min_edge, shape[2]*resampling_factor), pyramid_shapes[level-1][2]);
			else active_shape[2] = 1;
			/////////////////////////////////////////////////////////////////////////////////////////////

			//Check if pyramid level results in a reasonable sized image
			/////////////////////////////////////////////////////////////////////////////////////////////
			if (scaling_mode == "Ershov" && (active_shape[0] < min_edge || active_shape[1] < min_edge || (active_shape[2] < min_edge && shape[2] != 1)))
			{
				params->pyramid.nLevels = level;
				break;
			}
			/////////////////////////////////////////////////////////////////////////////////////////////

			pyramid_shapes.push_back(active_shape);

			if (active_shape[0] == min_edge && active_shape[1] == min_edge && (active_shape[2] == min_edge || shape[2] == 1))
			{
				params->pyramid.nLevels = level+1;
				break;
			}
		}

		//Preallocate sufficient memory for anything but level 0 on host
		/////////////////////////////////////////////////////////////////////////////////////////////
		if(allocate_memory)
		{
			if(params->pyramid.nLevels > 1)
			{
				long long int nslice_level1 = pyramid_shapes[1][0]*pyramid_shapes[1][1];
				long long int nstack_level1 = pyramid_shapes[1][2]*nslice_level1;
				active_frame = (float*) malloc(nstack_level1*sizeof(*active_frame));
			}
			else
				active_frame = (float*) malloc(0*sizeof(*active_frame));
		}
		/////////////////////////////////////////////////////////////////////////////////////////////

		return;
	}

	void ImagePyramid::resample_frame(float *frame, int shape[3], int level, optflow::ProtocolParameters *params, std::string interpolation_mode)
	{
		/////////////////////////////////////////////////////////////////////////////////////////////
		float scaling_factor = params->pyramid.scaling_factor;
		std::string scaling_mode = params->pyramid.scaling_mode;

		long long int nslice = shape[0]*shape[1];
		long long int nstack = nslice*shape[2];

		if (level == 0)
		{
			free(active_frame);
			active_frame = frame;
			return;
		}
		/////////////////////////////////////////////////////////////////////////////////////////////

		//When not specified, set interpolation to default for given mode
		/////////////////////////////////////////////////////////////////////////////////////////////
		     if (interpolation_mode == "cubic"  && scaling_mode == "Ershov") interpolation_mode = "cubic_unfiltered";
		else if (interpolation_mode == "linear" && scaling_mode == "Ershov") interpolation_mode = "linear_unfiltered";
		else if (interpolation_mode == "cubic"  && scaling_mode == "Liu")    interpolation_mode = "cubic_filtered";
		else if (interpolation_mode == "linear" && scaling_mode == "Liu")    interpolation_mode = "linear_filtered";
		else if (interpolation_mode == "cubic") interpolation_mode = "cubic_unfiltered";
		else if (interpolation_mode == "linear") interpolation_mode = "linear_unfiltered";
		/////////////////////////////////////////////////////////////////////////////////////////////

		//Unfiltered resampling
		/////////////////////////////////////////////////////////////////////////////////////////////
		if      (interpolation_mode == "cubic_unfiltered"){
			resample::cubic_coons(frame, shape, active_frame, pyramid_shapes[level]);
			return;
		}
		else if (interpolation_mode == "linear_unfiltered"){
			resample::linear_coons(frame, shape, active_frame, pyramid_shapes[level]);
			return;
		}
		/////////////////////////////////////////////////////////////////////////////////////////////

		//Filtered resampling
		/////////////////////////////////////////////////////////////////////////////////////////////

		int n_Liu = log(0.25)/log(scaling_factor);
		float sigma = ((1.f/scaling_factor)-1.f)*level;

		//Just do the filtering depending on previous level once
		///////////////////////////////////////////////////
		if (scaling_mode == "Liu" && level >= n_Liu)
		{
			if (level == params->pyramid.nLevels-1)
			{
				sigma = ((1.f/scaling_factor)-1.f)*n_Liu;
				buildBackupsLiu(frame, shape, n_Liu, sigma, interpolation_mode, level);
			}
			else
			{
				free(liu_frames[liu_frames.size()-1]);
				liu_frames.pop_back();
			}
			std::swap(active_frame,liu_frames[liu_frames.size()-1]);
			return;
		}
		else if (scaling_mode == "Liu" && liu_frames.size() != 0)
		{
			free(liu_frames[liu_frames.size()-1]);
			liu_frames.clear();

			free(active_frame);
			active_frame = (float*) malloc(nstack*sizeof(*active_frame));
		}
		///////////////////////////////////////////////////

		float *next_image = (float*) malloc(nstack*sizeof(*next_image));

		if (interpolation_mode.find((std::string) "antialiasing") == std::string::npos)
		{
			//Default Gaussian blur following Liu

			//Create Gaussian kernel for current level
			///////////////////////////////////////////////////
			int fsize = (int) (3*sigma);
			double sum = 0;

			std::vector<float> kernel(2*fsize+1, 0);
			for(uint16_t p=0; p < kernel.size();p++)
			{
				kernel[p] = exp(-((p-fsize)*(p-fsize))/(sigma*sigma*2));
				sum += kernel[p];
			}
			for(uint16_t p=0; p<kernel.size();p++) kernel[p] /= sum;
			///////////////////////////////////////////////////

			///////////////////////////////////////////////////
			//Interpolate blurred frame
			filter::apply_3Dconvolution_splitdims(frame, shape, kernel, next_image);
			///////////////////////////////////////////////////
		}
		else
		{
			//Set sigma for each axis according to Sun, Roth, Black: "A Quantitative Analysis of Current Practices in Optical Flow Estimation and the Principles behind Them"
			///////////////////////////////////////////////////
			float sigma0 = 1.f/sqrtf((2.f*pyramid_shapes[level][0])/((float) shape[0]));
			float sigma1 = 1.f/sqrtf((2.f*pyramid_shapes[level][1])/((float) shape[1]));
			float sigma2 = 1.f/sqrtf((2.f*pyramid_shapes[level][2])/((float) shape[2]));

			int fsize0 = (int) (3*sigma0);
			int fsize1 = (int) (3*sigma1);
			int fsize2 = (int) (3*sigma2);
			double sum0 = 0.0;
			double sum1 = 0.0;
			double sum2 = 0.0;

			std::vector<float> kernel0(2*fsize0+1, 0);
			std::vector<float> kernel1(2*fsize1+1, 0);
			std::vector<float> kernel2(2*fsize2+1, 0);
			for(uint16_t p=0; p < kernel0.size();p++){kernel0[p] = exp(-((p-fsize0)*(p-fsize0))/(sigma0*sigma0*2)); sum0 += kernel0[p];}
			for(uint16_t p=0; p < kernel1.size();p++){kernel1[p] = exp(-((p-fsize1)*(p-fsize1))/(sigma1*sigma1*2)); sum1 += kernel1[p];}
			for(uint16_t p=0; p < kernel2.size();p++){kernel2[p] = exp(-((p-fsize2)*(p-fsize2))/(sigma2*sigma2*2)); sum2 += kernel2[p];}

			for(uint16_t p=0; p<kernel0.size();p++) kernel0[p] /= sum0;
			for(uint16_t p=0; p<kernel1.size();p++) kernel1[p] /= sum1;
			for(uint16_t p=0; p<kernel2.size();p++) kernel2[p] /= sum2;

			filter::apply_3Dconvolution_splitdims(frame, shape, kernel0, kernel1, kernel2, next_image);
			///////////////////////////////////////////////////
		}

		     if (interpolation_mode == "cubic_filtered" || interpolation_mode == "cubic_antialiasing")  resample::cubic_coons(next_image, shape, active_frame, pyramid_shapes[level]);
		else if (interpolation_mode == "linear_filtered" || interpolation_mode == "linear_antialiasing") resample::linear_coons(next_image, shape, active_frame, pyramid_shapes[level]);

		free(next_image);
		///////////////////////////////////////////////////


		/////////////////////////////////////////////////////////////////////////////////////////////

		return;
	}

	void ImagePyramid::buildBackupsLiu(float *frame, int shape[3], int n_Liu, float sigma, std::string interpolation_mode, int maxlevel)
	{
		long long int nslice = shape[0]*shape[1];
		long long int nstack = nslice*shape[2];

		//Create Gaussian kernel
		///////////////////////////////////////////////////
		int fsize = (int) (3*sigma);
		double sum = 0;

		std::vector<float> kernel(2*fsize+1, 0);
		for(uint16_t p=0; p < kernel.size();p++)
		{
			kernel[p] = exp(-((p-fsize)*(p-fsize))/(sigma*sigma*2));
			sum += kernel[p];
		}
		for(uint16_t p=0; p<kernel.size();p++) kernel[p] /= sum;
		///////////////////////////////////////////////////

		for (int l = n_Liu; l <= maxlevel; l++)
		{
			float *filtered_image;
			float *downsampled;
			long long int ndown;

			if(l==n_Liu)
			{
				filtered_image = (float*) malloc(nstack*sizeof(*filtered_image));
				filter::apply_3Dconvolution_splitdims(frame, shape, kernel, filtered_image);

				ndown = pyramid_shapes[l][0]*pyramid_shapes[l][1]; ndown *= pyramid_shapes[l][2];
				downsampled = (float*) calloc(ndown, sizeof(*downsampled));

				     if (interpolation_mode == "cubic_filtered")  resample::cubic_coons(filtered_image, shape, downsampled, pyramid_shapes[l]);
				else if (interpolation_mode == "linear_filtered") resample::linear_coons(filtered_image, shape, downsampled, pyramid_shapes[l]);
			}
			else
			{
				long long int nstack2 = pyramid_shapes[l-1][0]*pyramid_shapes[l-1][1]; nstack2 *= pyramid_shapes[l-1][2];
				filtered_image = (float*) malloc(nstack2*sizeof(*filtered_image));
				filter::apply_3Dconvolution_splitdims(liu_frames[liu_frames.size()-1], pyramid_shapes[l-1], kernel, filtered_image);

				ndown = pyramid_shapes[l][0]*pyramid_shapes[l][1]; ndown *= pyramid_shapes[l][2];
				downsampled = (float*) calloc(ndown, sizeof(*downsampled));

					 if (interpolation_mode == "cubic_filtered")  resample::cubic_coons(filtered_image, pyramid_shapes[l-1], downsampled, pyramid_shapes[l]);
				else if (interpolation_mode == "linear_filtered") resample::linear_coons(filtered_image, pyramid_shapes[l-1], downsampled, pyramid_shapes[l]);
			}

			liu_frames.push_back(downsampled);
		}

		return;
	}
}
