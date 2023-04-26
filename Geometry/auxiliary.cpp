#include <iostream>
#include <vector>
#include <algorithm>
#include <omp.h>
#include <math.h>
#include "histogram.h"
#include "../protocol_parameters.h"
#include "hdcommunication.h"
#include "../Scaling/resampling.h"

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

    float* backup_imagestack(float* image, int shape[3])
    {
    	long long int nslice = shape[0]*shape[1];
    	long long int nstack = shape[2]*nslice;
    	float* output = (float*) malloc(nstack*sizeof(*output));

    	#pragma omp parallel for
    	for(long long int idx = 0; idx < nstack; idx++)
    		output[idx] = image[idx];

    	return output;
    }

    /*Histogram based normalization
    *********************************************************/
    void zeroshift_minimum(float *image0, float *image1, int shape[3])
    {
    	long long int nslice = shape[0]*shape[1];
    	long long int nstack = nslice*shape[2];

		float min_value = image0[0];

		#pragma omp parallel for reduction(min: min_value)
		for (long long int pos = 0; pos < nstack; pos++)
		{
			min_value = std::min(std::min(image0[pos], min_value), image1[pos]);
		}

		std::cout << "min_value of " << min_value << " shifted" << std::endl;

		#pragma omp parallel for
		for (long long int pos = 0; pos < nstack; pos++)
		{
			image0[pos] += min_value;
			image1[pos] += min_value;
		}
		return;
    }

    void transform_values(std::string transformation, float*image, int shape[3])
    {
    	long long int nslice = shape[0]*shape[1];
    	long long int nstack = shape[2]*nslice;

    	if (transformation == "sqrt")
    	{
			#pragma omp parallel for
			for (long long int idx = 0; idx < nstack; idx++)
				image[idx] = sqrtf(std::max(0.0f, image[idx]));
    	}
    	else if (transformation == "log")
    	{
			#pragma omp parallel for
			for (long long int idx = 0; idx < nstack; idx++)
				image[idx] = log(image[idx]);
    	}
    	else if (transformation == "minus_log")
		{
			#pragma omp parallel for
			for (long long int idx = 0; idx < nstack; idx++)
				image[idx] = -log(image[idx]);
		}
    	else if (transformation == "exp")
		{
			#pragma omp parallel for
			for (long long int idx = 0; idx < nstack; idx++)
				image[idx] = exp(image[idx]);
		}
    	else if (transformation.substr(0,3)  == "pow")
		{
    		double power = atof((transformation.substr(3)).c_str());
			#pragma omp parallel for
			for (long long int idx = 0; idx < nstack; idx++)
				image[idx] = pow(image[idx], power);
		}
    	else
    	{
    		std::cout << "Unknown input string to transform_values!" << std::endl;
    	}
    	return;
    }

    void normalize2frames_simple(float *image0, float *image1, int shape[3], optflow::ProtocolParameters *params)
    {
    	long long int nslice = shape[0]*shape[1];
    	long long int nstack = nslice*shape[2];

    	float max_value = image0[0];
    	float min_value = image0[0];

		#pragma omp parallel for reduction(max: max_value) reduction(min: min_value)
    	for (long long int pos = 0; pos < nstack; pos++)
    	{
    		max_value = std::max(std::max(image0[pos], max_value), image1[pos]);
    		min_value = std::min(std::min(image0[pos], min_value), image1[pos]);
    	}

		#pragma omp parallel for
    	for (long long int pos = 0; pos < nstack; pos++)
    	{
    		image0[pos] = (image0[pos]-min_value)/(max_value-min_value);
    		image1[pos] = (image1[pos]-min_value)/(max_value-min_value);
    	}

    	//adjust intensity range in parameters
    	params->constraint.intensityRange[0] = (params->constraint.intensityRange[0]-min_value)/(max_value-min_value);
    	params->constraint.intensityRange[1] = (params->constraint.intensityRange[1]-min_value)/(max_value-min_value);

    	return;
    }
    void normalize2frames_histogram(float *image0, float *image1, int shape[3], optflow::ProtocolParameters *params, int n_bins, double cutoff,
    		bool ignore_zero, bool extrapolate, bool rescale_zero)
	{
		histo::Histogram histo;
		histo.histocutoff = cutoff;
		histo.ignore_zero = ignore_zero;

		std::pair<float,float> bounds = histo.get_effectivehistogrambounds_2frame(image0, image1, shape, n_bins);

		long long int nslice = shape[0]*shape[1];
		long long int nstack = nslice*shape[2];

		float max_value = bounds.second;
		float min_value = bounds.first;

		#pragma omp parallel for
		for (long long int pos = 0; pos < nstack; pos++)
		{
			float val0 = image0[pos];
			float val1 = image1[pos];

			if (val0 != 0.0f || rescale_zero || !ignore_zero)
				val0 = (val0-min_value)/(max_value-min_value);
			if (val1 != 0.0f || rescale_zero || !ignore_zero)
				val1 = (val1-min_value)/(max_value-min_value);

			if (extrapolate)
			{
				image0[pos] = val0;
				image1[pos] = val1;
			}
			else
			{
				image0[pos] = std::max(0.f, std::min(val0, 1.0f));
				image1[pos] = std::max(0.f, std::min(val1, 1.0f));
			}
		}

		//adjust intensity range in parameters
		if(!extrapolate)
		{
			params->constraint.intensityRange[0] = (params->constraint.intensityRange[0]-min_value)/(max_value-min_value);
			params->constraint.intensityRange[1] = (params->constraint.intensityRange[1]-min_value)/(max_value-min_value);
		}

		return;
	}
    void normalize2frames_histogram_independent(float *image0, float *image1, int shape[3], optflow::ProtocolParameters *params, int n_bins, double cutoff,
    		bool ignore_zero, bool extrapolate, bool rescale_zero)
	{
    	histo::Histogram histo;
    	histo.histocutoff = cutoff;
    	histo.ignore_zero = ignore_zero;

    	std::vector<double> histo_edges0, histo_edges1;
    	std::vector<uint64_t> histo_bins0, histo_bins1;

    	std::pair<float,float> bounds0 = histo.get_effectivehistogrambounds(image0, shape, n_bins);
    	std::pair<float,float> bounds1 = histo.get_effectivehistogrambounds(image1, shape, n_bins);

		long long int nslice = shape[0]*shape[1];
		long long int nstack = nslice*shape[2];

		float max_value0 = bounds0.second;
		float min_value0 = bounds0.first;
		float max_value1 = bounds1.second;
		float min_value1 = bounds1.first;

		#pragma omp parallel for
		for (long long int pos = 0; pos < nstack; pos++)
		{
			float val0 = image0[pos];
			float val1 = image1[pos];

			if (val0 != 0.0f || rescale_zero || !ignore_zero)
				val0 = (val0-min_value0)/(max_value0-min_value0);
			if (val1 != 0.0f || rescale_zero || !ignore_zero)
				val1 = (val1-min_value1)/(max_value1-min_value1);

			if (extrapolate)
			{
				image0[pos] = val0;
				image1[pos] = val1;
			}
			else
			{
				image0[pos] = std::max(0.f, std::min(val0, 1.0f));
				image1[pos] = std::max(0.f, std::min(val1, 1.0f));
			}
		}

		if(!extrapolate)
		{
			//adjust intensity range in parameters
			float min_value = std::min(min_value0, min_value1);
			float max_value = std::max(max_value0, max_value1);
			params->constraint.intensityRange[0] = (params->constraint.intensityRange[0]-min_value)/(max_value-min_value);
			params->constraint.intensityRange[1] = (params->constraint.intensityRange[1]-min_value)/(max_value-min_value);
		}

		return;
	}
    void normalize2frames_histogram_mask(float *image0, float *image1, float *mask, int shape[3], optflow::ProtocolParameters *params, int n_bins, double cutoff,
    		bool extrapolate, bool rescale_zero)
	{
		histo::Histogram histo;
		histo.histocutoff = cutoff;
		histo.ignore_zero = true;

		long long int nslice = shape[0]*shape[1];
		long long int nstack = nslice*shape[2];

		float* tmpimage = (float*) calloc(nstack, sizeof(*tmpimage));

		#pragma omp parallel for
		for(long long int idx = 0; idx < nstack; idx++)
			if(mask[idx] != 0) tmpimage[idx] = image0[idx];

		std::pair<float,float> bounds = histo.get_effectivehistogrambounds(tmpimage, shape, n_bins);
		float max_value = bounds.second;
		float min_value = bounds.first;

		#pragma omp parallel for
		for (long long int pos = 0; pos < nstack; pos++)
		{
			float val0 = image0[pos];
			float val1 = image1[pos];

			if (val0 != 0.0f || rescale_zero)
				val0 = (val0-min_value)/(max_value-min_value);
			if (val1 != 0.0f || rescale_zero)
				val1 = (val1-min_value)/(max_value-min_value);

			if (extrapolate)
			{
				image0[pos] = val0;
				image1[pos] = val1;
			}
			else
			{
				image0[pos] = std::max(0.f, std::min(val0, 1.0f));
				image1[pos] = std::max(0.f, std::min(val1, 1.0f));
			}
		}

		if(!extrapolate)
		{
			//adjust intensity range in parameters
			params->constraint.intensityRange[0] = (params->constraint.intensityRange[0]-min_value)/(max_value-min_value);
			params->constraint.intensityRange[1] = (params->constraint.intensityRange[1]-min_value)/(max_value-min_value);
		}

		return;
	}
    void normalize1frame_histogramequalized(float *image, int shape[3], optflow::ProtocolParameters *params, int n_bins, int n_bins_out, double cutoff, bool ignore_zero)
	{
		histo::Histogram histo;
		histo.histocutoff = cutoff;
		histo.ignore_zero = ignore_zero;

		std::pair<float,float> bounds = histo.get_effectivehistogrambounds(image, shape, n_bins);
		std::vector<uint64_t> histobins;
		std::vector<double> histoedges;

		//switching to finer scale
		n_bins = n_bins_out;

		histo.calculatehistogram(image, shape, n_bins, bounds.first, bounds.second, histobins, histoedges);

		if (params->preprocessing.sqrt_equalization)
		{
			for (int p = 0; p < histobins.size(); p++)
					histobins[p] = sqrtf(histobins[p]);
		}

		//get cumulative histogram
		/////////////////////////////////////////////////////////
		for (int p = 1; p < histobins.size(); p++)
			histobins[p] += histobins[p-1];

		if (params->preprocessing.smoothed_equalization > 0)
		{
			std::vector<uint64_t> histobins1(histobins.size(), 0);

			for (int p = 0; p < histobins.size(); p++)
			{
				float sum = histobins[p];
				float count = 1;
				for (int r = 1; r <= params->preprocessing.smoothed_equalization/2; r++)
				{
					if (p-r >= 0){sum += histobins[p-r]; count++;}
					if (p+r < histobins.size()){sum += histobins[p+r]; count++;}
				}
				histobins1[p] = sum/count;
			}
			std::swap(histobins, histobins1);
		}
		/////////////////////////////////////////////////////////

		long long int nslice = shape[0]*shape[1];
		long long int nstack = nslice*shape[2];
		long long int cdfmin = histobins[0];

		float max_value = bounds.second;
		float min_value = bounds.first;

		//equalize
		#pragma omp parallel for
		for (long long int pos = 0; pos < nstack; pos++)
		{
			float val0 = (image[pos]-min_value)/(max_value-min_value);

			val0 = std::max(0.f, std::min(val0, 1.0f));

			int this_bin0 = std::min((int) (val0*n_bins), n_bins-1);

			image[pos] = ((float) (histobins[this_bin0]-cdfmin))/(histobins[histobins.size()-1]);
		}

		//adjust intensity range in parameters
		params->constraint.intensityRange[0] = (params->constraint.intensityRange[0]-min_value)/(max_value-min_value);
		params->constraint.intensityRange[1] = (params->constraint.intensityRange[1]-min_value)/(max_value-min_value);

		return;
	}
    void normalize2frames_histogramequalized(float *image0, float *image1, int shape[3], optflow::ProtocolParameters *params, int n_bins, int n_bins_out, double cutoff, bool ignore_zero)
	{
		histo::Histogram histo;
		histo.histocutoff = cutoff;
		histo.ignore_zero = ignore_zero;

		std::pair<float,float> bounds = histo.get_effectivehistogrambounds_2frame(image0, image1, shape, n_bins);
		std::vector<uint64_t> histobins_frame0, histobins_frame1;
		std::vector<double> histoedges_frame0, histoedges_frame1;

		//switching to finer scale
		n_bins = n_bins_out;

		histo.calculatehistogram(image0, shape, n_bins, bounds.first, bounds.second, histobins_frame0, histoedges_frame0);
		histo.calculatehistogram(image1, shape, n_bins, bounds.first, bounds.second, histobins_frame1, histoedges_frame1);

		if (params->preprocessing.sqrt_equalization)
		{
			for (int p = 0; p < histobins_frame0.size(); p++)
			{
					histobins_frame0[p] = sqrtf(histobins_frame0[p]+histobins_frame1[p]);
					histobins_frame1[p] = 0.0f;
			}
		}

		//get cumulative histogram
		/////////////////////////////////////////////////////////
		histobins_frame0[0] += histobins_frame1[0];
		for (int p = 1; p < histobins_frame0.size(); p++)
			histobins_frame0[p] += histobins_frame0[p-1]+histobins_frame1[p];

		if (params->preprocessing.smoothed_equalization > 0)
		{
			for (int p = 0; p < histobins_frame0.size(); p++)
			{
				float sum = histobins_frame0[p];
				float count = 1;
				for (int r = 1; r <= params->preprocessing.smoothed_equalization/2; r++)
				{
					if (p-r >= 0){sum += histobins_frame0[p-r]; count++;}
					if (p+r < histobins_frame0.size()){sum += histobins_frame0[p+r]; count++;}
				}
				histobins_frame1[p] = sum/count;
			}
			std::swap(histobins_frame0, histobins_frame1);
		}
		/////////////////////////////////////////////////////////

		long long int nslice = shape[0]*shape[1];
		long long int nstack = nslice*shape[2];
		long long int cdfmin = histobins_frame0[0];

		float max_value = bounds.second;
		float min_value = bounds.first;

		//equalize
		#pragma omp parallel for
		for (long long int pos = 0; pos < nstack; pos++)
		{
			float val0 = (image0[pos]-min_value)/(max_value-min_value);
			float val1 = (image1[pos]-min_value)/(max_value-min_value);

			val0 = std::max(0.f, std::min(val0, 1.0f));
			val1 = std::max(0.f, std::min(val1, 1.0f));

			int this_bin0 = std::min((int) (val0*n_bins), n_bins-1);
			int this_bin1 = std::min((int) (val1*n_bins), n_bins-1);

			image0[pos] = ((float) (histobins_frame0[this_bin0]-cdfmin))/(histobins_frame0[histobins_frame0.size()-1]);
			image1[pos] = ((float) (histobins_frame0[this_bin1]-cdfmin))/(histobins_frame0[histobins_frame0.size()-1]);
		}

		//adjust intensity range in parameters
		params->constraint.intensityRange[0] = (params->constraint.intensityRange[0]-min_value)/(max_value-min_value);
		params->constraint.intensityRange[1] = (params->constraint.intensityRange[1]-min_value)/(max_value-min_value);

		return;
	}
    void normalize2frames_histogramequalized_mask(float *image0, float *image1, float *mask, int shape[3], optflow::ProtocolParameters *params, int n_bins, int n_bins_out, double cutoff)
	{
		histo::Histogram histo;
		histo.histocutoff = cutoff;
		histo.ignore_zero = true;

		long long int nslice = shape[0]*shape[1];
		long long int nstack = nslice*shape[2];

		float* tmpimage = (float*) calloc(nstack, sizeof(*tmpimage));

		#pragma omp parallel for
		for(long long int idx = 0; idx < nstack; idx++)
			if(mask[idx] != 0) tmpimage[idx] = image0[idx];

		std::pair<float,float> bounds = histo.get_effectivehistogrambounds(tmpimage, shape, n_bins);
		std::vector<uint64_t> histobins_frame0;
		std::vector<double> histoedges_frame0;

		//switching to finer scale
		n_bins = n_bins_out;

		histo.calculatehistogram(tmpimage, shape, n_bins, bounds.first, bounds.second, histobins_frame0, histoedges_frame0);
		free(tmpimage);

		if (params->preprocessing.sqrt_equalization)
		{
			for (int p = 0; p < histobins_frame0.size(); p++)
					histobins_frame0[p] = sqrtf(histobins_frame0[p]);
		}

		//get cumulative histogram
		/////////////////////////////////////////////////////////
		for (int p = 1; p < histobins_frame0.size(); p++)
			histobins_frame0[p] += histobins_frame0[p-1];
		/////////////////////////////////////////////////////////

		if (params->preprocessing.smoothed_equalization > 0)
		{
			std::vector<uint64_t> histobins_frame1(histobins_frame0.size(), 0);

			for (int p = 0; p < histobins_frame0.size(); p++)
			{
				float sum = histobins_frame0[p];
				float count = 1;
				for (int r = 1; r <= params->preprocessing.smoothed_equalization/2; r++)
				{
					if (p-r >= 0){sum += histobins_frame0[p-r]; count++;}
					if (p+r < histobins_frame0.size()){sum += histobins_frame0[p+r]; count++;}
				}
				histobins_frame1[p] = sum/count;
			}
			std::swap(histobins_frame0, histobins_frame1);
		}

		long long int cdfmin = histobins_frame0[0];

		float max_value = bounds.second;
		float min_value = bounds.first;

		//equalize
		#pragma omp parallel for
		for (long long int pos = 0; pos < nstack; pos++)
		{
			float val0 = (image0[pos]-min_value)/(max_value-min_value);
			float val1 = (image1[pos]-min_value)/(max_value-min_value);

			val0 = std::max(0.f, std::min(val0, 1.0f));
			val1 = std::max(0.f, std::min(val1, 1.0f));

			if (val0 >= 0 && val0 <= 1.0)
			{
				int this_bin0 = std::min((int) (val0*n_bins), n_bins-1);
				image0[pos] = ((float) (histobins_frame0[this_bin0]-cdfmin))/(histobins_frame0[histobins_frame0.size()-1]);
			}
			else image0[pos] = val0;

			if(val1 >= 0 && val1 <= 1.0)
			{
			int this_bin1 = std::min((int) (val1*n_bins), n_bins-1);
			image1[pos] = ((float) (histobins_frame0[this_bin1]-cdfmin))/(histobins_frame0[histobins_frame0.size()-1]);
			}
			else image1[pos] = val1;
		}

		//adjust intensity range in parameters
		params->constraint.intensityRange[0] = (params->constraint.intensityRange[0]-min_value)/(max_value-min_value);
		params->constraint.intensityRange[1] = (params->constraint.intensityRange[1]-min_value)/(max_value-min_value);

		return;
	}
    void normalizeframe_histogram(float *image0, int shape[3])
	{
		histo::Histogram histo;
		histo.ignore_zero = false;

		std::pair<float,float> bounds = histo.get_effectivehistogrambounds(image0, shape, 1000);

		long long int nslice = shape[0]*shape[1];
		long long int nstack = nslice*shape[2];

		float max_value = bounds.second;
		float min_value = bounds.first;

		#pragma omp parallel for
		for (long long int pos = 0; pos < nstack; pos++)
		{
			float val0 = (image0[pos]-min_value)/(max_value-min_value);

			image0[pos] = std::max(0.f, std::min(val0, 1.0f));
		}
		return;
	}

    void set_initialguess(float* flowvector, int active_shape[3], int full_shape[3], float constant_guess[3], std::string previous_result_path)
    {
		long long int nslice = active_shape[0]*active_shape[1];
		long long int nstack = nslice*active_shape[2];

		if(previous_result_path == "zstrain")
		{
			float rel_origin = constant_guess[0];
			constant_guess[0] = 0.0;
			float zcog = active_shape[2]*rel_origin-0.5f;//active_shape[2]/2.-0.5f;

			#pragma omp parallel for
			for (long long int idx = 0; idx < nstack; idx++)
			{
				int z = idx/nslice;
				float shift = (z-zcog)*constant_guess[2];//*active_shape[2]/active_shape[2];
				flowvector[idx+2*nstack] = shift;
			}
		}
		else if(previous_result_path != "none")
    	{
			float *guess_dim0 = (float*) calloc(nstack, sizeof(*guess_dim0));
			float *guess_dim1 = (float*) calloc(nstack, sizeof(*guess_dim1));
			float *guess_dim2 = (float*) calloc(nstack, sizeof(*guess_dim2));

			hdcom::HdCommunication hdcom;
			int resshape[3];
			float* guess_activedim = hdcom.GetTif_unknowndim_32bit(previous_result_path+"/dx/", resshape);
			resample::linear_coons(guess_activedim, resshape, guess_dim0, active_shape, 1, true);
			free(guess_activedim);

			guess_activedim = hdcom.GetTif_unknowndim_32bit(previous_result_path+"/dy/", resshape);
			resample::linear_coons(guess_activedim, resshape, guess_dim1, active_shape, 1, true);
			free(guess_activedim);

			guess_activedim = hdcom.GetTif_unknowndim_32bit(previous_result_path+"/dz/", resshape);
			resample::linear_coons(guess_activedim, resshape, guess_dim2, active_shape, 1, true);
			free(guess_activedim);

			#pragma omp parallel for
			for (long long int idx = 0; idx < nstack; idx++)
			{
				flowvector[idx] = guess_dim0[idx];
				flowvector[idx+nstack] = guess_dim1[idx];
				flowvector[idx+2*nstack] = guess_dim2[idx];
			}
			free(guess_dim0); free(guess_dim1); free(guess_dim2);
    	}
    	else
    	{
			#pragma omp parallel for
			for (long long int idx = 0; idx < nstack; idx++)
			{
				flowvector[idx] = constant_guess[0]*((float) active_shape[0])/((float) full_shape[0]);
				flowvector[idx+nstack] = constant_guess[1]*((float) active_shape[1])/((float) full_shape[1]);
				flowvector[idx+2*nstack] = constant_guess[2]*((float) active_shape[2])/((float) full_shape[2]);
			}
    	}

		return;
    }
    void add_initial_ycompression(float* flowvector, int active_shape[3], float compression)
	{
		long long int nslice = active_shape[0]*active_shape[1];
		long long int nstack = nslice*active_shape[2];

		std::cout << "applying y-elongation on shape[1] = " << active_shape[1] << std::endl;
		for (int y = 0; y < active_shape[1]; y+=20)
		{

			float offset = 606.f-y;//(active_shape[1]/429.f*265.f)-y;
			float shift = -(compression*offset);
			std::cout << "y " << y << ": " << shift << std::endl;
		}

		#pragma omp parallel for
		for (long long int idx = 0; idx < nstack; idx++)
		{
			int z = idx/nslice;
			int y = (idx-z*nslice)/active_shape[0];
			//float offset = (active_shape[1]/2.)-y;
			//float shift = -(offset - ((1.f-compression)*offset));

			float offset = 606.f-y;//(active_shape[1]/429.f*265.f)-y;
			float shift = -(compression*offset);
			flowvector[idx+nstack] += shift-138.f;
		}

		return;
	}

    float* project_average_through_mask(float* img, float* mask, int shape[3], int dim, bool absolute_values, std::string tag)
	{
		long long int nslice = shape[0]*shape[1];
		long long int nstack = shape[2]*nslice;
		float* output;
		float* counter;

		float sum = 0;
		float overall_count = 0;

		int outshape[2] = {shape[1], shape[2]};
		if (dim == 1) outshape[0] = shape[0];
		else if (dim == 2) {outshape[0] = shape[0]; outshape[1] = shape[1];}

		long long int nslice_out = outshape[0]*outshape[1];
		output = (float*) calloc(nslice_out, sizeof(*output));
		counter = (float*) calloc(nslice_out, sizeof(*counter));

		for (long long int idx = 0; idx < nstack; idx++)
		{
			if (mask[idx] != 0)
			{
				int z = idx/nslice;
				int y = (idx-z*nslice)/shape[0];
				int x = idx-z*nslice-y*shape[0];

				float val = img[idx];
				if (absolute_values) val = fabs(val);

				long long int outidx;
				if(dim == 0) outidx = z*shape[1]+y;
				else if (dim == 1) outidx = z*shape[0]+x;
				else outidx = y*shape[0]+x;

				output[outidx] += val;
				counter[outidx]++;
				sum += val;
				overall_count++;
			}
		}

		#pragma omp parallel for
		for (long long int idx = 0; idx < nslice_out; idx++)
		{
			if(counter[idx] > 0) output[idx] /= counter[idx];
		}

		if(tag != "none")
			std::cout << "global " << tag << ": " << sum/overall_count << std::endl;

		return output;
	}

    std::vector<float> _calceigenvector_3x3symmetric(float Exx, float Eyy, float Ezz, float Exy, float Exz, float Eyz, float eigenval)
    	{
    		std::vector<float> eigenvector(3,0.0);

    		//https://www.geometrictools.com/Documentation/RobustEigenSymmetric3x3.pdf
    		///(might not be numerically stable)

    		double r0xr1[3] = {Exy*Eyz-Exz*(Eyy-eigenval), Exz*Exy-(Exx-eigenval)*Eyz, (Exx-eigenval)*(Eyy-eigenval)-Exy*Exy};
    		double r0xr2[3] = {Exy*(Ezz-eigenval)-Exz*Eyz, Exz*Exz-(Exx-eigenval)*(Ezz-eigenval), (Exx-eigenval)*Eyz-Exy*Exz};
    		double r1xr2[3] = {(Eyy-eigenval)*(Ezz-eigenval)-Eyz*Eyz, Eyz*Exz-Exy*(Ezz-eigenval), Exy*Eyz-(Eyy-eigenval)*Exz};

    		double d0 = r0xr1[0]*r0xr1[0] + r0xr1[1]*r0xr1[1] + r0xr1[2]*r0xr1[2];
    		double d1 = r0xr2[0]*r0xr2[0] + r0xr2[1]*r0xr2[1] + r0xr2[2]*r0xr2[2];
    		double d2 = r1xr2[0]*r1xr2[0] + r1xr2[1]*r1xr2[1] + r1xr2[2]*r1xr2[2];

    		double dmax = d0;
    		int imax = 0;

    		if (d1 > dmax) {dmax = d1; imax = 1;}
    		if (d2 > dmax) {imax = 2;}

    		if (imax == 0)
    		{
    			double sqrtd = sqrt(d0);
    			eigenvector[0] = r0xr1[0]/sqrtd;
    			eigenvector[1] = r0xr1[1]/sqrtd;
    			eigenvector[2] = r0xr1[2]/sqrtd;
    		}
    		else if (imax == 1)
    		{
    			double sqrtd = sqrt(d1);
    			eigenvector[0] = r0xr2[0]/sqrtd;
    			eigenvector[1] = r0xr2[1]/sqrtd;
    			eigenvector[2] = r0xr2[2]/sqrtd;
    		}
    		else
    		{
    			double sqrtd = sqrt(d2);
    			eigenvector[0] = r1xr2[0]/sqrtd;
    			eigenvector[1] = r1xr2[1]/sqrtd;
    			eigenvector[2] = r1xr2[2]/sqrtd;
    		}

    		return eigenvector;
    	}
    std::vector<std::vector<float>> eigenvalues_and_eigenvectors_3x3symmetric(float Exx, float Eyy, float Ezz, float Exy, float Exz, float Eyz)
    	{
    		//returns a vector with output[0] = three eigenvalues followed by the 3 eigenvectors
    		std::vector<std::vector<float>> output;

    		double eig1, eig2, eig3;

    		//https://en.wikipedia.org/wiki/Eigenvalue_algorithm
    		double p1 = Exy*Exy + Exz*Exz + Eyz*Eyz;
    		if (p1 == 0)
    		{
    			eig1 = Exx;
    			eig2 = Eyy;
    			eig3 = Ezz;

    			if (Eyy > Exx) {eig1 = Eyy; eig2 = Exx;}
    			if (Ezz > eig1){eig3 = eig2; eig2 = eig1; eig1 = Ezz;}
    			else if (Ezz > eig2) {eig3 = eig2; eig2 = Ezz;}
    		}
    		else
    		{
    			double third = 1./3.;
    			double q=(Exx+Eyy+Ezz)*third;

    			double p2 = (Exx-q)*(Exx-q) + (Eyy- q)*(Eyy-q) + (Ezz-q)*(Ezz-q) + 2.*p1;
    		    double p = sqrt(p2 / 6.);
    		    double Bxx = (1./p) * (Exx-q);
    		    double Byy = (1./p) * (Eyy-q);
    			double Bzz = (1./p) * (Ezz-q);
    		    double Bxy = (1./p)*Exy;
    		    double Bxz = (1./p)*Exz;
    		    double Byz = (1./p)*Eyz;
    		    double detB = Bxx*Byy*Bzz-Bxx*Byz*Byz+Bxy*(2.*Bxz*Byz-Bxy*Bzz)+Bxz*Bxz*(-Byy);
    		    double r = detB*0.5;
    		    double phi;

    		   if (r <= -1) phi =  3.14159265359*third;
    		   else if (r >= 1) phi = 0.;
    		   else phi = (acos(r))*third;

    		   //the eigenvalues satisfy eig3 <= eig2 <= eig1
    		   eig1 = q + 2 * p * cos(phi);
    		   eig3 = q + 2 * p * cos(phi + (2*3.14159265359*third));
    		   eig2 = 3 * q - eig1 - eig3;     //since trace(A) = eig1 + eig2 + eig3;
    		}

    		std::vector<float> eigenvec1 = _calceigenvector_3x3symmetric(Exx,Eyy,Ezz,Exy,Exz,Eyz, eig1);
    		std::vector<float> eigenvec2 = _calceigenvector_3x3symmetric(Exx,Eyy,Ezz,Exy,Exz,Eyz, eig2);
    		std::vector<float> eigenvec3 = _calceigenvector_3x3symmetric(Exx,Eyy,Ezz,Exy,Exz,Eyz, eig3);

    		output.push_back({(float) eig1, (float) eig2, (float) eig3});
    		output.push_back(eigenvec1);
    		output.push_back(eigenvec2);
    		output.push_back(eigenvec3);

    		return output;
    	}
}
