#include "histogram.h"
#include <numeric>
#include <algorithm>
#include <omp.h>

namespace histo
{
	std::pair<dtype,dtype> Histogram::get_effectivehistogrambounds(dtype *image, int shape[3], int n_bins)
	{
		long long int nslice = shape[0]*shape[1];
		long long int nstack = nslice*shape[2];

		dtype lowerbound = *std::min_element(image,image+nstack);
		dtype upperbound = *std::max_element(image,image+nstack);

		//First determine the appropriate lower and upper bound from a histogram covering all data
		std::vector<uint64_t> histobins_tmp0;
		std::vector<double> histoedges;
		calculatehistogram(image, shape, n_bins, lowerbound, upperbound, histobins_tmp0, histoedges);

		uint64_t running_sum = 0;
		for (int i = 0; i < n_bins; i++)
		{
			running_sum += histobins_tmp0[i];
			if (running_sum/((double) (nstack)) >= histocutoff)
			{
				//lower bound found
				lowerbound = histoedges[i];
				break;
			}
		}

		running_sum = 0;
		for (int i = n_bins-1; i >= 0; i--)
		{
			running_sum += histobins_tmp0[i];
			if (running_sum/((double) (nstack)) >= histocutoff)
			{
				//upper bound found
				upperbound = histoedges[i+1];
				break;
			}
		}

		std::pair<dtype,dtype> bounds = {lowerbound, upperbound};

		return bounds;
	}
	std::pair<dtype,dtype> Histogram::get_effectivehistogrambounds_2frame(dtype *frame0, dtype *frame1, int shape[3], int n_bins)
	{
		long long int nslice = shape[0]*shape[1];
		long long int nstack = nslice*shape[2];

		dtype lowerbound = *std::min_element(frame0,frame0+nstack);
		dtype upperbound = *std::max_element(frame0,frame0+nstack);
		lowerbound = std::min(*std::min_element(frame1,frame1+nstack), lowerbound);
		upperbound = std::max(*std::max_element(frame1,frame1+nstack), upperbound);

		//First determine the appropriate lower and upper bound from a histogram covering all data
		std::vector<uint64_t> histobins_tmp0, histobins_tmp1;
		std::vector<double> histoedges;
		calculatehistogram(frame0, shape, n_bins, lowerbound, upperbound, histobins_tmp0, histoedges);
		calculatehistogram(frame1, shape, n_bins, lowerbound, upperbound, histobins_tmp1, histoedges);

		uint64_t running_sum = 0;
		for (int i = 0; i < n_bins; i++)
		{
			histobins_tmp0[i] += histobins_tmp1[i];

			running_sum += histobins_tmp0[i];
			if (running_sum/((double) (2*nstack)) >= histocutoff)
			{
				//lower bound found
				lowerbound = histoedges[i];
				break;
			}
		}

		running_sum = 0;
		for (int i = n_bins-1; i >= 0; i--)
		{
			running_sum += histobins_tmp0[i];
			if (running_sum/((double) (2*nstack)) >= histocutoff)
			{
				//upper bound found
				upperbound = histoedges[i+1];
				break;
			}
		}

		std::pair<dtype,dtype> bounds = {lowerbound, upperbound};

		return bounds;
	}

    void Histogram::calculateeffectivehistogram(dtype *data, int shape[3], int n_bins, std::vector<uint64_t> &out_histobins, std::vector<double> &out_histoedges)
    {
    	long long int nslice = shape[0]*shape[1];
    	long long int nstack = nslice*shape[2];

        dtype lowerbound = *std::min_element(data,data+nstack);
        dtype upperbound = *std::max_element(data,data+nstack);

        //First determine the appropriate lower and upper bound from a histogram covering all data
        calculatehistogram(data, shape, n_bins, lowerbound, upperbound, out_histobins, out_histoedges);

        uint64_t running_sum = 0;
        for (int i = 0; i < n_bins; i++)
        {
            running_sum += out_histobins[i];
            if (running_sum/((double) nstack) >= histocutoff)
            {
                //lower bound found
                lowerbound = out_histoedges[i];
                break;
            }
        }

        running_sum = 0;
        for (int i = n_bins-1; i >= 0; i--)
        {
            running_sum += out_histobins[i];
            if (running_sum/((double) nstack) >= histocutoff)
            {
                //upper bound found
                upperbound = out_histoedges[i+1];
                break;
            }
        }
        //Now recalculate with effctive data range:
        calculatehistogram(data, shape, n_bins, lowerbound, upperbound, out_histobins, out_histoedges);

        return;
    }
    void Histogram::calculateeffectivehistogram_masked(dtype *data, float *mask, int shape[3], int n_bins, std::vector<double> &out_histobins, std::vector<double> &out_histoedges)
	{
		long long int nslice = shape[0]*shape[1];
		long long int nstack = nslice*shape[2];

		dtype lowerbound = 1e9;
		dtype upperbound = -1e9;

		#pragma omp parallel for reduction(max: upperbound), reduction(min: lowerbound)
		for (long long int idx = 0; idx < nstack; idx++)
		{
			float maskval = mask[idx];
			if (maskval == 0.0f) continue;

			dtype val = data[idx];

			if(val > upperbound) upperbound = val;
			if(val < lowerbound) lowerbound = val;
		}
		//First determine the appropriate lower and upper bound from a histogram covering all data
		calculatehistogram(data, mask, shape, n_bins, lowerbound, upperbound, out_histobins, out_histoedges);

		uint64_t running_sum = 0;
		for (int i = 0; i < n_bins; i++)
		{
			running_sum += out_histobins[i];
			if (running_sum/((double) nstack) >= histocutoff)
			{
				//lower bound found
				lowerbound = out_histoedges[i];
				break;
			}
		}

		running_sum = 0;
		for (int i = n_bins-1; i >= 0; i--)
		{
			running_sum += out_histobins[i];
			if (running_sum/((double) nstack) >= histocutoff)
			{
				//upper bound found
				upperbound = out_histoedges[i+1];
				break;
			}
		}
		//Now recalculate with effctive data range:
		calculatehistogram(data, mask, shape, n_bins, lowerbound, upperbound, out_histobins, out_histoedges);

		return;
	}
    void Histogram::calculateeffectivehistogram(dtype *data, int shape[3], int n_bins, std::vector<double> &out_histobins, std::vector<double> &out_histoedges)
    {
        std::vector<uint64_t> out_histobins_tmp;
        calculateeffectivehistogram(data, shape, n_bins, out_histobins_tmp, out_histoedges);
        out_histobins.assign(out_histobins_tmp.size(),0);
        for(uint64_t idx = 0; idx < out_histobins.size(); idx++)
            out_histobins[idx] = (double) out_histobins_tmp[idx];

        return;
    }

    void Histogram::calculatehistogram(dtype *data, int shape[3], int n_bins, dtype lowerbound, dtype upperbound, std::vector<uint64_t> &out_histobins, std::vector<double> &out_histoedges)
    {
    	long long int nslice = shape[0]*shape[1];
    	long long int nstack = nslice*shape[2];

        double lb = (double) lowerbound;
        double ub = (double) upperbound;
        double range = ub-lb;

        out_histoedges = linspace(lowerbound,upperbound,n_bins+1);
        out_histobins.clear();
        out_histobins.resize(n_bins,0);

        uint64_t this_bin;
        dtype datavalue;
        //Iterate one time to determine the appropriate lower and upper bound
        for (uint64_t idx = 0; idx < nstack; idx++)
        {
            datavalue = data[idx];

            if((datavalue < lowerbound) || (datavalue > upperbound))
                continue;

            this_bin = (datavalue-lowerbound)/range*n_bins;
            out_histobins[this_bin]++;
        }
        return;
    }
    void Histogram::calculatehistogram(dtype *data, int shape[3], int n_bins, dtype lowerbound, dtype upperbound, std::vector<double> &out_histobins, std::vector<double> &out_histoedges)
    {
    	long long int nslice = shape[0]*shape[1];
    	long long int nstack = nslice*shape[2];

        double lb = (double) lowerbound;
        double ub = (double) upperbound;
        double range = ub-lb;

        out_histoedges = linspace(lowerbound,upperbound,n_bins+1);
        out_histobins.clear();
        out_histobins.resize(n_bins,0);

        uint64_t this_bin;
        dtype datavalue;
        //Iterate one time to determine the appropriate lower and upper bound
        for (uint64_t idx = 0; idx < nstack; idx++)
        {
            datavalue = data[idx];

            if((datavalue < lowerbound) || (datavalue > upperbound))
                continue;

            this_bin = (datavalue-lowerbound)/range*n_bins;
            out_histobins[this_bin]++;
        }
        return;
    }
    void Histogram::calculatehistogram(dtype *data, float *mask, int shape[3], int n_bins, dtype lowerbound, dtype upperbound, std::vector<double> &out_histobins, std::vector<double> &out_histoedges)
	{
		long long int nslice = shape[0]*shape[1];
		long long int nstack = nslice*shape[2];

		double lb = (double) lowerbound;
		double ub = (double) upperbound;
		double range = ub-lb;

		out_histoedges = linspace(lowerbound,upperbound,n_bins+1);
		out_histobins.clear();
		out_histobins.resize(n_bins,0);

		uint64_t this_bin;
		dtype datavalue;
		//Iterate one time to determine the appropriate lower and upper bound
		for (uint64_t idx = 0; idx < nstack; idx++)
		{
			if (mask[idx] == 0.0f) continue;

			datavalue = data[idx];

			if((datavalue < lowerbound) || (datavalue > upperbound))
				continue;

			this_bin = (datavalue-lowerbound)/range*n_bins;
			out_histobins[this_bin]++;
		}
		return;
	}

    std::vector<double> Histogram::binedges2bincenter(std::vector<double> &binedges)
    {
        std::vector<double> bincenters(binedges.size()-1,0.);
        for (int i = 0; i < bincenters.size(); i++)
            bincenters[i] = (binedges[i]+binedges[i+1])/2.;
        return bincenters;
    }

    std::vector<double> Histogram::normalize(std::vector<uint64_t> histobins, std::string normalization)
    {
        std::vector<double> output(histobins.size(), 0.);
        double weight = 1.;

        if (normalization == "area")
            weight = std::accumulate(histobins.begin(), histobins.end(), 0.);
        else if (normalization == "height")
            weight = (double) *std::max_element(histobins.begin(), histobins.end());

        for(uint64_t idx; idx < histobins.size(); idx++)
            output[idx] = ((double) histobins[idx])/weight;

        return output;
    }
    std::vector<double> Histogram::normalize(std::vector<double> histobins, std::string normalization)
    {
        std::vector<double> output(histobins.size(), 0.);
        double weight = 1.;

        if (normalization == "area")
            weight = std::accumulate(histobins.begin(), histobins.end(), 0.);
        else if (normalization == "height")
            weight = *std::max_element(histobins.begin(), histobins.end());

        for(uint64_t idx; idx < histobins.size(); idx++)
            output[idx] = (histobins[idx])/weight;

        return output;
    }

    float* Histogram::calculate2DHistogram(float *frame0, float *frame1, int shape[3], std::vector<double> &histoedges0, std::vector<double> &histoedges1, int outshape[2])
    {
    	long long int nslice = shape[0]*shape[1];
    	long long int nstack = shape[2]*nslice;
    	long long int nrange = 0;

    	outshape[0] = histoedges0.size()-1;
		outshape[1] = histoedges1.size()-1;

		float lb0 = histoedges0[0];
		float ub0 = histoedges0[histoedges0.size()-1];
		float lb1 = histoedges1[0];
		float ub1 = histoedges1[histoedges1.size()-1];
		float range0 = ub0-lb0;
		float range1 = ub1-lb1;

    	float *output = (float*) calloc((outshape[0]*outshape[1]),sizeof(*output));

		//#pragma omp parallel for
		for (long long int pos = 0; pos < nstack; pos++)
		{
			float val0 = frame0[pos];
			float val1 = frame1[pos];

			int bin0 = ((val0-lb0)/range0)*outshape[0];
			int bin1 = ((val1-lb1)/range1)*outshape[1];

			if(bin0 < 0 || bin0 >= outshape[0] || bin1 < 0 || bin1 >= outshape[1])
				continue;

			nrange++;
			//bin0 = std::max(0,std::min(outshape[0]-1, bin0));
			//bin1 = std::max(0,std::min(outshape[1]-1, bin1));

			//#pragma omp critical
			//{
				output[bin1*outshape[0]+bin0]++;
			//}
		}

    	//normalize
		#pragma omp parallel for
    	for (long long int pos = 0; pos < outshape[0]*outshape[1]; pos++)
    		output[pos] /= nrange;

    	return output;
    }
    float* Histogram::calculate2DHistogram(float *frame0, float *frame1, float *mask, int shape[3], std::vector<double> &histoedges0, std::vector<double> &histoedges1, int outshape[2])
	{
		long long int nslice = shape[0]*shape[1];
		long long int nstack = shape[2]*nslice;
		long long int nrange = 0;

		outshape[0] = histoedges0.size()-1;
		outshape[1] = histoedges1.size()-1;

		float lb0 = histoedges0[0];
		float ub0 = histoedges0[histoedges0.size()-1];
		float lb1 = histoedges1[0];
		float ub1 = histoedges1[histoedges1.size()-1];
		float range0 = ub0-lb0;
		float range1 = ub1-lb1;

		float *output = (float*) calloc((outshape[0]*outshape[1]),sizeof(*output));

		//#pragma omp parallel for
		for (long long int pos = 0; pos < nstack; pos++)
		{
			if (mask[pos] == 0.0f) continue;

			float val0 = frame0[pos];
			float val1 = frame1[pos];

			int bin0 = ((val0-lb0)/range0)*outshape[0];
			int bin1 = ((val1-lb1)/range1)*outshape[1];

			if(bin0 < 0 || bin0 >= outshape[0] || bin1 < 0 || bin1 >= outshape[1])
				continue;

			nrange++;
			//bin0 = std::max(0,std::min(outshape[0]-1, bin0));
			//bin1 = std::max(0,std::min(outshape[1]-1, bin1));

			//#pragma omp critical
			//{
			output[bin1*outshape[0]+bin0]++;
			//}
		}

		//normalize
		#pragma omp parallel for
		for (long long int pos = 0; pos < outshape[0]*outshape[1]; pos++)
			output[pos] /= nrange;

		return output;
	}

    /*****************************************************************************************/

    std::vector<double> Histogram::linspace(double startval, double endval, uint64_t bins)
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
}
