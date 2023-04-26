#include <iostream>
#include <vector>
#include <algorithm>
#include <omp.h>
#include <fstream>
#include <math.h>

#include "../Geometry/histogram.h"
#include "../protocol_parameters.h"
#include "../Geometry/hdcommunication.h"
#include "../Geometry/filtering.h"

#include "histomatching.h"

namespace histomatch
{
    //When batchprocessing low-dose data normalization fails occasionally.
    //We need to be very rigorous in matching intensities or the dataterm explodes
	//We'll match the modes for now and consider std later.
    /**********************************************************/

	double HistogramOptimization::_get_crosscorrelation(std::vector<double> &histobins0, std::vector<double> &histobins1)
	{
		//ignores 0 bin and 1 bin
		double mean0 = 0.0; double mean1 = 0.0;
		double std0 = 0.0; double std1 = 0.0;
		double crosscorr = 0.0;
		double N = histobins0.size()-2;

		#pragma omp parallel for reduction(+: mean0, mean1, std0, std1, crosscorr)
		for (long long int idx = 1; idx < histobins0.size()-2; idx++)
		{
			double val0 = histobins0[idx];
			double val1 = histobins1[idx];
			mean0 += val0;
			mean1 += val1;
			std0 += val0*val0;
			std1 += val1*val1;
			crosscorr += val0*val1;
		}

		mean0 /= N;
		mean1 /= N;
		std0 = std::sqrt(std0/N-mean0*mean0);
		std1 = std::sqrt(std1/N-mean1*mean1);
		crosscorr = (crosscorr/N-mean0*mean1)/(std0*std1);

		return crosscorr;
	}
	double HistogramOptimization::_get_correlation_of_transformed_histogram(std::vector<double> a, std::vector<double> &histobins0, std::vector<double> &histobins1)
	{
		std::vector<double> histobins_out(histobins1.size(), 0.0);

		//walk through the reference axis
		#pragma omp parallel for
		for (long long int i = 0; i < histobins1.size(); i++)
		{
			//calculate the position in the transformed axis
			double i_new = a[0];
			for (int p = 1; p < a.size(); p++)
			{
				double aval = a[p];
				for (int k = 0; k < p; k++) aval *= i;
				i_new += aval;

			}
			if (a.size() == 1) i_new += i; //only shifts

			int i_floor = floor(i_new);
			int i_ceil = ceil(i_new);

			//interpolate linearly
			double outval = 0.0;
			double weight = i_new-i_floor;

			if(i_floor >= 0 && i_floor < histobins1.size()) outval += (1.-weight)*histobins1[i_floor];
			if(i_ceil >= 0 && i_ceil < histobins1.size()) outval += weight*histobins1[i_ceil];

			histobins_out[i] = outval;
		}

		double r = _get_crosscorrelation(histobins0, histobins_out);

		return r;
	}
	std::vector<double> HistogramOptimization::_gaussian_convolution1D(std::vector<double> &input, float sigma)
	{
		int n = input.size();

		//create kernel
		//////////////////////////////////////////////
		int fsize = (int) (3*sigma);
		double sum = 0;

		std::vector<float> kernel(2*fsize+1, 0);
		for(uint16_t p=0; p<kernel.size();p++)
		{
			kernel[p] = exp(-((p-fsize)*(p-fsize))/(sigma*sigma*2));
			sum += kernel[p];
		}
		for(uint16_t p=0; p<kernel.size();p++) kernel[p] /= sum;
		//////////////////////////////////////////////

		std::vector<double> output(input.size(), 0.0);

		#pragma omp parallel for
		for(int x = 0; x < n; x++)
		{
			float sum = 0.0f;

			for(int xi=-fsize; xi<=fsize; xi++)
			{
				int x0 = x+xi;

				//reflective boundaries
				if (x0 < 0) x0 = -x0;
				else if (x0 >= n) x0 = 2*n-x0-2;

				sum += kernel[xi+fsize]*input[x0];
			}

			output[x] = sum;
		}

		return output;
	}
	std::vector<int> HistogramOptimization::_get_extrema(std::vector<double> &derivative)
	{
		std::vector<int> extrema;
		bool nvalid = false; bool ncrossing = false;
		bool pvalid = false; bool pcrossing = false;
		double lastval = 0.0;
		int lastcross = -1;

		for(int i = (int) (boundary_distance*derivative.size()); i < derivative.size()-((int) (boundary_distance*derivative.size())); i++)
		{
			double val = derivative[i];

			if (val <= 0.0 && lastval > 0.0) {ncrossing = true; pcrossing = false; nvalid = false; lastcross = i;}
			if (val >= 0.0 && lastval < 0.0) {pcrossing = true; ncrossing = false; pvalid = false; lastcross = i;}

			//make sure it is a pronounced crossing
			if (val < -extrema_cutoff) nvalid = true;
			if (val >  extrema_cutoff) pvalid = true;

			if (nvalid && pvalid)
			{
				nvalid = false; pvalid = false;
				extrema.push_back(lastcross);
			}

			lastval = val;
		}

		return extrema;
	}

	std::vector<double> HistogramOptimization::_linearleastsquares(std::vector<std::vector<double>> &xy_pairs, double intercept_scaler, int excluded_index)
	{
		double x = 0.0; double y = 0.0; double xy = 0.0; double xx = 0.0;
		double N = xy_pairs.size();

		for (int i = 0; i < xy_pairs.size(); i++)
		{
			if(i == excluded_index){
				N -= 1;
				continue;
			}
			double valx = xy_pairs[i][0];
			double valy = xy_pairs[i][1];

			x += valx;
			y += valy;
			xy += valx*valy;
			xx += valx*valx;
		}

		double slope = (N*xy - x*y)/(N*xx-x*x);
		double intercept = (y - slope*x)/N;

		return {intercept*intercept_scaler, slope};
	}

	int HistogramOptimization::map_pairs_of_extrema(float* normalized_frame0, float* normalized_frame1, int shape[3], std::string normalization)
	{
		mapped_extrema_pairs.clear();

		//assuming an intensity from 0 to 1 we calculate both histograms
		histo::Histogram histo;
		histo.ignore_zero = true;
		std::vector<double> histobins0, histobins1,histoedges;
		histo.calculatehistogram(normalized_frame0,shape,n_bins,0.0,1.0,histobins0,histoedges);
		histo.calculatehistogram(normalized_frame1,shape,n_bins,0.0,1.0,histobins1,histoedges);
		std::vector<double> bincenters0 = histo.binedges2bincenter(histoedges);

		histobins0 = histo.normalize(histobins0, normalization);
		histobins1 = histo.normalize(histobins1, normalization);

		double cutoff_backup = extrema_cutoff;
		if (normalization == "area") extrema_cutoff /= n_bins;

		output_vectors.push_back(bincenters0); output_columns.push_back("intensity");
		output_vectors.push_back(histobins0); output_columns.push_back("frame0");
		output_vectors.push_back(histobins1); output_columns.push_back("frame1");

		//First order derivative of histogram for identifying extrema
		////////////////////////////////////////////////////////////////////////
		std::vector<double> derivative0 = {histobins0[1]-histobins0[0]};
		std::vector<double> derivative1 = {histobins1[1]-histobins1[0]};

		for (int i = 1; i < bincenters0.size()-1; i++)
		{
			derivative0.push_back((histobins0[i+1]-histobins0[i-1])/2);
			derivative1.push_back((histobins1[i+1]-histobins1[i-1])/2);
		}
		derivative0.push_back(histobins0[histobins0.size()-1]-histobins0[histobins0.size()-2]);
		derivative1.push_back(histobins0[histobins1.size()-1]-histobins1[histobins1.size()-2]);

		derivative0 = _gaussian_convolution1D(derivative0, derivative_sigma);
		derivative1 = _gaussian_convolution1D(derivative1, derivative_sigma);

		output_vectors.push_back(derivative0); output_columns.push_back("derivative_frame0");
		output_vectors.push_back(derivative1); output_columns.push_back("derivative_frame1");
		////////////////////////////////////////////////////////////////////////

		//Grab Zero crossings and create pairs with smallest distance by brute force
		////////////////////////////////////////////////////////////////////////
		std::vector<int> extrema0 = _get_extrema(derivative0);
		std::vector<int> extrema1 = _get_extrema(derivative1);

		std::vector<std::vector<double>> extrema_pairs;
		while(extrema0.size() > 0 && extrema1.size() > 0)
		{
			double bestdist = 1e9;
			std::pair<int,int> bestpair ={0.0,0.0};

			for (int i = 0; i < extrema0.size(); i++)
			{
				double x0 = bincenters0[extrema0[i]];
				double y0 = histobins0[extrema0[i]];

				for (int p = 0; p < extrema1.size(); p++)
				{
					double x1 = bincenters0[extrema1[p]];
					double y1 = histobins1[extrema1[p]];

					double this_dist = (x0-x1)*(x0-x1)+(y0-y1)*(y0-y1);

					if (this_dist < bestdist)
					{
						bestdist = this_dist;
						bestpair.first = p;
						bestpair.second = i;
					}
				}
			}

			extrema_pairs.push_back({bincenters0[extrema1[bestpair.first]],bincenters0[extrema0[bestpair.second]], bestdist});
			extrema0.erase(extrema0.begin()+bestpair.second);
			extrema1.erase(extrema1.begin()+bestpair.first);
		}
		////////////////////////////////////////////////////////////////////////

		//sort and eliminate wrap arounds
		////////////////////////////////////////////////////////////////////////

		std::sort(extrema_pairs.begin(), extrema_pairs.end());
		for (int i = 0; i < extrema_pairs.size();i++)
		{
			double val0 = extrema_pairs[i][1];
			double val1 = extrema_pairs[i][0];

			//if there is faulty ordering erase the pair with the bigger distance
			if (i < extrema_pairs.size()-1 && val0 > extrema_pairs[i+1][1])
			{
				double dist0 = extrema_pairs[i][2];
				double dist1 = extrema_pairs[i+1][2];

				if (dist1 > dist0) extrema_pairs.erase(extrema_pairs.begin()+i+1);
				else extrema_pairs.erase(extrema_pairs.begin()+i);

				i--; continue;
			}
			if (i > 0 && val0 < extrema_pairs[i-1][1])
			{
				double dist0 = extrema_pairs[i][2];
				double dist1 = extrema_pairs[i-1][2];

				if (dist1 > dist0) extrema_pairs.erase(extrema_pairs.begin()+i-1);
				else extrema_pairs.erase(extrema_pairs.begin()+i);

				i--; continue;
			}

			//std::cout << val1 << " " << val0 << std::endl;
		}

		mapped_extrema_pairs = extrema_pairs;
		////////////////////////////////////////////////////////////////////////

		extrema_cutoff = cutoff_backup;
		return mapped_extrema_pairs.size();
	}
	std::pair<double,double> HistogramOptimization::SelectLinearRegressionSubset(float* normalized_frame0, float* normalized_frame1, int shape[3], bool rescale_zeros)
	{
		//Will try to discard extrema pairs to improve regression

		//Calc Histograms and initial correlation
		/////////////////////////////////////////////////////////////////////
		histo::Histogram histo;
		histo.ignore_zero = true;
		std::vector<double> histobins0, histobins1,histoedges;
		histo.calculatehistogram(normalized_frame0,shape,n_bins,0.0,1.0,histobins0,histoedges);
		histo.calculatehistogram(normalized_frame1,shape,n_bins,0.0,1.0,histobins1,histoedges);
		std::vector<double> bincenters0 = histo.binedges2bincenter(histoedges);

		//if(output_vectors.size() != 0) {output_vectors.clear(); output_columns.clear();}
		output_vectors.push_back(bincenters0); output_columns.push_back("intensity");
		output_vectors.push_back(histobins0); output_columns.push_back("frame0");
		output_vectors.push_back(histobins1); output_columns.push_back("frame1");

		double crosscorr0 = _get_crosscorrelation(histobins1, histobins0);
		if (mapped_extrema_pairs.size()==0) return {crosscorr0, crosscorr0};
		/////////////////////////////////////////////////////////////////////

		//Find the best subset of extrema to improve correlation for the histograms
		/////////////////////////////////////////////////////////////////////
		double bestcorr = crosscorr0;
		std::vector<double> bestfit = {0.0,1.0};

		std::vector<std::vector<double>> active_extrema= mapped_extrema_pairs;
		std::vector<double> active_result = _linearleastsquares(active_extrema,(double) n_bins);
		double crosscorr1 = _get_correlation_of_transformed_histogram(active_result, histobins1, histobins0);
		if (crosscorr1 > bestcorr){bestcorr = crosscorr1; bestfit = active_result;}

		bool improvement = true;
		while(active_extrema.size() > 2 && improvement)
		{
			improvement = false;
			for (int i = 0; i < active_extrema.size(); i++)
			{
				active_result = _linearleastsquares(active_extrema,(double) n_bins, i); //exclude i
				crosscorr1 = _get_correlation_of_transformed_histogram(active_result, histobins1, histobins0);
				if (crosscorr1 > bestcorr)
				{
					bestcorr = crosscorr1;
					bestfit = active_result;
					improvement = true;
					active_extrema.erase(active_extrema.begin()+i);
					i--;
				}
			}
		}
		if (crosscorr0 >= bestcorr) return {crosscorr0, crosscorr0}; //no improvement
		/////////////////////////////////////////////////////////////////////

		//Apply the transformation:
		/////////////////////////////////////////////////////////////////////
		long long int nslice = shape[0]*shape[1];
		long long int nstack = shape[2]*nslice;
		double intercept = bestfit[0]/((double) n_bins);
		double slope = bestfit[1];

		#pragma omp parallel for
		for (long long int idx = 0; idx < nstack; idx++)
		{
			float val = normalized_frame1[idx];
			if (val != 0.0 || rescale_zeros)
				normalized_frame1[idx] = intercept + slope*val;
		}

		histobins1.clear();
		histo.calculatehistogram(normalized_frame1,shape,n_bins,0.0,1.0,histobins1,histoedges);

		output_vectors.push_back(histobins1); output_columns.push_back("frame1_lineartransformed");

		double crosscorr2 = _get_crosscorrelation(histobins1, histobins0);
		//std::cout << crosscorr0 << " " << crosscorr1 << " " << crosscorr2 << std::endl;
		/////////////////////////////////////////////////////////////////////

		return {crosscorr0, crosscorr2};
	}
	std::pair<double,double> HistogramOptimization::RegressMappedExtremaLinearLeastSquares(float* normalized_frame0, float* normalized_frame1, int shape[3], bool rescale_zeros)
	{
		//Regression
		/////////////////////////////////////////////////////////////////////
		double x = 0.0; double y = 0.0; double xy = 0.0; double xx = 0.0;
		double N = mapped_extrema_pairs.size();

		for (int i = 0; i < mapped_extrema_pairs.size(); i++)
		{
			double valx = mapped_extrema_pairs[i][0];
			double valy = mapped_extrema_pairs[i][1];

			x += valx;
			y += valy;
			xy += valx*valy;
			xx += valx*valx;
		}

		double slope = (N*xy - x*y)/(N*xx-x*x);
		double intercept = (y - slope*x)/N;

		std::vector<double> a = {intercept*n_bins, slope};
		//std::cout << intercept << " " << slope << std::endl;
		/////////////////////////////////////////////////////////////////////

		//check for improvement
		/////////////////////////////////////////////////////////////////////
		histo::Histogram histo;
		histo.ignore_zero = true;
		std::vector<double> histobins0, histobins1,histoedges;
		histo.calculatehistogram(normalized_frame0,shape,n_bins,0.0,1.0,histobins0,histoedges);
		histo.calculatehistogram(normalized_frame1,shape,n_bins,0.0,1.0,histobins1,histoedges);
		std::vector<double> bincenters0 = histo.binedges2bincenter(histoedges);

		//if(output_vectors.size() != 0) {output_vectors.clear(); output_columns.clear();}
		output_vectors.push_back(bincenters0); output_columns.push_back("intensity");
		output_vectors.push_back(histobins0); output_columns.push_back("frame0");
		output_vectors.push_back(histobins1); output_columns.push_back("frame1");

		double crosscorr0 = _get_crosscorrelation(histobins1, histobins0);
		double crosscorr1 = _get_correlation_of_transformed_histogram(a, histobins1, histobins0);

		if (crosscorr0 >= crosscorr1){
			//std::cout << "no improvement: " << crosscorr0 << " " << crosscorr1 << std::endl;
			return {crosscorr0, crosscorr0}; //no improvement
		}
		/////////////////////////////////////////////////////////////////////

		//Apply the transformation:
		/////////////////////////////////////////////////////////////////////
		long long int nslice = shape[0]*shape[1];
		long long int nstack = shape[2]*nslice;

		#pragma omp parallel for
		for (long long int idx = 0; idx < nstack; idx++)
		{
			float val = normalized_frame1[idx];
			if (val != 0.0 || rescale_zeros)
				normalized_frame1[idx] = intercept + slope*val;
		}

		histobins1.clear();
		histo.calculatehistogram(normalized_frame1,shape,n_bins,0.0,1.0,histobins1,histoedges);

		output_vectors.push_back(histobins1); output_columns.push_back("frame1_lineartransformed");

		double crosscorr2 = _get_crosscorrelation(histobins1, histobins0);
		//std::cout << crosscorr0 << " " << crosscorr1 << " " << crosscorr2 << std::endl;
		/////////////////////////////////////////////////////////////////////

		return {crosscorr0, crosscorr2};
	}
	void HistogramOptimization::export_csv(std::string outpath)
	{
		////////////////////////////////////////////////////////////////////////
		if (output_vectors.size() > 0)
		{
			std::ofstream histofile;
			histofile.open(outpath+"/histogram_outputs.csv", std::ofstream::out);
			for(int i = 0; i < output_vectors[0].size(); i++)
			{
				if(i == 0)
				{
					histofile << output_columns[0];
					for (int p = 1; p < output_columns.size(); p++)
						histofile << "," << output_columns[p];
					histofile << "\n";
				}
				histofile << output_vectors[0][i];
				for (int p = 1; p < output_vectors.size(); p++)
					histofile << "," << output_vectors[p][i];
				histofile << "\n";
			}
			histofile.close();
		}
		////////////////////////////////////////////////////////////////////////

		if(mapped_extrema_pairs.size() > 0)
		{
			std::ofstream extremafile;
			extremafile.open(outpath+"/histogram_extrema.csv", std::ofstream::out);
			for(int i = 0; i < mapped_extrema_pairs.size(); i++)
			{
				extremafile << mapped_extrema_pairs[i][0] << "," << mapped_extrema_pairs[i][1] << "\n";
			}
			extremafile.close();
		}
	}

	void HistogramOptimization::maximize_histogram_correlation(float* normalized_frame0, float* normalized_frame1, int shape[3])
	{
		//In case we ever need more than linear, try Nelder-Mead Simplex optimization or Expand the modal fit to a polyfit.
		//This problem appears to be non-convex (map out once)

		long long int nslice = shape[0]*shape[1];
		long long int nstack = shape[2]*nslice;

		int polylevel = 2;
		float h = 0.1;
		float minomega = 0.0001;
		float precision = 1e-5;
		std::vector<double> guess = {0.0, 1.0, 0.0, 0.0, 0.0};

		//assuming an intensity from 0 to 1 we calculate both histograms
		histo::Histogram histo;
		histo.ignore_zero = true;
		std::vector<double> histobins0, histobins1,histoedges;
		histo.calculatehistogram(normalized_frame0,shape,n_bins,0.0,1.0,histobins0,histoedges);
		histo.calculatehistogram(normalized_frame1,shape,n_bins,0.0,1.0,histobins1,histoedges);
		std::vector<double> bincenters0 = histo.binedges2bincenter(histoedges);

		//histobins0 = histo.normalize(histobins0, "height");
		//histobins1 = histo.normalize(histobins1, "height");

		//rough estimate of the gradient should suffice for our purpose

		std::vector<double> a_vals = {guess[0]};
		if (polylevel > 0) a_vals.push_back(guess[1]);
		for (int i = 1; i < polylevel; i++) a_vals.push_back(0.0);
		std::vector<double> r_vals(2*a_vals.size(), 0.0);
		std::vector<double> gradient(a_vals.size(), 0.0);

		double crosscorr = _get_correlation_of_transformed_histogram(a_vals, histobins1, histobins0);
		//crosscorr = _get_crosscorrelation(histobins0, histobins1);

		for (int iter = 0; iter < 10000; iter++)
		{
			double lastcorr = crosscorr;
			double gradlength = 0.0;

			for (int dim = 0; dim < a_vals.size(); dim++)
			{
				if(polylevel == 0 && dim == 1) continue;
				a_vals[dim] += h;
				r_vals[2*dim] = _get_correlation_of_transformed_histogram(a_vals, histobins1, histobins0);
				a_vals[dim] -= 2*h;
				r_vals[2*dim+1] = _get_correlation_of_transformed_histogram(a_vals, histobins1, histobins0);
				a_vals[dim] += h;
				gradient[dim] = (r_vals[2*dim]-r_vals[2*dim+1])/(2.*h);
				//gradlength += gradient[dim]*gradient[dim];
			}
			//gradlength = sqrt(gradlength);
			//for (int dim = 0; dim < a_vals.size(); dim++)
			//	gradient[dim] /= gradlength;


			//line search
			double stepsize = 4.0;
			while (stepsize >= minomega)
			{
				stepsize *= 0.5;
				bool improving = true;

				while(improving)
				{
					for (int dim = 0; dim < a_vals.size(); dim++) a_vals[dim] += stepsize*gradient[dim];

					double next_corr;
					next_corr = _get_correlation_of_transformed_histogram(a_vals, histobins1, histobins0);

					if (next_corr > crosscorr) crosscorr = next_corr;
					else
					{
						improving = false;
						for (int dim = 0; dim < a_vals.size(); dim++) a_vals[dim] -= stepsize*gradient[dim];
					}

					std::cout << "r: " << crosscorr << "      \r" ;
					std::cout.flush();
				}
			}

			if(crosscorr-lastcorr < precision)
				break;
		}
		std::cout << std::endl;
		for (int i = 0; i < a_vals.size(); i++) std::cout << "a" << i << ": " << a_vals[i] << " ";
		std::cout << "r: " << crosscorr << std::endl;

		//Apply the transformation:
		#pragma omp parallel for
		for (long long int idx = 0; idx < nstack; idx++)
		{
			float val = normalized_frame1[idx]*bincenters0.size(); //value as bin
			float outval = a_vals[0];
			for (int p = 1; p < a_vals.size(); p++)
			{
				double aval = a_vals[p];
				for (int k = 0; k < p; k++) aval *= val;
				outval += aval;
			}
			if(polylevel == 0) outval += val;

			if(normalized_frame1[idx] != 0.0)
				normalized_frame1[idx] = outval/bincenters0.size();
		}

		histobins1.clear();
		histo.calculatehistogram(normalized_frame1,shape,n_bins,0.0,1.0,histobins1,histoedges);
		//histobins1 = histo.normalize(histobins1, "height");
		output_vectors.push_back(histobins1); output_columns.push_back("frame1_gradascent");

		return;
	}
}
