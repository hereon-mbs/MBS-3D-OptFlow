#ifndef HISTOMATCHING_H
#define HISTOMATCHING_H

#include <vector>
#include <string.h>
#include <iostream>
#include <cstdint>

namespace histomatch
{
	class HistogramOptimization
	{
	public:
		int n_bins = 1000;
		float derivative_sigma = 2.0;
		double extrema_cutoff = 0.001; //make sure local extrema are pronounced
		double boundary_distance = 0.1; //more or less obsolet with SelectLinearRegressionSubset, limits evaluated dynamic range

		int map_pairs_of_extrema(float* normalized_frame0, float* normalized_frame1, int shape[3], std::string normalization);

		std::pair<double,double> SelectLinearRegressionSubset(float* normalized_frame0, float* normalized_frame1, int shape[3], bool rescale_zeros);
		std::pair<double,double> RegressMappedExtremaLinearLeastSquares(float* normalized_frame0, float* normalized_frame1, int shape[3], bool rescale_zeros);

		void export_csv(std::string outpath);

		//outdated:
		void maximize_histogram_correlation(float* normalized_frame0, float* normalized_frame1, int shape[3]);

	private:
		std::vector<std::vector<double>> mapped_extrema_pairs;
		std::vector<std::vector<double>> output_vectors;
		std::vector<std::string> output_columns;

		double _get_crosscorrelation(std::vector<double> &histobins0, std::vector<double> &histobins1);
		double _get_correlation_of_transformed_histogram(std::vector<double> a, std::vector<double> &histobins0, std::vector<double> &histobins1);
		std::vector<double> _gaussian_convolution1D(std::vector<double> &input, float sigma);
		std::vector<int> _get_extrema(std::vector<double> &derivative);

		std::vector<double> _linearleastsquares(std::vector<std::vector<double>> &xy_pairs, double intercept_scaler = 1., int excluded_index = -1);

	};
}

#endif //HISTOMATCHING_H
