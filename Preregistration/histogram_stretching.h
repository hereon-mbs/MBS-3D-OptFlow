#ifndef HISTOGRAM_STRETCHING_H
#define HISTOGRAM_STRETCHING_H

#include <iostream>
#include <string.h>
#include <cstdint>
#include <vector>
#include "Geometry/histogram.h"
#include "Geometry/auxiliary.h"
#include "Geometry/hdcommunication.h"

namespace int_transform
{
	//extract the two modes of the histogram with a GMM and stretch the intensities of frame 1

	int n_bins = 512;
	int radius = 20;

	std::vector<double> get_histogram_maxima_(int radius, std::vector<double> &histobins, std::vector<double> &histocenters)
	{
		std::vector<double> maxima;

		double active_maximum = histobins[0];
		int active_pos = 0;
		int nsteps_active = 0;

		for (int i = 1; i < histobins.size(); i++)
		{
			if (histobins[i] > active_maximum)
			{
				if(nsteps_active >= radius)
					maxima.push_back(histocenters[active_pos]);

				active_maximum = histobins[i];
				active_pos = i;
				nsteps_active = 0;
			}
			else
				nsteps_active++;
		}
		if(nsteps_active >= radius)
			maxima.push_back(histocenters[active_pos]);

		return maxima;
	}

	std::vector<double> stretch_frame1(float *frame0, float *frame1, int shape[3], std::string outpath)
	{
		long long int nslice = shape[0]*shape[1];
		long long int nstack = shape[2]*nslice;

		/////////////////////////////////////////////////////////////////////
		float *histomask = (float*) calloc(nstack, sizeof(*histomask));
		#pragma omp parallel for
		for (long long int idx = 0; idx < nstack; idx++)
		{
			if (frame1[idx] != 0.0f) histomask[idx] = 1.0f;
			else continue;
		}

		histo::Histogram histo;

		std::vector<double> histobins0, histobins1, histobins2;
		std::vector<double> histoedges0, histoedges1, histoedges2;
		histo.calculateeffectivehistogram_masked(frame0, histomask, shape, n_bins, histobins0, histoedges0);
		histo.calculateeffectivehistogram_masked(frame1, histomask, shape, n_bins, histobins1, histoedges1);
		std::vector<double> histocenters0 = histo.binedges2bincenter(histoedges0);
		std::vector<double> histocenters1 = histo.binedges2bincenter(histoedges1);
		/////////////////////////////////////////////////////////////////////

		/////////////////////////////////////////////////////////////////////
		std::vector<double> maxima_frame0 = get_histogram_maxima_(radius, histobins0, histocenters0);
		std::vector<double> maxima_frame1 = get_histogram_maxima_(radius, histobins1, histocenters1);

		std::cout << "found " << maxima_frame0.size() << " maxima in the histogram of frame0:";
		for (int i = 0; i < maxima_frame0.size(); i++)
			std::cout << " " << maxima_frame0[i];
		std::cout << std::endl;

		std::cout << "found " << maxima_frame1.size() << " maxima in the histogram of frame1:";
		for (int i = 0; i < maxima_frame1.size(); i++)
			std::cout << " " << maxima_frame1[i];
		std::cout << std::endl;
		/////////////////////////////////////////////////////////////////////

		/////////////////////////////////////////////////////////////////////
		if (maxima_frame0.size() == maxima_frame1.size() && maxima_frame0.size() > 0)
		{
			float range1 = maxima_frame1[1]-maxima_frame1[0];
			float range0 = maxima_frame0[1]-maxima_frame0[0];

			#pragma omp parallel for
			for (long long int idx = 0; idx < nstack; idx++)
			{
				if(histomask[idx] != 0)
				{
					//stretch frame1
					frame1[idx] = ((frame1[idx]-maxima_frame1[0])/range1)*range0 + maxima_frame0[0];
				}
			}
		}
		/////////////////////////////////////////////////////////////////////

		histo.calculateeffectivehistogram_masked(frame1, histomask, shape, n_bins, histobins2, histoedges2);
		std::vector<double> histocenters2 = histo.binedges2bincenter(histoedges2);

		histobins0 = histo.normalize(histobins0, "height");
		histobins1 = histo.normalize(histobins1, "height");
		histobins2 = histo.normalize(histobins2, "height");

		std::vector<std::vector<double>> outdata = {histocenters0, histobins0, histocenters1, histobins1, histocenters2, histobins2};
		hdcom::HdCommunication hdcom;
		hdcom.SaveColumnData(outdata, outpath, "histogram_comparison");

		free(histomask);
		std::vector<double> stretch_parameters = {maxima_frame0[0], maxima_frame0[1], maxima_frame1[0], maxima_frame1[1]};
		return stretch_parameters;
	}
	void apply_stretch_parameters(float *image, int shape[3], std::vector<double> &stretch_parameters)
	{
		long long int nslice = shape[0]*shape[1];
		long long int nstack = shape[2]*nslice;

		double lower0 = stretch_parameters[0];
		double upper0 = stretch_parameters[1];
		double lower1 = stretch_parameters[2];
		double upper1 = stretch_parameters[3];

		double range1 = upper1-lower1;
		double range0 = upper1-lower1;

		#pragma omp parallel for
		for (long long int idx = 0; idx < nstack; idx++)
			image[idx] = ((image[idx]-lower1)/range1)*range0 + lower0;

		return;
	}
}

#endif //HISTOGRAM_STRETCHING_H
