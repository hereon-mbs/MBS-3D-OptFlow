#ifndef HISTOGRAM_H
#define HISTOGRAM_H

#include <vector>
#include <string.h>
#include <iostream>
#include <cstdint>

namespace histo
{
    typedef float dtype;

    class Histogram
    {
    public:
        double histocutoff = 0.001;

        std::pair<dtype,dtype> get_effectivehistogrambounds(dtype *image, int shape[3], int n_bins);
        std::pair<dtype,dtype> get_effectivehistogrambounds_2frame(dtype *frame0, dtype *frame1, int shape[3], int n_bins);

        void calculateeffectivehistogram_masked(dtype *data, float *mask, int shape[3], int n_bins, std::vector<double> &out_histobins, std::vector<double> &out_histoedges);
        void calculateeffectivehistogram(dtype *data, int shape[3], int n_bins, std::vector<uint64_t> &out_histobins, std::vector<double> &out_histoedges);
        void calculateeffectivehistogram(dtype *data, int shape[3], int n_bins, std::vector<double> &out_histobins, std::vector<double> &out_histoedges);
        void calculatehistogram(dtype *data, int shape[3], int n_bins, dtype lowerbound, dtype upperbound, std::vector<uint64_t> &out_histobins, std::vector<double> &out_histoedges);
        void calculatehistogram(dtype *data, int shape[3], int n_bins, dtype lowerbound, dtype upperbound, std::vector<double> &out_histobins, std::vector<double> &out_histoedges);
        void calculatehistogram(dtype *data, float *mask, int shape[3], int n_bins, dtype lowerbound, dtype upperbound, std::vector<double> &out_histobins, std::vector<double> &out_histoedges);
        std::vector<double> binedges2bincenter(std::vector<double> &binedges);
        std::vector<double> normalize(std::vector<uint64_t> histobins, std::string normalization);
        std::vector<double> normalize(std::vector<double> histobins, std::string normalization);

        float* calculate2DHistogram(float *frame0, float *frame1,  int shape[3], std::vector<double> &histoedges0, std::vector<double> &histoedges1, int outshape[2]);
        float* calculate2DHistogram(float *frame0, float *frame1,  float *mask, int shape[3], std::vector<double> &histoedges0, std::vector<double> &histoedges1, int outshape[2]);

    private:
        std::vector<double> linspace(double startval, double endval, uint64_t bins);
    };
}

#endif // HISTOGRAM_H
