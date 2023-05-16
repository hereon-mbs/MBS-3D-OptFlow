#ifndef AUXILIARY_H
#define AUXILIARY_H

#include <iostream>
#include <string.h>
#include <cstdint>
#include <vector>
#include "../protocol_parameters.h"

namespace aux
{
    /*String-Manipulation
    *********************************************************/
    std::string zfill_int2string(int inint, const unsigned int &zfill);
    std::string get_active_directory();

    /*Numpy-like
    *********************************************************/
    std::vector<double> linspace(double startval, double endval, uint64_t bins);

    float* backup_imagestack(float* image, int shape[3]);

    /*Histogram based normalization
    *********************************************************/
    void zeroshift_minimum(float *image0, float *image1, int shape[3]);

    void transform_values(std::string transformation, float*image, int shape[3]);

    void normalize2frames_simple(float *image0, float *image1, int shape[3], optflow::ProtocolParameters *params);
    void normalize2frames_histogram(float *image0, float *image1, int shape[3], optflow::ProtocolParameters *params, int n_bins = 1000, double cutoff = 0.001,
    		bool ignore_zero = false, bool extrapolate = false, bool rescale_zero = true);
    void normalize2frames_histogram_independent(float *image0, float *image1, int shape[3], optflow::ProtocolParameters *params, int n_bins = 1000, double cutoff = 0.001,
    		bool ignore_zero = false, bool extrapolate = false, bool rescale_zero = true);
    void normalize2frames_histogram_mask(float *image0, float *image1, float *mask, int shape[3], optflow::ProtocolParameters *params, int n_bins, double cutoff,
    		bool extrapolate = false, bool rescale_zero = true);
    void normalize1frame_histogramequalized(float *image, int shape[3], optflow::ProtocolParameters *params, int n_bins, int n_bins_out, double cutoff, bool ignore_zero);
    void normalize2frames_histogramequalized(float *image0, float *image1, int shape[3], optflow::ProtocolParameters *params, int n_bins = 1000, int n_bins_out = 65536, double cutoff = 0.001, bool ignore_zero = false);
    void normalize2frames_histogramequalized_mask(float *image0, float *image1, float* mask, int shape[3], optflow::ProtocolParameters *params, int n_bins = 1000, int n_bins_out = 65536, double cutoff = 0.001);
    void normalizeframe_histogram(float *image0, int shape[3]);
    /*Helper functions for testing purposes*/
    void set_initialguess(float* flowvector, int active_shape[3], int full_shape[3], float constant_guess[3], std::string previous_result_path);
    void add_initial_ycompression(float* flowvector, int active_shape[3], float compression);

    float* project_average_through_mask(float* img, float* mask, int shape[3], int dim, bool absolute_values, std::string tag = "none");

    std::vector<float> _calceigenvector_3x3symmetric(float Exx, float Eyy, float Ezz, float Exy, float Exz, float Eyz, float eigenval);
    std::vector<std::vector<float>> eigenvalues_and_eigenvectors_3x3symmetric(float Exx, float Eyy, float Ezz, float Exy, float Exz, float Eyz);
}

#endif // AUXILIARY_H
