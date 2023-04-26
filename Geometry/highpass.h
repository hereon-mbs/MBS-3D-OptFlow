#ifndef HIGHPASSFILTER_H
#define HIGHPASSFILTER_H

#include <vector>
#include <stdexcept>
#include <cstdint>
#include <fftw3.h>
#include <omp.h>
#include "auxiliary.h"

/* Currently now parrallel dft compiled! */

namespace fourier_filtering
{
    void dft_highpass(float *imagestack, float *output, int shape[3], double sigma, int planning_rigor)
    {
        double pi = 3.14159265358979323846;
        double term1 = 1./sqrt(2.*sigma*sigma*pi);

        int nx = shape[0], ny = shape[1];
        uint64_t n_slice = shape[0]*shape[1];
        double *mask = (double*) malloc(sizeof(double) * nx * ((ny/2)+1));

        //calculate the relative frequencies for each axis
        std::vector<double> y_freq = aux::linspace(0., 1., ((ny/2)+1));
        std::vector<double> x_freq = aux::linspace(0., 1., ceil(nx/2.));
        std::vector<double> x_freq_inv = aux::linspace(1., 0., ceil(nx/2.));
        x_freq.insert(x_freq.end(), x_freq_inv.begin(), x_freq_inv.end());
        if(x_freq.size() > (uint64_t) nx) x_freq.erase(x_freq.begin() + nx);

        //Provide a mask for transformation
        uint64_t idx_mask = 0;
        for (uint64_t x = 0; x < (uint64_t) nx; x++)
        {
            for(uint64_t y = 0; y < (uint64_t) ((ny/2)+1); y++)
            {
                mask[idx_mask] = sqrt(x_freq[x]*x_freq[x]+y_freq[y]*y_freq[y]);  //calculate radial spatial frequency
                mask[idx_mask] = 1-(exp(-(mask[idx_mask]*mask[idx_mask])/(2*sigma*sigma)));
                idx_mask++;
            }
        }

        //maintain brightness:
        mask[0] = 1.;
        mask[(uint64_t) (nx-1) * ((ny/2)+1)] = 1.;

        //Set up plans
        double *in1, *result;
        fftw_complex *out1;

        //Allocate
        in1  = (double*) malloc(sizeof(double) * nx * ny);
        out1 = (fftw_complex*) fftw_malloc(sizeof(fftw_complex) * nx * ((ny/2)+1));
        result = (double*) malloc(sizeof(double) * nx * ny);

        //Note: plans can not run in parallel threads. Would need to prepare a plan for every thread beforehand
        std::cout << "planning fft..." << std::endl;
        fftw_plan plan1 = fftw_plan_dft_r2c_2d(nx, ny, in1, out1, planning_rigor);         //forward plan
        fftw_plan plan3 = fftw_plan_dft_c2r_2d(nx, ny, out1, result, planning_rigor);      //backward plan

        for (uint64_t z = 0; z < (uint64_t) shape[2]; z++)
        {
            std::cout << "slice " << (z+1) << "/" << shape[2] << "\r";

            //fill in grayvalues
            //need to convert to row major format!!!!
            uint64_t inputidx = 0;
            for (uint64_t y = 0; y < (uint64_t) shape[1]; y++)
            {
                for (uint64_t x = 0; x < (uint64_t) shape[0]; x++)
                {
                    uint64_t idx = x*shape[1]+y;
                    in1[idx] = imagestack[inputidx+(z*n_slice)];
                    inputidx++;
                }
            }

            //Fourier transform
            fftw_execute(plan1);

            //Merge the DFTs
            double paganin_weight = 1.;
            for(uint64_t idx = 0; idx < (uint64_t) (nx * ((ny/2)+1)); idx++)
            {
                out1[idx][0] *= mask[idx];
                out1[idx][1] *= mask[idx];
            }

            fftw_execute(plan3);

            //Write output by converting back to column major format
            uint64_t idx_ifft = 0;
            for (uint64_t x = 0; x < (uint64_t) nx; x++)
            {
                for(uint64_t y = 0; y < (uint64_t) ny; y++)
                {
                    uint64_t idx_output = y*nx + x + (z*n_slice);
                    output[idx_output] = result[idx_ifft]/n_slice; //Output scales with number of elements
                    idx_ifft++;
                }
            }
        }
        std::cout << std::endl;

        free(mask);
        fftw_free(out1);
        free(in1);
        free(result);
        fftw_destroy_plan(plan1);
        fftw_destroy_plan(plan3);

        return;
    }
}

#endif // HIGHPASSFILTER_H
