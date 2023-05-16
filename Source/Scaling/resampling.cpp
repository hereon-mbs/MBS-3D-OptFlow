#include <iostream>
#include <vector>
#include <algorithm>
#include <omp.h>
#include <math.h>

namespace resample
{
    //Pointwise functions:
    ////////////////////////////////////////////////////////////////////////
    float interpolate_cubic(float y0, float y1, float y2, float y3, float mu)
    {
        float mu2 = mu*mu;

        float a0 = y3-y2-y0+y1;
        float a1 = y0-y1-a0;
        float a2 = y2-y0;
        float a3 = y1;

        return a0*mu*mu2+a1*mu2+a2*mu+a3;
    }
    ////////////////////////////////////////////////////////////////////////

    //Resampling with Coons-patches (recommended)
    ////////////////////////////////////////////////////////////////////////
    void linear_coons(float* input, int inshape[3], float* output, int outshape[3], int vector_dims, bool scale_intensity)
    {
        float xratio = ((float) outshape[0])/((float) inshape[0]);
        float yratio = ((float) outshape[1])/((float) inshape[1]);
        float zratio = ((float) outshape[2])/((float) inshape[2]);

        float intensity_multiplier[3] = {1.f,1.f,1.f};
		if (scale_intensity)
		{
			intensity_multiplier[0] = xratio;
			intensity_multiplier[1] = yratio;
			intensity_multiplier[2] = zratio;
		}

        //size of input
        int nx0 = inshape[0];
        long long int nslice0 = inshape[0]*inshape[1];
        long long int nstack0 = inshape[2]*nslice0;

        //size of output
        int nx1 = outshape[0];
        long long int nslice1 = outshape[0]*outshape[1];
        long long int nstack1 = outshape[2]*nslice1;

        float dx = 1./xratio;
        float dy = 1./yratio;
        float dz = 1./zratio;

        //Resample in z-direction when necessary
        if (outshape[2] != inshape[2])
        {
            long long int nstack2 = outshape[2]*nslice0;
            float* tmpout = (float*) malloc((vector_dims*nstack2)*sizeof(*tmpout));

			for (int dim = 0; dim < vector_dims; dim++)
			{
				#pragma omp parallel for
				for (long long int pos = 0; pos < nstack2; pos++)
				{
					int z = pos/nslice0;
					int y = (pos-z*nslice0)/nx0;
					int x = (pos-z*nslice0-y*nx0);

					float z0 = z*dz;
					int zf = floor(z0);
					int zc = std::min((int) ceil(z0), inshape[2]-1);

					float w = z0-zf;

					float P000 = input[dim*nstack0+zf*nslice0+y*nx0 + x];
					float P001 = input[dim*nstack0+zc*nslice0+y*nx0 + x];

					tmpout[dim*nstack2+z*nslice0+y*nx0+x] = (P001-P000)*w+P000;
				}
			}

			input = tmpout;
            nstack0 = nstack2;
        }

        //Resample in xy-direction
        for (int dim = 0; dim < vector_dims; dim++)
        {
			#pragma omp parallel for
			for (long long int pos = 0; pos < nstack1; pos++)
			{
				int z = pos/nslice1;
				int y = (pos-z*nslice1)/nx1;
				int x = (pos-z*nslice1-y*nx1);

				float x0 = x*dx;
				int xf = floor(x0);
				int xc = std::min((int) ceil(x0), nx0-1);

				float y0 = y*dy;
				int yf = floor(y0);
				int yc = std::min((int) ceil(y0), inshape[1]-1);

				float u = x0-xf;
				float v = y0-yf;

				float P00 = input[dim*nstack0 + z*nslice0+yf*nx0 + xf];
				float P10 = input[dim*nstack0 + z*nslice0+yf*nx0 + xc];
				float P01 = input[dim*nstack0 + z*nslice0+yc*nx0 + xf];
				float P11 = input[dim*nstack0 + z*nslice0+yc*nx0 + xc];

				float glv = (P01-P00)*v+P00; //left
				float grv = (P11-P10)*v+P10; //right
				float gtu = (P10-P00)*u+P00; //top
				float gbu = (P11-P01)*u+P01; //bottom

				float sigma_lr = (1.-u)*glv + u*grv;
				float sigma_bt = (1.-v)*gtu + v*gbu;
				float corr_lrbt = P00*(1.-v)*(1.-u) + P01*v*(1.-u) + P10*(1.-v)*u + P11*u*v;

				output[dim*nstack1 + z*nslice1+y*nx1+x] = (sigma_lr+sigma_bt-corr_lrbt)*intensity_multiplier[dim];
			}
        }
        return;
    }
    void cubic_coons(float* input, int inshape[3], float* output, int outshape[3], int vector_dims, bool scale_intensity)
    {
        float xratio = ((float) outshape[0])/((float) inshape[0]);
        float yratio = ((float) outshape[1])/((float) inshape[1]);
        float zratio = ((float) outshape[2])/((float) inshape[2]);

        float intensity_multiplier[3] = {1.f,1.f,1.f};
        if (scale_intensity)
        {
        	intensity_multiplier[0] = xratio;
        	intensity_multiplier[1] = yratio;
        	intensity_multiplier[2] = zratio;
        }

        //size of input
        int nx0 = inshape[0];
        long long int nslice0 = inshape[0]*inshape[1];
        long long int nstack0 = inshape[2]*nslice0;

        //size of output
        int nx1 = outshape[0];
        long long int nslice1 = outshape[0]*outshape[1];
        long long int nstack1 = outshape[2]*nslice1;

        float dx = 1./xratio;
        float dy = 1./yratio;
        float dz = 1./zratio;

        float* tmpout;

        //Resample in z-direction when necessary
        if (outshape[2] != inshape[2])
        {
            long long int nstack2 = outshape[2]*nslice0;
            tmpout = (float*) malloc((vector_dims*nstack2)*sizeof(*tmpout));

            for (int dim = 0; dim < vector_dims; dim++)
            {
				#pragma omp parallel for
				for (long long int pos = 0; pos < nstack2; pos++)
				{
					int z = pos/nslice0;
					int y = (pos-z*nslice0)/nx0;
					int x = (pos-z*nslice0-y*nx0);

					float z0 = z*dz;
					int zf = floor(z0);
					int zc = std::min((int) ceil(z0), inshape[2]-1);

					//extrapolate with zero-gradient
					int zf2 = std::max(0, zf-1);
					int zc2 = std::min(zc+1, inshape[2]-1);

					float w = z0-zf;

					float P000 = input[dim*nstack0 + zf2*nslice0+y*nx0 + x];
					float P001 = input[dim*nstack0 + zf *nslice0+y*nx0 + x];
					float P002 = input[dim*nstack0 + zc *nslice0+y*nx0 + x];
					float P003 = input[dim*nstack0 + zc2*nslice0+y*nx0 + x];

					tmpout[dim*nstack2 + z*nslice0+y*nx0+x] = interpolate_cubic(P000,P001,P002,P003,w);
				}
            }

           input = tmpout;
           nstack0 = nstack2;
        }

        //Resample in xy-direction
        for (int dim = 0; dim < vector_dims; dim++)
        {
			#pragma omp parallel for
			for (long long int pos = 0; pos < nstack1; pos++)
			{
				int z = pos/nslice1;
				int y = (pos-z*nslice1)/nx1;
				int x = (pos-z*nslice1-y*nx1);

				float x0 = x*dx;
				int xf = floor(x0);
				int xc = std::min((int) ceil(x0), nx0-1);

				//extrapolate with zero-gradient
				int xf2 = std::max(0, xf-1);
				int xc2 = std::min(xc+1, nx0-1);

				float y0 = y*dy;
				int yf = floor(y0);
				int yc = std::min((int) ceil(y0), inshape[1]-1);

				//extrapolate with zero-gradient
				int yf2 = std::max(0, yf-1);
				int yc2 = std::min(yc+1, inshape[1]-1);

				float v = y0-yf;
				float u = x0-xf;

				float P10 = input[dim*nstack0 + z*nslice0+yf2*nx0 + xf];
				float P20 = input[dim*nstack0 + z*nslice0+yf2*nx0 + xc];

				float P01 = input[dim*nstack0 + z*nslice0+yf*nx0 + xf2];
				float P11 = input[dim*nstack0 + z*nslice0+yf*nx0 + xf];
				float P21 = input[dim*nstack0 + z*nslice0+yf*nx0 + xc];
				float P31 = input[dim*nstack0 + z*nslice0+yf*nx0 + xc2];

				float P02 = input[dim*nstack0 + z*nslice0+yc*nx0 + xf2];
				float P12 = input[dim*nstack0 + z*nslice0+yc*nx0 + xf];
				float P22 = input[dim*nstack0 + z*nslice0+yc*nx0 + xc];
				float P32 = input[dim*nstack0 + z*nslice0+yc*nx0 + xc2];

				float P13 = input[dim*nstack0 + z*nslice0+yc2*nx0 + xf];
				float P23 = input[dim*nstack0 + z*nslice0+yc2*nx0 + xc];

				float gtu = interpolate_cubic(P01,P11,P21,P31,u);
				float gbu = interpolate_cubic(P02,P12,P22,P32,u);

				float glv = interpolate_cubic(P10,P11,P12,P13,v);
				float grv = interpolate_cubic(P20,P21,P22,P23,v);

				float sigma_lr = (1.-u)*glv + u*grv;
				float sigma_bt = (1.-v)*gtu + v*gbu;
				float corr_lrbt = P11*(1.-v)*(1.-u) + P12*v*(1.-u) + P21*(1.-v)*u + P22*u*v;

				output[dim*nstack1 + z*nslice1+y*nx1+x] = (sigma_lr+sigma_bt-corr_lrbt)*intensity_multiplier[dim];
			}
        }

        if (outshape[2] != inshape[2]) free(tmpout);
        return;
    }
    ////////////////////////////////////////////////////////////////////////

    //Resampling by majority
	////////////////////////////////////////////////////////////////////////
	float* downsample_majority_bin2(float* input, int inshape[3], bool keep_bigger_value = false)
	{
		int outshape[3] = {std::max(1,(int) roundf(inshape[0]/2.)), std::max(1,(int) roundf(inshape[1]/2.)), std::max(1,(int) roundf(inshape[2]/2.))};
		int nx = outshape[0]; int ny = outshape[1]; int nz = outshape[2];
		long long int nslice = nx*ny;
		long long int nstack = nslice*nz;
		float* output = (float*) calloc(nstack, sizeof(*output));

		int nx0 = inshape[0];
		long long int nslice0 = inshape[0]*inshape[1];

		//#pragma omp parallel for
		for (long long int idx = 0; idx < nstack; idx++)
		{
			int z = idx/nslice;
			int y = (idx-z*nslice)/nx;
			int x = idx-z*nslice-y*nx;

			int x0 = 2*x; int y0 = 2*y; int z0 = 2*z;
			int z1 = std::min(inshape[2]-1, z0+1);
			int y1 = std::min(inshape[1]-1, y0+1);
			int x1 = std::min(inshape[0]-1, x0+1);

			int occurences[256] = {0};

			occurences[(int) input[z0*nslice0+y0*nx0+x0]]++;
			occurences[(int) input[z0*nslice0+y0*nx0+x1]]++;
			occurences[(int) input[z0*nslice0+y1*nx0+x0]]++;
			occurences[(int) input[z0*nslice0+y1*nx0+x1]]++;
			occurences[(int) input[z1*nslice0+y0*nx0+x0]]++;
			occurences[(int) input[z1*nslice0+y0*nx0+x1]]++;
			occurences[(int) input[z1*nslice0+y1*nx0+x0]]++;
			occurences[(int) input[z1*nslice0+y1*nx0+x1]]++;

			int sum = occurences[0];
			int maxvote = sum;
			uint8_t pos = 0;
			uint8_t outval = 0;

			if (keep_bigger_value)
			{
				while(8-sum >= maxvote)
				{
					pos++;
					int count = occurences[pos];
					sum += count;
					if (count >= maxvote){maxvote = count; outval = pos;}
				}
			}
			else
			{
				while(8-sum > maxvote)
				{
					pos++;
					int count = occurences[pos];
					sum += count;
					if (count > maxvote){maxvote = count; outval = pos;}
				}
			}

			output[idx] = outval;
		}

		return output;
	}
	////////////////////////////////////////////////////////////////////////

    //Resampling with interpolation
    ////////////////////////////////////////////////////////////////////////
    void downsample_linear(float* input, int inshape[3], float* output, int outshape[3], bool average)
    {
        //with average = false only boundary voxels are interpolated,
        //else all voxels in the interpolation volume contribute

        double base_weight = 0.0;
        if (average) base_weight = 1.0;

        double xratio = ((double) outshape[0])/((double) inshape[0]);
        double yratio = ((double) outshape[1])/((double) inshape[1]);
        double zratio = ((double) outshape[2])/((double) inshape[2]);

        int nx0 = inshape[0];
        long long int nslice0 = inshape[0]*inshape[1];
        int nx1 = outshape[0];
        long long int nslice1 = outshape[0]*outshape[1];
        long long int nstack1 = std::max(outshape[2],1)*nslice1;

        #pragma omp parallel for
        for (long long int pos = 0; pos < nstack1; pos++)
        {
            int z = pos/nslice1;
            int y = (pos-z*nslice1)/nx1;
            int x = (pos-z*nslice1-y*nx1);

            float z0 = z/zratio;
            float z1 = (z+1)/zratio;
            int zf0 = floor(z0);
            int zf1 = floor(z1);

            float y0 = y/yratio;
            float y1 = (y+1)/yratio;
            int yf0 = floor(y0);
            int yf1 = floor(y1);

            float x0 = x/xratio;
            float x1 = (x+1)/xratio;
            int xf0 = floor(x0);
            int xf1 = floor(x1);

            double value = 0.0;
            double weightsum = 0.0;

            for (int yi = yf0; yi < y1; yi++)
            {
                double wy = base_weight;

                if(yi == yf0) wy = 1.-(y0-yf0);
                else if(yi+1 > x1) wy = (y1-yf1);

                for (int xi = xf0; xi < x1; xi++)
                {
                    double wx = base_weight;

                    if(xi == xf0) wx = 1.-(x0-xf0);
                    else if(xi+1 > x1) wx = (x1-xf1);

                    if(inshape[2] > 1)
                    {
                        for (int zi = zf0; zi < z1; zi++)
                        {
                            double wz = base_weight;

                            if(zi == zf0) wz = 1.-(z0-zf0);
                            else if(zi+1 > z1) wz = (z1-zf1);

                            value += input[zi*nslice0+yi*nx0 + xi]*wx*wy*wz;
                            weightsum += wx*wy*wz;
                        }
                    }
                    else
                    {
                        value += input[z*nslice0+yi*nx0 + xi]*wx*wy;
                        weightsum += wx*wy;
                    }
                }
            }

            output[z*nslice1+y*nx1+x] = value/weightsum;
        }

        return;
    }
    void upsample_linear(float* input, int inshape[3], float* output, int outshape[3])
    {
        double xratio = ((double) outshape[0])/((double) inshape[0]);
        double yratio = ((double) outshape[1])/((double) inshape[1]);
        double zratio = ((double) outshape[2])/((double) inshape[2]);

        int nx0 = inshape[0];
        long long int nslice0 = inshape[0]*inshape[1];
        int nx1 = outshape[0];
        long long int nslice1 = outshape[0]*outshape[1];

        for (int z = 0; z < std::max(outshape[2],1); z++)
        {
            float z0 = z/zratio;
            int zf = floor(z0);
            int zc = ceil(z0);

            double wzf = zc-z0;
            double wzc = 1.-wzf;

            zc = std::min(zc, inshape[2]-1);

            #pragma omp parallel for
            for (int y = 0; y < outshape[1]; y++)
            {
                float y0 = y/yratio;
                int yf = floor(y0);
                int yc = ceil(y0);

                double wyf = yc-y0;
                double wyc = 1.-wyf;

                yc = std::min(yc, inshape[1]-1);

                for (int x = 0; x < outshape[0]; x++)
                {
                    float x0 = x/xratio;
                    int xf = floor(x0);
                    int xc = ceil(x0);

                    double wxf = xc-x0;
                    double wxc = 1.-wxf;

                    xc = std::min(xc, nx0-1);

                    if(outshape[2] == 1)
                    {
                        output[y*nx1+x] = wxf*wyf*input[yf*nx0 + xf] + wxc*wyf*input[yf*nx0 + xc] + wxf*wyc*input[yf*nx0 + xf] + wxc*wyc*input[yc*nx0 + xc];
                    }
                    else
                    {
                         double value = wzf*wxf*wyf*input[zf*nslice0+yf*nx0 + xf] + wzf*wxc*wyf*input[zf*nslice0+yf*nx0 + xc] + wzf*wxf*wyc*input[zf*nslice0+yf*nx0 + xf] + wzf*wxc*wyc*input[zf*nslice0+yc*nx0 + xc];
                         value += wzc*wxf*wyf*input[zc*nslice0+yf*nx0 + xf] + wzc*wxc*wyf*input[zc*nslice0+yf*nx0 + xc] + wzc*wxf*wyc*input[zc*nslice0+yf*nx0 + xf] + wzc*wxc*wyc*input[zc*nslice0+yc*nx0 + xc];

                         output[z*nslice1+y*nx1+x] = value;
                    }
                }
            }
        }

        return;
    }
    void upsample_cubic(float* input, int inshape[3], float* output, int outshape[3])
    {
        //Doing it in one pass = less write operations, more read operations

        double xratio = ((double) outshape[0])/((double) inshape[0]);
        double yratio = ((double) outshape[1])/((double) inshape[1]);
        double zratio = ((double) outshape[2])/((double) inshape[2]);

        int nx0 = inshape[0];
        long long int nslice0 = inshape[0]*inshape[1];
        int nx1 = outshape[0];
        long long int nslice1 = outshape[0]*outshape[1];

        float dx = 1.f/xratio;
        float dy = 1.f/yratio;
        float dz = 1.f/zratio;

        for (int z = 0; z < std::max(outshape[2],1); z++)
        {
            float z0 = z/zratio;
            int zf = floor(z0);
            int zc = ceil(z0);

            zc = std::min(zc, inshape[2]-1);

            //extrapolate with zero-gradient
            int zf2 = std::max(0, zf-1);
            int zc2 = std::min(zc+1, inshape[2]-1);

            #pragma omp parallel for
            for (int y = 0; y < outshape[1]; y++)
            {
                float y0 = y/yratio;
                int yf = floor(y0);
                int yc = ceil(y0);

                yc = std::min(yc, inshape[1]-1);

                //extrapolate with zero-gradient
                int yf2 = std::max(0, yf-1);
                int yc2 = std::min(yc+1, inshape[1]-1);

                for (int x = 0; x < outshape[0]; x++)
                {
                    float x0 = x/xratio;
                    int xf = floor(x0);
                    int xc = ceil(x0);

                    xc = std::min(xc, nx0-1);

                    //extrapolate with zero-gradient
                    int xf2 = std::max(0, xf-1);
                    int xc2 = std::min(xc+1, nx0-1);

                    if(outshape[2] == 1)
                    {
                        float val_00 = input[yf2*nx0 + xf2];
                        float val_10 = input[yf2*nx0 + xf];
                        float val_20 = input[yf2*nx0 + xc];
                        float val_30 = input[yf2*nx0 + xc2];

                        float val_01 = input[yf*nx0 + xf2];
                        float val_11 = input[yf*nx0 + xf];
                        float val_21 = input[yf*nx0 + xc];
                        float val_31 = input[yf*nx0 + xc2];

                        float val_02 = input[yc*nx0 + xf2];
                        float val_12 = input[yc*nx0 + xf];
                        float val_22 = input[yc*nx0 + xc];
                        float val_32 = input[yc*nx0 + xc2];

                        float val_03 = input[yc2*nx0 + xf2];
                        float val_13 = input[yc2*nx0 + xf];
                        float val_23 = input[yc2*nx0 + xc];
                        float val_33 = input[yc2*nx0 + xc2];

                        //Interpolate in x
                        float val_x0 = interpolate_cubic(val_00, val_10, val_20, val_30, dx);
                        float val_x1 = interpolate_cubic(val_01, val_11, val_21, val_31, dx);
                        float val_x2 = interpolate_cubic(val_02, val_12, val_22, val_32, dx);
                        float val_x3 = interpolate_cubic(val_03, val_13, val_23, val_33, dx);

                        //Interpolate in y
                        output[y*nx1+x] = interpolate_cubic(val_x0, val_x1, val_x2, val_x3, dy);
                    }
                    else
                    {
                        float val_000 = input[zf2*nslice0+yf2*nx0 + xf2];
                        float val_100 = input[zf2*nslice0+yf2*nx0 + xf];
                        float val_200 = input[zf2*nslice0+yf2*nx0 + xc];
                        float val_300 = input[zf2*nslice0+yf2*nx0 + xc2];

                        float val_010 = input[zf2*nslice0+yf*nx0 + xf2];
                        float val_110 = input[zf2*nslice0+yf*nx0 + xf];
                        float val_210 = input[zf2*nslice0+yf*nx0 + xc];
                        float val_310 = input[zf2*nslice0+yf*nx0 + xc2];

                        float val_020 = input[zf2*nslice0+yc*nx0 + xf2];
                        float val_120 = input[zf2*nslice0+yc*nx0 + xf];
                        float val_220 = input[zf2*nslice0+yc*nx0 + xc];
                        float val_320 = input[zf2*nslice0+yc*nx0 + xc2];

                        float val_030 = input[zf2*nslice0+yc2*nx0 + xf2];
                        float val_130 = input[zf2*nslice0+yc2*nx0 + xf];
                        float val_230 = input[zf2*nslice0+yc2*nx0 + xc];
                        float val_330 = input[zf2*nslice0+yc2*nx0 + xc2];

                        float val_001 = input[zf*nslice0+yf2*nx0 + xf2];
                        float val_101 = input[zf*nslice0+yf2*nx0 + xf];
                        float val_201 = input[zf*nslice0+yf2*nx0 + xc];
                        float val_301 = input[zf*nslice0+yf2*nx0 + xc2];

                        float val_011 = input[zf*nslice0+yf*nx0 + xf2];
                        float val_111 = input[zf*nslice0+yf*nx0 + xf];
                        float val_211 = input[zf*nslice0+yf*nx0 + xc];
                        float val_311 = input[zf*nslice0+yf*nx0 + xc2];

                        float val_021 = input[zf*nslice0+yc*nx0 + xf2];
                        float val_121 = input[zf*nslice0+yc*nx0 + xf];
                        float val_221 = input[zf*nslice0+yc*nx0 + xc];
                        float val_321 = input[zf*nslice0+yc*nx0 + xc2];

                        float val_031 = input[zf*nslice0+yc2*nx0 + xf2];
                        float val_131 = input[zf*nslice0+yc2*nx0 + xf];
                        float val_231 = input[zf*nslice0+yc2*nx0 + xc];
                        float val_331 = input[zf*nslice0+yc2*nx0 + xc2];

                        float val_002 = input[zc*nslice0+yf2*nx0 + xf2];
                        float val_102 = input[zc*nslice0+yf2*nx0 + xf];
                        float val_202 = input[zc*nslice0+yf2*nx0 + xc];
                        float val_302 = input[zc*nslice0+yf2*nx0 + xc2];

                        float val_012 = input[zc*nslice0+yf*nx0 + xf2];
                        float val_112 = input[zc*nslice0+yf*nx0 + xf];
                        float val_212 = input[zc*nslice0+yf*nx0 + xc];
                        float val_312 = input[zc*nslice0+yf*nx0 + xc2];

                        float val_022 = input[zc*nslice0+yc*nx0 + xf2];
                        float val_122 = input[zc*nslice0+yc*nx0 + xf];
                        float val_222 = input[zc*nslice0+yc*nx0 + xc];
                        float val_322 = input[zc*nslice0+yc*nx0 + xc2];

                        float val_032 = input[zc*nslice0+yc2*nx0 + xf2];
                        float val_132 = input[zc*nslice0+yc2*nx0 + xf];
                        float val_232 = input[zc*nslice0+yc2*nx0 + xc];
                        float val_332 = input[zc*nslice0+yc2*nx0 + xc2];

                        float val_003 = input[zc2*nslice0+yf2*nx0 + xf2];
                        float val_103 = input[zc2*nslice0+yf2*nx0 + xf];
                        float val_203 = input[zc2*nslice0+yf2*nx0 + xc];
                        float val_303 = input[zc2*nslice0+yf2*nx0 + xc2];

                        float val_013 = input[zc2*nslice0+yf*nx0 + xf2];
                        float val_113 = input[zc2*nslice0+yf*nx0 + xf];
                        float val_213 = input[zc2*nslice0+yf*nx0 + xc];
                        float val_313 = input[zc2*nslice0+yf*nx0 + xc2];

                        float val_023 = input[zc2*nslice0+yc*nx0 + xf2];
                        float val_123 = input[zc2*nslice0+yc*nx0 + xf];
                        float val_223 = input[zc2*nslice0+yc*nx0 + xc];
                        float val_323 = input[zc2*nslice0+yc*nx0 + xc2];

                        float val_033 = input[zc2*nslice0+yc2*nx0 + xf2];
                        float val_133 = input[zc2*nslice0+yc2*nx0 + xf];
                        float val_233 = input[zc2*nslice0+yc2*nx0 + xc];
                        float val_333 = input[zc2*nslice0+yc2*nx0 + xc2];

                        //Interpolate in x
                        float val_x00 = interpolate_cubic(val_000, val_100, val_200, val_300, dx);
                        float val_x10 = interpolate_cubic(val_010, val_110, val_210, val_310, dx);
                        float val_x20 = interpolate_cubic(val_020, val_120, val_220, val_320, dx);
                        float val_x30 = interpolate_cubic(val_030, val_130, val_230, val_330, dx);
                        float val_x01 = interpolate_cubic(val_001, val_101, val_201, val_301, dx);
                        float val_x11 = interpolate_cubic(val_011, val_111, val_211, val_311, dx);
                        float val_x21 = interpolate_cubic(val_021, val_121, val_221, val_321, dx);
                        float val_x31 = interpolate_cubic(val_031, val_131, val_231, val_331, dx);
                        float val_x02 = interpolate_cubic(val_002, val_102, val_202, val_302, dx);
                        float val_x12 = interpolate_cubic(val_012, val_112, val_212, val_312, dx);
                        float val_x22 = interpolate_cubic(val_022, val_122, val_222, val_322, dx);
                        float val_x32 = interpolate_cubic(val_032, val_132, val_232, val_332, dx);
                        float val_x03 = interpolate_cubic(val_003, val_103, val_203, val_303, dx);
                        float val_x13 = interpolate_cubic(val_013, val_113, val_213, val_313, dx);
                        float val_x23 = interpolate_cubic(val_023, val_123, val_223, val_323, dx);
                        float val_x33 = interpolate_cubic(val_033, val_133, val_233, val_333, dx);

                        //Interpolate in y
                        float val_xy0 = interpolate_cubic(val_x00, val_x10, val_x20, val_x30, dy);
                        float val_xy1 = interpolate_cubic(val_x01, val_x11, val_x21, val_x31, dy);
                        float val_xy2 = interpolate_cubic(val_x02, val_x12, val_x22, val_x32, dy);
                        float val_xy3 = interpolate_cubic(val_x03, val_x13, val_x23, val_x33, dy);

                        //Interpolate in y
                        output[z*nslice1+y*nx1+x] = interpolate_cubic(val_xy0, val_xy1, val_xy2, val_xy3, dz);
                    }
                }
            }
        }

        return;
    }
    ////////////////////////////////////////////////////////////////////////

    //Upscaling a flow vector
    ////////////////////////////////////////////////////////////////////////
    void upscalevector(float *&u, int last_shape[3], int next_shape[3], int ndims, std::string interpolation_mode)
    {
    	long long int nslice0 = last_shape[0]*last_shape[1];
    	long long int nstack0 = last_shape[2]*nslice0;
    	long long int nslice1 = next_shape[0]*next_shape[1];
    	long long int nstack1 = next_shape[2]*nslice1;

    	if (last_shape[0] == next_shape[0] && last_shape[1] == next_shape[1] && last_shape[2] == next_shape[2])
    		return;

    	float *output = (float*) malloc((ndims*nstack1)*sizeof(*output));

    	if (interpolation_mode == "linear")
    		linear_coons(u, last_shape, output, next_shape, ndims, true);
    	else if (interpolation_mode == "cubic")
    		cubic_coons(u, last_shape, output, next_shape, ndims, true);

    	free(u);
    	u = output;

    	//Deep copy
		/*#pragma omp parallel for
    	for (long long int pos = 0; pos < (ndims*nstack1); pos++)
    	{
    		u[pos] = output[pos];
    	}
    	std::cout << "through" << std::endl;*/

    	//std::swap(u, output);
    	//free(output);

    	return;
    }
    ////////////////////////////////////////////////////////////////////////
}
