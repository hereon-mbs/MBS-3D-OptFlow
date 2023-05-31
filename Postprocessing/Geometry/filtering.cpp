#include <iostream>
#include <vector>
#include <algorithm>
#include <omp.h>
#include <math.h>

namespace filter
{
    float* apply_1Dconvolution(int dim, float* image, int shape[3], std::vector<float> &kernel1D)
    {
        int nx = shape[0];
        int ny = shape[1];
        int nz = shape[2];
        long long int nslice = shape[0]*shape[1];
        long long int nstack = nslice*shape[2];
        int fsize = kernel1D.size()/2;

        float *outimg = (float*) malloc(nstack*sizeof(*outimg));

        //Dim0
        if (dim == 0)
        {
            #pragma omp parallel for
            for(int y = 0; y < ny; y++)
            {
                for(int z = 0; z < std::max(nz, 1); z++)
                {
                    long long int posyz = z*nslice+y*nx;

                    for(int x = 0; x < nx; x++)
                    {
                        float sum = 0.0f;

                        for(int xi=-fsize; xi<=fsize; xi++)
                        {
                            int x0 = x+xi;

                            //reflective boundaries
                            if (x0 < 0) x0 = -x0;
                            else if (x0 >= nx) x0 = 2*nx-x0-2;

                            sum += kernel1D[xi+fsize]*image[posyz+x0];
                        }

                        outimg[posyz+x] = sum;
                    }
                }
            }
        }
        else if (dim == 1)
        {
            #pragma omp parallel for
            for(int z = 0; z < std::max(nz, 1); z++)
            {
                for(int x = 0; x < nx; x++)
                {
                    long long int posxz = z*nslice+x;

                    for(int y = 0; y < ny; y++)
                    {
                        float sum = 0.0f;

                        for(int yi=-fsize; yi<=fsize; yi++)
                        {
                            int y0 = y+yi;

                            //reflective boundaries
                            if (y0 < 0) y0 = -y0;
                            else if (y0 >= ny) y0 = 2*ny-y0-2;

                            sum += kernel1D[yi+fsize]*image[posxz+y0*nx];
                        }

                        outimg[posxz+y*nx] = sum;
                    }
                }
            }
		}
		else if (dim == 2)
		{
			int offset = 0;
			if (nz <= fsize)
			{
				//renormalize
				offset = fsize-nz+1;
				float sum = 0.0f;

				for(uint16_t p=offset; p<kernel1D.size()-offset;p++)
					sum += kernel1D[p];
				for(uint16_t p=offset; p<kernel1D.size()-offset;p++)
					kernel1D[p]/sum;
			}

			//Dim2
			#pragma omp parallel for
			for(int y = 0; y < ny; y++)
			{
				for(int x = 0; x < nx; x++)
				{
					long long int posxy = y*nx+x;

					for(int z = 0; z < nz; z++)
					{
						float sum = 0.0f;

						for(int zi=-(fsize-offset); zi<=(fsize-offset); zi++)
						{
							int z0 = z+zi;

							//reflective boundaries
							if (z0 < 0) z0 = -z0;
							else if (z0 >= nz) z0 = 2*nz-z0-2;

							sum += kernel1D[zi+fsize]*image[posxy+z0*nslice];
						}

						outimg[posxy+z*nslice] = sum;
					}
				}
			}
		}
		else
            std::cout << "Unknown dim for 1D convolution!" << std::endl;

        return outimg;
    }

	//////////////////////////////////////////////////////////////////////////////////////////////////////////
	void apply_2Dconvolution_splitdims(float* image2D, int shape[2], std::vector<float> &kernel1D, float* &outimg)
    {
        int nx = shape[0];
        int ny = shape[1];
        long long int nslice = shape[0]*shape[1];
        int fsize = kernel1D.size()/2;

        float *outimg2 = (float*) malloc(nslice*sizeof(*outimg2));

        //Dim0
		#pragma omp parallel for
        for(int y = 0; y < ny; y++)
        {
            for(int x = 0; x < nx; x++)
            {
                float sum = 0.0f;

                for(int xi=-fsize; xi<=fsize; xi++)
                {
                    int x0 = x+xi;

                    //reflective boundaries
                    if (x0 < 0) x0 = -x0;
                    else if (x0 >= nx) x0 = 2*nx-x0-2;

                    sum += kernel1D[xi+fsize]*image2D[y*nx+x0];
                }

                outimg[y*nx+x] = sum;
            }
        }

        //Dim1
		#pragma omp parallel for
        for(int x = 0; x < nx; x++)
        {
            for(int y = 0; y < ny; y++)
            {
                float sum = 0.0f;

                for(int yi=-fsize; yi<=fsize; yi++)
                {
                    int y0 = y+yi;

                    //reflective boundaries
                    if (y0 < 0) y0 = -y0;
                    else if (y0 >= ny) y0 = 2*ny-y0-2;

                    sum += kernel1D[yi+fsize]*outimg[x+y0*nx];
                }

                outimg2[x+y*nx] = sum;
            }
		}

		std::swap(outimg2, outimg);
		free(outimg2);

        return;
    }
	void apply_3Dconvolution_splitdims(float* image, int shape[3], std::vector<float> &kernel1D, float* &outimg)
    {
        int nx = shape[0];
        int ny = shape[1];
        int nz = shape[2];
        long long int nslice = shape[0]*shape[1];
        long long int nstack = nslice*shape[2];
        int fsize = kernel1D.size()/2;

        float *outimg2 = (float*) malloc(nstack*sizeof(*outimg2));

        //Dim0
		#pragma omp parallel for
        for(int y = 0; y < ny; y++)
        {
			for(int z = 0; z < std::max(nz, 1); z++)
			{
				long long int posyz = z*nslice+y*nx;

				for(int x = 0; x < nx; x++)
				{
					float sum = 0.0f;

					for(int xi=-fsize; xi<=fsize; xi++)
					{
						int x0 = x+xi;

						//reflective boundaries
						if (x0 < 0) x0 = -x0;
						else if (x0 >= nx) x0 = 2*nx-x0-2;

						sum += kernel1D[xi+fsize]*image[posyz+x0];
					}

					outimg[posyz+x] = sum;
				}
			}
        }

        //Dim1
		#pragma omp parallel for
		for(int z = 0; z < std::max(nz, 1); z++)
		{
			for(int x = 0; x < nx; x++)
			{
				long long int posxz = z*nslice+x;

				for(int y = 0; y < ny; y++)
				{
					float sum = 0.0f;

					for(int yi=-fsize; yi<=fsize; yi++)
					{
						int y0 = y+yi;

						//reflective boundaries
						if (y0 < 0) y0 = -y0;
						else if (y0 >= ny) y0 = 2*ny-y0-2;

						sum += kernel1D[yi+fsize]*outimg[posxz+y0*nx];
					}

					outimg2[posxz+y*nx] = sum;
				}
			}
		}

		if(shape[2] == 1)
		{
			std::swap(outimg2, outimg);
		}
		else
		{
			int offset = 0;

			if (nz <= fsize)
			{
				//renormalize

				offset = fsize-nz+1;
				float sum = 0.0f;

				for(uint16_t p=offset; p<kernel1D.size()-offset;p++)
					sum += kernel1D[p];
				for(uint16_t p=offset; p<kernel1D.size()-offset;p++)
					kernel1D[p]/sum;
			}

			//Dim2
			#pragma omp parallel for
			for(int y = 0; y < ny; y++)
			{
				for(int x = 0; x < nx; x++)
				{
					long long int posxy = y*nx+x;

					for(int z = 0; z < nz; z++)
					{
						float sum = 0.0f;

						for(int zi=-(fsize-offset); zi<=(fsize-offset); zi++)
						{
							int z0 = z+zi;

							//reflective boundaries
							if (z0 < 0) z0 = -z0;
							else if (z0 >= nz) z0 = 2*nz-z0-2;

							sum += kernel1D[zi+fsize]*outimg2[posxy+z0*nslice];
						}

						outimg[posxy+z*nslice] = sum;
					}
				}
			}
		}

		free(outimg2);

        return;
    }
    void apply_3Dconvolution_splitdims_labelonly(float* image, int shape[3], std::vector<float> &kernel1D, float* &outimg, float* labelimage, float label)
    {
        int nx = shape[0];
        int ny = shape[1];
        int nz = shape[2];
        long long int nslice = shape[0]*shape[1];
        long long int nstack = nslice*shape[2];
        int fsize = kernel1D.size()/2;

        float *outimg2 = (float*) malloc(nstack*sizeof(*outimg2));

        //Dim0
		#pragma omp parallel for
        for(int y = 0; y < ny; y++)
        {
			for(int z = 0; z < std::max(nz, 1); z++)
			{
				long long int posyz = z*nslice+y*nx;

				for(int x = 0; x < nx; x++)
				{
					float sum = 0.0f;
                    float weightsum = 0.0f;

					for(int xi=-fsize; xi<=fsize; xi++)
					{
						int x0 = x+xi;

						//reflective boundaries
						if (x0 < 0) x0 = -x0;
						else if (x0 >= nx) x0 = 2*nx-x0-2;

						if (labelimage[posyz+x0] == label)
						{
                            sum += kernel1D[xi+fsize]*image[posyz+x0];
                            weightsum += kernel1D[xi+fsize];
						}
					}

					if (weightsum > 0.0f)
                        outimg[posyz+x] = sum/weightsum;
                    else outimg[posyz+x] = image[posyz+x];
				}
			}
        }

        //Dim1
		#pragma omp parallel for
		for(int z = 0; z < std::max(nz, 1); z++)
		{
			for(int x = 0; x < nx; x++)
			{
				long long int posxz = z*nslice+x;

				for(int y = 0; y < ny; y++)
				{
					float sum = 0.0f;
					float weightsum = 0.0f;

					for(int yi=-fsize; yi<=fsize; yi++)
					{
						int y0 = y+yi;

						//reflective boundaries
						if (y0 < 0) y0 = -y0;
						else if (y0 >= ny) y0 = 2*ny-y0-2;

						if (labelimage[posxz+y0*nx] == label)
						{
                            sum += kernel1D[yi+fsize]*outimg[posxz+y0*nx];
                            weightsum += kernel1D[yi+fsize];
						}

					}

					if (weightsum > 0.0f)
                        outimg2[posxz+y*nx] = sum/weightsum;
                    else outimg2[posxz+y*nx] = outimg[posxz+y*nx];
				}
			}
		}

		if(shape[2] == 1)
		{
			std::swap(outimg2, outimg);
		}
		else
		{
			int offset = 0;

			if (nz <= fsize)
			{
				//renormalize

				offset = fsize-nz+1;
				float sum = 0.0f;

				for(uint16_t p=offset; p<kernel1D.size()-offset;p++)
					sum += kernel1D[p];
				for(uint16_t p=offset; p<kernel1D.size()-offset;p++)
					kernel1D[p]/sum;
			}

			//Dim2
			#pragma omp parallel for
			for(int y = 0; y < ny; y++)
			{
				for(int x = 0; x < nx; x++)
				{
					long long int posxy = y*nx+x;

					for(int z = 0; z < nz; z++)
					{
						float sum = 0.0f;
						float weightsum = 0.0f;

						for(int zi=-(fsize-offset); zi<=(fsize-offset); zi++)
						{
							int z0 = z+zi;

							//reflective boundaries
							if (z0 < 0) z0 = -z0;
							else if (z0 >= nz) z0 = 2*nz-z0-2;

							if (labelimage[posxy+z0*nslice] == label)
                            {
                                sum += kernel1D[zi+fsize]*outimg2[posxy+z0*nslice];
                                weightsum += kernel1D[zi+fsize];
                            }

						}

						if (weightsum > 0.0f)
                            outimg[posxy+z*nslice] = sum/weightsum;
                        else
                            outimg[posxy+z*nslice] = outimg2[posxy+z*nslice];
					}
				}
			}
		}

		free(outimg2);

        return;
    }
    void apply_3Dconvolution_splitdims_interfaceonly(float* image, int shape[3], std::vector<float> &kernel1D, float* &outimg, uint8_t* phase_image)
    {
        int nx = shape[0];
        int ny = shape[1];
        int nz = shape[2];
        long long int nslice = shape[0]*shape[1];
        long long int nstack = nslice*shape[2];
        int fsize = kernel1D.size()/2;

        float *outimg2 = (float*) malloc(nstack*sizeof(*outimg2));

        //Dim0
		#pragma omp parallel for
        for(int y = 0; y < ny; y++)
        {
			for(int z = 0; z < std::max(nz, 1); z++)
			{
				long long int posyz = z*nslice+y*nx;

				for(int x = 0; x < nx; x++)
				{
					float sum = 0.0f;
                    float weightsum = 0.0f;

					for(int xi=-fsize; xi<=fsize; xi++)
					{
						int x0 = x+xi;

						//reflective boundaries
						if (x0 < 0) x0 = -x0;
						else if (x0 >= nx) x0 = 2*nx-x0-2;

						int this_phase = phase_image[posyz+x0];

						if ((x0+1 < nx && this_phase != phase_image[posyz+x0+1]) || (x0-1 >= 0 && this_phase != phase_image[posyz+x0-1]))
						{
                            sum += kernel1D[xi+fsize]*image[posyz+x0];
                            weightsum += kernel1D[xi+fsize];
						}
					}

					if (weightsum > 0.0f)
                        outimg[posyz+x] = sum/weightsum;
                    else outimg[posyz+x] = image[posyz+x];
				}
			}
        }
        //Dim1
		#pragma omp parallel for
		for(int z = 0; z < std::max(nz, 1); z++)
		{
			for(int x = 0; x < nx; x++)
			{
				long long int posxz = z*nslice+x;

				for(int y = 0; y < ny; y++)
				{
					float sum = 0.0f;
					float weightsum = 0.0f;

					for(int yi=-fsize; yi<=fsize; yi++)
					{
						int y0 = y+yi;

						//reflective boundaries
						if (y0 < 0) y0 = -y0;
						else if (y0 >= ny) y0 = 2*ny-y0-2;

						int this_phase = phase_image[posxz+y0];
						if ((y0+1 < ny && this_phase != phase_image[posxz+y0+nx]) || (y0-1 >= 0 && this_phase != phase_image[posxz+y0-nx]))
						{
                            sum += kernel1D[yi+fsize]*outimg[posxz+y0*nx];
                            weightsum += kernel1D[yi+fsize];
						}

					}

					if (weightsum > 0.0f)
                        outimg2[posxz+y*nx] = sum/weightsum;
                    else outimg2[posxz+y*nx] = outimg[posxz+y*nx];
				}
			}
		}

		if(shape[2] == 1)
		{
			std::swap(outimg2, outimg);
		}
		else
		{
			int offset = 0;

			if (nz <= fsize)
			{
				//renormalize

				offset = fsize-nz+1;
				float sum = 0.0f;

				for(uint16_t p=offset; p<kernel1D.size()-offset;p++)
					sum += kernel1D[p];
				for(uint16_t p=offset; p<kernel1D.size()-offset;p++)
					kernel1D[p]/sum;
			}

			//Dim2
			#pragma omp parallel for
			for(int y = 0; y < ny; y++)
			{
				for(int x = 0; x < nx; x++)
				{
					long long int posxy = y*nx+x;

					for(int z = 0; z < nz; z++)
					{
						float sum = 0.0f;
						float weightsum = 0.0f;

						for(int zi=-(fsize-offset); zi<=(fsize-offset); zi++)
						{
							int z0 = z+zi;

							//reflective boundaries
							if (z0 < 0) z0 = -z0;
							else if (z0 >= nz) z0 = 2*nz-z0-2;

							int this_phase = phase_image[posxy+z0];
                            if ((z0+1 < nz && this_phase != phase_image[posxy+z0+nslice]) || (z0-1 >= 0 && this_phase != phase_image[posxy+z0-nslice]))
                            {
                                sum += kernel1D[zi+fsize]*outimg2[posxy+z0*nslice];
                                weightsum += kernel1D[zi+fsize];
                            }

						}

						if (weightsum > 0.0f)
                            outimg[posxy+z*nslice] = sum/weightsum;
                        else
                            outimg[posxy+z*nslice] = outimg2[posxy+z*nslice];
					}
				}
			}
		}

		free(outimg2);

        return;
    }
	void apply_3Dconvolution_splitdims(float* image, int shape[3], std::vector<float> &kernel_dim0, std::vector<float> &kernel_dim1, std::vector<float> &kernel_dim2, float* &outimg)
	{
		int nx = shape[0];
		int ny = shape[1];
		int nz = shape[2];
		long long int nslice = shape[0]*shape[1];
		long long int nstack = nslice*shape[2];

		float *outimg2 = (float*) malloc(nstack*sizeof(*outimg2));

		//Dim0
		int fsize = kernel_dim0.size()/2;

		#pragma omp parallel for
		for(int y = 0; y < ny; y++)
		{
			for(int z = 0; z < std::max(nz, 1); z++)
			{
				long long int posyz = z*nslice+y*nx;

				for(int x = 0; x < nx; x++)
				{
					float sum = 0.0f;

					for(int xi=-fsize; xi<=fsize; xi++)
					{
						int x0 = x+xi;

						//reflective boundaries
						if (x0 < 0) x0 = -x0;
						else if (x0 >= nx) x0 = 2*nx-x0-2;

						sum += kernel_dim0[xi+fsize]*image[posyz+x0];
					}

					outimg[posyz+x] = sum;
				}
			}
		}

		//Dim1
		fsize = kernel_dim1.size()/2;

		#pragma omp parallel for
		for(int z = 0; z < std::max(nz, 1); z++)
		{
			for(int x = 0; x < nx; x++)
			{
				long long int posxz = z*nslice+x;

				for(int y = 0; y < ny; y++)
				{
					float sum = 0.0f;

					for(int yi=-fsize; yi<=fsize; yi++)
					{
						int y0 = y+yi;

						//reflective boundaries
						if (y0 < 0) y0 = -y0;
						else if (y0 >= ny) y0 = 2*ny-y0-2;

						sum += kernel_dim1[yi+fsize]*outimg[posxz+y0*nx];
					}

					outimg2[posxz+y*nx] = sum;
				}
			}
		}

		if(shape[2] == 1)
		{
			std::swap(outimg2, outimg);
		}
		else
		{
			int offset = 0;
			fsize = kernel_dim2.size()/2;

			if (nz <= fsize)
			{
				//renormalize

				offset = fsize-nz+1;
				float sum = 0.0f;

				for(uint16_t p=offset; p<kernel_dim2.size()-offset;p++)
					sum += kernel_dim2[p];
				for(uint16_t p=offset; p<kernel_dim2.size()-offset;p++)
					kernel_dim2[p]/sum;
			}

			//Dim2
			#pragma omp parallel for
			for(int y = 0; y < ny; y++)
			{
				for(int x = 0; x < nx; x++)
				{
					long long int posxy = y*nx+x;

					for(int z = 0; z < nz; z++)
					{
						float sum = 0.0f;

						for(int zi=-(fsize-offset); zi<=(fsize-offset); zi++)
						{
							int z0 = z+zi;

							//reflective boundaries
							if (z0 < 0) z0 = -z0;
							else if (z0 >= nz) z0 = 2*nz-z0-2;

							sum += kernel_dim2[zi+fsize]*outimg2[posxy+z0*nslice];
						}

						outimg[posxy+z*nslice] = sum;
					}
				}
			}
		}

		free(outimg2);

		return;
	}
    void apply_3Dconvolution_precise(float* image, int shape[3], std::vector<float> &kernel1D, float* outimg)
    {
        int nx = shape[0];
        long long int nslice = shape[0]*shape[1];
        long long int nstack = nslice*shape[2];
        int fsize = kernel1D.size()/2;

        #pragma omp parallel for
        for (int64_t pos = 0; pos < nstack; pos++)
        {
            int z0 = pos/nslice;
            int y0 = (pos-z0*nslice)/nx;
            int x0 = (pos-z0*nslice-y0*nx);

            float value = 0;

            for (int yi=-fsize; yi<=fsize; yi++) //filter in dim1
            {
                int y = y0+yi;

                //reflective boundaries
                if (y < 0) y = -y;
                if (y >= shape[1]) y = 2*shape[1]-y-2;

                for (int xi=-fsize; xi<=fsize; xi++) //filter in dim0
                {
                    double kernel_outerproduct = kernel1D[xi+fsize]*kernel1D[yi+fsize];

                    int x = x0+xi;

                    //reflective boundaries
                    if (x < 0) x = -x;
                    if (x >= nx) x = 2*nx-x-2;

                    if(shape[2] > 1) //3D case
                    {
                        for (int zi=-fsize; zi<=fsize; zi++)
                        {
                            kernel_outerproduct = kernel1D[xi+fsize]*kernel1D[yi+fsize]*kernel1D[zi+fsize];

                            int z = z0+zi;

                            //reflective boundaries
                            if (z < 0) z = -z;
                            if (z >= shape[2]) z = 2*shape[2]-z-2;

                            value += image[z*nslice+y*nx+x]*kernel_outerproduct;
                        }
                    }
                    else //2D case
                        value += image[z0*nslice+y*nx+x]*kernel_outerproduct;
                }
            }

            outimg[pos] = value;
        }

        return;
    }
    //////////////////////////////////////////////////////////////////////////////////////////////////////////

    //////////////////////////////////////////////////////////////////////////////////////////////////////////
    void apply_2DanisotropicGaussian_spatialdomain(float* image, int shape[3], float sigma0, float sigma1, float theta, float* outimg)
	{
    	float costheta = cos(theta*0.0174533f);
    	float sintheta = sin(theta*0.0174533f);

    	float term0 = 1.f/(2.f*sigma0*sigma0);
    	float term1 = 1.f/(2.f*sigma1*sigma1);

		int nx = shape[0];
		long long int nslice = shape[0]*shape[1];
		long long int nstack = nslice*shape[2];

		int fsize = (int) ceil(3.*sqrtf(sigma0*sigma0+sigma1*sigma1));
		int nk = 2*fsize+1;
		long long int kernelsize = nk*nk;
		float sum = 0.0f;
		float x_origin = fsize; float y_origin = fsize;

		//Create anisotropic Gaussian kernel
		///////////////////////////////////////////////////
		float *kernel = (float*) malloc (kernelsize*sizeof(*kernel));

		#pragma omp parallel for reduction(+: sum)
		for (long long int pos = 0; pos < kernelsize; pos++)
		{
			int y = (pos/nk);
			int x = pos-y*nk;

			float xrot = costheta*(x-x_origin) - sintheta*(y-y_origin);
			float yrot = sintheta*(x-x_origin) + costheta*(y-y_origin);

			float val = expf(-((xrot*xrot)*term0+(yrot*yrot)*term1));

			sum += val;
			kernel[pos] = val;
		}
		#pragma omp parallel for
		for (long long int pos = 0; pos < kernelsize; pos++) kernel[pos] /= sum;
		///////////////////////////////////////////////////

		//and apply
		///////////////////////////////////////////////////
		//#pragma omp parallel for
		for (long long int pos = 0; pos < nstack; pos++)
		{
			int z0 = pos/nslice;
			int y0 = (pos-z0*nslice)/nx;
			int x0 = (pos-z0*nslice-y0*nx);

			float value = 0;

			for (int yi=-fsize; yi<=fsize; yi++) //filter in dim1
			{
				int y = y0+yi;

				//reflective boundaries
				if (y < 0) y = -y;
				if (y >= shape[1]) y = 2*shape[1]-y-2;

				for (int xi=-fsize; xi<=fsize; xi++) //filter in dim0
				{
					int x = x0+xi;

					//reflective boundaries
					if (x < 0) x = -x;
					if (x >= nx) x = 2*nx-x-2;

					value += image[z0*nslice+y*nx+x]*kernel[(yi+fsize)*nk+(xi+fsize)];
				}
			}

			outimg[pos] = value;
		}
		///////////////////////////////////////////////////

		free(kernel);

		return;
	}
    //////////////////////////////////////////////////////////////////////////////////////////////////////////

    //////////////////////////////////////////////////////////////////////////////////////////////////////////
    void apply_3DGaussianFilter(float* &image, int shape[3], int fsize)
    {
        //Create Gaussian kernel
        ///////////////////////////////////////////////////
        float sigma = fsize/3.f;

        double sum = 0;

        std::vector<float> kernel(2*fsize+1, 0);
        for(uint16_t p=0; p<kernel.size();p++)
        {
            kernel[p] = exp(-((p-fsize)*(p-fsize))/(sigma*sigma*2));
            sum += kernel[p];
        }
        for(uint16_t p=0; p<kernel.size();p++) kernel[p] /= sum;
        ///////////////////////////////////////////////////

        long long int nslice = shape[0]*shape[1];
        long long int nstack = nslice*shape[2];
        float *outimg = (float*) malloc(nstack*sizeof(*outimg));

        apply_3Dconvolution_splitdims(image, shape, kernel, outimg);

        std::swap(image, outimg);
        free(outimg);

        return;
    }
    void apply_2DGaussianFilter(float* &image, int shape[2], float sigma)
	{
		//Create Gaussian kernel
		///////////////////////////////////////////////////
		int fsize = (int) (3*sigma);
		double sum = 0;

		std::vector<float> kernel(2*fsize+1, 0);
		for(uint16_t p=0; p<kernel.size();p++)
		{
			kernel[p] = exp(-((p-fsize)*(p-fsize))/(sigma*sigma*2));
			sum += kernel[p];
		}
		for(uint16_t p=0; p<kernel.size();p++) kernel[p] /= sum;
		///////////////////////////////////////////////////

		long long int nslice = shape[0]*shape[1];
		float *outimg = (float*) malloc(nslice*sizeof(*outimg));

		apply_2Dconvolution_splitdims(image, shape, kernel, outimg);

		std::swap(image, outimg);
		free(outimg);

		return;
	}
    void apply_3DGaussianFilter(float* &image, int shape[3], float sigma)
	{
		//Create Gaussian kernel
		///////////////////////////////////////////////////
		int fsize = (int) (3*sigma);
		double sum = 0;

		std::vector<float> kernel(2*fsize+1, 0);
		for(uint16_t p=0; p<kernel.size();p++)
		{
			kernel[p] = exp(-((p-fsize)*(p-fsize))/(sigma*sigma*2));
			sum += kernel[p];
		}
		for(uint16_t p=0; p<kernel.size();p++) kernel[p] /= sum;
		///////////////////////////////////////////////////

		long long int nslice = shape[0]*shape[1];
		long long int nstack = nslice*shape[2];
		float *outimg = (float*) malloc(nstack*sizeof(*outimg));

		apply_3Dconvolution_splitdims(image, shape, kernel, outimg);

		std::swap(image, outimg);
		free(outimg);

		return;
	}
	void apply_3DGaussianFilter2Label(float* &image, int shape[3], float sigma, float* labelimage, float label)
	{
		//Create Gaussian kernel
		///////////////////////////////////////////////////
		int fsize = (int) (3*sigma);
		double sum = 0;

		std::vector<float> kernel(2*fsize+1, 0);
		for(uint16_t p=0; p<kernel.size();p++)
		{
			kernel[p] = exp(-((p-fsize)*(p-fsize))/(sigma*sigma*2));
			sum += kernel[p];
		}
		for(uint16_t p=0; p<kernel.size();p++) kernel[p] /= sum;
		///////////////////////////////////////////////////

		long long int nslice = shape[0]*shape[1];
		long long int nstack = nslice*shape[2];
		float *outimg = (float*) malloc(nstack*sizeof(*outimg));

		apply_3Dconvolution_splitdims_labelonly(image, shape, kernel, outimg, labelimage, label);

		std::swap(image, outimg);
		free(outimg);

		return;
	}
    void apply_3DGaussianFilter2Vector(float* vectorimg, int shape[3], float sigma, int ndims)
	{
		//Create Gaussian kernel
		///////////////////////////////////////////////////
		int fsize = (int) (3*sigma);
		double sum = 0;

		std::vector<float> kernel(2*fsize+1, 0);
		for(uint16_t p=0; p<kernel.size();p++)
		{
			kernel[p] = exp(-((p-fsize)*(p-fsize))/(sigma*sigma*2));
			sum += kernel[p];
		}
		for(uint16_t p=0; p<kernel.size();p++) kernel[p] /= sum;
		///////////////////////////////////////////////////

		long long int nslice = shape[0]*shape[1];
		long long int nstack = nslice*shape[2];
		float *outimg = (float*) malloc(nstack*sizeof(*outimg));

		for (int dim = 0; dim < ndims; dim++)
		{
			apply_3Dconvolution_splitdims(vectorimg+dim*nstack, shape, kernel, outimg);

			#pragma omp parallel for
			for(uint64_t pos = 0; pos < nstack; pos++)
				vectorimg[dim*nstack+pos] = outimg[pos];
		}

		free(outimg);

		return;
	}
	void apply_3DGaussianFilter2Vector(float* &ux, float* &uy, float* &uz, int shape[3], float sigma)
	{
		//Create Gaussian kernel
		///////////////////////////////////////////////////
		int fsize = (int) (3*sigma);
		double sum = 0;

		std::vector<float> kernel(2*fsize+1, 0);
		for(uint16_t p=0; p<kernel.size();p++)
		{
			kernel[p] = std::max(1e-12f, (float) exp(-((p-fsize)*(p-fsize))/(sigma*sigma*2)));
			sum += kernel[p];
		}
		for(uint16_t p=0; p<kernel.size();p++) kernel[p] /= sum;
		///////////////////////////////////////////////////

		long long int nslice = shape[0]*shape[1];
		long long int nstack = nslice*shape[2];

		float *outimg = (float*) malloc(nstack*sizeof(*outimg));

		apply_3Dconvolution_splitdims(ux, shape, kernel, outimg);
		std::swap(outimg, ux);
		apply_3Dconvolution_splitdims(uy, shape, kernel, outimg);
		std::swap(outimg, uy);
		apply_3Dconvolution_splitdims(uz, shape, kernel, outimg);
		std::swap(outimg, uz);

		free(outimg);

		return;
	}
	void apply_3DGaussianFilter2Interface(float* &image, int shape[3], float sigma, uint8_t* phase_image)
	{
		//Create Gaussian kernel
		///////////////////////////////////////////////////
		int fsize = (int) (3*sigma);
		double sum = 0;

		std::vector<float> kernel(2*fsize+1, 0);
		for(uint16_t p=0; p<kernel.size();p++)
		{
			kernel[p] = exp(-((p-fsize)*(p-fsize))/(sigma*sigma*2));
			sum += kernel[p];
		}
		for(uint16_t p=0; p<kernel.size();p++) kernel[p] /= sum;
		///////////////////////////////////////////////////

		long long int nslice = shape[0]*shape[1];
		long long int nstack = nslice*shape[2];

		float *outimg = (float*) malloc(nstack*sizeof(*outimg));

		apply_3Dconvolution_splitdims_interfaceonly(image, shape, kernel, outimg, phase_image);
		std::swap(outimg, image);
		free(outimg);

		return;
	}

    void apply_3DMedianFilter_cubic(float* &image, int shape[3], float radius)
    {
    	int nx = shape[0];
		int ny = shape[1];
		int nz = shape[2];
		long long int nslice = shape[0]*shape[1];
		long long int nstack = nslice*shape[2];

		int radiusz = std::min(radius, nz-1.f);
		if (shape[2] <= 1) radiusz = 0;

		int nkernel = (2*radius+1)*(2*radius+1)*(2*radiusz+1);
		int mpos = nkernel/2;

		float* output = (float*) malloc(nstack*sizeof(*output));

		#pragma omp parallel for
		for (long long int pos = 0; pos < nstack; pos++)
		{
			int z = pos/nslice;
			int y = (pos-z*nslice)/nx;
			int x = (pos-z*nslice-y*nx);

			float *rankkernel = (float*) malloc(nkernel*sizeof(*rankkernel));

			int kpos = 0;
			for (int zi = -radiusz; zi <= radiusz; zi++)
			{
				int z0 = z+zi;

				//reflective
				if (z0 < 0) z0 = -z0;
				if (z0 >= nz) z0 = 2*nz-z0-2;

				for (int yi = -radius; yi <= radius; yi++)
				{
					int y0 = y+yi;
					if (y0 < 0) y0 = -y0;
					if (y0 >= ny) y0 = 2*ny-y0-2;

					for (int xi = -radius; xi <= radius; xi++)
					{
						int x0 = x+xi;
						if (x0 < 0) x0 = -x0;
						if (x0 >= nx) x0 = 2*nx-x0-2;

						rankkernel[kpos] = image[z0*nslice+y0*nx+x0];

						kpos++;
					}
				}
			}

			//insertion sort
			for (kpos = 0; kpos < nkernel; kpos++)
			{
				float value = rankkernel[kpos];
				int kpos2 = kpos-1;

				for (;kpos2 >= 0 && value < rankkernel[kpos2]; kpos2--)
					rankkernel[kpos2+1] = rankkernel[kpos2];

				rankkernel[kpos2+1] = value;
			}

			output[pos] = rankkernel[mpos];

			free(rankkernel);
		}

		std::swap(image, output);
		free(output);

		return;
    }
    void apply_3DMedianFilter_spheric(float* &image, int shape[3], float radius)
	{
		int nx = shape[0];
		int ny = shape[1];
		int nz = shape[2];
		long long int nslice = shape[0]*shape[1];
		long long int nstack = nslice*shape[2];

		float radiusz = std::min(radius, nz-1.f);
		if (shape[2] <= 1) radiusz = 0;

		//////////////////////////////////////////////////////
		int rz0 = ceil(radiusz);
		int r0 = ceil(radius);
		float r0sq = radius*radius;
		int nkernel = 0;

		//count n-kernel members
		for (int zi = -rz0; zi <= rz0; zi++)
			for (int yi = -r0; yi <= r0; yi++)
				for (int xi = -r0; xi <= r0; xi++)
				{
					float r2 = xi*xi+yi*yi+zi*zi;
					if(r2 <= r0sq) nkernel++;
				}

		//position(s) of median value
		int mpos = nkernel/2;
		//int mpos2 = nkernel/2;
		//if ((nkernel%2) == 0) std::max(0, mpos2-1); //cannot happen
		//////////////////////////////////////////////////////

		float* output = (float*) malloc(nstack*sizeof(*output));

		#pragma omp parallel for
		for (long long int pos = 0; pos < nstack; pos++)
		{
			int z = pos/nslice;
			int y = (pos-z*nslice)/nx;
			int x = (pos-z*nslice-y*nx);

			float *rankkernel = (float*) malloc(nkernel*sizeof(*rankkernel));

			int kpos = 0;
			for (int zi = -rz0; zi <= rz0; zi++)
			{
				int z0 = z+zi;

				//reflective
				if (z0 < 0) z0 = -z0;
				else if (z0 >= nz) z0 = 2*nz-z0-2;

				for (int yi = -radius; yi <= radius; yi++)
				{
					int y0 = y+yi;
					if (y0 < 0) y0 = -y0;
					else if (y0 >= ny) y0 = 2*ny-y0-2;

					for (int xi = -radius; xi <= radius; xi++)
					{
						int x0 = x+xi;
						if (x0 < 0) x0 = -x0;
						else if (x0 >= nx) x0 = 2*nx-x0-2;

						if(xi*xi+yi*yi+zi*zi <= r0sq)
						{
							rankkernel[kpos] = image[z0*nslice+y0*nx+x0];
							kpos++;
						}
					}
				}
			}

			//insertion sort
			for (kpos = 0; kpos < nkernel; kpos++)
			{
				float value = rankkernel[kpos];
				int kpos2 = kpos-1;

				for (;kpos2 >= 0 && value < rankkernel[kpos2]; kpos2--)
					rankkernel[kpos2+1] = rankkernel[kpos2];

				rankkernel[kpos2+1] = value;
			}

			output[pos] = rankkernel[mpos];
			//if (mpos == mpos2) output[pos] = rankkernel[mpos];
			//else output[pos] = (rankkernel[mpos]+rankkernel[mpos2])*0.5f;

			free(rankkernel);
		}

		std::swap(image, output);
		free(output);

		return;
	}
    void apply2vector_3DMedianFilter_spheric(float* &vectorimage, int ndim, int shape[3], float radius)
	{
		int nx = shape[0];
		int ny = shape[1];
		int nz = shape[2];
		long long int nslice = shape[0]*shape[1];
		long long int nstack = nslice*shape[2];

		float radiusz = std::min(radius, nz-1.f);
		if (shape[2] <= 1) radiusz = 0;

		//////////////////////////////////////////////////////
		int rz0 = ceil(radiusz);
		int r0 = ceil(radius);
		float r0sq = radius*radius;
		int nkernel = 0;

		//count n-kernel members
		for (int zi = -rz0; zi <= rz0; zi++)
			for (int yi = -r0; yi <= r0; yi++)
				for (int xi = -r0; xi <= r0; xi++)
				{
					float r2 = xi*xi+yi*yi+zi*zi;
					if(r2 <= r0sq) nkernel++;
				}

		//position(s) of median value
		int mpos = nkernel/2;
		//int mpos2 = nkernel/2;
		//if ((nkernel%2) == 0) std::max(0, mpos2-1); //cannot happen
		//////////////////////////////////////////////////////

		float* output = (float*) malloc((ndim*nstack)*sizeof(*output));

		for (int dim = 0; dim < ndim; dim++)
		{
			#pragma omp parallel for
			for (long long int pos = 0; pos < nstack; pos++)
			{
				int z = pos/nslice;
				int y = (pos-z*nslice)/nx;
				int x = (pos-z*nslice-y*nx);

				float *rankkernel = (float*) malloc(nkernel*sizeof(*rankkernel));

				int kpos = 0;
				for (int zi = -rz0; zi <= rz0; zi++)
				{
					int z0 = z+zi;

					//reflective
					if (z0 < 0) z0 = -z0;
					else if (z0 >= nz) z0 = 2*nz-z0-2;

					for (int yi = -radius; yi <= radius; yi++)
					{
						int y0 = y+yi;
						if (y0 < 0) y0 = -y0;
						else if (y0 >= ny) y0 = 2*ny-y0-2;

						for (int xi = -radius; xi <= radius; xi++)
						{
							int x0 = x+xi;
							if (x0 < 0) x0 = -x0;
							else if (x0 >= nx) x0 = 2*nx-x0-2;

							if(xi*xi+yi*yi+zi*zi <= r0sq)
							{
								rankkernel[kpos] = vectorimage[dim*nstack + z0*nslice+y0*nx+x0];
								kpos++;
							}
						}
					}
				}

				//insertion sort
				for (kpos = 0; kpos < nkernel; kpos++)
				{
					float value = rankkernel[kpos];
					int kpos2 = kpos-1;

					for (;kpos2 >= 0 && value < rankkernel[kpos2]; kpos2--)
						rankkernel[kpos2+1] = rankkernel[kpos2];

					rankkernel[kpos2+1] = value;
				}

				output[dim*nstack + pos] = rankkernel[mpos];
				//if (mpos == mpos2) output[pos] = rankkernel[mpos];
				//else output[pos] = (rankkernel[mpos]+rankkernel[mpos2])*0.5f;

				free(rankkernel);
			}
		}

		std::swap(vectorimage, output);
		free(output);

		return;
	}
    void apply2vector_3DMedianFilter_spheric(float* &vectorimage, float *mask, int ndim, int shape[3], float radius)
	{
		int nx = shape[0];
		int ny = shape[1];
		int nz = shape[2];
		long long int nslice = shape[0]*shape[1];
		long long int nstack = nslice*shape[2];

		float radiusz = std::min(radius, nz-1.f);
		if (shape[2] <= 1) radiusz = 0;

		//////////////////////////////////////////////////////
		int rz0 = ceil(radiusz);
		int r0 = ceil(radius);
		float r0sq = radius*radius;
		int nkernel = 0;

		//count n-kernel members
		for (int zi = -rz0; zi <= rz0; zi++)
			for (int yi = -r0; yi <= r0; yi++)
				for (int xi = -r0; xi <= r0; xi++)
				{
					float r2 = xi*xi+yi*yi+zi*zi;
					if(r2 <= r0sq) nkernel++;
				}

		//position(s) of median value
		int mpos = nkernel/2;
		//int mpos2 = nkernel/2;
		//if ((nkernel%2) == 0) std::max(0, mpos2-1); //cannot happen
		//////////////////////////////////////////////////////

		float* output = (float*) malloc((ndim*nstack)*sizeof(*output));

		for (int dim = 0; dim < ndim; dim++)
		{
			#pragma omp parallel for
			for (long long int pos = 0; pos < nstack; pos++)
			{
				int z = pos/nslice;
				int y = (pos-z*nslice)/nx;
				int x = (pos-z*nslice-y*nx);

				if(mask[pos] != 0.0f)
				{
					float *rankkernel = (float*) malloc(nkernel*sizeof(*rankkernel));

					int kpos = 0;
					for (int zi = -rz0; zi <= rz0; zi++)
					{
						int z0 = z+zi;

						//reflective
						if (z0 < 0) z0 = -z0;
						else if (z0 >= nz) z0 = 2*nz-z0-2;

						for (int yi = -radius; yi <= radius; yi++)
						{
							int y0 = y+yi;
							if (y0 < 0) y0 = -y0;
							else if (y0 >= ny) y0 = 2*ny-y0-2;

							for (int xi = -radius; xi <= radius; xi++)
							{
								int x0 = x+xi;
								if (x0 < 0) x0 = -x0;
								else if (x0 >= nx) x0 = 2*nx-x0-2;

								if(xi*xi+yi*yi+zi*zi <= r0sq)
								{
									rankkernel[kpos] = vectorimage[dim*nstack + z0*nslice+y0*nx+x0];
									kpos++;
								}
							}
						}
					}

					//insertion sort
					for (kpos = 0; kpos < nkernel; kpos++)
					{
						float value = rankkernel[kpos];
						int kpos2 = kpos-1;

						for (;kpos2 >= 0 && value < rankkernel[kpos2]; kpos2--)
							rankkernel[kpos2+1] = rankkernel[kpos2];

						rankkernel[kpos2+1] = value;
					}

					output[dim*nstack + pos] = rankkernel[mpos];
					//if (mpos == mpos2) output[pos] = rankkernel[mpos];
					//else output[pos] = (rankkernel[mpos]+rankkernel[mpos2])*0.5f;

					free(rankkernel);
				}
				else
					output[dim*nstack + pos] = vectorimage[dim*nstack + pos];
			}
		}

		std::swap(vectorimage, output);
		free(output);

		return;
	}

    void apply_3DImageFilter_2frame(float* &image0, float *&image1, int shape[3], float sigma, std::string filtername)
	{
    	if (filtername == "gaussian")
    	{
			//Create Gaussian kernel
			///////////////////////////////////////////////////
			int fsize = (int) (3*sigma);
			double sum = 0;

			std::vector<float> kernel(2*fsize+1, 0);
			for(uint16_t p=0; p<kernel.size();p++)
			{
				kernel[p] = exp(-((p-fsize)*(p-fsize))/(sigma*sigma*2));
				sum += kernel[p];
			}
			for(uint16_t p=0; p<kernel.size();p++)kernel[p] /= sum;
			///////////////////////////////////////////////////

			long long int nslice = shape[0]*shape[1];
			long long int nstack = nslice*shape[2];

			float *outimg = (float*) malloc(nstack*sizeof(*outimg));
			apply_3Dconvolution_splitdims(image0, shape, kernel, outimg);

			std::swap(image0, outimg);

			apply_3Dconvolution_splitdims(image1, shape, kernel, outimg);
			std::swap(image1, outimg);

			free(outimg);
    	}
    	else if (filtername == "median_simple")
    	{
    		apply_3DMedianFilter_cubic(image0, shape, sigma);
    		apply_3DMedianFilter_cubic(image1, shape, sigma);
    	}
    	else if (filtername == "median")
    	{
    		apply_3DMedianFilter_spheric(image0, shape, sigma);
    		apply_3DMedianFilter_spheric(image1, shape, sigma);
    	}
    	else if (filtername == "none")
    	{
			/*std::vector<float> kernel = {1.0f};

			long long int nslice = shape[0]*shape[1];
			long long int nstack = nslice*shape[2];

			float *outimg = (float*) malloc(nstack*sizeof(*outimg));

			//apply_3Dconvolution_splitdims(image0, shape, kernel, outimg);
			//std::swap(image0, outimg);

			free(outimg);*/
    	}
    	else std::cout << "Warning! " << filtername << " is an unknown 2frame filter!" << std::endl;

		return;
	}

    void maxfilter27neighbourhood(float* &image, int shape[3])
    {
        int nx = shape[0]; int ny = shape[1]; int nz = shape[2];
        long long int nslice = nx*ny;
        long long int nstack = nz*nslice;

        float* output = (float*) calloc(nstack, sizeof(*output));

        #pragma omp parallel for
        for(long long int idx = 0; idx < nstack; idx++)
        {
            int z = idx/nslice;
            int y = (idx-z*nslice)/nx;
            int x = idx-z*nslice-y*nx;

            float maxval = image[idx];

            for (int r = -1; r <= 1; r++)
            for (int q = -1; q <= 1; q++)
            for (int p = -1; p <= 1; p++)
            {
                if (r == 0 && p == 0 && q == 0) continue;

                int x1 = std::max(0,std::min(nx-1, x+p));
                int y1 = std::max(0,std::min(ny-1, y+q));
                int z1 = std::max(0,std::min(nz-1, z+r));

                maxval = std::max(maxval, image[z1*nslice+y1*nx+x1]);
            }

            output[idx] = maxval;
        }

        std::swap(output, image);
        free(output);

        return;
    }

    //////////////////////////////////////////////////////////////////////////////////////////////////////////
    float* calc_blurred_derivative(int dim, float* image, int shape[3], int stencil, int fd_order)
    {
        //Create a Gaussian kernel
        ///////////////////////////////////////////////////
        int fsize = (stencil-3)/2;
        float sigma = fsize/3.f;

        double normalizer = .5;

        //assuming 2nd order FD, when higher increase stencil
        if (fd_order == 4)
        {
            stencil += 2;
            normalizer = 1./12.;
        }
        else if (fd_order == 6)
        {
            stencil += 4;
            normalizer = 1./60.;
        }

        double sum = 0;
        std::vector<float> gkernel(2*fsize+1, 0);
        for(uint16_t p=0; p<gkernel.size();p++)
        {
            gkernel[p] = exp(-((p-fsize)*(p-fsize))/(sigma*sigma*2));
            sum += gkernel[p];
        }
        for(uint16_t p=0; p<gkernel.size();p++) gkernel[p] /= sum;
        normalizer /= sum;

        std::vector<float> diff_kernel(stencil, 0);

        if (fd_order == 4)
        {
            for(uint16_t p=0; p<diff_kernel.size();p++)
            {
                diff_kernel[p] = normalizer*(
                   8.*exp(-((p-(fsize+2)-1)*(p-(fsize+2)-1))/(sigma*sigma*2))
                 - 8.*exp(-((p-(fsize+2)+1)*(p-(fsize+2)+1))/(sigma*sigma*2))
                 - 1.*exp(-((p-(fsize+2)-2)*(p-(fsize+2)-2))/(sigma*sigma*2))
                 + 1.*exp(-((p-(fsize+2)+2)*(p-(fsize+2)+2))/(sigma*sigma*2)));
            }
        }
        else if (fd_order == 6)
        {
            for(uint16_t p=0; p<diff_kernel.size();p++)
            {
                diff_kernel[p] = normalizer*(
                   45.*exp(-((p-(fsize+3)-1)*(p-(fsize+3)-1))/(sigma*sigma*2))
                 - 45.*exp(-((p-(fsize+3)+1)*(p-(fsize+3)+1))/(sigma*sigma*2))
                 - 9.*exp(-((p-(fsize+3)-2)*(p-(fsize+3)-2))/(sigma*sigma*2))
                 + 9.*exp(-((p-(fsize+3)+2)*(p-(fsize+3)+2))/(sigma*sigma*2))
                 + 1.*exp(-((p-(fsize+3)-3)*(p-(fsize+3)-3))/(sigma*sigma*2))
                 - 1.*exp(-((p-(fsize+3)+3)*(p-(fsize+3)+3))/(sigma*sigma*2)));
            }
        }
        else
        {
            for(uint16_t p=0; p<diff_kernel.size();p++)
            {
                diff_kernel[p] = normalizer*exp(-((p-(fsize+1)-1)*(p-(fsize+1)-1))/(sigma*sigma*2))
                 - normalizer*exp(-((p-(fsize+1)+1)*(p-(fsize+1)+1))/(sigma*sigma*2));
            }

        }

        //Kernel is separable...
        ///////////////////////////////////////////////////
        float *output1, *output2;

        if (dim != 0) output1 = apply_1Dconvolution(0, image, shape, gkernel);
        else output1 = apply_1Dconvolution(0, image, shape, diff_kernel);

        if (dim != 1) output2 = apply_1Dconvolution(1, output1, shape, gkernel);
        else output2 = apply_1Dconvolution(1, output1, shape, diff_kernel);
        free(output1);

        if (dim != 2) output1 = apply_1Dconvolution(2, output2, shape, gkernel);
        else output1 = apply_1Dconvolution(2, output2, shape, diff_kernel);
        free(output2);
        ///////////////////////////////////////////////////

        return output1;
    }
    float* calc_Farid_derivative(float* image, int shape[3], int dim, int radius, int order, bool use_interpolator)
    {
        //Following Farid and Simoncelli 2004: "Differentation of Discrete Multidimensional Signals"
        //
        // dim = direction of derivative
        // radius = sample number
        // order = order of the derivative
        // blur = 0th order in orthogonal directions or not
        //

        long long int nslice = shape[0]*shape[1];
        long long int nstack = shape[2]*nslice;

        float *output1, *output2;

        radius = abs(radius); order = abs(order);
        std::vector<float> pkernel, dkernel;

        if (order == 0)
        {
            //just the interpolator
            if (radius == 1) pkernel = { 0.229879, 0.540242, 0.229879};
            else if (radius == 2) pkernel = { 0.037659, 0.249153, 0.426375, 0.249153, 0.037659};
            else if (radius == 3) pkernel = { 0.004711, 0.069321, 0.245410, 0.361117, 0.245410, 0.069321, 0.004711};
            else if (radius == 4) pkernel = { 0.000721, 0.015486, 0.090341, 0.234494, 0.317916, 0.234494, 0.090341, 0.015486, 0.000721};

            //correct imprecision
            if (radius == 2) for (int i = 0; i < pkernel.size(); i++) pkernel[i] /= 0.999999;

            dkernel = pkernel;
        }
        if (order == 1)
        {
            if (radius == 1)
            {
                pkernel = { 0.229879, 0.540242, 0.229879};
                dkernel = {-0.425287, 0.000000, 0.425287};
            }
            else if (radius == 2)
            {
                pkernel = { 0.037659, 0.249153, 0.426375, 0.249153, 0.037659};
                dkernel = {-0.109604,-0.276691, 0.000000, 0.276691, 0.109604};

                for (int i = 0; i < pkernel.size(); i++) pkernel[i] /= 0.999999;
            }
            else if (radius == 3)
            {
                pkernel = { 0.004711, 0.069321, 0.245410, 0.361117, 0.245410, 0.069321, 0.004711};
                dkernel = {-0.018708,-0.125376,-0.193091, 0.000000, 0.193091, 0.125376, 0.018708};
            }
            else if (radius == 4)
            {
                pkernel = { 0.000721, 0.015486, 0.090341, 0.234494, 0.317916, 0.234494, 0.090341, 0.015486, 0.000721};
                dkernel = {-0.003059,-0.035187,-0.118739,-0.143928, 0.000000, 0.143928, 0.118739, 0.035187, 0.003059};
            }
        }
        else if (order == 2)
        {
            if (radius < 2)
            {
                std::cout << "Warning! Minimal radius of 2 required for 2nd order derivatives!" << std::endl;
                radius = 2;
            }

            if (radius == 2)
            {
            pkernel = { 0.030320, 0.249724, 0.439911, 0.249724, 0.030320};
            //dkernel1 = {-0.104550,-0.292315, 0.000000, 0.292315, 0.104550};
            dkernel = { 0.232905, 0.002668,-0.471147, 0.002668, 0.232905};

            for (int i = 0; i < pkernel.size(); i++){
                pkernel[i] /= 0.999999;
                //dkernel[i] -= (-9.9977478385e-7)/5.;
                }
            }
            else if (radius == 3)
            {
                pkernel = { 0.003992, 0.067088, 0.246217, 0.365406, 0.246217, 0.067088, 0.003992};
                //dkernel1 = {-0.015964,-0.121482,-0.193357, 0.000000, 0.193357, 0.121482, 0.015964};
                dkernel = { 0.054174, 0.147520,-0.057325,-0.288736,-0.057325, 0.147520, 0.054174};
                //dkernel3 = {-0.111680, 0.012759, 0.336539, 0.000000,-0.336539,-0.012759, 0.111680};

                for (int i = 0; i < pkernel.size(); i++) dkernel[i] -= 2.8759295999999987e-7;
            }
            else if (radius == 4)
            {
                pkernel = { 0.000721, 0.015486, 0.090341, 0.234494, 0.317916, 0.234494, 0.090341, 0.015486, 0.000721};
                //dkernel1 = {-0.003059,-0.035187,-0.118739,-0.143928, 0.000000, 0.143928, 0.118739, 0.035187, 0.003059};
                dkernel = {0.010257, 0.061793,0.085598,-0.061661,-0.191974,-0.061661,0.085598,0.061793,0.010257};
                //dkernel3 = {-0.027205,-0.065929,0.053614,0.203718, 0.000000,-0.203718,-0.053614,0.065929,0.027205};
            }
        }
        else if (order == 3)
        {
            if (radius < 3)
            {
                std::cout << "Warning! Minimal radius of 3 required for 3rd order derivatives!" << std::endl;
                radius = 3;
            }

            if (radius == 3)
            {
                pkernel = { 0.003992, 0.067088, 0.246217, 0.365406, 0.246217, 0.067088, 0.003992};
                //dkernel1 = {-0.015964,-0.121482,-0.193357, 0.000000, 0.193357, 0.121482, 0.015964};
                //dkernel2 = { 0.054174, 0.147520,-0.057325,-0.288736,-0.057325, 0.147520, 0.054174};
                dkernel = {-0.111680, 0.012759, 0.336539, 0.000000,-0.336539,-0.012759, 0.111680};
            }
            else if (radius == 4)
            {
                pkernel = { 0.000721, 0.015486, 0.090341, 0.234494, 0.317916, 0.234494, 0.090341, 0.015486, 0.000721};
                //dkernel1 = {-0.003059,-0.035187,-0.118739,-0.143928, 0.000000, 0.143928, 0.118739, 0.035187, 0.003059};
                //dkernel2 = {0.010257, 0.061793,0.085598,-0.061661,-0.191974,-0.061661,0.085598,0.061793,0.010257};
                dkernel = {-0.027205,-0.065929,0.053614,0.203718, 0.000000,-0.203718,-0.053614,0.065929,0.027205};
            }
        }

        if (radius > 4 || order > 3 || radius == 0)
        {
            std::cout << "Warning! Farid derivatives only supported until radius 4, 3rd order derivative! Returning all zeros!" << std::endl;
            output1 = (float*) calloc(nstack, sizeof(*output1));
            return output1;
        }

        //Kernel is separable...
        ///////////////////////////////////////////////////
        if (!use_interpolator)
        {
            output1 = apply_1Dconvolution(dim, image, shape, dkernel);
            return output1;
        }

        /*output1 = apply_1Dconvolution(0, image, shape, pkernel);
        output2 = apply_1Dconvolution(1, output1, shape, pkernel);
        free(output1);
        output1 = apply_1Dconvolution(2, output2, shape, pkernel);
        free(output2);
        if (dim == 0) output2 = apply_1Dconvolution(0, output1, shape, dkernel);
        else if (dim == 1) output2 = apply_1Dconvolution(1, output1, shape, dkernel);
        else output2 = apply_1Dconvolution(2, output1, shape, dkernel);
        free(output1);
        return output2;*/


        if (dim != 0) output1 = apply_1Dconvolution(0, image, shape, pkernel);
        else
            output1 = apply_1Dconvolution(0, image, shape, dkernel);

        if (dim != 1) output2 = apply_1Dconvolution(1, output1, shape, pkernel);
        else output2 = apply_1Dconvolution(1, output1, shape, dkernel);
        free(output1);

        if (dim != 2) output1 = apply_1Dconvolution(2, output2, shape, pkernel);
        else output1 = apply_1Dconvolution(2, output2, shape, dkernel);

        free(output2);
        ///////////////////////////////////////////////////

        return output1;
    }
    //////////////////////////////////////////////////////////////////////////////////////////////////////////
}

