#ifndef DISPLACEMENT_DERIVATIVES_H
#define DISPLACEMENT_DERIVATIVES_H

#include <iostream>
#include <vector>
#include <string.h>
#include <omp.h>
#include <math.h>

#include "polar_decomposition.h"
#include "../Geometry/filtering.h"
#include "../Geometry/hdcommunication.h"

namespace derive
{
    float _determinant3x3(float M[9])
	{
		//assuming row first
		return (M[0]*M[4]*M[8]) + (M[1]*M[5]*M[6]) + (M[2]*M[3]*M[7]) - (M[6]*M[4]*M[2]) - (M[7]*M[5]*M[0]) - (M[8]*M[3]*M[1]);
	}
	void __calc_invariants(float Exx, float Eyy, float Ezz, float Exy, float Exz, float Eyz, double &outI1, double &outI2, double &outI3)
	{
		outI1 = Exx+Eyy+Ezz;
		outI2 = Exx*Eyy-Exy*Exy+Exx*Ezz-Exz*Exz+Eyy*Ezz-Eyz*Eyz;
		float M[9] = {Exx, Exy, Exz, Exy, Eyy, Eyz, Exz,Eyz,Ezz};
		outI3 = _determinant3x3(M);
		return;
	}

	float* calc_divergence(float* ux, float *uy, float* uz, int shape[3], std::string finitedifference_type="fourthorder")
	{
		int nx = shape[0]; int ny = shape[1]; int nz = shape[2];
		long long int nslice = nx*ny;
		long long int nstack = nz*nslice;

		float *output = (float*) calloc(nstack,sizeof(*output));

		float normalizer = 1.f/2.f;

		if (finitedifference_type == "centraldifference" || finitedifference_type == "secondorder")
			normalizer = 1.f/2.f;
		else if (finitedifference_type == "Barron" || finitedifference_type == "fourthorder")
			normalizer = 1.f/12.f;
		else
			std::cout << "Warning! unknown finite difference for plot_divergence. Using central difference." << std::endl;

		#pragma omp parallel for
		for (long long int pos = 0; pos < nstack; pos++)
		{
			int z = pos/nslice;
			int y = (pos-z*nslice)/nx;
			int x = pos-z*nslice-y*nx;

			int zp = z+1; int zn = z-1;
			int yp = y+1; int yn = y-1;
			int xp = x+1; int xn = x-1;
			int zp2 = z+2; int zn2 = z-2;
			int yp2 = y+2; int yn2 = y-2;
			int xp2 = x+2; int xn2 = x-2;

			//Reflective boundary conditions (mirrored on first/last value)
			if (zp == nz) zp -= 2; if (zn < 0) zn = 1;
			if (yp == ny) yp -= 2; if (yn < 0) yn = 1;
			if (xp == nx) xp -= 2; if (xn < 0) xn = 1;
			if (zp2 >= nz) zp2 -= 2*nz-zp2-2; if (zn2 < 0) zn2 = -zn2;
			if (yp2 >= ny) yp2 -= 2*ny-yp2-2; if (yn2 < 0) yn2 = -yn2;
			if (xp2 >= nx) xp2 -= 2*nx-xp2-2; if (xn2 < 0) xn2 = -xn2;

			float divergence = 0.0f;

			if (finitedifference_type == "Barron" || finitedifference_type == "fourthorder")
			{
				//x-derivative of ux displacement
				float val_xn2 = ux[z*nslice + y*nx + xn2];
				float val_xn  = ux[z*nslice + y*nx + xn ];
				float val_xp  = ux[z*nslice + y*nx + xp ];
				float val_xp2 = ux[z*nslice + y*nx + xp2];

				float eps_xx = normalizer*(val_xn2 -8*val_xn + 8*val_xp - val_xp2);

				//y-derivative of uy displacement
				float val_yn2 = uy[z*nslice + yn2*nx + x];
				float val_yn  = uy[z*nslice + yn*nx + x ];
				float val_yp  = uy[z*nslice + yp*nx + x ];
				float val_yp2 = uy[z*nslice + yp2*nx + x];

				float eps_yy = normalizer*(val_yn2 -8*val_yn + 8*val_yp - val_yp2);

				//z-derivative of uz displacement
				float val_zn2 = uz[zn2*nslice + y*nx + x];
				float val_zn  = uz[zn*nslice  + y*nx + x ];
				float val_zp  = uz[zp*nslice  + y*nx + x ];
				float val_zp2 = uz[zp2*nslice + y*nx + x];

				float eps_zz = normalizer*(val_zn2 -8*val_zn + 8*val_zp - val_zp2);

				divergence = eps_xx+eps_yy+eps_zz;
			}
			else
			{
				//x-derivative of ux displacement
				float val_xn  = ux[z*nslice + y*nx + xn ];
				float val_xp  = ux[z*nslice + y*nx + xp ];

				float eps_xx = normalizer*(-val_xn + val_xp);

				//y-derivative of uy displacement
				float val_yn  = uy[z*nslice + yn*nx + x ];
				float val_yp  = uy[z*nslice + yp*nx + x ];

				float eps_yy = normalizer*(-val_yn + val_yp);

				//z-derivative of uz displacement
				float val_zn  = uz[zn*nslice  + y*nx + x ];
				float val_zp  = uz[zp*nslice  + y*nx + x ];

				float eps_zz = normalizer*(-val_zn + val_zp);

				divergence = eps_xx+eps_yy+eps_zz;
			}

			output[pos] = divergence;
		}

		return output;
	}
	float* calc_volumetric_strain(float* ux, float *uy, float* uz, int shape[3], std::string finitedifference_type="fourthorder")
	{
		int nx = shape[0]; int ny = shape[1]; int nz = shape[2];
		long long int nslice = nx*ny;
		long long int nstack = nz*nslice;

		float *output = (float*) calloc(nstack,sizeof(*output));

		float normalizer = 1.f/2.f;

		if (finitedifference_type == "centraldifference" || finitedifference_type == "secondorder")
			normalizer = 1.f/2.f;
		else if (finitedifference_type == "Barron" || finitedifference_type == "fourthorder")
			normalizer = 1.f/12.f;
		else
			std::cout << "Warning! unknown finite difference for plot_divergence. Using central difference." << std::endl;

		#pragma omp parallel for
		for (long long int pos = 0; pos < nstack; pos++)
		{
			int z = pos/nslice;
			int y = (pos-z*nslice)/nx;
			int x = pos-z*nslice-y*nx;

			int zp = z+1; int zn = z-1;
			int yp = y+1; int yn = y-1;
			int xp = x+1; int xn = x-1;
			int zp2 = z+2; int zn2 = z-2;
			int yp2 = y+2; int yn2 = y-2;
			int xp2 = x+2; int xn2 = x-2;

			//Reflective boundary conditions (mirrored on first/last value)
			if (zp == nz) zp -= 2; if (zn < 0) zn = 1;
			if (yp == ny) yp -= 2; if (yn < 0) yn = 1;
			if (xp == nx) xp -= 2; if (xn < 0) xn = 1;
			if (zp2 >= nz) zp2 -= 2*nz-zp2-2; if (zn2 < 0) zn2 = -zn2;
			if (yp2 >= ny) yp2 -= 2*ny-yp2-2; if (yn2 < 0) yn2 = -yn2;
			if (xp2 >= nx) xp2 -= 2*nx-xp2-2; if (xn2 < 0) xn2 = -xn2;

			float strain = 0.0f;

			if (finitedifference_type == "Barron" || finitedifference_type == "fourthorder")
			{
				//x-derivative of ux displacement
				float val_xn2 = ux[z*nslice + y*nx + xn2];
				float val_xn  = ux[z*nslice + y*nx + xn ];
				float val_xp  = ux[z*nslice + y*nx + xp ];
				float val_xp2 = ux[z*nslice + y*nx + xp2];

				float eps_xx = normalizer*(val_xn2 -8*val_xn + 8*val_xp - val_xp2);

				//y-derivative of ux displacement
				float val_yn2 = ux[z*nslice + yn2*nx + x];
				float val_yn  = ux[z*nslice + yn*nx + x ];
				float val_yp  = ux[z*nslice + yp*nx + x ];
				float val_yp2 = ux[z*nslice + yp2*nx + x];

				float eps_xy = normalizer*(val_yn2 -8*val_yn + 8*val_yp - val_yp2);

				//z-derivative of ux displacement
				float val_zn2 = ux[zn2*nslice + y*nx + x];
				float val_zn  = ux[zn*nslice  + y*nx + x ];
				float val_zp  = ux[zp*nslice  + y*nx + x ];
				float val_zp2 = ux[zp2*nslice + y*nx + x];

				float eps_xz = normalizer*(val_zn2 -8*val_zn + 8*val_zp - val_zp2);

				//x-derivative of uy displacement
				val_xn2 = uy[z*nslice + y*nx + xn2];
				val_xn  = uy[z*nslice + y*nx + xn ];
				val_xp  = uy[z*nslice + y*nx + xp ];
				val_xp2 = uy[z*nslice + y*nx + xp2];

				float eps_yx = normalizer*(val_xn2 -8*val_xn + 8*val_xp - val_xp2);

				//y-derivative of uy displacement
				val_yn2 = uy[z*nslice + yn2*nx + x];
				val_yn  = uy[z*nslice + yn*nx + x ];
				val_yp  = uy[z*nslice + yp*nx + x ];
				val_yp2 = uy[z*nslice + yp2*nx + x];

				float eps_yy = normalizer*(val_yn2 -8*val_yn + 8*val_yp - val_yp2);

				//z-derivative of uy displacement
				val_zn2 = uy[zn2*nslice + y*nx + x];
				val_zn  = uy[zn*nslice  + y*nx + x ];
				val_zp  = uy[zp*nslice  + y*nx + x ];
				val_zp2 = uy[zp2*nslice + y*nx + x];

				float eps_yz = normalizer*(val_zn2 -8*val_zn + 8*val_zp - val_zp2);

				//x-derivative of uz displacement
				val_xn2 = uz[z*nslice + y*nx + xn2];
				val_xn  = uz[z*nslice + y*nx + xn ];
				val_xp  = uz[z*nslice + y*nx + xp ];
				val_xp2 = uz[z*nslice + y*nx + xp2];

				float eps_zx = normalizer*(val_xn2 -8*val_xn + 8*val_xp - val_xp2);

				//y-derivative of uz displacement
				val_yn2 = uz[z*nslice + yn2*nx + x];
				val_yn  = uz[z*nslice + yn*nx + x ];
				val_yp  = uz[z*nslice + yp*nx + x ];
				val_yp2 = uz[z*nslice + yp2*nx + x];

				float eps_zy = normalizer*(val_yn2 -8*val_yn + 8*val_yp - val_yp2);

				//z-derivative of uz displacement
				val_zn2 = uz[zn2*nslice + y*nx + x];
				val_zn  = uz[zn*nslice  + y*nx + x ];
				val_zp  = uz[zp*nslice  + y*nx + x ];
				val_zp2 = uz[zp2*nslice + y*nx + x];

				float eps_zz = normalizer*(val_zn2 -8*val_zn + 8*val_zp - val_zp2);

				//volumetric strain calculated as det(F)-1

				strain = eps_xx*eps_yy*eps_zz + eps_xy*eps_yz*eps_zx + eps_xz*eps_yx*eps_zy - eps_zx*eps_yy*eps_xz - eps_zy*eps_yz*eps_xx - eps_zz*eps_yx*eps_xy-1;
			}
			else
			{
				//x-derivative of ux displacement
				float val_xn  = ux[z*nslice + y*nx + xn ];
				float val_xp  = ux[z*nslice + y*nx + xp ];

				float eps_xx = normalizer*(-val_xn + val_xp);

				//y-derivative of ux displacement
				float val_yn  = ux[z*nslice + yn*nx + x ];
				float val_yp  = ux[z*nslice + yp*nx + x ];

				float eps_xy = normalizer*(-val_yn + val_yp);

				//z-derivative of ux displacement
				float val_zn  = ux[zn*nslice  + y*nx + x ];
				float val_zp  = ux[zp*nslice  + y*nx + x ];

				float eps_xz = normalizer*(-val_zn + val_zp);

				//x-derivative of uy displacement
				val_xn  = uy[z*nslice + y*nx + xn ];
				val_xp  = uy[z*nslice + y*nx + xp ];

				float eps_yx = normalizer*(-val_xn + val_xp);

				//y-derivative of uy displacement
				val_yn  = uy[z*nslice + yn*nx + x ];
				val_yp  = uy[z*nslice + yp*nx + x ];

				float eps_yy = normalizer*(-val_yn + val_yp);

				//z-derivative of uy displacement
				val_zn  = uy[zn*nslice  + y*nx + x ];
				val_zp  = uy[zp*nslice  + y*nx + x ];

				float eps_yz = normalizer*(-val_zn + val_zp);

				//x-derivative of uz displacement
				val_xn  = uz[z*nslice + y*nx + xn ];
				val_xp  = uz[z*nslice + y*nx + xp ];

				float eps_zx = normalizer*(-val_xn + val_xp);

				//y-derivative of uy displacement
				val_zn  = uz[z*nslice + yn*nx + x ];
				val_zp  = uz[z*nslice + yp*nx + x ];

				float eps_zy = normalizer*(-val_yn + val_yp);

				//z-derivative of uz displacement
				val_zn  = uz[zn*nslice  + y*nx + x ];
				val_zp  = uz[zp*nslice  + y*nx + x ];

				float eps_zz = normalizer*(-val_zn + val_zp);

				strain = eps_xx*eps_yy*eps_zz + eps_xy*eps_yz*eps_zx + eps_xz*eps_yx*eps_zy - eps_zx*eps_yy*eps_xz - eps_zy*eps_yz*eps_xx - eps_zz*eps_yx*eps_xy-1;
			}

			output[pos] = strain;
		}

		return output;
	}
	float* calc_directional_stretch(float* ui, int shape[3])
	{
		int nx = shape[0]; int ny = shape[1]; int nz = shape[2];
		long long int nslice = nx*ny;
		long long int nstack = nz*nslice;

		float *output = (float*) calloc(nstack,sizeof(*output));

		//calculate center of gravity
		double shift = 0.0;

		#pragma omp parallel for reduction(+: shift)
		for (long long int pos = 0; pos < nstack; pos++)
            shift += ui[pos];

        double cog = nz/2.-0.5 + shift;

        #pragma omp parallel for
		for (long long int pos = 0; pos < nstack; pos++)
		{
            int z = pos/nslice;
            double dist = z-cog;
            double stretch = ui[pos];

            if (dist > 0)
                stretch = (dist+stretch)/dist;
            else
                stretch = fabs(stretch);

			output[pos] = stretch;
        }

		return output;
	}
	float* calc_maximum_shear_strain(float* ux, float *uy, float* uz, int shape[3], std::string finitedifference_type="fourthorder")
	{
		int nx = shape[0]; int ny = shape[1]; int nz = shape[2];
		long long int nslice = nx*ny;
		long long int nstack = nz*nslice;

		float *output = (float*) calloc(nstack,sizeof(*output));

		float normalizer = 1.f/2.f;

		if (finitedifference_type == "centraldifference" || finitedifference_type == "secondorder")
			normalizer = 1.f/2.f;
		else if (finitedifference_type == "Barron" || finitedifference_type == "fourthorder")
			normalizer = 1.f/12.f;
		else
			std::cout << "Warning! unknown finite difference for plot_divergence. Using central difference." << std::endl;

		#pragma omp parallel for
		for (long long int pos = 0; pos < nstack; pos++)
		{
			int z = pos/nslice;
			int y = (pos-z*nslice)/nx;
			int x = pos-z*nslice-y*nx;

			int zp = z+1; int zn = z-1;
			int yp = y+1; int yn = y-1;
			int xp = x+1; int xn = x-1;
			int zp2 = z+2; int zn2 = z-2;
			int yp2 = y+2; int yn2 = y-2;
			int xp2 = x+2; int xn2 = x-2;

			//Reflective boundary conditions (mirrored on first/last value)
			if (zp == nz) zp -= 2; if (zn < 0) zn = 1;
			if (yp == ny) yp -= 2; if (yn < 0) yn = 1;
			if (xp == nx) xp -= 2; if (xn < 0) xn = 1;
			if (zp2 >= nz) zp2 -= 2*nz-zp2-2; if (zn2 < 0) zn2 = -zn2;
			if (yp2 >= ny) yp2 -= 2*ny-yp2-2; if (yn2 < 0) yn2 = -yn2;
			if (xp2 >= nx) xp2 -= 2*nx-xp2-2; if (xn2 < 0) xn2 = -xn2;

			float shear_strain = 0.0f;

			if (finitedifference_type == "Barron" || finitedifference_type == "fourthorder")
			{
				//x-derivative of ux displacement
				float val_xn2 = ux[z*nslice + y*nx + xn2];
				float val_xn  = ux[z*nslice + y*nx + xn ];
				float val_xp  = ux[z*nslice + y*nx + xp ];
				float val_xp2 = ux[z*nslice + y*nx + xp2];

				float eps_xx = normalizer*(val_xn2 -8*val_xn + 8*val_xp - val_xp2);

				//y-derivative of uy displacement
				float val_yn2 = uy[z*nslice + yn2*nx + x];
				float val_yn  = uy[z*nslice + yn*nx + x ];
				float val_yp  = uy[z*nslice + yp*nx + x ];
				float val_yp2 = uy[z*nslice + yp2*nx + x];

				float eps_yy = normalizer*(val_yn2 -8*val_yn + 8*val_yp - val_yp2);

				//x-derivative of uy displacement
				val_xn2 = uy[z*nslice + y*nx + xn2];
				val_xn  = uy[z*nslice + y*nx + xn ];
				val_xp  = uy[z*nslice + y*nx + xp ];
				val_xp2 = uy[z*nslice + y*nx + xp2];

				float eps_yx = normalizer*(val_xn2 -8*val_xn + 8*val_xp - val_xp2);

				//z-derivative of uz displacement
				float val_zn2 = uz[zn2*nslice + y*nx + x];
				float val_zn  = uz[zn*nslice  + y*nx + x ];
				float val_zp  = uz[zp*nslice  + y*nx + x ];
				float val_zp2 = uz[zp2*nslice + y*nx + x];

				float eps_zz = normalizer*(val_zn2 -8*val_zn + 8*val_zp - val_zp2);

				//x-derivative of uz displacement
				val_xn2 = uz[z*nslice + y*nx + xn2];
				val_xn  = uz[z*nslice + y*nx + xn ];
				val_xp  = uz[z*nslice + y*nx + xp ];
				val_xp2 = uz[z*nslice + y*nx + xp2];

				float eps_zx = normalizer*(val_xn2 -8*val_xn + 8*val_xp - val_xp2);

				//y-derivative of uz displacement
				val_yn2 = uz[z*nslice + yn2*nx + x];
				val_yn  = uz[z*nslice + yn*nx + x ];
				val_yp  = uz[z*nslice + yp*nx + x ];
				val_yp2 = uz[z*nslice + yp2*nx + x];

				float eps_zy = normalizer*(val_yn2 -8*val_yn + 8*val_yp - val_yp2);

				shear_strain = 1.f/3.f*sqrt(2.f*(eps_xx-eps_yy)*(eps_xx-eps_yy) + 2.f*(eps_xx-eps_zz)*(eps_xx-eps_zz) + 2.f*(eps_yy-eps_zz)*(eps_yy-eps_zz)
						+ 12.f*eps_yx*eps_yx + 12.f*eps_zx*eps_zx + 12.f*eps_zy*eps_zy);
			}

			else
			{
				//x-derivative of ux displacement
				float val_xn  = ux[z*nslice + y*nx + xn ];
				float val_xp  = ux[z*nslice + y*nx + xp ];

				float eps_xx = normalizer*(-val_xn + val_xp);

				//y-derivative of uy displacement
				float val_yn  = uy[z*nslice + yn*nx + x ];
				float val_yp  = uy[z*nslice + yp*nx + x ];

				float eps_yy = normalizer*(-val_yn + val_yp);

				//x-derivative of uy displacement
				val_xn  = uy[z*nslice + y*nx + xn ];
				val_xp  = uy[z*nslice + y*nx + xp ];

				float eps_yx = normalizer*(-val_xn + val_xp);

				//z-derivative of uz displacement
				float val_zn  = uz[zn*nslice  + y*nx + x ];
				float val_zp  = uz[zp*nslice  + y*nx + x ];

				float eps_zz = normalizer*(-val_zn + val_zp);

				//x-derivative of uz displacement
				val_xn  = uz[z*nslice + y*nx + xn ];
				val_xp  = uz[z*nslice + y*nx + xp ];

				float eps_zx = normalizer*(-val_xn + val_xp);

				//y-derivative of uy displacement
				val_zn  = uz[z*nslice + yn*nx + x ];
				val_zp  = uz[z*nslice + yp*nx + x ];

				float eps_zy = normalizer*(-val_yn + val_yp);

				shear_strain = 1.f/3.f*sqrt(2.f*(eps_xx-eps_yy)*(eps_xx-eps_yy) + 2.f*(eps_xx-eps_zz)*(eps_xx-eps_zz) + 2.f*(eps_yy-eps_zz)*(eps_yy-eps_zz)
							+ 12.f*eps_yx*eps_yx + 12.f*eps_zx*eps_zx + 12.f*eps_zy*eps_zy);
			}

			output[pos] = shear_strain;
		}

		return output;
	}
	float* calc_from_green_strain(float* ux, float *uy, float* uz, int shape[3], std::string outval, std::string finitedifference_type="fourthorder")
	{
		int nx = shape[0]; int ny = shape[1]; int nz = shape[2];
		long long int nslice = nx*ny;
		long long int nstack = nz*nslice;

		float *output = (float*) calloc(nstack,sizeof(*output));

        float *dxx_field, *dxy_field, *dxz_field, *dyx_field, *dyy_field, *dyz_field, *dzx_field, *dzy_field, *dzz_field;


		float normalizer = 1.f/2.f;

		if (finitedifference_type == "centraldifference" || finitedifference_type == "secondorder")
			normalizer = 1.f/2.f;
		else if (finitedifference_type == "Barron" || finitedifference_type == "fourthorder")
			normalizer = 1.f/12.f;
        else if (finitedifference_type.substr(0,5) == "Farid")
        {
            int radius = 2;
            dxx_field = filter::calc_Farid_derivative(ux, shape, 0, radius, 1);
            dxy_field = filter::calc_Farid_derivative(ux, shape, 1, radius, 1);
            dxz_field = filter::calc_Farid_derivative(ux, shape, 2, radius, 1);
            dyx_field = filter::calc_Farid_derivative(uy, shape, 0, radius, 1);
            dyy_field = filter::calc_Farid_derivative(uy, shape, 1, radius, 1);
            dyz_field = filter::calc_Farid_derivative(uy, shape, 2, radius, 1);
            dzx_field = filter::calc_Farid_derivative(uz, shape, 0, radius, 1);
            dzy_field = filter::calc_Farid_derivative(uz, shape, 1, radius, 1);
            dzz_field = filter::calc_Farid_derivative(uz, shape, 2, radius, 1);
        }
		else
			std::cout << "Warning! unknown finite difference for plot_divergence. Using central difference." << std::endl;

		#pragma omp parallel for
		for (long long int pos = 0; pos < nstack; pos++)
		{
			int z = pos/nslice;
			int y = (pos-z*nslice)/nx;
			int x = pos-z*nslice-y*nx;

			float strain = 0.0f;
			float dxx, dxy, dxz, dyx, dyy, dyz, dzx, dzy, dzz;

			if (finitedifference_type == "Barron" || finitedifference_type == "fourthorder")
			{
                int zp = z+1; int zn = z-1;
                int yp = y+1; int yn = y-1;
                int xp = x+1; int xn = x-1;
                int zp2 = z+2; int zn2 = z-2;
                int yp2 = y+2; int yn2 = y-2;
                int xp2 = x+2; int xn2 = x-2;

                //Reflective boundary conditions (mirrored on first/last value)
                if (zp == nz) zp -= 2; if (zn < 0) zn = 1;
                if (yp == ny) yp -= 2; if (yn < 0) yn = 1;
                if (xp == nx) xp -= 2; if (xn < 0) xn = 1;
                if (zp2 == nz) zp2 = nz-4;
                if (yp2 == ny) yp2 = ny-4;
                if (xp2 == nx) xp2 = nx-4;
                if (zp2 > nz) zp2 = nz-3; if (zn2 < 0) zn2 = -zn2;
                if (yp2 > ny) yp2 = ny-3; if (yn2 < 0) yn2 = -yn2;
                if (xp2 > nx) xp2 = nx-3; if (xn2 < 0) xn2 = -xn2;

				//x-derivative of ux displacement
				float val_xn2 = ux[z*nslice + y*nx + xn2];
				float val_xn  = ux[z*nslice + y*nx + xn ];
				float val_xp  = ux[z*nslice + y*nx + xp ];
				float val_xp2 = ux[z*nslice + y*nx + xp2];

				dxx = normalizer*(val_xn2 -8*val_xn + 8*val_xp - val_xp2);

				//y-derivative of ux displacement
				float val_yn2 = ux[z*nslice + yn2*nx + x];
				float val_yn  = ux[z*nslice + yn*nx + x ];
				float val_yp  = ux[z*nslice + yp*nx + x ];
				float val_yp2 = ux[z*nslice + yp2*nx + x];

				dxy = normalizer*(val_yn2 -8*val_yn + 8*val_yp - val_yp2);

				//z-derivative of ux displacement
				float val_zn2 = ux[zn2*nslice + y*nx + x];
				float val_zn  = ux[zn*nslice  + y*nx + x ];
				float val_zp  = ux[zp*nslice  + y*nx + x ];
				float val_zp2 = ux[zp2*nslice + y*nx + x];

				dxz = normalizer*(val_zn2 -8*val_zn + 8*val_zp - val_zp2);

				//x-derivative of uy displacement
				val_xn2 = uy[z*nslice + y*nx + xn2];
				val_xn  = uy[z*nslice + y*nx + xn ];
				val_xp  = uy[z*nslice + y*nx + xp ];
				val_xp2 = uy[z*nslice + y*nx + xp2];

				dyx = normalizer*(val_xn2 -8*val_xn + 8*val_xp - val_xp2);

				//y-derivative of uy displacement
				val_yn2 = uy[z*nslice + yn2*nx + x];
				val_yn  = uy[z*nslice + yn*nx + x ];
				val_yp  = uy[z*nslice + yp*nx + x ];
				val_yp2 = uy[z*nslice + yp2*nx + x];

				dyy = normalizer*(val_yn2 -8*val_yn + 8*val_yp - val_yp2);

				//z-derivative of uy displacement
				val_zn2 = uy[zn2*nslice + y*nx + x];
				val_zn  = uy[zn*nslice  + y*nx + x ];
				val_zp  = uy[zp*nslice  + y*nx + x ];
				val_zp2 = uy[zp2*nslice + y*nx + x];

				dyz = normalizer*(val_zn2 -8*val_zn + 8*val_zp - val_zp2);

				//x-derivative of uz displacement
				val_xn2 = uz[z*nslice + y*nx + xn2];
				val_xn  = uz[z*nslice + y*nx + xn ];
				val_xp  = uz[z*nslice + y*nx + xp ];
				val_xp2 = uz[z*nslice + y*nx + xp2];

				dzx = normalizer*(val_xn2 -8*val_xn + 8*val_xp - val_xp2);

				//y-derivative of uz displacement
				val_yn2 = uz[z*nslice + yn2*nx + x];
				val_yn  = uz[z*nslice + yn*nx + x ];
				val_yp  = uz[z*nslice + yp*nx + x ];
				val_yp2 = uz[z*nslice + yp2*nx + x];

				dzy = normalizer*(val_yn2 -8*val_yn + 8*val_yp - val_yp2);

				//z-derivative of uz displacement
				val_zn2 = uz[zn2*nslice + y*nx + x];
				val_zn  = uz[zn*nslice  + y*nx + x ];
				val_zp  = uz[zp*nslice  + y*nx + x ];
				val_zp2 = uz[zp2*nslice + y*nx + x];

				dzz = normalizer*(val_zn2 -8*val_zn + 8*val_zp - val_zp2);
			}
			else if (finitedifference_type.substr(0,5) == "Farid")
			{
                dxx = dxx_field[pos]; dxy = dxy_field[pos]; dxz = dxz_field[pos];
                dyx = dyx_field[pos]; dyy = dyy_field[pos]; dyz = dyz_field[pos];
                dzx = dzx_field[pos]; dyz = dzy_field[pos]; dzz = dzz_field[pos];
			}
			else
			{
                int zp = z+1; int zn = z-1;
                int yp = y+1; int yn = y-1;
                int xp = x+1; int xn = x-1;

                //Reflective boundary conditions (mirrored on first/last value)
                if (zp == nz) zp -= 2; if (zn < 0) zn = 1;
                if (yp == ny) yp -= 2; if (yn < 0) yn = 1;
                if (xp == nx) xp -= 2; if (xn < 0) xn = 1;

				//x-derivative of ux displacement
				float val_xn  = ux[z*nslice + y*nx + xn ];
				float val_xp  = ux[z*nslice + y*nx + xp ];

				dxx = normalizer*(-val_xn + val_xp);

				//y-derivative of ux displacement
				float val_yn  = ux[z*nslice + yn*nx + x ];
				float val_yp  = ux[z*nslice + yp*nx + x ];

				dxy = normalizer*(-val_yn + val_yp);

				//z-derivative of ux displacement
				float val_zn  = ux[zn*nslice  + y*nx + x ];
				float val_zp  = ux[zp*nslice  + y*nx + x ];

				dxz = normalizer*(-val_zn + val_zp);

				//x-derivative of uy displacement
				val_xn  = uy[z*nslice + y*nx + xn ];
				val_xp  = uy[z*nslice + y*nx + xp ];

				dyx = normalizer*(-val_xn + val_xp);

				//y-derivative of uy displacement
				val_yn  = uy[z*nslice + yn*nx + x ];
				val_yp  = uy[z*nslice + yp*nx + x ];

				dyy = normalizer*(-val_yn + val_yp);

				//z-derivative of uy displacement
				val_zn  = uy[zn*nslice  + y*nx + x ];
				val_zp  = uy[zp*nslice  + y*nx + x ];

				dyz = normalizer*(-val_zn + val_zp);

				//x-derivative of uz displacement
				val_xn  = uz[z*nslice + y*nx + xn ];
				val_xp  = uz[z*nslice + y*nx + xp ];

				dzx = normalizer*(-val_xn + val_xp);

				//y-derivative of uy displacement
				val_zn  = uz[z*nslice + yn*nx + x ];
				val_zp  = uz[z*nslice + yp*nx + x ];

				dzy = normalizer*(-val_yn + val_yp);

				//z-derivative of uz displacement
				val_zn  = uz[zn*nslice  + y*nx + x ];
				val_zp  = uz[zp*nslice  + y*nx + x ];

				dzz = normalizer*(-val_zn + val_zp);
			}

			//Green-Lagrange strain tensor
			float Exx = dxx + 0.5f*(dxx*dxx+dyx*dyx+dzx*dzx);
			float Eyy = dyy + 0.5f*(dxy*dxy+dyy*dyy+dzy*dzy);
			float Ezz = dzz + 0.5f*(dxz*dxz+dyz*dyz+dzz*dzz);
			float Exy = 0.5f*(dxy+dyx)+0.5f*(dxx*dxy+dyx*dyy+dzx*dzy);
			float Exz = 0.5f*(dxz+dzx)+0.5f*(dxx*dxz+dyx*dyz+dzx*dzz);
			float Eyz = 0.5f*(dyz+dzz)+0.5f*(dxy*dxz+dyy*dyz+dzy*dzz);

			if (outval == "maxshear"){
				//strain = 1.f/3.f*sqrt(2.f*(dxx-dyy)*(dxx-dyy) + 2.f*(dxx-dzz)*(dxx-dzz) + 2.f*(dyy-Ezz)*(dyy-dzz)
				//		+ 12.f*dxy*dxy + 12.f*dxz*dxz + 12.f*dyz*dyz);
				strain = 1.f/3.f*sqrt(2.f*(Exx-Eyy)*(Exx-Eyy) + 2.f*(Exx-Ezz)*(Exx-Ezz) + 2.f*(Eyy-Ezz)*(Eyy-Ezz)
						+ 12.f*Exy*Exy + 12.f*Exz*Exz + 12.f*Eyz*Eyz);
			}
			else if (outval == "volstrain"){
				//adding identity gives the deformation gradient F
				dxx += 1.f;
				dyy += 1.f;
				dzz += 1.f;

				//volumetric strain calculated as det(F)-1
				strain = dxx*dyy*dzz + dxy*dyz*dxz + dxz*dxy*dyz - dxz*dyy*dxz - dyz*dyz*dxx - dzz*dxy*dxy-1;
				//strain = Exx*Eyy*Ezz + Exy*Eyz*Exz + Exz*Exy*Eyz - Exz*Eyy*Exz - Eyz*Eyz*Exx - Ezz*Exy*Exy-1;
			}
			else if (outval == "divergence")
				strain = Exx+Eyy+Ezz;
			else if (outval == "dilatation")
				strain = (Exx+Eyy+Ezz)/3.f;
			else if (outval == "volstrain2")
			{
				//adding identity gives the deformation gradient F
				dxx += 1.f;
				dyy += 1.f;
				dzz += 1.f;

				//decompose deformation gradient in rotation and strain and get divergence
				float F[9] = {dxx, dxy, dxz, dyx, dyy, dyz, dzx, dzy, dzz};
				strain = polardec::get_volumetricstrain(F);
			}
			else if (outval == "Exx")
				strain = Exx;
			else if (outval == "Eyy")
				strain = Eyy;
			else if (outval == "Ezz")
				strain = Ezz;
            else if (outval == "Exy")
				strain = Exy;
            else if (outval == "Eyz")
				strain = Eyz;
			else if (outval == "Exz")
				strain = Exz;
			else if (outval == "deviatoric_Eyy")
			{
				float dilatation = (Exx+Eyy+Ezz)/3.f; //hydrostatic strain
				strain = Eyy-dilatation;
			}
			else if (outval == "deviatoric_Ezz")
			{
				float dilatation = (Exx+Eyy+Ezz)/3.f; //hydrostatic strain
				strain = Ezz-dilatation;
			}
			else if (outval == "equivalent_strain")
			{
				//float dilatation = (Exx+Eyy+Ezz)/3.f; //hydrostatic strain

				//deviatoric strains (https://dianafea.com/manuals/d944/Analys/node405.html)
				float exx = (2.*Exx-Eyy-Ezz)/3.;
				float eyy = (2.*Eyy-Exx-Ezz)/3.;
				float ezz = (2.*Ezz-Eyy-Exx)/3.;

				//engineering strains
				float gamma_xy = 2.f*Exy;
				float gamma_xz = 2.f*Exz;
				float gamma_yz = 2.f*Eyz;

				//von Mises equivalent strain
				strain = (2.f/3.f)*sqrtf((3.f*(exx*exx+eyy*eyy+ezz*ezz)/2.f)+(3.f*(gamma_xy*gamma_xy+gamma_xz*gamma_xz+gamma_yz*gamma_yz)/4.f));
			}
			else if (outval == "principal_strain_max")
			{
				//refer: https://www.continuummechanics.org/principalstrain.html

				//calculate invariants
				double I1, I2, I3;
				__calc_invariants(Exx,Eyy,Ezz,Exy,Exz,Eyz,I1,I2,I3);

				//intermediate quantities
				double Q = (3.*I2-I1*I1)/9.;
				double R = (2.*I1*I1*I1-9.*I1*I2+27.*I3*I3*I3)/54.;
				double theta = acos(R/sqrt(-Q*Q*Q));

				double pi = 3.1415926535897932384626433832795;
				float sqrtconst = 2.*sqrt(-Q);
				float e_max = sqrtconst*cos(theta/3.) + 1./3.*I1;
				//float e_min = sqrtconst*cos((theta+2.*pi)/3.) + 1./3.*I1;
				//float e_shear = sqrtconst*cos((theta+4.*pi)/3.) + 1./3.*I1;

				if (std::isnan(e_max))
					strain = 0.0f;
				else
					strain = e_max;
			}
			else if (outval == "principal_strain_min")
			{
				//refer: https://www.continuummechanics.org/principalstrain.html

				//calculate invariants
				double I1, I2, I3;
				__calc_invariants(Exx,Eyy,Ezz,Exy,Exz,Eyz,I1,I2,I3);

				//intermediate quantities
				double Q = (3.*I2-I1*I1)/9.;
				double R = (2.*I1*I1*I1-9.*I1*I2+27.*I3*I3*I3)/54.;
				double theta = acos(R/sqrt(-Q*Q*Q));
				double pi = 3.1415926535897932384626433832795;
				float sqrtconst = 2.*sqrt(-Q);

				//float e_max = sqrtconst*cos(theta/3.) + 1./3.*I1;
				float e_min = sqrtconst*cos((theta+2.*pi)/3.) + 1./3.*I1;

				//float e_shear = sqrtconst*cos((theta+4.*pi)/3.) + 1./3.*I1;

				if (std::isnan(e_min))
					strain = 0.0f;
				else
					strain = e_min;
			}
			else if (outval == "principal_strain_shear")
			{
				//refer: https://www.continuummechanics.org/principalstrain.html

				//calculate invariants
				double I1, I2, I3;
				__calc_invariants(Exx,Eyy,Ezz,Exy,Exz,Eyz,I1,I2,I3);

				//intermediate quantities
				double Q = (3.*I2-I1*I1)/9.;
				double R = (2.*I1*I1*I1-9.*I1*I2+27.*I3*I3*I3)/54.;
				double theta = acos(R/sqrt(-Q*Q*Q));

				double pi = 3.1415926535897932384626433832795;
				float sqrtconst = 2.*sqrt(-Q);
				if (Q > 0) sqrtconst = 0.0;
				//float e_max = sqrtconst*cos(theta/3.) + 1./3.*I1;
				//float e_min = sqrtconst*cos((theta+2.*pi)/3.) + 1./3.*I1;
				float e_shear = sqrtconst*cos((theta+4.*pi)/3.) + 1./3.*I1;

				if (std::isnan(e_shear))
					strain = 0.0f;
				else
					strain = e_shear;
			}
			else
				std::cout << "unknown outval" << std::endl;

			output[pos] = strain;
		}

		if (finitedifference_type.substr(0,5) == "Farid")
		{
            free(dxx_field); free(dxy_field); free(dxz_field);
            free(dyx_field); free(dyy_field); free(dyz_field);
            free(dzx_field); free(dzy_field); free(dzz_field);
		}

		return output;
	}
	uint8_t* calc_localdisplacement_maxima(float* ux, float *uy, float* uz, int shape[3])
	{
		int nx = shape[0]; int ny = shape[1]; int nz = shape[2];
		long long int nslice = shape[0]*shape[1];
		long long int nstack = nslice*shape[2];

		float* magnitude = (float*) calloc(nstack,sizeof(*magnitude));
		uint8_t* output = (uint8_t*) calloc(nstack,sizeof(*output));

		#pragma omp parallel for
		for (long long int idx = 0; idx < nstack; idx++)
			magnitude[idx] = sqrtf(ux[idx]*ux[idx]+uy[idx]*uy[idx]+uz[idx]*uz[idx]);

		#pragma omp parallel for
		for (long long int idx = 0; idx < nstack; idx++)
		{
			int z = idx/nslice;
			int y = (idx-z*nslice)/nx;
			int x = idx-z*nslice-y*nx;

			//not on boundary
			if (x > 0 && x < nx-1 && y > 0 && y < ny-1 && z > 0 && z < nz-1)
			{
				float val = magnitude[idx];

				if(val < magnitude[idx-1]) continue;
				if(val < magnitude[idx+1]) continue;
				if(val < magnitude[idx-nx]) continue;
				if(val < magnitude[idx+nx]) continue;
				if(val < magnitude[idx-nslice]) continue;
				if(val < magnitude[idx+nslice]) continue;

				if(val < magnitude[idx-1-nx]) continue;
				if(val < magnitude[idx+1-nx]) continue;
				if(val < magnitude[idx-1+nx]) continue;
				if(val < magnitude[idx+1+nx]) continue;
				if(val < magnitude[idx-1-nslice]) continue;
				if(val < magnitude[idx+1-nslice]) continue;
				if(val < magnitude[idx-1+nslice]) continue;
				if(val < magnitude[idx+1+nslice]) continue;
				if(val < magnitude[idx-nx-nslice]) continue;
				if(val < magnitude[idx+nx-nslice]) continue;
				if(val < magnitude[idx-nx+nslice]) continue;
				if(val < magnitude[idx+nx+nslice]) continue;

				if(val < magnitude[idx-1-nx-nslice]) continue;
				if(val < magnitude[idx+1-nx-nslice]) continue;
				if(val < magnitude[idx-1+nx-nslice]) continue;
				if(val < magnitude[idx+1+nx-nslice]) continue;
				if(val < magnitude[idx-1-nx+nslice]) continue;
				if(val < magnitude[idx+1-nx+nslice]) continue;
				if(val < magnitude[idx-1+nx+nslice]) continue;
				if(val < magnitude[idx+1+nx+nslice]) continue;

				output[idx] = 128;
			}
		}

		return output;
	}
}

#endif //DISPLACEMENT_DERIVATIVES_H

