#include <iostream>
#include <omp.h>
#include <algorithm>
#include <math.h>
#include "smoothnessterm_cpu.h"

namespace optflow
{
namespace cpu2d
{
	void update_smoothnessterm_Barron(optflow_type *u, optflow_type *du,  optflow_type *phi, mathtype_solver epsilon_phi_squared, int shape[3], mathtype_solver hx, mathtype_solver hy)
	{
		//Fourth order approximation

		int nx = shape[0];
		int ny = shape[1];
		int nz = shape[2];
		idx_type nslice = nx*ny;
		idx_type nstack = nz*nslice;

		mathtype_solver normalizer_x = -1.f/(12.f*hx);
		mathtype_solver normalizer_y = -1.f/(12.f*hy);

		#pragma omp parallel for
		for (idx_type pos = 0; pos < nstack; pos++)
		{
			int y = pos/nx;
			int x = pos-y*nx;

			int yp = y+1; int yn = y-1;
			int xp = x+1; int xn = x-1;
			int yp2 = y+2; int yn2 = y-2;
			int xp2 = x+2; int xn2 = x-2;

			//Reflective boundary conditions (mirrored on first/last value)
			if (yp == ny) yp -= 2; if (yn < 0) yn = 1;
			if (xp == nx) xp -= 2; if (xn < 0) xn = 1;
			if (yp2 >= ny) yp2 = 2*ny-yp2-2; if (yn2 < 0) yn2 = -yn2;
			if (xp2 >= nx) xp2 = 2*nx-xp2-2; if (xn2 < 0) xn2 = -xn2;

			//acquire from global memory
			//////////////////////////////////////////////////////////////
			optflow_type u_xn2 = u[y*nx  +xn2];
			optflow_type u_xn =  u[y*nx  + xn];
			optflow_type u_xp =  u[y*nx+xp];
			optflow_type u_xp2 = u[y*nx+xp2];
			optflow_type u_yn2 = u[yn2*nx+  x];
			optflow_type u_yn =  u[yn*nx + x ];
			optflow_type u_yp =  u[yp*nx+x];
			optflow_type u_yp2 = u[yp2*nx+x];

			optflow_type v_xn2 = u[nstack + y*nx  +xn2];
			optflow_type v_xn =  u[nstack + y*nx  + xn];
			optflow_type v_xp =  u[nstack + y*nx+xp];
			optflow_type v_xp2 = u[nstack + y*nx+xp2];
			optflow_type v_yn2 = u[nstack + yn2*nx+  x];
			optflow_type v_yn =  u[nstack + yn*nx + x ];
			optflow_type v_yp =  u[nstack + yp*nx+x];
			optflow_type v_yp2 = u[nstack + yp2*nx+x];

			u_xn2 += du[y*nx  +xn2];
			u_xn +=  du[y*nx  + xn];
			u_xp +=  du[y*nx+xp];
			u_xp2 += du[y*nx+xp2];
			u_yn2 += du[yn2*nx+  x];
			u_yn +=  du[yn*nx + x ];
			u_yp +=  du[yp*nx+x];
			u_yp2 += du[yp2*nx+x];

			v_xn2 += du[nstack + y*nx  +xn2];
			v_xn +=  du[nstack + y*nx  + xn];
			v_xp +=  du[nstack + y*nx+xp];
			v_xp2 += du[nstack + y*nx+xp2];
			v_yn2 += du[nstack + yn2*nx+  x];
			v_yn +=  du[nstack + yn*nx + x ];
			v_yp +=  du[nstack + yp*nx+x];
			v_yp2 += du[nstack + yp2*nx+x];
			//////////////////////////////////////////////////////////////

			 //partial derivatives
			//////////////////////////////////////////////////////////////
			mathtype_solver ux = normalizer_x*(-u_xn2 +8*u_xn -8*u_xp +u_xp2);
			mathtype_solver uy = normalizer_y*(-u_yn2 +8*u_yn -8*u_yp +u_yp2);
			mathtype_solver vx = normalizer_x*(-v_xn2 +8*v_xn -8*v_xp +v_xp2);
			mathtype_solver vy = normalizer_y*(-v_yn2 +8*v_yn -8*v_yp +v_yp2);
			//////////////////////////////////////////////////////////////

			//smoothness term (phi)
			//////////////////////////////////////////////////////////////
			mathtype_solver val1 = ux*ux + uy*uy + vx*vx + vy*vy;

			phi[pos] = 0.5f/sqrtf(val1 + epsilon_phi_squared);
		}

		return;
	}
	void update_smoothnessterm_centralDiff(optflow_type *u, optflow_type *du,  optflow_type *phi, mathtype_solver epsilon_phi_squared, int shape[3],mathtype_solver hx, mathtype_solver hy)
	{
		int nx = shape[0];
		int ny = shape[1];
		int nz = shape[2];
		idx_type nslice = shape[0]*shape[1];
		idx_type nstack = shape[2]*nslice;

		mathtype_solver normalizer_x = -1.f/(2.f*hx);
		mathtype_solver normalizer_y = -1.f/(2.f*hy);

		#pragma omp parallel for
		for (idx_type pos = 0; pos < nstack; pos++)
		{
			int y = pos/nx;
			int x = pos-y*nx;

			int yp = y+1; int yn = y-1;
			int xp = x+1; int xn = x-1;

			//Reflective boundary conditions (mirrored on first/last value)
			if (yp == ny) yp -= 2; if (yn < 0) yn = 1;
			if (xp == nx) xp -= 2; if (xn < 0) xn = 1;

			//acquire from global memory
			//////////////////////////////////////////////////////////////
			optflow_type u_xn =  u[y*nx  + xn];
			optflow_type u_xp =  u[y*nx+xp];
			optflow_type u_yn =  u[yn*nx + x ];
			optflow_type u_yp =  u[yp*nx+x];

			optflow_type v_xn =  u[nstack + y*nx  + xn];
			optflow_type v_xp =  u[nstack + y*nx+xp];
			optflow_type v_yn =  u[nstack + yn*nx + x ];
			optflow_type v_yp =  u[nstack + yp*nx+x];

			u_xn +=  du[y*nx  + xn];
			u_xp +=  du[y*nx+xp];
			u_yn +=  du[yn*nx + x ];
			u_yp +=  du[yp*nx+x];

			v_xn +=  du[nstack + y*nx  + xn];
			v_xp +=  du[nstack + y*nx+xp];
			v_yn +=  du[nstack + yn*nx + x ];
			v_yp +=  du[nstack + yp*nx+x];
			//////////////////////////////////////////////////////////////

			 //partial derivatives
			//////////////////////////////////////////////////////////////
			mathtype_solver ux = normalizer_x*(u_xp-u_xn);
			mathtype_solver uy = normalizer_y*(u_yp-u_yn);
			mathtype_solver vx = normalizer_x*(v_xp-v_xn);
			mathtype_solver vy = normalizer_y*(v_yp-v_yn);
			//////////////////////////////////////////////////////////////

			//smoothness term (phi)
			//////////////////////////////////////////////////////////////
			mathtype_solver val1 = ux*ux + uy*uy + vx*vx + vy*vy;
			phi[pos] = 0.5f/sqrtf(val1 + epsilon_phi_squared);
		}

		return;
	}
	void update_smoothnessterm_forwardDiff(optflow_type *u, optflow_type *du,  optflow_type *phi, mathtype_solver epsilon_phi_squared, int shape[3], mathtype_solver hx, mathtype_solver hy)
	{
		int nx = shape[0];
		int ny = shape[1];
		int nz = shape[2];
		idx_type nslice = shape[0]*shape[1];
		idx_type nstack = shape[2]*nslice;

		mathtype_solver normalizer_x = -1.f/hx;
		mathtype_solver normalizer_y = -1.f/hy;

		#pragma omp parallel for
		for (idx_type pos = 0; pos < nstack; pos++)
		{
			int y = pos/nx;
			int x = pos-y*nx;

			int yp = y+1; int yn = y-1;
			int xp = x+1; int xn = x-1;

			//Reflective boundary conditions (mirrored on first/last value)
			if (yp == ny) yp -= 2; if (yn < 0) yn = 1;
			if (xp == nx) xp -= 2; if (xn < 0) xn = 1;

			//acquire from global memory
			//////////////////////////////////////////////////////////////
			optflow_type u0 =  u[pos];
			optflow_type u_xp =  u[y*nx+xp];
			optflow_type u_yp =  u[yp*nx+x];

			optflow_type v0 =  u[nstack + pos];
			optflow_type v_xp =  u[nstack + y*nx+xp];
			optflow_type v_yp =  u[nstack + yp*nx+x];

			u0 += du[pos];
			u_xp +=  du[y*nx+xp];
			u_yp +=  du[yp*nx+x];

			v0 += du[nstack+pos];
			v_xp +=  du[nstack + y*nx+xp];
			v_yp +=  du[nstack + yp*nx+x];
			//////////////////////////////////////////////////////////////

			 //partial derivatives
			//////////////////////////////////////////////////////////////
			mathtype_solver ux = normalizer_x*(u_xp-u0);
			mathtype_solver uy = normalizer_y*(u_yp-u0);
			mathtype_solver vx = normalizer_x*(v_xp-v0);
			mathtype_solver vy = normalizer_y*(v_yp-v0);
			//////////////////////////////////////////////////////////////

			//smoothness term (phi)
			//////////////////////////////////////////////////////////////
			mathtype_solver val1 = ux*ux + uy*uy + vx*vx + vy*vy;
			phi[pos] = 0.5f/sqrtf(val1 + epsilon_phi_squared);
		}

		return;
	}
}

namespace cpu3d
{
	void update_smoothnessterm_Barron(optflow_type *u, optflow_type *du,  optflow_type *phi, mathtype_solver epsilon_phi_squared, int shape[3], mathtype_solver hx, mathtype_solver hy, mathtype_solver hz)
	{
		//Fourth order approximation

		int nx = shape[0];
		int ny = shape[1];
		int nz = shape[2];
		idx_type nslice = nx*ny;
		idx_type nstack = nz*nslice;

		mathtype_solver normalizer_x = -1.f/(12.f*hx);
		mathtype_solver normalizer_y = -1.f/(12.f*hy);
		mathtype_solver normalizer_z = -1.f/(12.f*hz);

		#pragma omp parallel for
		for (idx_type pos = 0; pos < nstack; pos++)
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
			if (zp2 >= nz) zp2 = 2*nz-zp2-2; if (zn2 < 0) zn2 = -zn2;
			if (yp2 >= ny) yp2 = 2*ny-yp2-2; if (yn2 < 0) yn2 = -yn2;
			if (xp2 >= nx) xp2 = 2*nx-xp2-2; if (xn2 < 0) xn2 = -xn2;

			//acquire from global memory
			//////////////////////////////////////////////////////////////
			optflow_type u_xn2 = u[z*nslice + y*nx  +xn2];
			optflow_type u_xn =  u[z*nslice + y*nx  + xn];
			optflow_type u_xp =  u[z*nslice + y*nx+xp];
			optflow_type u_xp2 = u[z*nslice + y*nx+xp2];
			optflow_type u_yn2 = u[z*nslice + yn2*nx+  x];
			optflow_type u_yn =  u[z*nslice + yn*nx + x ];
			optflow_type u_yp =  u[z*nslice + yp*nx+x];
			optflow_type u_yp2 = u[z*nslice + yp2*nx+x];
			optflow_type u_zn2 = u[zn2*nslice + y*nx + x];
			optflow_type u_zn = u[zn*nslice + y*nx + x];
			optflow_type u_zp = u[zp*nslice + y*nx + x];
			optflow_type u_zp2 = u[zp2*nslice + y*nx + x];

			optflow_type v_xn2 = u[nstack + z*nslice + y*nx  +xn2];
			optflow_type v_xn =  u[nstack + z*nslice + y*nx  + xn];
			optflow_type v_xp =  u[nstack + z*nslice + y*nx+xp];
			optflow_type v_xp2 = u[nstack + z*nslice + y*nx+xp2];
			optflow_type v_yn2 = u[nstack + z*nslice + yn2*nx+  x];
			optflow_type v_yn =  u[nstack + z*nslice + yn*nx + x ];
			optflow_type v_yp =  u[nstack + z*nslice + yp*nx+x];
			optflow_type v_yp2 = u[nstack + z*nslice + yp2*nx+x];
			optflow_type v_zn2 = u[nstack + zn2*nslice + y*nx + x];
			optflow_type v_zn =  u[nstack + zn*nslice + y*nx + x];
			optflow_type v_zp =  u[nstack + zp*nslice + y*nx + x];
			optflow_type v_zp2 = u[nstack + zp2*nslice + y*nx + x];

			optflow_type w_xn2 = u[2*nstack + z*nslice + y*nx  +xn2];
			optflow_type w_xn =  u[2*nstack + z*nslice + y*nx  + xn];
			optflow_type w_xp =  u[2*nstack + z*nslice + y*nx+xp];
			optflow_type w_xp2 = u[2*nstack + z*nslice + y*nx+xp2];
			optflow_type w_yn2 = u[2*nstack + z*nslice + yn2*nx+  x];
			optflow_type w_yn =  u[2*nstack + z*nslice + yn*nx + x ];
			optflow_type w_yp =  u[2*nstack + z*nslice + yp*nx+x];
			optflow_type w_yp2 = u[2*nstack + z*nslice + yp2*nx+x];
			optflow_type w_zn2 = u[2*nstack + zn2*nslice + y*nx + x];
			optflow_type w_zn =  u[2*nstack + zn*nslice + y*nx + x];
			optflow_type w_zp =  u[2*nstack + zp*nslice + y*nx + x];
			optflow_type w_zp2 = u[2*nstack + zp2*nslice + y*nx + x];

			u_xn2 += du[z*nslice + y*nx  +xn2];
			u_xn +=  du[z*nslice + y*nx  + xn];
			u_xp +=  du[z*nslice + y*nx+xp];
			u_xp2 += du[z*nslice + y*nx+xp2];
			u_yn2 += du[z*nslice + yn2*nx+  x];
			u_yn +=  du[z*nslice + yn*nx + x ];
			u_yp +=  du[z*nslice + yp*nx+x];
			u_yp2 += du[z*nslice + yp2*nx+x];
			u_zn2 += du[zn2*nslice+ y*nx+x];
			u_zn +=  du[zn*nslice + y*nx+x];
			u_zp +=  du[zp*nslice + y*nx+x];
			u_zp2 += du[zp2*nslice+ y*nx+x];

			v_xn2 += du[nstack + z*nslice + y*nx  +xn2];
			v_xn +=  du[nstack + z*nslice + y*nx  + xn];
			v_xp +=  du[nstack + z*nslice + y*nx+xp];
			v_xp2 += du[nstack + z*nslice + y*nx+xp2];
			v_yn2 += du[nstack + z*nslice + yn2*nx+  x];
			v_yn +=  du[nstack + z*nslice + yn*nx + x ];
			v_yp +=  du[nstack + z*nslice + yp*nx+x];
			v_yp2 += du[nstack + z*nslice + yp2*nx+x];
			v_zn2 += du[nstack + zn2*nslice + y*nx+  x];
			v_zn +=  du[nstack + zn*nslice + y*nx + x ];
			v_zp +=  du[nstack + zp*nslice + y*nx+x];
			v_zp2 += du[nstack + zp2*nslice + y*nx+x];

			w_xn2 += du[2*nstack + z*nslice + y*nx  +xn2];
			w_xn +=  du[2*nstack + z*nslice + y*nx  + xn];
			w_xp +=  du[2*nstack + z*nslice + y*nx+xp];
			w_xp2 += du[2*nstack + z*nslice + y*nx+xp2];
			w_yn2 += du[2*nstack + z*nslice + yn2*nx+  x];
			w_yn +=  du[2*nstack + z*nslice + yn*nx + x ];
			w_yp +=  du[2*nstack + z*nslice + yp*nx+x];
			w_yp2 += du[2*nstack + z*nslice + yp2*nx+x];
			w_zn2 += du[2*nstack + zn2*nslice + y*nx+  x];
			w_zn +=  du[2*nstack + zn*nslice + y*nx + x ];
			w_zp +=  du[2*nstack + zp*nslice + y*nx+x];
			w_zp2 += du[2*nstack + zp2*nslice + y*nx+x];
			//////////////////////////////////////////////////////////////

			 //partial derivatives
			//////////////////////////////////////////////////////////////
			mathtype_solver ux = normalizer_x*(-u_xn2 +8*u_xn -8*u_xp +u_xp2);
			mathtype_solver uy = normalizer_y*(-u_yn2 +8*u_yn -8*u_yp +u_yp2);
			mathtype_solver uz = normalizer_z*(-u_zn2 +8*u_zn -8*u_zp +u_zp2);
			mathtype_solver vx = normalizer_x*(-v_xn2 +8*v_xn -8*v_xp +v_xp2);
			mathtype_solver vy = normalizer_y*(-v_yn2 +8*v_yn -8*v_yp +v_yp2);
			mathtype_solver vz = normalizer_z*(-v_zn2 +8*v_zn -8*v_zp +v_zp2);
			mathtype_solver wx = normalizer_x*(-w_xn2 +8*w_xn -8*w_xp +w_xp2);
			mathtype_solver wy = normalizer_y*(-w_yn2 +8*w_yn -8*w_yp +w_yp2);
			mathtype_solver wz = normalizer_z*(-w_zn2 +8*w_zn -8*w_zp +w_zp2);
			//////////////////////////////////////////////////////////////

			//smoothness term (phi)
			//////////////////////////////////////////////////////////////
			mathtype_solver val1 = (ux*ux + uy*uy + uz*uz) + (vx*vx + vy*vy + vz*vz) + (wx*wx + wy*wy + wz*wz);

			phi[pos] = 0.5f/sqrtf(val1 + epsilon_phi_squared);
		}

		return;
	}
	void update_smoothnessterm_centralDiff(optflow_type *u, optflow_type *du,  optflow_type *phi, mathtype_solver epsilon_phi_squared, int shape[3], mathtype_solver hx, mathtype_solver hy, mathtype_solver hz)
	{
		int nx = shape[0];
		int ny = shape[1];
		int nz = shape[2];
		idx_type nslice = shape[0]*shape[1];
		idx_type nstack = shape[2]*nslice;

		mathtype_solver normalizer_x = -1.f/(2.f*hx);
		mathtype_solver normalizer_y = -1.f/(2.f*hy);
		mathtype_solver normalizer_z = -1.f/(2.f*hz);

		#pragma omp parallel for
		for (idx_type pos = 0; pos < nstack; pos++)
		{
			int z = pos/nslice;
			int y = (pos-z*nslice)/nx;
			int x = pos-z*nslice-y*nx;

			int zp = z+1; int zn = z-1;
			int yp = y+1; int yn = y-1;
			int xp = x+1; int xn = x-1;

			//Reflective boundary conditions (mirrored on first/last value)
			if (zp == nz) zp -= 2; if (zn < 0) zn = 1;
			if (yp == ny) yp -= 2; if (yn < 0) yn = 1;
			if (xp == nx) xp -= 2; if (xn < 0) xn = 1;

			//acquire from global memory
			//////////////////////////////////////////////////////////////
			optflow_type u_xn =  u[z*nslice+ y*nx  + xn];
			optflow_type u_xp =  u[z*nslice+ y*nx+xp];
			optflow_type u_yn =  u[z*nslice+ yn*nx + x ];
			optflow_type u_yp =  u[z*nslice+ yp*nx+x];
			optflow_type u_zn =  u[zn*nslice+ y*nx + x];
			optflow_type u_zp =  u[zp*nslice+ y*nx + x];

			optflow_type v_xn =  u[nstack +z*nslice+  y*nx  + xn];
			optflow_type v_xp =  u[nstack +z*nslice+  y*nx+xp];
			optflow_type v_yn =  u[nstack +z*nslice+  yn*nx + x ];
			optflow_type v_yp =  u[nstack +z*nslice+  yp*nx+x];
			optflow_type v_zn =  u[nstack +zn*nslice+  y*nx + x ];
			optflow_type v_zp =  u[nstack +zp*nslice+  y*nx+x];

			optflow_type w_xn =  u[2*nstack +z*nslice+  y*nx  + xn];
			optflow_type w_xp =  u[2*nstack +z*nslice+  y*nx+xp];
			optflow_type w_yn =  u[2*nstack +z*nslice+  yn*nx + x ];
			optflow_type w_yp =  u[2*nstack +z*nslice+  yp*nx+x];
			optflow_type w_zn =  u[2*nstack +zn*nslice+  y*nx + x ];
			optflow_type w_zp =  u[2*nstack +zp*nslice+  y*nx+x];

			u_xn +=  du[z*nslice+y*nx  + xn];
			u_xp +=  du[z*nslice+y*nx+xp];
			u_yn +=  du[z*nslice+yn*nx + x ];
			u_yp +=  du[z*nslice+yp*nx+x];
			u_zn +=  du[zn*nslice+y*nx + x ];
			u_zp +=  du[zp*nslice+y*nx+x];

			v_xn +=  du[nstack + z*nslice+ y*nx  + xn];
			v_xp +=  du[nstack + z*nslice+ y*nx+xp];
			v_yn +=  du[nstack + z*nslice+ yn*nx + x ];
			v_yp +=  du[nstack + z*nslice+ yp*nx+x];
			v_zn +=  du[nstack + zn*nslice+ y*nx + x ];
			v_zp +=  du[nstack + zp*nslice+ y*nx+x];

			w_xn +=  du[2*nstack + z*nslice+ y*nx  + xn];
			w_xp +=  du[2*nstack + z*nslice+ y*nx+xp];
			w_yn +=  du[2*nstack + z*nslice+ yn*nx + x ];
			w_yp +=  du[2*nstack + z*nslice+ yp*nx+x];
			w_zn +=  du[2*nstack + zn*nslice+ y*nx + x ];
			w_zp +=  du[2*nstack + zp*nslice+ y*nx+x];
			//////////////////////////////////////////////////////////////

			 //partial derivatives
			//////////////////////////////////////////////////////////////
			mathtype_solver ux = normalizer_x*(u_xp-u_xn);
			mathtype_solver uy = normalizer_y*(u_yp-u_yn);
			mathtype_solver uz = normalizer_z*(u_zp-u_zn);
			mathtype_solver vx = normalizer_x*(v_xp-v_xn);
			mathtype_solver vy = normalizer_y*(v_yp-v_yn);
			mathtype_solver vz = normalizer_z*(v_zp-v_zn);
			mathtype_solver wx = normalizer_x*(w_xp-w_xn);
			mathtype_solver wy = normalizer_y*(w_yp-w_yn);
			mathtype_solver wz = normalizer_z*(w_zp-w_zn);
			//////////////////////////////////////////////////////////////

			//smoothness term (phi)
			//////////////////////////////////////////////////////////////
			mathtype_solver val1 = (ux*ux + uy*uy + uz*uz) + (vx*vx + vy*vy + vz*vz) + (wx*wx + wy*wy + wz*wz);
			phi[pos] = 0.5f/sqrtf(val1 + epsilon_phi_squared);
		}

		return;
	}
	void update_smoothnessterm_forwardDiff(optflow_type *u, optflow_type *du,  optflow_type *phi, mathtype_solver epsilon_phi_squared, int shape[3], mathtype_solver hx, mathtype_solver hy, mathtype_solver hz)
	{
		int nx = shape[0];
		int ny = shape[1];
		int nz = shape[2];
		idx_type nslice = shape[0]*shape[1];
		idx_type nstack = shape[2]*nslice;

		mathtype_solver normalizer_x = -1.f/hx;
		mathtype_solver normalizer_y = -1.f/hy;
		mathtype_solver normalizer_z = -1.f/hz;

		#pragma omp parallel for
		for (idx_type pos = 0; pos < nstack; pos++)
		{
			int z = pos/nslice;
			int y = (pos-z*nslice)/nx;
			int x = pos-z*nslice-y*nx;

			int zp = z+1; int zn = z-1;
			int yp = y+1; int yn = y-1;
			int xp = x+1; int xn = x-1;

			//Reflective boundary conditions (mirrored on first/last value)
			if (zp == nz) zp -= 2; if (zn < 0) zn = 1;
			if (yp == ny) yp -= 2; if (yn < 0) yn = 1;
			if (xp == nx) xp -= 2; if (xn < 0) xn = 1;

			//acquire from global memory
			//////////////////////////////////////////////////////////////
			optflow_type u0 =  u[pos];
			optflow_type u_xp =  u[z*nslice+y*nx+xp];
			optflow_type u_yp =  u[z*nslice+yp*nx+x];
			optflow_type u_zp =  u[zp*nslice+y*nx+x];

			optflow_type   v0 =  u[nstack + z*nslice+ pos];
			optflow_type v_xp =  u[nstack + z*nslice+ y*nx+xp];
			optflow_type v_yp =  u[nstack + z*nslice+ yp*nx+x];
			optflow_type v_zp =  u[nstack + zp*nslice+ y*nx+x];

			optflow_type   w0 =  u[2*nstack + z*nslice+ pos];
			optflow_type w_xp =  u[2*nstack + z*nslice+ y*nx+xp];
			optflow_type w_yp =  u[2*nstack + z*nslice+ yp*nx+x];
			optflow_type w_zp =  u[2*nstack + zp*nslice+ y*nx+x];

			u0 += du[pos];
			u_xp += du[ z*nslice +  y*nx + xp];
			u_yp += du[ z*nslice + yp*nx +  x];
			u_zp += du[zp*nslice +  y*nx +  x];

			v0 += du[nstack+pos];
			v_xp +=  du[nstack + z*nslice + y*nx+xp];
			v_yp +=  du[nstack + z*nslice + yp*nx+x];
			v_zp +=  du[nstack + zp*nslice + y*nx+x];

			w0 += du[2*nstack+pos];
			w_xp +=  du[2*nstack + z*nslice + y*nx+xp];
			w_yp +=  du[2*nstack + z*nslice + yp*nx+x];
			w_zp +=  du[2*nstack + zp*nslice + y*nx+x];
			//////////////////////////////////////////////////////////////

			 //partial derivatives
			//////////////////////////////////////////////////////////////
			mathtype_solver ux = normalizer_x*(u_xp-u0);
			mathtype_solver uy = normalizer_y*(u_yp-u0);
			mathtype_solver uz = normalizer_z*(u_zp-u0);
			mathtype_solver vx = normalizer_x*(v_xp-v0);
			mathtype_solver vy = normalizer_y*(v_yp-v0);
			mathtype_solver vz = normalizer_z*(v_zp-v0);
			mathtype_solver wx = normalizer_x*(w_xp-w0);
			mathtype_solver wy = normalizer_y*(w_yp-w0);
			mathtype_solver wz = normalizer_z*(w_zp-w0);
			//////////////////////////////////////////////////////////////

			//smoothness term (phi)
			//////////////////////////////////////////////////////////////
			mathtype_solver val1 = (ux*ux + uy*uy + uz*uz) + (vx*vx + vy*vy + vz*vz) + (wx*wx + wy*wy + wz*wz);
			phi[pos] = 0.5f/sqrtf(val1 + epsilon_phi_squared);
		}

		return;
	}
}
}
