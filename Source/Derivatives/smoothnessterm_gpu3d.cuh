#ifndef SMOOTHNESSTERM_GPU3D_CUH
#define SMOOTHNESSTERM_GPU3D_CUH

#include <iostream>
#include <cuda.h>
#include "../Solver/optflow_base.h"
#include "../Solver/gpu_constants.cuh"

namespace optflow
{
namespace gpu3d
{
	// All functions return the derivative of sqrt(x+eps), i.e 1/(2*sqrt(x+eps))
	//
	//11.2021 minor changes:
	//		- Corrected the derivatives in the smoothness term which were all inverted. Doesn't matter since we take the norm.
	//		- Corrected the upper side reflective boundary which wrapped around before. It read zp2 -= 2*nz-zp2-2 and is know zp2 = 2*nz-zp2-2

	__global__ void update_smoothnessterm_Barron(optflow_type *u, optflow_type *du,  optflow_type *phi, optflow_type *adaptivity)
	{
		//Fourth order approximation

		//acquire constants and position
		/////////////////////////////////////////////
		int nx = gpu_const::nx_c;
		int ny = gpu_const::ny_c;
		int nz = gpu_const::nz_c;

		mathtype_solver epsilon_phi_squared = gpu_const::epsilon_phi_squared_c;
		mathtype_solver hx = gpu_const::hx_c;
		mathtype_solver hy = gpu_const::hy_c;
		mathtype_solver hz = gpu_const::hz_c;

		bool anisotropic = gpu_const::anisotropic_smoothness_c;
		bool decoupled = gpu_const::decoupled_smoothness_c;
		bool adaptive_smoothness = gpu_const::adaptive_smoothness_c;
		bool complementary_smoothness = gpu_const::complementary_smoothness_c;

		idx_type nslice = nx*ny;
		idx_type nstack = nz*nslice;

		idx_type pos = (blockIdx.x*blockDim.x+threadIdx.x);
		if (pos >= nstack) {pos = threadIdx.x;}

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

		mathtype_solver normalizer_x = 1.f/(12.f*hx);
		mathtype_solver normalizer_y = 1.f/(12.f*hy);
		mathtype_solver normalizer_z = 1.f/(12.f*hz);
		/////////////////////////////////////////////

		//acquire from global memory
		//////////////////////////////////////////////////////////////
		__syncthreads();
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

		//Rotate vectors to edge aligned basis
		//////////////////////////////////////////////////////////////
		if (adaptive_smoothness)
		{
			__syncthreads();
			mathtype_solver theta = adaptivity[pos];
			mathtype_solver ksi = adaptivity[pos+nstack];
			mathtype_solver sintheta, costheta, sinksi, cosksi;

			__sincosf(theta, &sintheta, &costheta);
			__sincosf(ksi, &sinksi, &cosksi);

			float ux1 = ux*(costheta*cosksi) -uy*sintheta + uz*(costheta*sinksi);
			float uy1 = ux*(sintheta*cosksi) +uy*costheta + uz*(sintheta*sinksi);
			float uz1 = ux*(-sinksi)                      + uz*cosksi;

			float vx1 = vx*(costheta*cosksi) -vy*sintheta + vz*(costheta*sinksi);
			float vy1 = vx*(sintheta*cosksi) +vy*costheta + vz*(sintheta*sinksi);
			float vz1 = vx*(-sinksi)                      + vz*cosksi;

			float wx1 = wx*(costheta*cosksi) -wy*sintheta + wz*(costheta*sinksi);
			float wy1 = wx*(sintheta*cosksi) +wy*costheta + wz*(sintheta*sinksi);
			float wz1 = wx*(-sinksi)                      + wz*cosksi;

			ux = ux1; uy = uy1; uz = uz1;
			vx = vx1; vy = vy1; vz = vz1;
			wx = wx1; wy = wy1; wz = wz1;
		}
		//////////////////////////////////////////////////////////////

		//calculate smoothness term
		//////////////////////////////////////////////////////////////
		if (!decoupled)
		{
			mathtype_solver value;

			if(!anisotropic) value = 0.5f/sqrtf((ux*ux + uy*uy + uz*uz) + (vx*vx + vy*vy + vz*vz) + (wx*wx + wy*wy + wz*wz) + epsilon_phi_squared); //Isotropic flow driven := phi(tr(nabla_u1*nabla_u1^T + nabla_u2^T)
			else if(!adaptive_smoothness || !complementary_smoothness){
				value = 0.5f/(sqrtf(ux*ux + vx*vx + wx*wx + epsilon_phi_squared)
						    +sqrtf(uy*uy + vy*vy + wy*wy + epsilon_phi_squared)
						    +sqrtf(uz*uz + vz*vz + wz*wz + epsilon_phi_squared)); //Anisotropic flow driven := tr(phi(nabla_u1*nabla_u1^T + nabla_u2^T)
			}
			else{
				value = 0.166666667f/sqrtf(ux*ux + vx*vx + wx*wx + epsilon_phi_squared)
						   + 0.333333333f*sqrtf(uy*uy + vy*vy + wy*wy + epsilon_phi_squared)
						   + 0.333333333f*sqrtf(uz*uz + vz*vz + wz*wz + epsilon_phi_squared); //complementary following Zimmer2011:Optical flow in Harmony
			}

			__syncthreads();
			phi[pos] = value;
		}
		else
		{
			mathtype_solver value1, value2, value3;

			if(!anisotropic){
				//decoupled isotropic:
				value1 = 0.5f/sqrtf(ux*ux + uy*uy + uz*uz + epsilon_phi_squared);
				value2 = 0.5f/sqrtf(vx*vx + vy*vy + vz*vz + epsilon_phi_squared);
				value3 = 0.5f/sqrtf(wx*wx + wy*wy + wz*wz + epsilon_phi_squared);
			}
			else if (!adaptive_smoothness || !complementary_smoothness){
				//decoupled anisotropic:
				value1 = 0.5f/(sqrtf(ux*ux + epsilon_phi_squared)+sqrtf(uy*uy + epsilon_phi_squared)+sqrtf(uz*uz + epsilon_phi_squared));
				value2 = 0.5f/(sqrtf(vx*vx + epsilon_phi_squared)+sqrtf(vy*vy + epsilon_phi_squared)+sqrtf(vz*vz + epsilon_phi_squared));
				value3 = 0.5f/(sqrtf(wx*wx + epsilon_phi_squared)+sqrtf(wy*wy + epsilon_phi_squared)+sqrtf(wz*wz + epsilon_phi_squared));
			}
			else
			{
				//decoupled complementary
				value1 = 0.166666667f/sqrtf(ux*ux + epsilon_phi_squared)+0.333333333f*sqrtf(uy*uy + epsilon_phi_squared)+0.333333333f*sqrtf(uz*uz + epsilon_phi_squared);
				value2 = 0.166666667f/sqrtf(vx*vx + epsilon_phi_squared)+0.333333333f*sqrtf(vy*vy + epsilon_phi_squared)+0.333333333f*sqrtf(vz*vz + epsilon_phi_squared);
				value3 = 0.166666667f/sqrtf(wx*wx + epsilon_phi_squared)+0.333333333f*sqrtf(wy*wy + epsilon_phi_squared)+0.333333333f*sqrtf(wz*wz + epsilon_phi_squared);
			}

			__syncthreads();
			phi[pos] = value1;
			phi[nstack+pos] = value2;
			phi[2*nstack+pos] = value3;
		}
		//////////////////////////////////////////////////////////////

		return;
	}
	__global__ void update_smoothnessterm_centralDiff(optflow_type *u, optflow_type *du,  optflow_type *phi, optflow_type *adaptivity)
	{
		//acquire constants and position
		/////////////////////////////////////////////
		int nx = gpu_const::nx_c;
		int ny = gpu_const::ny_c;
		int nz = gpu_const::nz_c;

		mathtype_solver epsilon_phi_squared = gpu_const::epsilon_phi_squared_c;
		mathtype_solver hx = gpu_const::hx_c;
		mathtype_solver hy = gpu_const::hy_c;
		mathtype_solver hz = gpu_const::hz_c;

		bool anisotropic = gpu_const::anisotropic_smoothness_c;
		bool decoupled = gpu_const::decoupled_smoothness_c;
		bool adaptive_smoothness = gpu_const::adaptive_smoothness_c;
		bool complementary_smoothness = gpu_const::complementary_smoothness_c;

		idx_type nslice = nx*ny;
		idx_type nstack = nz*nslice;

		idx_type pos = (blockIdx.x*blockDim.x+threadIdx.x);
		if (pos >= nstack) pos = threadIdx.x;

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

		mathtype_solver normalizer_x = 1.f/(2.f*hx);
		mathtype_solver normalizer_y = 1.f/(2.f*hy);
		mathtype_solver normalizer_z = 1.f/(2.f*hz);
		/////////////////////////////////////////////

		//acquire from global memory
		//////////////////////////////////////////////////////////////
		__syncthreads();
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

		//Rotate vectors to edge aligned basis
		//////////////////////////////////////////////////////////////
		if (adaptive_smoothness)
		{
			__syncthreads();
			mathtype_solver theta = adaptivity[pos];
			mathtype_solver ksi = adaptivity[pos+nstack];
			mathtype_solver sintheta, costheta, sinksi, cosksi;

			__sincosf(theta, &sintheta, &costheta);
			__sincosf(ksi, &sinksi, &cosksi);

			float ux1 = ux*(costheta*cosksi) -uy*sintheta + uz*(costheta*sinksi);
			float uy1 = ux*(sintheta*cosksi) +uy*costheta + uz*(sintheta*sinksi);
			float uz1 = ux*(-sinksi)                      + uz*cosksi;

			float vx1 = vx*(costheta*cosksi) -vy*sintheta + vz*(costheta*sinksi);
			float vy1 = vx*(sintheta*cosksi) +vy*costheta + vz*(sintheta*sinksi);
			float vz1 = vx*(-sinksi)                      + vz*cosksi;

			float wx1 = wx*(costheta*cosksi) -wy*sintheta + wz*(costheta*sinksi);
			float wy1 = wx*(sintheta*cosksi) +wy*costheta + wz*(sintheta*sinksi);
			float wz1 = wx*(-sinksi)                      + wz*cosksi;

			ux = ux1; uy = uy1; uz = uz1;
			vx = vx1; vy = vy1; vz = vz1;
			wx = wx1; wy = wy1; wz = wz1;
		}
		//////////////////////////////////////////////////////////////

		//calculate smoothness term
		//////////////////////////////////////////////////////////////
		if (!decoupled)
		{
			mathtype_solver value;

			if(!anisotropic) value = 0.5f/sqrtf((ux*ux + uy*uy + uz*uz) + (vx*vx + vy*vy + vz*vz) + (wx*wx + wy*wy + wz*wz) + epsilon_phi_squared); //Isotropic flow driven := phi(tr(nabla_u1*nabla_u1^T + nabla_u2^T)
			else if(!adaptive_smoothness || !complementary_smoothness){
				value = 0.5f/(sqrtf(ux*ux + vx*vx + wx*wx + epsilon_phi_squared)
						    +sqrtf(uy*uy + vy*vy + wy*wy + epsilon_phi_squared)
						    +sqrtf(uz*uz + vz*vz + wz*wz + epsilon_phi_squared)); //Anisotropic flow driven := tr(phi(nabla_u1*nabla_u1^T + nabla_u2^T)
			}
			else{
				value = 0.166666667f/sqrtf(ux*ux + vx*vx + wx*wx + epsilon_phi_squared)
						   + 0.333333333f*sqrtf(uy*uy + vy*vy + wy*wy + epsilon_phi_squared)
						   + 0.333333333f*sqrtf(uz*uz + vz*vz + wz*wz + epsilon_phi_squared); //complementary following Zimmer2011:Optical flow in Harmony
			}

			__syncthreads();
			phi[pos] = value;
		}
		else
		{
			mathtype_solver value1, value2, value3;

			if(!anisotropic){
				//decoupled isotropic:
				value1 = 0.5f/sqrtf(ux*ux + uy*uy + uz*uz + epsilon_phi_squared);
				value2 = 0.5f/sqrtf(vx*vx + vy*vy + vz*vz + epsilon_phi_squared);
				value3 = 0.5f/sqrtf(wx*wx + wy*wy + wz*wz + epsilon_phi_squared);
			}
			else if (!adaptive_smoothness || !complementary_smoothness){
				//decoupled anisotropic:
				value1 = 0.5f/(sqrtf(ux*ux + epsilon_phi_squared)+sqrtf(uy*uy + epsilon_phi_squared)+sqrtf(uz*uz + epsilon_phi_squared));
				value2 = 0.5f/(sqrtf(vx*vx + epsilon_phi_squared)+sqrtf(vy*vy + epsilon_phi_squared)+sqrtf(vz*vz + epsilon_phi_squared));
				value3 = 0.5f/(sqrtf(wx*wx + epsilon_phi_squared)+sqrtf(wy*wy + epsilon_phi_squared)+sqrtf(wz*wz + epsilon_phi_squared));
			}
			else
			{
				//decoupled complementary
				value1 = 0.166666667f/sqrtf(ux*ux + epsilon_phi_squared)+0.333333333f*sqrtf(uy*uy + epsilon_phi_squared)+0.333333333f*sqrtf(uz*uz + epsilon_phi_squared);
				value2 = 0.166666667f/sqrtf(vx*vx + epsilon_phi_squared)+0.333333333f*sqrtf(vy*vy + epsilon_phi_squared)+0.333333333f*sqrtf(vz*vz + epsilon_phi_squared);
				value3 = 0.166666667f/sqrtf(wx*wx + epsilon_phi_squared)+0.333333333f*sqrtf(wy*wy + epsilon_phi_squared)+0.333333333f*sqrtf(wz*wz + epsilon_phi_squared);
			}

			__syncthreads();
			phi[pos] = value1;
			phi[nstack+pos] = value2;
			phi[2*nstack+pos] = value3;
		}
		//////////////////////////////////////////////////////////////

		return;
	}
	__global__ void update_smoothnessterm_forwardDiff(optflow_type *u, optflow_type *du,  optflow_type *phi, optflow_type *adaptivity)
	{
		//acquire constants and position
		/////////////////////////////////////////////
		int nx = gpu_const::nx_c;
		int ny = gpu_const::ny_c;
		int nz = gpu_const::nz_c;

		mathtype_solver epsilon_phi_squared = gpu_const::epsilon_phi_squared_c;
		mathtype_solver hx = gpu_const::hx_c;
		mathtype_solver hy = gpu_const::hy_c;
		mathtype_solver hz = gpu_const::hz_c;

		bool anisotropic = gpu_const::anisotropic_smoothness_c;
		bool decoupled = gpu_const::decoupled_smoothness_c;
		bool adaptive_smoothness = gpu_const::adaptive_smoothness_c;
		bool complementary_smoothness = gpu_const::complementary_smoothness_c;

		idx_type nslice = nx*ny;
		idx_type nstack = nz*nslice;

		idx_type pos = (blockIdx.x*blockDim.x+threadIdx.x);
		if (pos >= nstack) pos = threadIdx.x;

		int z = pos/nslice;
		int y = (pos-z*nslice)/nx;
		int x = pos-z*nslice-y*nx;

		int zp = z+1; //int zn = z-1;
		int yp = y+1; //int yn = y-1;
		int xp = x+1; //int xn = x-1;

		//Reflective boundary conditions (mirrored on first/last value)
		if (zp == nz) zp -= 2; //if (zn < 0) zn = 1;
		if (yp == ny) yp -= 2; //if (yn < 0) yn = 1;
		if (xp == nx) xp -= 2; //if (xn < 0) xn = 1;

		mathtype_solver normalizer_x = 1.f/hx;
		mathtype_solver normalizer_y = 1.f/hy;
		mathtype_solver normalizer_z = 1.f/hz;
		/////////////////////////////////////////////

		//acquire from global memory
		//////////////////////////////////////////////////////////////
		__syncthreads();
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

		//Rotate vectors to edge aligned basis
		//////////////////////////////////////////////////////////////
		if (adaptive_smoothness)
		{
			__syncthreads();
			mathtype_solver theta = adaptivity[pos];
			mathtype_solver ksi = adaptivity[pos+nstack];
			mathtype_solver sintheta, costheta, sinksi, cosksi;

			__sincosf(theta, &sintheta, &costheta);
			__sincosf(ksi, &sinksi, &cosksi);

			float ux1 = ux*(costheta*cosksi) -uy*sintheta + uz*(costheta*sinksi);
			float uy1 = ux*(sintheta*cosksi) +uy*costheta + uz*(sintheta*sinksi);
			float uz1 = ux*(-sinksi)                      + uz*cosksi;

			float vx1 = vx*(costheta*cosksi) -vy*sintheta + vz*(costheta*sinksi);
			float vy1 = vx*(sintheta*cosksi) +vy*costheta + vz*(sintheta*sinksi);
			float vz1 = vx*(-sinksi)                      + vz*cosksi;

			float wx1 = wx*(costheta*cosksi) -wy*sintheta + wz*(costheta*sinksi);
			float wy1 = wx*(sintheta*cosksi) +wy*costheta + wz*(sintheta*sinksi);
			float wz1 = wx*(-sinksi)                      + wz*cosksi;

			ux = ux1; uy = uy1; uz = uz1;
			vx = vx1; vy = vy1; vz = vz1;
			wx = wx1; wy = wy1; wz = wz1;
		}
		//////////////////////////////////////////////////////////////

		//calculate smoothness term
		//////////////////////////////////////////////////////////////
		if (!decoupled)
		{
			mathtype_solver value;

			if(!anisotropic) value = 0.5f/sqrtf((ux*ux + uy*uy + uz*uz) + (vx*vx + vy*vy + vz*vz) + (wx*wx + wy*wy + wz*wz) + epsilon_phi_squared); //Isotropic flow driven := phi(tr(nabla_u1*nabla_u1^T + nabla_u2^T)
			else if(!adaptive_smoothness || !complementary_smoothness){
				value = 0.5f/(sqrtf(ux*ux + vx*vx + wx*wx + epsilon_phi_squared)
						    +sqrtf(uy*uy + vy*vy + wy*wy + epsilon_phi_squared)
						    +sqrtf(uz*uz + vz*vz + wz*wz + epsilon_phi_squared)); //Anisotropic flow driven := tr(phi(nabla_u1*nabla_u1^T + nabla_u2^T)
			}
			else{
				value = 0.166666667f/sqrtf(ux*ux + vx*vx + wx*wx + epsilon_phi_squared)
						   + 0.333333333f*sqrtf(uy*uy + vy*vy + wy*wy + epsilon_phi_squared)
						   + 0.333333333f*sqrtf(uz*uz + vz*vz + wz*wz + epsilon_phi_squared); //complementary following Zimmer2011:Optical flow in Harmony
			}

			__syncthreads();
			phi[pos] = value;
		}
		else
		{
			mathtype_solver value1, value2, value3;

			if(!anisotropic){
				//decoupled isotropic:
				value1 = 0.5f/sqrtf(ux*ux + uy*uy + uz*uz + epsilon_phi_squared);
				value2 = 0.5f/sqrtf(vx*vx + vy*vy + vz*vz + epsilon_phi_squared);
				value3 = 0.5f/sqrtf(wx*wx + wy*wy + wz*wz + epsilon_phi_squared);
			}
			else if (!adaptive_smoothness || !complementary_smoothness){
				//decoupled anisotropic:
				value1 = 0.5f/(sqrtf(ux*ux + epsilon_phi_squared)+sqrtf(uy*uy + epsilon_phi_squared)+sqrtf(uz*uz + epsilon_phi_squared));
				value2 = 0.5f/(sqrtf(vx*vx + epsilon_phi_squared)+sqrtf(vy*vy + epsilon_phi_squared)+sqrtf(vz*vz + epsilon_phi_squared));
				value3 = 0.5f/(sqrtf(wx*wx + epsilon_phi_squared)+sqrtf(wy*wy + epsilon_phi_squared)+sqrtf(wz*wz + epsilon_phi_squared));
			}
			else
			{
				//decoupled complementary
				value1 = 0.166666667f/sqrtf(ux*ux + epsilon_phi_squared)+0.333333333f*sqrtf(uy*uy + epsilon_phi_squared)+0.333333333f*sqrtf(uz*uz + epsilon_phi_squared);
				value2 = 0.166666667f/sqrtf(vx*vx + epsilon_phi_squared)+0.333333333f*sqrtf(vy*vy + epsilon_phi_squared)+0.333333333f*sqrtf(vz*vz + epsilon_phi_squared);
				value3 = 0.166666667f/sqrtf(wx*wx + epsilon_phi_squared)+0.333333333f*sqrtf(wy*wy + epsilon_phi_squared)+0.333333333f*sqrtf(wz*wz + epsilon_phi_squared);
			}

			__syncthreads();
			phi[pos] = value1;
			phi[nstack+pos] = value2;
			phi[2*nstack+pos] = value3;
		}
		//////////////////////////////////////////////////////////////

		return;
	}

	//Derivatives accroding to Farid and Simoncelli 2004: "Differentation of Discrete Multidimensional Signals"
	//They are also used by: Kochba et al. 2015: "A Fast Iterative Digital Volume Correlation Algorithm for Large Deformations"
	__global__ void update_smoothnessterm_Farid3(optflow_type *u, optflow_type *du,  optflow_type *phi, optflow_type *adaptivity)
	{
		//Derivative accroding to Farid and Simoncelli 2004: "Differentation of Discrete Multidimensional Signals"
		//using a 3tap stencil

		//acquire constants and position
		/////////////////////////////////////////////
		int nx = gpu_const::nx_c;
		int ny = gpu_const::ny_c;
		int nz = gpu_const::nz_c;

		mathtype_solver epsilon_phi_squared = gpu_const::epsilon_phi_squared_c;
		mathtype_solver hx = gpu_const::hx_c;
		mathtype_solver hy = gpu_const::hy_c;
		mathtype_solver hz = gpu_const::hz_c;

		bool anisotropic = gpu_const::anisotropic_smoothness_c;
		bool decoupled = gpu_const::decoupled_smoothness_c;
		bool adaptive_smoothness = gpu_const::adaptive_smoothness_c;
		bool complementary_smoothness = gpu_const::complementary_smoothness_c;

		idx_type nslice = nx*ny;
		idx_type nstack = nz*nslice;

		idx_type pos = (blockIdx.x*blockDim.x+threadIdx.x);
		if (pos >= nstack) pos = threadIdx.x;

		int z = pos/nslice;
		int y = (pos-z*nslice)/nx;
		int x = pos-z*nslice-y*nx;
		/////////////////////////////////////////////

		 //partial derivatives
		//////////////////////////////////////////////////////////////
		mathtype_solver normalizer_x = 1.f/hx;
		mathtype_solver normalizer_y = 1.f/hy;
		mathtype_solver normalizer_z = 1.f/hz;

		mathtype_solver ux = 0.0f; mathtype_solver uy = 0.0f; mathtype_solver uz = 0.0f;
		mathtype_solver vx = 0.0f; mathtype_solver vy = 0.0f; mathtype_solver vz = 0.0f;
		mathtype_solver wx = 0.0f; mathtype_solver wy = 0.0f; mathtype_solver wz = 0.0f;
		//////////////////////////////////////////////////////////////

		float farid[4] = {0.0f, 0.12412487720f, 0.05281651765f, 0.02247401886f};

		__syncthreads();
		for(int r = -1; r <= 1; r++)
		{
			int z2 = z+r;
			if (z2 == nz) z2 -= 2; if (z2 < 0) z2 = 1;

			int n_edges_r = abs(r);

			for (int q = -1; q <= 1; q++)
			{
				int y2 = y+q;
				if (y2 == ny) y2 -= 2; if (y2 < 0) y2 = 1;
				int n_edges_q = n_edges_r + abs(q);

				for(int p = -1; p <= 1; p++)
				{
					int x2 = x+p;
					if (x2 == nx) x2 -= 2; if (x2 < 0) x2 = 1;

					int n_edges = n_edges_q + abs(p);
					if (n_edges == 0) continue;

					__syncthreads();
					optflow_type this_u = u[         z2*nslice+y2*nx+x2] + du[         z2*nslice+y2*nx+x2];
					optflow_type this_v = u[  nstack+z2*nslice+y2*nx+x2] + du[  nstack+z2*nslice+y2*nx+x2];
					optflow_type this_w = u[2*nstack+z2*nslice+y2*nx+x2] + du[2*nstack+z2*nslice+y2*nx+x2];

					float this_weight = farid[n_edges];

					ux += normalizer_x*p*this_weight*this_u;
					vx += normalizer_x*p*this_weight*this_v;
					wx += normalizer_x*p*this_weight*this_w;

					uy += normalizer_y*q*this_weight*this_u;
					vy += normalizer_y*q*this_weight*this_v;
					wy += normalizer_y*q*this_weight*this_w;

					uz += normalizer_z*r*this_weight*this_u;
					vz += normalizer_z*r*this_weight*this_v;
					wz += normalizer_z*r*this_weight*this_w;
				}
			}
		}
		//////////////////////////////////////////////////////////////

		//Rotate vectors to edge aligned basis
		//////////////////////////////////////////////////////////////
		if (adaptive_smoothness)
		{
			__syncthreads();
			mathtype_solver theta = adaptivity[pos];
			mathtype_solver ksi = adaptivity[pos+nstack];
			mathtype_solver sintheta, costheta, sinksi, cosksi;

			__sincosf(theta, &sintheta, &costheta);
			__sincosf(ksi, &sinksi, &cosksi);

			float ux1 = ux*(costheta*cosksi) -uy*sintheta + uz*(costheta*sinksi);
			float uy1 = ux*(sintheta*cosksi) +uy*costheta + uz*(sintheta*sinksi);
			float uz1 = ux*(-sinksi)                      + uz*cosksi;

			float vx1 = vx*(costheta*cosksi) -vy*sintheta + vz*(costheta*sinksi);
			float vy1 = vx*(sintheta*cosksi) +vy*costheta + vz*(sintheta*sinksi);
			float vz1 = vx*(-sinksi)                      + vz*cosksi;

			float wx1 = wx*(costheta*cosksi) -wy*sintheta + wz*(costheta*sinksi);
			float wy1 = wx*(sintheta*cosksi) +wy*costheta + wz*(sintheta*sinksi);
			float wz1 = wx*(-sinksi)                      + wz*cosksi;

			ux = ux1; uy = uy1; uz = uz1;
			vx = vx1; vy = vy1; vz = vz1;
			wx = wx1; wy = wy1; wz = wz1;
		}
		//////////////////////////////////////////////////////////////

		//calculate smoothness term
		//////////////////////////////////////////////////////////////
		if (!decoupled)
		{
			mathtype_solver value;

			if(!anisotropic) value = 0.5f/sqrtf((ux*ux + uy*uy + uz*uz) + (vx*vx + vy*vy + vz*vz) + (wx*wx + wy*wy + wz*wz) + epsilon_phi_squared); //Isotropic flow driven := phi(tr(nabla_u1*nabla_u1^T + nabla_u2^T)
			else if(!adaptive_smoothness || !complementary_smoothness){
				value = 0.5f/(sqrtf(ux*ux + vx*vx + wx*wx + epsilon_phi_squared)
							+sqrtf(uy*uy + vy*vy + wy*wy + epsilon_phi_squared)
							+sqrtf(uz*uz + vz*vz + wz*wz + epsilon_phi_squared)); //Anisotropic flow driven := tr(phi(nabla_u1*nabla_u1^T + nabla_u2^T)
			}
			else{
				value = 0.166666667f/sqrtf(ux*ux + vx*vx + wx*wx + epsilon_phi_squared)
						   + 0.333333333f*sqrtf(uy*uy + vy*vy + wy*wy + epsilon_phi_squared)
						   + 0.333333333f*sqrtf(uz*uz + vz*vz + wz*wz + epsilon_phi_squared); //complementary following Zimmer2011:Optical flow in Harmony
			}

			__syncthreads();
			phi[pos] = value;
		}
		else{
			mathtype_solver value1, value2, value3;

			if(!anisotropic){
				//decoupled isotropic:
				value1 = 0.5f/sqrtf(ux*ux + uy*uy + uz*uz + epsilon_phi_squared);
				value2 = 0.5f/sqrtf(vx*vx + vy*vy + vz*vz + epsilon_phi_squared);
				value3 = 0.5f/sqrtf(wx*wx + wy*wy + wz*wz + epsilon_phi_squared);
			}
			else if (!adaptive_smoothness || !complementary_smoothness){
				//decoupled anisotropic:
				value1 = 0.5f/(sqrtf(ux*ux + epsilon_phi_squared)+sqrtf(uy*uy + epsilon_phi_squared)+sqrtf(uz*uz + epsilon_phi_squared));
				value2 = 0.5f/(sqrtf(vx*vx + epsilon_phi_squared)+sqrtf(vy*vy + epsilon_phi_squared)+sqrtf(vz*vz + epsilon_phi_squared));
				value3 = 0.5f/(sqrtf(wx*wx + epsilon_phi_squared)+sqrtf(wy*wy + epsilon_phi_squared)+sqrtf(wz*wz + epsilon_phi_squared));
			}
			else
			{
				//decoupled complementary
				value1 = 0.166666667f/sqrtf(ux*ux + epsilon_phi_squared)+0.333333333f*sqrtf(uy*uy + epsilon_phi_squared)+0.333333333f*sqrtf(uz*uz + epsilon_phi_squared);
				value2 = 0.166666667f/sqrtf(vx*vx + epsilon_phi_squared)+0.333333333f*sqrtf(vy*vy + epsilon_phi_squared)+0.333333333f*sqrtf(vz*vz + epsilon_phi_squared);
				value3 = 0.166666667f/sqrtf(wx*wx + epsilon_phi_squared)+0.333333333f*sqrtf(wy*wy + epsilon_phi_squared)+0.333333333f*sqrtf(wz*wz + epsilon_phi_squared);
			}

			__syncthreads();
			phi[pos] = value1;
			phi[nstack+pos] = value2;
			phi[2*nstack+pos] = value3;
		}
		//////////////////////////////////////////////////////////////

		return;
	}
	__global__ void update_smoothnessterm_Farid5(optflow_type *u, optflow_type *du,  optflow_type *phi, optflow_type *adaptivity)
	{
		//Derivative accroding to Farid and Simoncelli 2004: "Differentation of Discrete Multidimensional Signals"
		//using a 5tap stencil

		//acquire constants and position
		/////////////////////////////////////////////
		int nx = gpu_const::nx_c;
		int ny = gpu_const::ny_c;
		int nz = gpu_const::nz_c;

		mathtype_solver epsilon_phi_squared = gpu_const::epsilon_phi_squared_c;
		mathtype_solver hx = gpu_const::hx_c;
		mathtype_solver hy = gpu_const::hy_c;
		mathtype_solver hz = gpu_const::hz_c;

		bool anisotropic = gpu_const::anisotropic_smoothness_c;
		bool decoupled = gpu_const::decoupled_smoothness_c;
		bool adaptive_smoothness = gpu_const::adaptive_smoothness_c;
		bool complementary_smoothness = gpu_const::complementary_smoothness_c;

		idx_type nslice = nx*ny;
		idx_type nstack = nz*nslice;

		idx_type pos = (blockIdx.x*blockDim.x+threadIdx.x);
		if (pos >= nstack) pos = threadIdx.x;

		int z = pos/nslice;
		int y = (pos-z*nslice)/nx;
		int x = pos-z*nslice-y*nx;
		/////////////////////////////////////////////

		 //partial derivatives
		//////////////////////////////////////////////////////////////
		mathtype_solver normalizer_x = 1.f/hx;
		mathtype_solver normalizer_y = 1.f/hy;
		mathtype_solver normalizer_z = 1.f/hz;

		mathtype_solver ux = 0.0f; mathtype_solver uy = 0.0f; mathtype_solver uz = 0.0f;
		mathtype_solver vx = 0.0f; mathtype_solver vy = 0.0f; mathtype_solver vz = 0.0f;
		mathtype_solver wx = 0.0f; mathtype_solver wy = 0.0f; mathtype_solver wz = 0.0f;
		//////////////////////////////////////////////////////////////

		float farid[27] = {
				0.0f, 0.0503013134f,  0.01992556825f,
				0.0f, 0.02939366736f, 0.01164354291f,
				0.0f, 0.004442796111f,0.001759899082f,

				0.0f, 0.02939366549f,  0.01164354291f,
				0.0f, 0.01717624255f,  0.006803925615f,
				0.0f, 0.002596156206f, 0.001028400264f,

				0.0f, 0.004442796111f, 0.001759899198f,
				0.0f, 0.002596156206f, 0.001028400264f,
				0.0f, 0.0003924040357f, 0.0001554407354f
		};

		__syncthreads();
		for(int r = -2; r <= 2; r++)
		{
			int z2 = z+r;
			if (z2 >= nz) z2 = 2*nz-z2-2; if (z2 < 0) z2 = -z2;

			int absr = abs(r);
			mathtype_solver sign_z = r < 0 ? -normalizer_z : normalizer_z;

			for (int q = -2; q <= 2; q++)
			{
				int y2 = y+q;
				if (y2 >= ny) y2 = 2*ny-y2-2; if (y2 < 0) y2 = -y2;
				int absq = abs(q);
				mathtype_solver sign_y = q < 0 ? -normalizer_y : normalizer_y;

				for(int p = -2; p <= 2; p++)
				{
					int x2 = x+p;
					if (x2 >= nx) x2 = 2*nx-x2-2; if (x2 < 0) x2 = -x2;
					if (p == 0 && q == 0 && r == 0) continue;
					int absp = abs(p);
					mathtype_solver sign_x = p < 0 ? -normalizer_x : normalizer_x;

					__syncthreads();
					optflow_type this_u = u[         z2*nslice+y2*nx+x2] + du[         z2*nslice+y2*nx+x2];
					optflow_type this_v = u[  nstack+z2*nslice+y2*nx+x2] + du[  nstack+z2*nslice+y2*nx+x2];
					optflow_type this_w = u[2*nstack+z2*nslice+y2*nx+x2] + du[2*nstack+z2*nslice+y2*nx+x2];

					int idx_x = absr*9 + absq*3 + absp;
					int idx_y = absr*9 + absp*3 + absq;
					int idx_z = absq*9 + absp*3 + absr;

					float this_weight_x = sign_x*farid[idx_x];
					float this_weight_y = sign_y*farid[idx_y];
					float this_weight_z = sign_z*farid[idx_z];

					ux += this_weight_x*this_u;
					vx += this_weight_x*this_v;
					wx += this_weight_x*this_w;

					uy += this_weight_y*this_u;
					vy += this_weight_y*this_v;
					wy += this_weight_y*this_w;

					uz += this_weight_z*this_u;
					vz += this_weight_z*this_v;
					wz += this_weight_z*this_w;
				}
			}
		}
		//////////////////////////////////////////////////////////////

		//Rotate vectors to edge aligned basis
		//////////////////////////////////////////////////////////////
		if (adaptive_smoothness)
		{
			__syncthreads();
			mathtype_solver theta = adaptivity[pos];
			mathtype_solver ksi = adaptivity[pos+nstack];
			mathtype_solver sintheta, costheta, sinksi, cosksi;

			__sincosf(theta, &sintheta, &costheta);
			__sincosf(ksi, &sinksi, &cosksi);

			float ux1 = ux*(costheta*cosksi) -uy*sintheta + uz*(costheta*sinksi);
			float uy1 = ux*(sintheta*cosksi) +uy*costheta + uz*(sintheta*sinksi);
			float uz1 = ux*(-sinksi)                      + uz*cosksi;

			float vx1 = vx*(costheta*cosksi) -vy*sintheta + vz*(costheta*sinksi);
			float vy1 = vx*(sintheta*cosksi) +vy*costheta + vz*(sintheta*sinksi);
			float vz1 = vx*(-sinksi)                      + vz*cosksi;

			float wx1 = wx*(costheta*cosksi) -wy*sintheta + wz*(costheta*sinksi);
			float wy1 = wx*(sintheta*cosksi) +wy*costheta + wz*(sintheta*sinksi);
			float wz1 = wx*(-sinksi)                      + wz*cosksi;

			ux = ux1; uy = uy1; uz = uz1;
			vx = vx1; vy = vy1; vz = vz1;
			wx = wx1; wy = wy1; wz = wz1;
		}
		//////////////////////////////////////////////////////////////

		//calculate smoothness term
		//////////////////////////////////////////////////////////////
		if (!decoupled)
		{
			mathtype_solver value;

			if(!anisotropic) value = 0.5f/sqrtf((ux*ux + uy*uy + uz*uz) + (vx*vx + vy*vy + vz*vz) + (wx*wx + wy*wy + wz*wz) + epsilon_phi_squared); //Isotropic flow driven := phi(tr(nabla_u1*nabla_u1^T + nabla_u2^T)
			else if(!adaptive_smoothness || !complementary_smoothness){
				value = 0.5f/(sqrtf(ux*ux + vx*vx + wx*wx + epsilon_phi_squared)
							+sqrtf(uy*uy + vy*vy + wy*wy + epsilon_phi_squared)
							+sqrtf(uz*uz + vz*vz + wz*wz + epsilon_phi_squared)); //Anisotropic flow driven := tr(phi(nabla_u1*nabla_u1^T + nabla_u2^T)
			}
			else{
				value = 0.166666667f/sqrtf(ux*ux + vx*vx + wx*wx + epsilon_phi_squared)
						   + 0.333333333f*sqrtf(uy*uy + vy*vy + wy*wy + epsilon_phi_squared)
						   + 0.333333333f*sqrtf(uz*uz + vz*vz + wz*wz + epsilon_phi_squared); //complementary following Zimmer2011:Optical flow in Harmony
			}

			__syncthreads();
			phi[pos] = value;
		}
		else{
			mathtype_solver value1, value2, value3;

			if(!anisotropic){
				//decoupled isotropic:
				value1 = 0.5f/sqrtf(ux*ux + uy*uy + uz*uz + epsilon_phi_squared);
				value2 = 0.5f/sqrtf(vx*vx + vy*vy + vz*vz + epsilon_phi_squared);
				value3 = 0.5f/sqrtf(wx*wx + wy*wy + wz*wz + epsilon_phi_squared);
			}
			else if (!adaptive_smoothness || !complementary_smoothness){
				//decoupled anisotropic:
				value1 = 0.5f/(sqrtf(ux*ux + epsilon_phi_squared)+sqrtf(uy*uy + epsilon_phi_squared)+sqrtf(uz*uz + epsilon_phi_squared));
				value2 = 0.5f/(sqrtf(vx*vx + epsilon_phi_squared)+sqrtf(vy*vy + epsilon_phi_squared)+sqrtf(vz*vz + epsilon_phi_squared));
				value3 = 0.5f/(sqrtf(wx*wx + epsilon_phi_squared)+sqrtf(wy*wy + epsilon_phi_squared)+sqrtf(wz*wz + epsilon_phi_squared));
			}
			else
			{
				//decoupled complementary
				value1 = 0.166666667f/sqrtf(ux*ux + epsilon_phi_squared)+0.333333333f*sqrtf(uy*uy + epsilon_phi_squared)+0.333333333f*sqrtf(uz*uz + epsilon_phi_squared);
				value2 = 0.166666667f/sqrtf(vx*vx + epsilon_phi_squared)+0.333333333f*sqrtf(vy*vy + epsilon_phi_squared)+0.333333333f*sqrtf(vz*vz + epsilon_phi_squared);
				value3 = 0.166666667f/sqrtf(wx*wx + epsilon_phi_squared)+0.333333333f*sqrtf(wy*wy + epsilon_phi_squared)+0.333333333f*sqrtf(wz*wz + epsilon_phi_squared);
			}

			__syncthreads();
			phi[pos] = value1;
			phi[nstack+pos] = value2;
			phi[2*nstack+pos] = value3;
		}
		//////////////////////////////////////////////////////////////

		return;
	}
	__global__ void update_smoothnessterm_Farid7(optflow_type *u, optflow_type *du,  optflow_type *phi, optflow_type *adaptivity)
	{
		//Derivative accroding to Farid and Simoncelli 2004: "Differentation of Discrete Multidimensional Signals"
		//using a 7tap stencil

		//acquire constants and position
		/////////////////////////////////////////////
		int nx = gpu_const::nx_c;
		int ny = gpu_const::ny_c;
		int nz = gpu_const::nz_c;

		mathtype_solver epsilon_phi_squared = gpu_const::epsilon_phi_squared_c;
		mathtype_solver hx = gpu_const::hx_c;
		mathtype_solver hy = gpu_const::hy_c;
		mathtype_solver hz = gpu_const::hz_c;

		bool anisotropic = gpu_const::anisotropic_smoothness_c;
		bool decoupled = gpu_const::decoupled_smoothness_c;
		bool adaptive_smoothness = gpu_const::adaptive_smoothness_c;
		bool complementary_smoothness = gpu_const::complementary_smoothness_c;

		idx_type nslice = nx*ny;
		idx_type nstack = nz*nslice;

		idx_type pos = (blockIdx.x*blockDim.x+threadIdx.x);
		if (pos >= nstack) pos = threadIdx.x;

		int z = pos/nslice;
		int y = (pos-z*nslice)/nx;
		int x = pos-z*nslice-y*nx;
		/////////////////////////////////////////////

		 //partial derivatives
		//////////////////////////////////////////////////////////////
		mathtype_solver normalizer_x = 1.f/hx;
		mathtype_solver normalizer_y = 1.f/hy;
		mathtype_solver normalizer_z = 1.f/hz;

		mathtype_solver ux = 0.0f; mathtype_solver uy = 0.0f; mathtype_solver uz = 0.0f;
		mathtype_solver vx = 0.0f; mathtype_solver vy = 0.0f; mathtype_solver vz = 0.0f;
		mathtype_solver wx = 0.0f; mathtype_solver wy = 0.0f; mathtype_solver wz = 0.0f;
		//////////////////////////////////////////////////////////////

		float farid[64] = {
				0.0f, 0.02518012933f, 0.01634971797f, 0.002439625794f,
				0.0f, 0.01711205766f, 0.01111103687f, 0.001657935209f,
				0.0f, 0.004833645653f,0.003138536355f,0.0004683172156f,
				0.0f, 0.0003284906852f,0.0002132924128f,3.182646469e-05f,

				0.0f,0.01711205766f,0.01111103687f,0.001657935092f,
				0.0f,0.01162911206f,0.00755090313f,0.001126709278f,
				0.0f,0.003284876933f,0.002132904949f,0.0003182617365f,
				0.0f,0.0002232376137f,0.0001449505071f,2.162881356e-05f,

				0.0f,0.004833645653f,0.003138536355f,0.0004683171865f,
				0.0f,0.003284876933f,0.002132904716f,0.0003182617365f,
				0.0f,0.0009278797079f,0.000602481945f,8.989944035e-05f,
				0.0f,6.305796705e-05f,4.094419273e-05f,6.109494279e-06f,

				0.0f,0.0003284907143f,0.0002132924274f,3.182646105e-05f,
				0.0f,0.0002232376137f,0.0001449505071f,2.162881356e-05f,
				0.0f,6.305796705e-05f,4.094419273e-05f,6.109494279e-06f,
				0.0f,4.285368959e-06f,2.782534466e-06f,4.151963537e-07f,
		};

		__syncthreads();
		for(int r = -3; r <= 3; r++)
		{
			int z2 = z+r;
			if (z2 >= nz) z2 = 2*nz-z2-2; if (z2 < 0) z2 = -z2;

			int absr = abs(r);
			mathtype_solver sign_z = r < 0 ? -normalizer_z : normalizer_z;

			for (int q = -3; q <= 3; q++)
			{
				int y2 = y+q;
				if (y2 >= ny) y2 = 2*ny-y2-2; if (y2 < 0) y2 = -y2;
				int absq = abs(q);
				mathtype_solver sign_y = q < 0 ? -normalizer_y : normalizer_y;

				for(int p = -3; p <= 3; p++)
				{
					int x2 = x+p;
					if (x2 >= nx) x2 = 2*nx-x2-2; if (x2 < 0) x2 = -x2;
					if (p == 0 && q == 0 && r == 0) continue;
					int absp = abs(p);
					mathtype_solver sign_x = p < 0 ? -normalizer_x : normalizer_x;

					__syncthreads();
					optflow_type this_u = u[         z2*nslice+y2*nx+x2] + du[         z2*nslice+y2*nx+x2];
					optflow_type this_v = u[  nstack+z2*nslice+y2*nx+x2] + du[  nstack+z2*nslice+y2*nx+x2];
					optflow_type this_w = u[2*nstack+z2*nslice+y2*nx+x2] + du[2*nstack+z2*nslice+y2*nx+x2];

					int idx_x = absr*16 + absq*4 + absp;
					int idx_y = absr*16 + absp*4 + absq;
					int idx_z = absq*16 + absp*4 + absr;

					float this_weight_x = sign_x*farid[idx_x];
					float this_weight_y = sign_y*farid[idx_y];
					float this_weight_z = sign_z*farid[idx_z];

					ux += this_weight_x*this_u;
					vx += this_weight_x*this_v;
					wx += this_weight_x*this_w;

					uy += this_weight_y*this_u;
					vy += this_weight_y*this_v;
					wy += this_weight_y*this_w;

					uz += this_weight_z*this_u;
					vz += this_weight_z*this_v;
					wz += this_weight_z*this_w;
				}
			}
		}
		//////////////////////////////////////////////////////////////

		//Rotate vectors to edge aligned basis
		//////////////////////////////////////////////////////////////
		if (adaptive_smoothness)
		{
			__syncthreads();
			mathtype_solver theta = adaptivity[pos];
			mathtype_solver ksi = adaptivity[pos+nstack];
			mathtype_solver sintheta, costheta, sinksi, cosksi;

			__sincosf(theta, &sintheta, &costheta);
			__sincosf(ksi, &sinksi, &cosksi);

			float ux1 = ux*(costheta*cosksi) -uy*sintheta + uz*(costheta*sinksi);
			float uy1 = ux*(sintheta*cosksi) +uy*costheta + uz*(sintheta*sinksi);
			float uz1 = ux*(-sinksi)                      + uz*cosksi;

			float vx1 = vx*(costheta*cosksi) -vy*sintheta + vz*(costheta*sinksi);
			float vy1 = vx*(sintheta*cosksi) +vy*costheta + vz*(sintheta*sinksi);
			float vz1 = vx*(-sinksi)                      + vz*cosksi;

			float wx1 = wx*(costheta*cosksi) -wy*sintheta + wz*(costheta*sinksi);
			float wy1 = wx*(sintheta*cosksi) +wy*costheta + wz*(sintheta*sinksi);
			float wz1 = wx*(-sinksi)                      + wz*cosksi;

			ux = ux1; uy = uy1; uz = uz1;
			vx = vx1; vy = vy1; vz = vz1;
			wx = wx1; wy = wy1; wz = wz1;
		}
		//////////////////////////////////////////////////////////////

		//calculate smoothness term
		//////////////////////////////////////////////////////////////
		if (!decoupled)
		{
			mathtype_solver value;

			if(!anisotropic) value = 0.5f/sqrtf((ux*ux + uy*uy + uz*uz) + (vx*vx + vy*vy + vz*vz) + (wx*wx + wy*wy + wz*wz) + epsilon_phi_squared); //Isotropic flow driven := phi(tr(nabla_u1*nabla_u1^T + nabla_u2^T)
			else if(!adaptive_smoothness || !complementary_smoothness){
				value = 0.5f/(sqrtf(ux*ux + vx*vx + wx*wx + epsilon_phi_squared)
							+sqrtf(uy*uy + vy*vy + wy*wy + epsilon_phi_squared)
							+sqrtf(uz*uz + vz*vz + wz*wz + epsilon_phi_squared)); //Anisotropic flow driven := tr(phi(nabla_u1*nabla_u1^T + nabla_u2^T)
			}
			else{
				value = 0.166666667f/sqrtf(ux*ux + vx*vx + wx*wx + epsilon_phi_squared)
						   + 0.333333333f*sqrtf(uy*uy + vy*vy + wy*wy + epsilon_phi_squared)
						   + 0.333333333f*sqrtf(uz*uz + vz*vz + wz*wz + epsilon_phi_squared); //complementary following Zimmer2011:Optical flow in Harmony
			}

			__syncthreads();
			phi[pos] = value;
		}
		else{
			mathtype_solver value1, value2, value3;

			if(!anisotropic){
				//decoupled isotropic:
				value1 = 0.5f/sqrtf(ux*ux + uy*uy + uz*uz + epsilon_phi_squared);
				value2 = 0.5f/sqrtf(vx*vx + vy*vy + vz*vz + epsilon_phi_squared);
				value3 = 0.5f/sqrtf(wx*wx + wy*wy + wz*wz + epsilon_phi_squared);
			}
			else if (!adaptive_smoothness || !complementary_smoothness){
				//decoupled anisotropic:
				value1 = 0.5f/(sqrtf(ux*ux + epsilon_phi_squared)+sqrtf(uy*uy + epsilon_phi_squared)+sqrtf(uz*uz + epsilon_phi_squared));
				value2 = 0.5f/(sqrtf(vx*vx + epsilon_phi_squared)+sqrtf(vy*vy + epsilon_phi_squared)+sqrtf(vz*vz + epsilon_phi_squared));
				value3 = 0.5f/(sqrtf(wx*wx + epsilon_phi_squared)+sqrtf(wy*wy + epsilon_phi_squared)+sqrtf(wz*wz + epsilon_phi_squared));
			}
			else
			{
				//decoupled complementary
				value1 = 0.166666667f/sqrtf(ux*ux + epsilon_phi_squared)+0.333333333f*sqrtf(uy*uy + epsilon_phi_squared)+0.333333333f*sqrtf(uz*uz + epsilon_phi_squared);
				value2 = 0.166666667f/sqrtf(vx*vx + epsilon_phi_squared)+0.333333333f*sqrtf(vy*vy + epsilon_phi_squared)+0.333333333f*sqrtf(vz*vz + epsilon_phi_squared);
				value3 = 0.166666667f/sqrtf(wx*wx + epsilon_phi_squared)+0.333333333f*sqrtf(wy*wy + epsilon_phi_squared)+0.333333333f*sqrtf(wz*wz + epsilon_phi_squared);
			}

			__syncthreads();
			phi[pos] = value1;
			phi[nstack+pos] = value2;
			phi[2*nstack+pos] = value3;
		}
		//////////////////////////////////////////////////////////////

		return;
	}
	__global__ void update_smoothnessterm_Farid9(optflow_type *u, optflow_type *du,  optflow_type *phi, optflow_type *adaptivity)
	{
		//Derivative accroding to Farid and Simoncelli 2004: "Differentation of Discrete Multidimensional Signals"
		//using a 9tap stencil

		//acquire constants and position
		/////////////////////////////////////////////
		int nx = gpu_const::nx_c;
		int ny = gpu_const::ny_c;
		int nz = gpu_const::nz_c;

		mathtype_solver epsilon_phi_squared = gpu_const::epsilon_phi_squared_c;
		mathtype_solver hx = gpu_const::hx_c;
		mathtype_solver hy = gpu_const::hy_c;
		mathtype_solver hz = gpu_const::hz_c;

		bool anisotropic = gpu_const::anisotropic_smoothness_c;
		bool decoupled = gpu_const::decoupled_smoothness_c;
		bool adaptive_smoothness = gpu_const::adaptive_smoothness_c;
		bool complementary_smoothness = gpu_const::complementary_smoothness_c;

		idx_type nslice = nx*ny;
		idx_type nstack = nz*nslice;

		idx_type pos = (blockIdx.x*blockDim.x+threadIdx.x);
		if (pos >= nstack) pos = threadIdx.x;

		int z = pos/nslice;
		int y = (pos-z*nslice)/nx;
		int x = pos-z*nslice-y*nx;
		/////////////////////////////////////////////

		 //partial derivatives
		//////////////////////////////////////////////////////////////
		mathtype_solver normalizer_x = 1.f/hx;
		mathtype_solver normalizer_y = 1.f/hy;
		mathtype_solver normalizer_z = 1.f/hz;

		mathtype_solver ux = 0.0f; mathtype_solver uy = 0.0f; mathtype_solver uz = 0.0f;
		mathtype_solver vx = 0.0f; mathtype_solver vy = 0.0f; mathtype_solver vz = 0.0f;
		mathtype_solver wx = 0.0f; mathtype_solver wy = 0.0f; mathtype_solver wz = 0.0f;
		//////////////////////////////////////////////////////////////

		float farid[125] = {
		        0.0f,0.01454688795f,0.01200102083f,0.003556370735f,0.0003091749386f,
		        0.0f,0.01072974596f,0.008851921186f,0.00262316945f,0.0002280466142f,
		        0.0f,0.004133734852f,0.003410285106f,0.001010600477f,8.785707905e-05f,
		        0.0f,0.000708593172f,0.0005845814594f,0.0001732342935f,1.506021363e-05f,
		        0.0f,3.299081072e-05f,2.721705096e-05f,8.065474503e-06f,7.01176134e-07f,

		        0.0f,0.01072974596f,0.008851921186f,0.00262316945f,0.0002280465997f,
		        0.0f,0.00791423209f,0.006529153325f,0.001934842789f,0.0001682065777f,
		        0.0f,0.003049031831f,0.002515417291f,0.0007454162696f,6.480314914e-05f,
		        0.0f,0.0005226564244f,0.0004311857338f,0.0001277771516f,1.110837366e-05f,
		        0.0f,2.433393274e-05f,2.00752238e-05f,5.949072147e-06f,5.171856401e-07f,

		        0.0f,0.004133734852f,0.003410285106f,0.001010600594f,8.785708633e-05f,
		        0.0f,0.003049031831f,0.002515417291f,0.0007454162696f,6.480314914e-05f,
		        0.0f,0.001174667967f,0.0009690879378f,0.0002871785546f,2.496601883e-05f,
		        0.0f,0.0002013582707f,0.0001661183342f,4.922733933e-05f,4.279604582e-06f,
		        0.0f,9.374874935e-06f,7.734167411e-06f,2.291935516e-06f,1.992506071e-07f,

		        0.0f,0.000708593172f,0.0005845814594f,0.0001732342935f,1.506021363e-05f,
		        0.0f,0.0005226564244f,0.0004311857338f,0.0001277771516f,1.110837366e-05f,
		        0.0f,0.0002013582707f,0.0001661183342f,4.922733933e-05f,4.279604582e-06f,
		        0.0f,3.451626617e-05f,2.847553696e-05f,8.438411896e-06f,7.335977443e-07f,
		        0.0f,1.60701461e-06f,1.325769176e-06f,3.928771548e-07f,3.415497574e-08f,

		        0.0f,3.299081072e-05f,2.721705096e-05f,8.065474503e-06f,7.011761909e-07f,
		        0.0f,2.433393456e-05f,2.00752238e-05f,5.949072147e-06f,5.171856969e-07f,
		        0.0f,9.374874935e-06f,7.734167411e-06f,2.291935516e-06f,1.992506071e-07f,
		        0.0f,1.607014724e-06f,1.32576929e-06f,3.928771264e-07f,3.415497929e-08f,
		        0.0f,7.481968112e-08f,6.172540878e-08f,1.829164553e-08f,1.590193643e-09f
		};

		__syncthreads();
		for(int r = -4; r <= 4; r++)
		{
			int z2 = z+r;
			if (z2 >= nz) z2 = 2*nz-z2-2; if (z2 < 0) z2 = -z2;

			int absr = abs(r);
			mathtype_solver sign_z = r < 0 ? -normalizer_z : normalizer_z;

			for (int q = -4; q <= 4; q++)
			{
				int y2 = y+q;
				if (y2 >= ny) y2 = 2*ny-y2-2; if (y2 < 0) y2 = -y2;
				int absq = abs(q);
				mathtype_solver sign_y = q < 0 ? -normalizer_y : normalizer_y;

				for(int p = -4; p <= 4; p++)
				{
					int x2 = x+p;
					if (x2 >= nx) x2 = 2*nx-x2-2; if (x2 < 0) x2 = -x2;
					if (p == 0 && q == 0 && r == 0) continue;
					int absp = abs(p);
					mathtype_solver sign_x = p < 0 ? -normalizer_x : normalizer_x;

					__syncthreads();
					optflow_type this_u = u[         z2*nslice+y2*nx+x2] + du[         z2*nslice+y2*nx+x2];
					optflow_type this_v = u[  nstack+z2*nslice+y2*nx+x2] + du[  nstack+z2*nslice+y2*nx+x2];
					optflow_type this_w = u[2*nstack+z2*nslice+y2*nx+x2] + du[2*nstack+z2*nslice+y2*nx+x2];

					int idx_x = absr*25 + absq*5 + absp;
					int idx_y = absr*25 + absp*5 + absq;
					int idx_z = absq*25 + absp*5 + absr;

					float this_weight_x = sign_x*farid[idx_x];
					float this_weight_y = sign_y*farid[idx_y];
					float this_weight_z = sign_z*farid[idx_z];

					ux += this_weight_x*this_u;
					vx += this_weight_x*this_v;
					wx += this_weight_x*this_w;

					uy += this_weight_y*this_u;
					vy += this_weight_y*this_v;
					wy += this_weight_y*this_w;

					uz += this_weight_z*this_u;
					vz += this_weight_z*this_v;
					wz += this_weight_z*this_w;
				}
			}
		}
		//////////////////////////////////////////////////////////////

		//Rotate vectors to edge aligned basis
		//////////////////////////////////////////////////////////////
		if (adaptive_smoothness)
		{
			__syncthreads();
			mathtype_solver theta = adaptivity[pos];
			mathtype_solver ksi = adaptivity[pos+nstack];
			mathtype_solver sintheta, costheta, sinksi, cosksi;

			__sincosf(theta, &sintheta, &costheta);
			__sincosf(ksi, &sinksi, &cosksi);

			float ux1 = ux*(costheta*cosksi) -uy*sintheta + uz*(costheta*sinksi);
			float uy1 = ux*(sintheta*cosksi) +uy*costheta + uz*(sintheta*sinksi);
			float uz1 = ux*(-sinksi)                      + uz*cosksi;

			float vx1 = vx*(costheta*cosksi) -vy*sintheta + vz*(costheta*sinksi);
			float vy1 = vx*(sintheta*cosksi) +vy*costheta + vz*(sintheta*sinksi);
			float vz1 = vx*(-sinksi)                      + vz*cosksi;

			float wx1 = wx*(costheta*cosksi) -wy*sintheta + wz*(costheta*sinksi);
			float wy1 = wx*(sintheta*cosksi) +wy*costheta + wz*(sintheta*sinksi);
			float wz1 = wx*(-sinksi)                      + wz*cosksi;

			ux = ux1; uy = uy1; uz = uz1;
			vx = vx1; vy = vy1; vz = vz1;
			wx = wx1; wy = wy1; wz = wz1;
		}
		//////////////////////////////////////////////////////////////

		//calculate smoothness term
		//////////////////////////////////////////////////////////////
		if (!decoupled)
		{
			mathtype_solver value;

			if(!anisotropic) value = 0.5f/sqrtf((ux*ux + uy*uy + uz*uz) + (vx*vx + vy*vy + vz*vz) + (wx*wx + wy*wy + wz*wz) + epsilon_phi_squared); //Isotropic flow driven := phi(tr(nabla_u1*nabla_u1^T + nabla_u2^T)
			else if(!adaptive_smoothness || !complementary_smoothness){
				value = 0.5f/(sqrtf(ux*ux + vx*vx + wx*wx + epsilon_phi_squared)
							+sqrtf(uy*uy + vy*vy + wy*wy + epsilon_phi_squared)
							+sqrtf(uz*uz + vz*vz + wz*wz + epsilon_phi_squared)); //Anisotropic flow driven := tr(phi(nabla_u1*nabla_u1^T + nabla_u2^T)
			}
			else{
				value = 0.166666667f/sqrtf(ux*ux + vx*vx + wx*wx + epsilon_phi_squared)
						   + 0.333333333f*sqrtf(uy*uy + vy*vy + wy*wy + epsilon_phi_squared)
						   + 0.333333333f*sqrtf(uz*uz + vz*vz + wz*wz + epsilon_phi_squared); //complementary following Zimmer2011:Optical flow in Harmony
			}

			__syncthreads();
			phi[pos] = value;
		}
		else{
			mathtype_solver value1, value2, value3;

			if(!anisotropic){
				//decoupled isotropic:
				value1 = 0.5f/sqrtf(ux*ux + uy*uy + uz*uz + epsilon_phi_squared);
				value2 = 0.5f/sqrtf(vx*vx + vy*vy + vz*vz + epsilon_phi_squared);
				value3 = 0.5f/sqrtf(wx*wx + wy*wy + wz*wz + epsilon_phi_squared);
			}
			else if (!adaptive_smoothness || !complementary_smoothness){
				//decoupled anisotropic:
				value1 = 0.5f/(sqrtf(ux*ux + epsilon_phi_squared)+sqrtf(uy*uy + epsilon_phi_squared)+sqrtf(uz*uz + epsilon_phi_squared));
				value2 = 0.5f/(sqrtf(vx*vx + epsilon_phi_squared)+sqrtf(vy*vy + epsilon_phi_squared)+sqrtf(vz*vz + epsilon_phi_squared));
				value3 = 0.5f/(sqrtf(wx*wx + epsilon_phi_squared)+sqrtf(wy*wy + epsilon_phi_squared)+sqrtf(wz*wz + epsilon_phi_squared));
			}
			else
			{
				//decoupled complementary
				value1 = 0.166666667f/sqrtf(ux*ux + epsilon_phi_squared)+0.333333333f*sqrtf(uy*uy + epsilon_phi_squared)+0.333333333f*sqrtf(uz*uz + epsilon_phi_squared);
				value2 = 0.166666667f/sqrtf(vx*vx + epsilon_phi_squared)+0.333333333f*sqrtf(vy*vy + epsilon_phi_squared)+0.333333333f*sqrtf(vz*vz + epsilon_phi_squared);
				value3 = 0.166666667f/sqrtf(wx*wx + epsilon_phi_squared)+0.333333333f*sqrtf(wy*wy + epsilon_phi_squared)+0.333333333f*sqrtf(wz*wz + epsilon_phi_squared);
			}

			__syncthreads();
			phi[pos] = value1;
			phi[nstack+pos] = value2;
			phi[2*nstack+pos] = value3;
		}
		//////////////////////////////////////////////////////////////

		return;
	}
}
}

#endif //SMOOTHNESSTERM_GPU3D_CUH
