#ifndef SMOOTHNESSTERM_GPU2D_CUH
#define SMOOTHNESSTERM_GPU2D_CUH

#include <iostream>
#include <cuda.h>
#include "../Solver/optflow_base.h"
#include "../Solver/gpu_constants.cuh"

//References:
//	Weickert and Schnoerr 2001: "A Theoretical Framework for Convex Regularizers in PDE-Based Computation of Image Motion"
//	Sun et al. 2008
//	Zimmer et al. 2011

namespace optflow
{
namespace gpu2d
{
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

		bool anisotropic = gpu_const::anisotropic_smoothness_c;
		bool decoupled = gpu_const::decoupled_smoothness_c;
		bool adaptive_smoothness = gpu_const::adaptive_smoothness_c;
		bool complementary_smoothness = gpu_const::complementary_smoothness_c;

		idx_type nslice = nx*ny;
		idx_type nstack = nz*nslice;

		idx_type pos = (blockIdx.x*blockDim.x+threadIdx.x);
		if (pos >= nstack) pos = threadIdx.x;

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

		mathtype_solver normalizer_x = 1.f/(12.f*hx);
		mathtype_solver normalizer_y = 1.f/(12.f*hy);
		/////////////////////////////////////////////

		//acquire from global memory
		//////////////////////////////////////////////////////////////
		__syncthreads();

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

		//Rotate vectors to edge aligned basis
		//////////////////////////////////////////////////////////////
		if (adaptive_smoothness)
		{
			mathtype_solver theta = adaptivity[pos];
			mathtype_solver sintheta, costheta;

			__sincosf(theta, &sintheta, &costheta);

			float tmp_val = ux*costheta - uy*sintheta;
					   uy = ux*sintheta + uy*costheta;
					   ux = tmp_val;

			tmp_val = vx*costheta - vy*sintheta;
				 vy = vx*sintheta + vy*costheta;
				 vx = tmp_val;
		}
		//////////////////////////////////////////////////////////////

		//calculate smoothness term
		//////////////////////////////////////////////////////////////
		if (!decoupled)
		{
			mathtype_solver value;

			if(!anisotropic) value = 0.5f/sqrtf(ux*ux + vx*vx + uy*uy + vy*vy + epsilon_phi_squared); //Isotropic flow driven := phi(tr(nabla_u1*nabla_u1^T + nabla_u2^T)
			else if(!adaptive_smoothness || !complementary_smoothness) value = 0.5f/(sqrtf(ux*ux + vx*vx + epsilon_phi_squared)+sqrtf(uy*uy + vy*vy + epsilon_phi_squared)); //Anisotropic flow driven := tr(phi(nabla_u1*nabla_u1^T + nabla_u2^T)
			else value = 0.25f/sqrtf(ux*ux + vx*vx + epsilon_phi_squared) + 0.5f*sqrtf(uy*uy + vy*vy + epsilon_phi_squared); //following Zimmer2011:Optical flow in Harmony

			__syncthreads();
			phi[pos] = value;
		}
		else
		{
			mathtype_solver value1, value2;

			if(!anisotropic){
				//decoupled isotropic:
				value1 = 0.5f/sqrtf(ux*ux + uy*uy + epsilon_phi_squared);
				value2 = 0.5f/sqrtf(vx*vx + vy*vy + epsilon_phi_squared);
			}
			else if (!adaptive_smoothness || !complementary_smoothness){
				//decoupled anisotropic:
				value1 = 0.5f/(sqrtf(ux*ux + epsilon_phi_squared)+sqrtf(uy*uy + epsilon_phi_squared));
				value2 = 0.5f/(sqrtf(vx*vx + epsilon_phi_squared)+sqrtf(vy*vy + epsilon_phi_squared));
			}
			else
			{
				//decoupled complementary
				value1 = 0.25f/sqrtf(ux*ux + epsilon_phi_squared)+0.5f*sqrtf(uy*uy + epsilon_phi_squared);
				value2 = 0.25f/sqrtf(vx*vx + epsilon_phi_squared)+0.5f*sqrtf(vy*vy + epsilon_phi_squared);
			}

			__syncthreads();
			phi[pos] = value1;
			phi[nstack+pos] = value2;
		}
		//////////////////////////////////////////////////////////////

		return;
	}
	__global__ void update_smoothnessterm_Weickert(optflow_type *u, optflow_type *du,  optflow_type *phi, optflow_type *adaptivity)
	{
		//acquire constants and position
		/////////////////////////////////////////////
		int nx = gpu_const::nx_c;
		int ny = gpu_const::ny_c;
		int nz = gpu_const::nz_c;

		mathtype_solver epsilon_phi_squared = gpu_const::epsilon_phi_squared_c;
		mathtype_solver hx = gpu_const::hx_c;
		mathtype_solver hy = gpu_const::hy_c;

		bool anisotropic = gpu_const::anisotropic_smoothness_c;
		bool decoupled = gpu_const::decoupled_smoothness_c;
		bool adaptive_smoothness = gpu_const::adaptive_smoothness_c;
		bool complementary_smoothness = gpu_const::complementary_smoothness_c;

		idx_type nslice = nx*ny;
		idx_type nstack = nz*nslice;

		idx_type pos = (blockIdx.x*blockDim.x+threadIdx.x);
		if (pos >= nstack) pos = threadIdx.x;

		int y = pos/nx;
		int x = pos-y*nx;

		int yp = y+1; int yn = y-1;
		int xp = x+1; int xn = x-1;
		int yp2 = y+2; int yn2 = y-2;
		int xp2 = x+2; int xn2 = x-2;
		int yp3 = y+3; int yn3 = y-3;
		int xp3 = x+3; int xn3 = x-3;

		//Reflective boundary conditions (mirrored on first/last value)
		if (yp == ny) yp -= 2; if (yn < 0) yn = 1;
		if (xp == nx) xp -= 2; if (xn < 0) xn = 1;
		if (yp2 >= ny) yp2 = 2*ny-yp2-2; if (yn2 < 0) yn2 = -yn2;
		if (xp2 >= nx) xp2 = 2*nx-xp2-2; if (xn2 < 0) xn2 = -xn2;
		if (yp3 >= ny) yp3 = 2*ny-yp3-3; if (yn3 < 0) yn3 = -yn3;
		if (xp3 >= nx) xp3 = 2*nx-xp3-3; if (xn3 < 0) xn3 = -xn3;

		mathtype_solver normalizer_x = 1.f/(60.f*hx);
		mathtype_solver normalizer_y = 1.f/(60.f*hy);
		/////////////////////////////////////////////

		//acquire from global memory
		//////////////////////////////////////////////////////////////
		__syncthreads();
		optflow_type u_xn3 = u[y*nx  +xn3];
		optflow_type u_xn2 = u[y*nx  +xn2];
		optflow_type u_xn =  u[y*nx  + xn];
		optflow_type u_xp =  u[y*nx+xp];
		optflow_type u_xp2 = u[y*nx+xp2];
		optflow_type u_xp3 = u[y*nx+xp3];
		optflow_type u_yn3 = u[yn3*nx+  x];
		optflow_type u_yn2 = u[yn2*nx+  x];
		optflow_type u_yn =  u[yn*nx + x ];
		optflow_type u_yp =  u[yp*nx+x];
		optflow_type u_yp2 = u[yp2*nx+x];
		optflow_type u_yp3 = u[yp3*nx+x];

		optflow_type v_xn3 = u[nstack + y*nx  +xn3];
		optflow_type v_xn2 = u[nstack + y*nx  +xn2];
		optflow_type v_xn =  u[nstack + y*nx  + xn];
		optflow_type v_xp =  u[nstack + y*nx+xp];
		optflow_type v_xp2 = u[nstack + y*nx+xp2];
		optflow_type v_xp3 = u[nstack + y*nx+xp3];
		optflow_type v_yn3 = u[nstack + yn3*nx+  x];
		optflow_type v_yn2 = u[nstack + yn2*nx+  x];
		optflow_type v_yn =  u[nstack + yn*nx + x ];
		optflow_type v_yp =  u[nstack + yp*nx+x];
		optflow_type v_yp2 = u[nstack + yp2*nx+x];
		optflow_type v_yp3 = u[nstack + yp3*nx+x];

		u_xn3 += du[y*nx  +xn3];
		u_xn2 += du[y*nx  +xn2];
		u_xn +=  du[y*nx  + xn];
		u_xp +=  du[y*nx+xp];
		u_xp2 += du[y*nx+xp2];
		u_xp3 += du[y*nx  +xp3];
		u_yn3 += du[yn3*nx+  x];
		u_yn2 += du[yn2*nx+  x];
		u_yn +=  du[yn*nx + x ];
		u_yp +=  du[yp*nx+x];
		u_yp2 += du[yp2*nx+x];
		u_yp3 += du[yp3*nx+x];

		v_xn3 += du[nstack + y*nx  +xn3];
		v_xn2 += du[nstack + y*nx  +xn2];
		v_xn +=  du[nstack + y*nx  + xn];
		v_xp +=  du[nstack + y*nx+xp];
		v_xp2 += du[nstack + y*nx+xp2];
		v_xp3 += du[nstack + y*nx+xp3];
		v_yn3 += du[nstack + yn3*nx+  x];
		v_yn2 += du[nstack + yn2*nx+  x];
		v_yn +=  du[nstack + yn*nx + x ];
		v_yp +=  du[nstack + yp*nx+x];
		v_yp2 += du[nstack + yp2*nx+x];
		v_yp3 += du[nstack + yp3*nx+x];
		//////////////////////////////////////////////////////////////

		//partial derivatives
		//////////////////////////////////////////////////////////////
		mathtype_solver ux = normalizer_x*(u_xn3 -9*u_xn2 +45*u_xn -45*u_xp + 9*u_xp2 -u_xp3);
		mathtype_solver uy = normalizer_y*(u_yn3 -9*u_yn2 +45*u_yn -45*u_yp + 9*u_yp2 -u_yp3);
		mathtype_solver vx = normalizer_x*(v_xn3 -9*v_xn2 +45*v_xn -45*v_xp + 9*v_xp2 -v_xp3);
		mathtype_solver vy = normalizer_y*(v_yn3 -9*v_yn2 +45*v_yn -45*v_yp + 9*v_yp2 -v_yp3);
		//////////////////////////////////////////////////////////////

		//Rotate vectors to edge aligned basis
		//////////////////////////////////////////////////////////////
		if (adaptive_smoothness)
		{
			mathtype_solver theta = adaptivity[pos];
			mathtype_solver sintheta, costheta;

			__sincosf(theta, &sintheta, &costheta);

			float tmp_val = ux*costheta - uy*sintheta;
					   uy = ux*sintheta + uy*costheta;
					   ux = tmp_val;

			tmp_val = vx*costheta - vy*sintheta;
				 vy = vx*sintheta + vy*costheta;
				 vx = tmp_val;
		}
		//////////////////////////////////////////////////////////////

		//calculate smoothness term
		//////////////////////////////////////////////////////////////
		if (!decoupled)
		{
			mathtype_solver value;

			if(!anisotropic) value = 0.5f/sqrtf(ux*ux + vx*vx + uy*uy + vy*vy + epsilon_phi_squared); //Isotropic flow driven := phi(tr(nabla_u1*nabla_u1^T + nabla_u2^T)
			else if(!adaptive_smoothness || !complementary_smoothness) value = 0.5f/(sqrtf(ux*ux + vx*vx + epsilon_phi_squared)+sqrtf(uy*uy + vy*vy + epsilon_phi_squared)); //Anisotropic flow driven := tr(phi(nabla_u1*nabla_u1^T + nabla_u2^T)
			else value = 0.25f/sqrtf(ux*ux + vx*vx + epsilon_phi_squared) + 0.5f*sqrtf(uy*uy + vy*vy + epsilon_phi_squared); //following Zimmer2011:Optical flow in Harmony

			__syncthreads();
			phi[pos] = value;
		}
		else
		{
			mathtype_solver value1, value2;

			if(!anisotropic){
				//decoupled isotropic:
				value1 = 0.5f/sqrtf(ux*ux + uy*uy + epsilon_phi_squared);
				value2 = 0.5f/sqrtf(vx*vx + vy*vy + epsilon_phi_squared);
			}
			else if (!adaptive_smoothness || !complementary_smoothness){
				//decoupled anisotropic:
				value1 = 0.5f/(sqrtf(ux*ux + epsilon_phi_squared)+sqrtf(uy*uy + epsilon_phi_squared));
				value2 = 0.5f/(sqrtf(vx*vx + epsilon_phi_squared)+sqrtf(vy*vy + epsilon_phi_squared));
			}
			else
			{
				//decoupled complementary
				value1 = 0.25f/sqrtf(ux*ux + epsilon_phi_squared)+0.5f*sqrtf(uy*uy + epsilon_phi_squared);
				value2 = 0.25f/sqrtf(vx*vx + epsilon_phi_squared)+0.5f*sqrtf(vy*vy + epsilon_phi_squared);
			}

			__syncthreads();
			phi[pos] = value1;
			phi[nstack+pos] = value2;
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

		bool anisotropic = gpu_const::anisotropic_smoothness_c;
		bool decoupled = gpu_const::decoupled_smoothness_c;
		bool adaptive_smoothness = gpu_const::adaptive_smoothness_c;
		bool complementary_smoothness = gpu_const::complementary_smoothness_c;

		idx_type nslice = nx*ny;
		idx_type nstack = nz*nslice;

		idx_type pos = (blockIdx.x*blockDim.x+threadIdx.x);
		if (pos >= nstack) pos = threadIdx.x;

		int y = pos/nx;
		int x = pos-y*nx;

		int yp = y+1; int yn = y-1;
		int xp = x+1; int xn = x-1;

		//Reflective boundary conditions (mirrored on first/last value)
		if (yp == ny) yp -= 2; if (yn < 0) yn = 1;
		if (xp == nx) xp -= 2; if (xn < 0) xn = 1;

		mathtype_solver normalizer_x = 1.f/(2.f*hx);
		mathtype_solver normalizer_y = 1.f/(2.f*hy);
		/////////////////////////////////////////////

		//acquire from global memory
		//////////////////////////////////////////////////////////////
		__syncthreads();
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


		//Rotate vectors to edge aligned basis
		//////////////////////////////////////////////////////////////
		if (adaptive_smoothness)
		{
			mathtype_solver theta = adaptivity[pos];
			mathtype_solver sintheta, costheta;

			__sincosf(theta, &sintheta, &costheta);

			float tmp_val = ux*costheta - uy*sintheta;
					   uy = ux*sintheta + uy*costheta;
					   ux = tmp_val;

			tmp_val = vx*costheta - vy*sintheta;
				 vy = vx*sintheta + vy*costheta;
				 vx = tmp_val;
		}
		//////////////////////////////////////////////////////////////

		//calculate smoothness term
		//////////////////////////////////////////////////////////////
		if (!decoupled)
		{
			mathtype_solver value;

			if(!anisotropic) value = 0.5f/sqrtf(ux*ux + vx*vx + uy*uy + vy*vy + epsilon_phi_squared); //Isotropic flow driven := phi(tr(nabla_u1*nabla_u1^T + nabla_u2^T)
			else if(!adaptive_smoothness || !complementary_smoothness) value = 0.5f/(sqrtf(ux*ux + vx*vx + epsilon_phi_squared)+sqrtf(uy*uy + vy*vy + epsilon_phi_squared)); //Anisotropic flow driven := tr(phi(nabla_u1*nabla_u1^T + nabla_u2^T)
			else value = 0.25f/sqrtf(ux*ux + vx*vx + epsilon_phi_squared) + 0.5f*sqrtf(uy*uy + vy*vy + epsilon_phi_squared); //following Zimmer2011:Optical flow in Harmony

			__syncthreads();
			phi[pos] = value;
		}
		else
		{
			mathtype_solver value1, value2;

			if(!anisotropic){
				//decoupled isotropic:
				value1 = 0.5f/sqrtf(ux*ux + uy*uy + epsilon_phi_squared);
				value2 = 0.5f/sqrtf(vx*vx + vy*vy + epsilon_phi_squared);
			}
			else if (!adaptive_smoothness || !complementary_smoothness){
				//decoupled anisotropic:
				value1 = 0.5f/(sqrtf(ux*ux + epsilon_phi_squared)+sqrtf(uy*uy + epsilon_phi_squared));
				value2 = 0.5f/(sqrtf(vx*vx + epsilon_phi_squared)+sqrtf(vy*vy + epsilon_phi_squared));
			}
			else
			{
				//decoupled complementary
				value1 = 0.25f/sqrtf(ux*ux + epsilon_phi_squared)+0.5f*sqrtf(uy*uy + epsilon_phi_squared);
				value2 = 0.25f/sqrtf(vx*vx + epsilon_phi_squared)+0.5f*sqrtf(vy*vy + epsilon_phi_squared);
			}

			__syncthreads();
			phi[pos] = value1;
			phi[nstack+pos] = value2;
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

		bool anisotropic = gpu_const::anisotropic_smoothness_c;
		bool decoupled = gpu_const::decoupled_smoothness_c;
		bool adaptive_smoothness = gpu_const::adaptive_smoothness_c;
		bool complementary_smoothness = gpu_const::complementary_smoothness_c;

		idx_type nslice = nx*ny;
		idx_type nstack = nz*nslice;

		idx_type pos = (blockIdx.x*blockDim.x+threadIdx.x);
		if (pos >= nstack) pos = threadIdx.x;

		int y = pos/nx;
		int x = pos-y*nx;

		int yp = y+1; int yn = y-1;
		int xp = x+1; int xn = x-1;

		//Reflective boundary conditions (mirrored on first/last value)
		if (yp == ny) yp -= 2; if (yn < 0) yn = 1;
		if (xp == nx) xp -= 2; if (xn < 0) xn = 1;

		mathtype_solver normalizer_x = 1.f/hx;
		mathtype_solver normalizer_y = 1.f/hy;
		/////////////////////////////////////////////

		//acquire from global memory
		//////////////////////////////////////////////////////////////
		__syncthreads();
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

		//Rotate vectors to edge aligned basis
		//////////////////////////////////////////////////////////////
		if (adaptive_smoothness)
		{
			mathtype_solver theta = adaptivity[pos];
			mathtype_solver sintheta, costheta;

			__sincosf(theta, &sintheta, &costheta);

			float tmp_val = ux*costheta - uy*sintheta;
					   uy = ux*sintheta + uy*costheta;
					   ux = tmp_val;

			tmp_val = vx*costheta - vy*sintheta;
				 vy = vx*sintheta + vy*costheta;
				 vx = tmp_val;
		}
		//////////////////////////////////////////////////////////////

		//calculate smoothness term
		//////////////////////////////////////////////////////////////
		if (!decoupled)
		{
			mathtype_solver value;

			if(!anisotropic) value = 0.5f/sqrtf(ux*ux + vx*vx + uy*uy + vy*vy + epsilon_phi_squared); //Isotropic flow driven := phi(tr(nabla_u1*nabla_u1^T + nabla_u2^T)
			else if(!adaptive_smoothness || !complementary_smoothness) value = 0.5f/(sqrtf(ux*ux + vx*vx + epsilon_phi_squared)+sqrtf(uy*uy + vy*vy + epsilon_phi_squared)); //Anisotropic flow driven := tr(phi(nabla_u1*nabla_u1^T + nabla_u2^T)
			else value = 0.25f/sqrtf(ux*ux + vx*vx + epsilon_phi_squared) + 0.5f*sqrtf(uy*uy + vy*vy + epsilon_phi_squared); //following Zimmer2011:Optical flow in Harmony

			__syncthreads();
			phi[pos] = value;
		}
		else
		{
			mathtype_solver value1, value2;

			if(!anisotropic){
				//decoupled isotropic:
				value1 = 0.5f/sqrtf(ux*ux + uy*uy + epsilon_phi_squared);
				value2 = 0.5f/sqrtf(vx*vx + vy*vy + epsilon_phi_squared);
			}
			else if (!adaptive_smoothness || !complementary_smoothness){
				//decoupled anisotropic:
				value1 = 0.5f/(sqrtf(ux*ux + epsilon_phi_squared)+sqrtf(uy*uy + epsilon_phi_squared));
				value2 = 0.5f/(sqrtf(vx*vx + epsilon_phi_squared)+sqrtf(vy*vy + epsilon_phi_squared));
			}
			else
			{
				//decoupled complementary
				value1 = 0.25f/sqrtf(ux*ux + epsilon_phi_squared)+0.5f*sqrtf(uy*uy + epsilon_phi_squared);
				value2 = 0.25f/sqrtf(vx*vx + epsilon_phi_squared)+0.5f*sqrtf(vy*vy + epsilon_phi_squared);
			}

			__syncthreads();
			phi[pos] = value1;
			phi[nstack+pos] = value2;
		}
		//////////////////////////////////////////////////////////////

		return;
	}
	__global__ void update_smoothnessterm_LBM(optflow_type *u, optflow_type *du,  optflow_type *phi, optflow_type *adaptivity)
	{
		//Lattice Boltzmann style finite differences
		//
		//Ramadugu et al. 2013: "Lattice differential operators for computational physics"
		//

		//acquire constants and position
		/////////////////////////////////////////////
		int nx = gpu_const::nx_c;
		int ny = gpu_const::ny_c;
		int nz = gpu_const::nz_c;

		mathtype_solver epsilon_phi_squared = gpu_const::epsilon_phi_squared_c;
		mathtype_solver hx = gpu_const::hx_c;
		mathtype_solver hy = gpu_const::hy_c;

		bool anisotropic = gpu_const::anisotropic_smoothness_c;
		bool decoupled = gpu_const::decoupled_smoothness_c;
		bool adaptive_smoothness = gpu_const::adaptive_smoothness_c;
		bool complementary_smoothness = gpu_const::complementary_smoothness_c;

		idx_type nslice = nx*ny;
		idx_type nstack = nz*nslice;

		idx_type pos = (blockIdx.x*blockDim.x+threadIdx.x);
		if (pos >= nstack) pos = threadIdx.x;

		int y = pos/nx;
		int x = pos-y*nx;

		int yp = y+1; int yn = y-1;
		int xp = x+1; int xn = x-1;

		//Reflective boundary conditions (mirrored on first/last value)
		if (yp == ny) yp -= 2; if (yn < 0) yn = 1;
		if (xp == nx) xp -= 2; if (xn < 0) xn = 1;

		mathtype_solver w2 = 0.02777777777777777f;
		mathtype_solver w1 = 4*w2;

		mathtype_solver normalizer_x = 3.0f/hx;
		mathtype_solver normalizer_y = 3.0f/hy;
		/////////////////////////////////////////////

		//acquire from global memory
		//////////////////////////////////////////////////////////////
		__syncthreads();
		optflow_type u_x0yn =  u[yn*nx + x ];
		optflow_type u_x0yp =  u[yp*nx + x];
		optflow_type u_xpy0 =  u[y*nx  + xp];
		optflow_type u_xpyn =  u[yn*nx + xp];
		optflow_type u_xpyp =  u[yp*nx + xp];
		optflow_type u_xny0 =  u[y*nx  + xn];
		optflow_type u_xnyn =  u[yn*nx + xn];
		optflow_type u_xnyp =  u[yp*nx + xn];

		optflow_type v_x0yn =  u[nstack + yn*nx + x ];
		optflow_type v_x0yp =  u[nstack + yp*nx + x];
		optflow_type v_xpy0 =  u[nstack + y*nx  + xp];
		optflow_type v_xpyn =  u[nstack + yn*nx + xp];
		optflow_type v_xpyp =  u[nstack + yp*nx + xp];
		optflow_type v_xny0 =  u[nstack + y*nx  + xn];
		optflow_type v_xnyn =  u[nstack + yn*nx + xn];
		optflow_type v_xnyp =  u[nstack + yp*nx + xn];

		u_x0yn += du[yn*nx + x ];
		u_x0yp += du[yp*nx + x];
		u_xpy0 += du[y*nx  + xp];
		u_xpyn += du[yn*nx + xp];
		u_xpyp += du[yp*nx + xp];
		u_xny0 += du[y*nx  + xn];
		u_xnyn += du[yn*nx + xn];
		u_xnyp += du[yp*nx + xn];

		v_x0yn += du[nstack + yn*nx + x ];
		v_x0yp += du[nstack + yp*nx + x];
		v_xpy0 += du[nstack + y*nx  + xp];
		v_xpyn += du[nstack + yn*nx + xp];
		v_xpyp += du[nstack + yp*nx + xp];
		v_xny0 += du[nstack + y*nx  + xn];
		v_xnyn += du[nstack + yn*nx + xn];
		v_xnyp += du[nstack + yp*nx + xn];
		//////////////////////////////////////////////////////////////

		 //partial derivatives
		//////////////////////////////////////////////////////////////
		mathtype_solver ux = normalizer_x*(w1*(u_xpy0 - u_xny0) + w2*(u_xpyn +u_xpyp -u_xnyn -u_xnyp));
		mathtype_solver uy = normalizer_y*(w1*(u_x0yp - u_x0yn) + w2*(-u_xpyn +u_xpyp -u_xnyn +u_xnyp));
		mathtype_solver vx = normalizer_x*(w1*(v_xpy0 - v_xny0) + w2*(v_xpyn +v_xpyp -v_xnyn -v_xnyp));
		mathtype_solver vy = normalizer_y*(w1*(v_x0yp - v_x0yn) + w2*(-v_xpyn +v_xpyp -v_xnyn +v_xnyp));
		//////////////////////////////////////////////////////////////

		//Rotate vectors to edge aligned basis
		//////////////////////////////////////////////////////////////
		if (adaptive_smoothness)
		{
			mathtype_solver theta = adaptivity[pos];
			mathtype_solver sintheta, costheta;

			__sincosf(theta, &sintheta, &costheta);

			float tmp_val = ux*costheta - uy*sintheta;
					   uy = ux*sintheta + uy*costheta;
					   ux = tmp_val;

			tmp_val = vx*costheta - vy*sintheta;
				 vy = vx*sintheta + vy*costheta;
				 vx = tmp_val;
		}
		//////////////////////////////////////////////////////////////

		//calculate smoothness term
		//////////////////////////////////////////////////////////////
		if (!decoupled)
		{
			mathtype_solver value;

			if(!anisotropic) value = 0.5f/sqrtf(ux*ux + vx*vx + uy*uy + vy*vy + epsilon_phi_squared); //Isotropic flow driven := phi(tr(nabla_u1*nabla_u1^T + nabla_u2^T)
			else if(!adaptive_smoothness || !complementary_smoothness) value = 0.5f/(sqrtf(ux*ux + vx*vx + epsilon_phi_squared)+sqrtf(uy*uy + vy*vy + epsilon_phi_squared)); //Anisotropic flow driven := tr(phi(nabla_u1*nabla_u1^T + nabla_u2^T)
			else value = 0.25f/sqrtf(ux*ux + vx*vx + epsilon_phi_squared) + 0.5f*sqrtf(uy*uy + vy*vy + epsilon_phi_squared); //following Zimmer2011:Optical flow in Harmony

			__syncthreads();
			phi[pos] = value;
		}
		else
		{
			mathtype_solver value1, value2;

			if(!anisotropic){
				//decoupled isotropic:
				value1 = 0.5f/sqrtf(ux*ux + uy*uy + epsilon_phi_squared);
				value2 = 0.5f/sqrtf(vx*vx + vy*vy + epsilon_phi_squared);
			}
			else if (!adaptive_smoothness || !complementary_smoothness){
				//decoupled anisotropic:
				value1 = 0.5f/(sqrtf(ux*ux + epsilon_phi_squared)+sqrtf(uy*uy + epsilon_phi_squared));
				value2 = 0.5f/(sqrtf(vx*vx + epsilon_phi_squared)+sqrtf(vy*vy + epsilon_phi_squared));
			}
			else
			{
				//decoupled complementary
				value1 = 0.25f/sqrtf(ux*ux + epsilon_phi_squared)+0.5f*sqrtf(uy*uy + epsilon_phi_squared);
				value2 = 0.25f/sqrtf(vx*vx + epsilon_phi_squared)+0.5f*sqrtf(vy*vy + epsilon_phi_squared);
			}

			__syncthreads();
			phi[pos] = value1;
			phi[nstack+pos] = value2;
		}
		//////////////////////////////////////////////////////////////

		return;
	}
	__global__ void update_smoothnessterm_Leclaire_FIII(optflow_type *u, optflow_type *du,  optflow_type *phi, optflow_type *adaptivity)
	{
		//Leclaire et al. 2011: "Isotropic color gradient for simulating very high-density ratios with a two-phase flow lattice Boltzmann model"

		//acquire constants and position
		/////////////////////////////////////////////
		int nx = gpu_const::nx_c;
		int ny = gpu_const::ny_c;
		int nz = gpu_const::nz_c;

		mathtype_solver epsilon_phi_squared = gpu_const::epsilon_phi_squared_c;
		mathtype_solver hx = gpu_const::hx_c;
		mathtype_solver hy = gpu_const::hy_c;

		bool anisotropic = gpu_const::anisotropic_smoothness_c;
		bool decoupled = gpu_const::decoupled_smoothness_c;
		bool adaptive_smoothness = gpu_const::adaptive_smoothness_c;
		bool complementary_smoothness = gpu_const::complementary_smoothness_c;

		idx_type nslice = nx*ny;
		idx_type nstack = nz*nslice;

		idx_type pos = (blockIdx.x*blockDim.x+threadIdx.x);
		if (pos >= nstack) pos = threadIdx.x;

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

		mathtype_solver w1 = 0.2666666666666667f;
		mathtype_solver w2 = 0.1f;
		mathtype_solver w3 = 0.00833333333333333f;

		mathtype_solver normalizer_x = 1.f/hx;
		mathtype_solver normalizer_y = 1.f/hy;
		/////////////////////////////////////////////

		//acquire from global memory
		//////////////////////////////////////////////////////////////
		__syncthreads();
		optflow_type u_x0yn =  u[yn*nx + x ];
		optflow_type u_x0yp =  u[yp*nx + x];
		optflow_type u_xpy0 =  u[y*nx  + xp];
		optflow_type u_xpyn =  u[yn*nx + xp];
		optflow_type u_xpyp =  u[yp*nx + xp];
		optflow_type u_xny0 =  u[y*nx  + xn];
		optflow_type u_xnyn =  u[yn*nx + xn];
		optflow_type u_xnyp =  u[yp*nx + xn];
		optflow_type u_xn2y0 =  u[y*nx + xn2];
		optflow_type u_xp2y0 =  u[y*nx + xp2];
		optflow_type u_x0yn2 =  u[yn2*nx + x];
		optflow_type u_x0yp2 =  u[yp2*nx + x];

		optflow_type v_x0yn =  u[nstack + yn*nx + x ];
		optflow_type v_x0yp =  u[nstack + yp*nx + x];
		optflow_type v_xpy0 =  u[nstack + y*nx  + xp];
		optflow_type v_xpyn =  u[nstack + yn*nx + xp];
		optflow_type v_xpyp =  u[nstack + yp*nx + xp];
		optflow_type v_xny0 =  u[nstack + y*nx  + xn];
		optflow_type v_xnyn =  u[nstack + yn*nx + xn];
		optflow_type v_xnyp =  u[nstack + yp*nx + xn];
		optflow_type v_xn2y0 =  u[nstack + y*nx + xn2];
		optflow_type v_xp2y0 =  u[nstack + y*nx + xp2];
		optflow_type v_x0yn2 =  u[nstack + yn2*nx + x];
		optflow_type v_x0yp2 =  u[nstack + yp2*nx + x];

		u_x0yn += du[yn*nx + x ];
		u_x0yp += du[yp*nx + x];
		u_xpy0 += du[y*nx  + xp];
		u_xpyn += du[yn*nx + xp];
		u_xpyp += du[yp*nx + xp];
		u_xny0 += du[y*nx  + xn];
		u_xnyn += du[yn*nx + xn];
		u_xnyp += du[yp*nx + xn];
		u_xn2y0 += du[y*nx + xn2];
		u_xp2y0 += du[y*nx + xp2];
		u_x0yn2 += du[yn2*nx + x];
		u_x0yp2 += du[yp2*nx + x];

		v_x0yn += du[nstack + yn*nx + x ];
		v_x0yp += du[nstack + yp*nx + x];
		v_xpy0 += du[nstack + y*nx  + xp];
		v_xpyn += du[nstack + yn*nx + xp];
		v_xpyp += du[nstack + yp*nx + xp];
		v_xny0 += du[nstack + y*nx  + xn];
		v_xnyn += du[nstack + yn*nx + xn];
		v_xnyp += du[nstack + yp*nx + xn];
		v_xn2y0 += du[nstack + y*nx + xn2];
		v_xp2y0 += du[nstack + y*nx + xp2];
		v_x0yn2 += du[nstack + yn2*nx + x];
		v_x0yp2 += du[nstack + yp2*nx + x];
		//////////////////////////////////////////////////////////////

		 //partial derivatives
		//////////////////////////////////////////////////////////////
		mathtype_solver ux = normalizer_x*(w1*(u_xpy0 - u_xny0) + w2*(u_xpyn +u_xpyp -u_xnyn -u_xnyp) + w3*(u_xp2y0-u_xn2y0));
		mathtype_solver uy = normalizer_y*(w1*(u_x0yp - u_x0yn) + w2*(-u_xpyn +u_xpyp -u_xnyn +u_xnyp)+ w3*(u_x0yp2-u_x0yn2));
		mathtype_solver vx = normalizer_x*(w1*(v_xpy0 - v_xny0) + w2*(v_xpyn +v_xpyp -v_xnyn -v_xnyp) + w3*(v_xp2y0-v_xn2y0));
		mathtype_solver vy = normalizer_y*(w1*(v_x0yp - v_x0yn) + w2*(-v_xpyn +v_xpyp -v_xnyn +v_xnyp)+ w3*(v_x0yp2-v_x0yn2));
		//////////////////////////////////////////////////////////////

		//Rotate vectors to edge aligned basis
		//////////////////////////////////////////////////////////////
		if (adaptive_smoothness)
		{
			mathtype_solver theta = adaptivity[pos];
			mathtype_solver sintheta, costheta;

			__sincosf(theta, &sintheta, &costheta);

			float tmp_val = ux*costheta - uy*sintheta;
					   uy = ux*sintheta + uy*costheta;
					   ux = tmp_val;

			tmp_val = vx*costheta - vy*sintheta;
				 vy = vx*sintheta + vy*costheta;
				 vx = tmp_val;
		}
		//////////////////////////////////////////////////////////////

		//calculate smoothness term
		//////////////////////////////////////////////////////////////
		if (!decoupled)
		{
			mathtype_solver value;

			if(!anisotropic) value = 0.5f/sqrtf(ux*ux + vx*vx + uy*uy + vy*vy + epsilon_phi_squared); //Isotropic flow driven := phi(tr(nabla_u1*nabla_u1^T + nabla_u2^T)
			else if(!adaptive_smoothness || !complementary_smoothness) value = 0.5f/(sqrtf(ux*ux + vx*vx + epsilon_phi_squared)+sqrtf(uy*uy + vy*vy + epsilon_phi_squared)); //Anisotropic flow driven := tr(phi(nabla_u1*nabla_u1^T + nabla_u2^T)
			else value = 0.25f/sqrtf(ux*ux + vx*vx + epsilon_phi_squared) + 0.5f*sqrtf(uy*uy + vy*vy + epsilon_phi_squared); //following Zimmer2011:Optical flow in Harmony

			__syncthreads();
			phi[pos] = value;
		}
		else
		{
			mathtype_solver value1, value2;

			if(!anisotropic){
				//decoupled isotropic:
				value1 = 0.5f/sqrtf(ux*ux + uy*uy + epsilon_phi_squared);
				value2 = 0.5f/sqrtf(vx*vx + vy*vy + epsilon_phi_squared);
			}
			else if (!adaptive_smoothness || !complementary_smoothness){
				//decoupled anisotropic:
				value1 = 0.5f/(sqrtf(ux*ux + epsilon_phi_squared)+sqrtf(uy*uy + epsilon_phi_squared));
				value2 = 0.5f/(sqrtf(vx*vx + epsilon_phi_squared)+sqrtf(vy*vy + epsilon_phi_squared));
			}
			else
			{
				//decoupled complementary
				value1 = 0.25f/sqrtf(ux*ux + epsilon_phi_squared)+0.5f*sqrtf(uy*uy + epsilon_phi_squared);
				value2 = 0.25f/sqrtf(vx*vx + epsilon_phi_squared)+0.5f*sqrtf(vy*vy + epsilon_phi_squared);
			}

			__syncthreads();
			phi[pos] = value1;
			phi[nstack+pos] = value2;
		}
		//////////////////////////////////////////////////////////////

		return;
	}
	__global__ void update_smoothnessterm_Leclaire_FIV(optflow_type *u, optflow_type *du,  optflow_type *phi, optflow_type *adaptivity)
	{
		//Leclaire et al. 2011: "Isotropic color gradient for simulating very high-density ratios with a two-phase flow lattice Boltzmann model"

		//acquire constants and position
		/////////////////////////////////////////////
		int nx = gpu_const::nx_c;
		int ny = gpu_const::ny_c;
		int nz = gpu_const::nz_c;

		mathtype_solver epsilon_phi_squared = gpu_const::epsilon_phi_squared_c;
		mathtype_solver hx = gpu_const::hx_c;
		mathtype_solver hy = gpu_const::hy_c;

		bool anisotropic = gpu_const::anisotropic_smoothness_c;
		bool decoupled = gpu_const::decoupled_smoothness_c;
		bool adaptive_smoothness = gpu_const::adaptive_smoothness_c;
		bool complementary_smoothness = gpu_const::complementary_smoothness_c;

		idx_type nslice = nx*ny;
		idx_type nstack = nz*nslice;

		idx_type pos = (blockIdx.x*blockDim.x+threadIdx.x);
		if (pos >= nstack) pos = threadIdx.x;

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

		mathtype_solver w1 = 0.19047619f;
		mathtype_solver w2 = 0.08888888888888889f;
		mathtype_solver w3 = 0.01666666666666667f;
		mathtype_solver w4 = 0.006349206f;
		mathtype_solver w5 = 0.000198413f;

		mathtype_solver normalizer_x = 1.f/hx;
		mathtype_solver normalizer_y = 1.f/hy;
		/////////////////////////////////////////////

		//acquire from global memory
		//////////////////////////////////////////////////////////////
		__syncthreads();
		optflow_type u_x0yn =  u[yn*nx + x ];
		optflow_type u_x0yp =  u[yp*nx + x];
		optflow_type u_xpy0 =  u[y*nx  + xp];
		optflow_type u_xpyn =  u[yn*nx + xp];
		optflow_type u_xpyp =  u[yp*nx + xp];
		optflow_type u_xny0 =  u[y*nx  + xn];
		optflow_type u_xnyn =  u[yn*nx + xn];
		optflow_type u_xnyp =  u[yp*nx + xn];
		optflow_type u_xn2y0 =  u[y*nx + xn2];
		optflow_type u_xp2y0 =  u[y*nx + xp2];
		optflow_type u_x0yn2 =  u[yn2*nx + x];
		optflow_type u_x0yp2 =  u[yp2*nx + x];

		optflow_type u_xn2yn2 =  u[yn2*nx + xn2];
		optflow_type u_xn2yn  =  u[yn*nx  + xn2];
		optflow_type u_xn2yp  =  u[yp*nx  + xn2];
		optflow_type u_xn2yp2 =  u[yp2*nx + xn2];
		optflow_type u_xp2yn2 =  u[yn2*nx + xp2];
		optflow_type u_xp2yn  =  u[yn*nx  + xp2];
		optflow_type u_xp2yp  =  u[yp*nx  + xp2];
		optflow_type u_xp2yp2 =  u[yp2*nx + xp2];
		optflow_type u_xnyn2 =  u[yn2*nx + xn2];
		optflow_type u_xpyn2 =  u[yn2*nx + xn2];
		optflow_type u_xnyp2 =  u[yp2*nx + xp2];
		optflow_type u_xpyp2 =  u[yp2*nx + xp2];

		optflow_type v_x0yn =  u[nstack + yn*nx + x ];
		optflow_type v_x0yp =  u[nstack + yp*nx + x];
		optflow_type v_xpy0 =  u[nstack + y*nx  + xp];
		optflow_type v_xpyn =  u[nstack + yn*nx + xp];
		optflow_type v_xpyp =  u[nstack + yp*nx + xp];
		optflow_type v_xny0 =  u[nstack + y*nx  + xn];
		optflow_type v_xnyn =  u[nstack + yn*nx + xn];
		optflow_type v_xnyp =  u[nstack + yp*nx + xn];
		optflow_type v_xn2y0 =  u[nstack + y*nx + xn2];
		optflow_type v_xp2y0 =  u[nstack + y*nx + xp2];
		optflow_type v_x0yn2 =  u[nstack + yn2*nx + x];
		optflow_type v_x0yp2 =  u[nstack + yp2*nx + x];

		optflow_type v_xn2yn2 =  u[nstack + yn2*nx + xn2];
		optflow_type v_xn2yn  =  u[nstack + yn*nx  + xn2];
		optflow_type v_xn2yp  =  u[nstack + yp*nx  + xn2];
		optflow_type v_xn2yp2 =  u[nstack + yp2*nx + xn2];
		optflow_type v_xp2yn2 =  u[nstack + yn2*nx + xp2];
		optflow_type v_xp2yn  =  u[nstack + yn*nx  + xp2];
		optflow_type v_xp2yp  =  u[nstack + yp*nx  + xp2];
		optflow_type v_xp2yp2 =  u[nstack + yp2*nx + xp2];
		optflow_type v_xnyn2 =  u[nstack + yn2*nx + xn2];
		optflow_type v_xpyn2 =  u[nstack + yn2*nx + xn2];
		optflow_type v_xnyp2 =  u[nstack + yp2*nx + xp2];
		optflow_type v_xpyp2 =  u[nstack + yp2*nx + xp2];

		u_x0yn += du[yn*nx + x ];
		u_x0yp += du[yp*nx + x];
		u_xpy0 += du[y*nx  + xp];
		u_xpyn += du[yn*nx + xp];
		u_xpyp += du[yp*nx + xp];
		u_xny0 += du[y*nx  + xn];
		u_xnyn += du[yn*nx + xn];
		u_xnyp += du[yp*nx + xn];
		u_xn2y0 += du[y*nx + xn2];
		u_xp2y0 += du[y*nx + xp2];
		u_x0yn2 += du[yn2*nx + x];
		u_x0yp2 += du[yp2*nx + x];

		u_xn2yn2 += du[yn2*nx + xn2];
		u_xn2yn  += du[yn*nx  + xn2];
		u_xn2yp  += du[yp*nx  + xn2];
		u_xn2yp2 += du[yp2*nx + xn2];
		u_xp2yn2 += du[yn2*nx + xp2];
		u_xp2yn  += du[yn*nx  + xp2];
		u_xp2yp  += du[yp*nx  + xp2];
		u_xp2yp2 += du[yp2*nx + xp2];
		u_xnyn2 += du[yn2*nx + xn2];
		u_xpyn2 += du[yn2*nx + xn2];
		u_xnyp2 += du[yp2*nx + xp2];
		u_xpyp2 += du[yp2*nx + xp2];

		v_x0yn += du[nstack + yn*nx + x ];
		v_x0yp += du[nstack + yp*nx + x];
		v_xpy0 += du[nstack + y*nx  + xp];
		v_xpyn += du[nstack + yn*nx + xp];
		v_xpyp += du[nstack + yp*nx + xp];
		v_xny0 += du[nstack + y*nx  + xn];
		v_xnyn += du[nstack + yn*nx + xn];
		v_xnyp += du[nstack + yp*nx + xn];
		v_xn2y0 += du[nstack + y*nx + xn2];
		v_xp2y0 += du[nstack + y*nx + xp2];
		v_x0yn2 += du[nstack + yn2*nx + x];
		v_x0yp2 += du[nstack + yp2*nx + x];

		v_xn2yn2 += du[nstack + yn2*nx + xn2];
		v_xn2yn  += du[nstack + yn*nx  + xn2];
		v_xn2yp  += du[nstack + yp*nx  + xn2];
		v_xn2yp2 += du[nstack + yp2*nx + xn2];
		v_xp2yn2 += du[nstack + yn2*nx + xp2];
		v_xp2yn  += du[nstack + yn*nx  + xp2];
		v_xp2yp  += du[nstack + yp*nx  + xp2];
		v_xp2yp2 += du[nstack + yp2*nx + xp2];
		v_xnyn2 += du[nstack + yn2*nx + xn2];
		v_xpyn2 += du[nstack + yn2*nx + xn2];
		v_xnyp2 += du[nstack + yp2*nx + xp2];
		v_xpyp2 += du[nstack + yp2*nx + xp2];
		//////////////////////////////////////////////////////////////

		 //partial derivatives
		//////////////////////////////////////////////////////////////
		mathtype_solver ux = normalizer_x*(w1*(u_xpy0 - u_xny0) + w2*(u_xpyn +u_xpyp -u_xnyn -u_xnyp) + w3*(u_xp2y0-u_xn2y0) + w4*(u_xp2yp+u_xp2yn-u_xn2yp-u_xn2yn)
				+ w5*(u_xp2yp2+u_xp2yn2-u_xn2yp2-u_xn2yn2));
		mathtype_solver uy = normalizer_y*(w1*(u_x0yp - u_x0yn) + w2*(-u_xpyn +u_xpyp -u_xnyn +u_xnyp)+ w3*(u_x0yp2-u_x0yn2) + w4*(u_xpyp2+u_xnyp2-u_xpyn2-u_xnyn2)
				+ w5*(u_xp2yp2-u_xp2yn2+u_xn2yp2-u_xn2yn2));
		mathtype_solver vx = normalizer_x*(w1*(v_xpy0 - v_xny0) + w2*(v_xpyn +v_xpyp -v_xnyn -v_xnyp) + w3*(v_xp2y0-v_xn2y0) + w4*(v_xp2yp+v_xp2yn-v_xn2yp-v_xn2yn)
				+ w5*(v_xp2yp2+v_xp2yn2-v_xn2yp2-v_xn2yn2));
		mathtype_solver vy = normalizer_y*(w1*(v_x0yp - v_x0yn) + w2*(-v_xpyn +v_xpyp -v_xnyn +v_xnyp)+ w3*(v_x0yp2-v_x0yn2) + w4*(v_xpyp2+v_xnyp2-v_xpyn2-v_xnyn2)
				+ w5*(v_xp2yp2-v_xp2yn2+v_xn2yp2-v_xn2yn2));
		//////////////////////////////////////////////////////////////

		//Rotate vectors to edge aligned basis
		//////////////////////////////////////////////////////////////
		if (adaptive_smoothness)
		{
			mathtype_solver theta = adaptivity[pos];
			mathtype_solver sintheta, costheta;

			__sincosf(theta, &sintheta, &costheta);

			float tmp_val = ux*costheta - uy*sintheta;
					   uy = ux*sintheta + uy*costheta;
					   ux = tmp_val;

			tmp_val = vx*costheta - vy*sintheta;
				 vy = vx*sintheta + vy*costheta;
				 vx = tmp_val;
		}
		//////////////////////////////////////////////////////////////

		//calculate smoothness term
		//////////////////////////////////////////////////////////////
		if (!decoupled)
		{
			mathtype_solver value;

			if(!anisotropic) value = 0.5f/sqrtf(ux*ux + vx*vx + uy*uy + vy*vy + epsilon_phi_squared); //Isotropic flow driven := phi(tr(nabla_u1*nabla_u1^T + nabla_u2^T)
			else if(!adaptive_smoothness || !complementary_smoothness) value = 0.5f/(sqrtf(ux*ux + vx*vx + epsilon_phi_squared)+sqrtf(uy*uy + vy*vy + epsilon_phi_squared)); //Anisotropic flow driven := tr(phi(nabla_u1*nabla_u1^T + nabla_u2^T)
			else value = 0.25f/sqrtf(ux*ux + vx*vx + epsilon_phi_squared) + 0.5f*sqrtf(uy*uy + vy*vy + epsilon_phi_squared); //following Zimmer2011:Optical flow in Harmony

			__syncthreads();
			phi[pos] = value;
		}
		else
		{
			mathtype_solver value1, value2;

			if(!anisotropic){
				//decoupled isotropic:
				value1 = 0.5f/sqrtf(ux*ux + uy*uy + epsilon_phi_squared);
				value2 = 0.5f/sqrtf(vx*vx + vy*vy + epsilon_phi_squared);
			}
			else if (!adaptive_smoothness || !complementary_smoothness){
				//decoupled anisotropic:
				value1 = 0.5f/(sqrtf(ux*ux + epsilon_phi_squared)+sqrtf(uy*uy + epsilon_phi_squared));
				value2 = 0.5f/(sqrtf(vx*vx + epsilon_phi_squared)+sqrtf(vy*vy + epsilon_phi_squared));
			}
			else
			{
				//decoupled complementary
				value1 = 0.25f/sqrtf(ux*ux + epsilon_phi_squared)+0.5f*sqrtf(uy*uy + epsilon_phi_squared);
				value2 = 0.25f/sqrtf(vx*vx + epsilon_phi_squared)+0.5f*sqrtf(vy*vy + epsilon_phi_squared);
			}

			__syncthreads();
			phi[pos] = value1;
			phi[nstack+pos] = value2;
		}
		//////////////////////////////////////////////////////////////

		return;
	}
	__global__ void update_smoothnessterm_Scharr3(optflow_type *u, optflow_type *du,  optflow_type *phi, optflow_type *adaptivity)
	{
		//acquire constants and position
		/////////////////////////////////////////////
		int nx = gpu_const::nx_c;
		int ny = gpu_const::ny_c;
		int nz = gpu_const::nz_c;

		mathtype_solver epsilon_phi_squared = gpu_const::epsilon_phi_squared_c;
		mathtype_solver hx = gpu_const::hx_c;
		mathtype_solver hy = gpu_const::hy_c;

		bool anisotropic = gpu_const::anisotropic_smoothness_c;
		bool decoupled = gpu_const::decoupled_smoothness_c;
		bool adaptive_smoothness = gpu_const::adaptive_smoothness_c;
		bool complementary_smoothness = gpu_const::complementary_smoothness_c;

		idx_type nslice = nx*ny;
		idx_type nstack = nz*nslice;

		idx_type pos = (blockIdx.x*blockDim.x+threadIdx.x);
		if (pos >= nstack) pos = threadIdx.x;

		int y = pos/nx;
		int x = pos-y*nx;

		int yp = y+1; int yn = y-1;
		int xp = x+1; int xn = x-1;

		//Reflective boundary conditions (mirrored on first/last value)
		if (yp == ny) yp -= 2; if (yn < 0) yn = 1;
		if (xp == nx) xp -= 2; if (xn < 0) xn = 1;

		mathtype_solver w1 = .6328125f;
		mathtype_solver w2 = .18359375f;

		mathtype_solver normalizer_x = 0.5f/hx;
		mathtype_solver normalizer_y = 0.5f/hy;
		/////////////////////////////////////////////

		//acquire from global memory
		//////////////////////////////////////////////////////////////
		__syncthreads();
		optflow_type u_x0yn =  u[yn*nx + x ];
		optflow_type u_x0yp =  u[yp*nx + x];
		optflow_type u_xpy0 =  u[y*nx  + xp];
		optflow_type u_xpyn =  u[yn*nx + xp];
		optflow_type u_xpyp =  u[yp*nx + xp];
		optflow_type u_xny0 =  u[y*nx  + xn];
		optflow_type u_xnyn =  u[yn*nx + xn];
		optflow_type u_xnyp =  u[yp*nx + xn];

		optflow_type v_x0yn =  u[nstack + yn*nx + x ];
		optflow_type v_x0yp =  u[nstack + yp*nx + x];
		optflow_type v_xpy0 =  u[nstack + y*nx  + xp];
		optflow_type v_xpyn =  u[nstack + yn*nx + xp];
		optflow_type v_xpyp =  u[nstack + yp*nx + xp];
		optflow_type v_xny0 =  u[nstack + y*nx  + xn];
		optflow_type v_xnyn =  u[nstack + yn*nx + xn];
		optflow_type v_xnyp =  u[nstack + yp*nx + xn];

		u_x0yn += du[yn*nx + x ];
		u_x0yp += du[yp*nx + x];
		u_xpy0 += du[y*nx  + xp];
		u_xpyn += du[yn*nx + xp];
		u_xpyp += du[yp*nx + xp];
		u_xny0 += du[y*nx  + xn];
		u_xnyn += du[yn*nx + xn];
		u_xnyp += du[yp*nx + xn];

		v_x0yn += du[nstack + yn*nx + x ];
		v_x0yp += du[nstack + yp*nx + x];
		v_xpy0 += du[nstack + y*nx  + xp];
		v_xpyn += du[nstack + yn*nx + xp];
		v_xpyp += du[nstack + yp*nx + xp];
		v_xny0 += du[nstack + y*nx  + xn];
		v_xnyn += du[nstack + yn*nx + xn];
		v_xnyp += du[nstack + yp*nx + xn];
		//////////////////////////////////////////////////////////////

		 //partial derivatives
		//////////////////////////////////////////////////////////////
		mathtype_solver ux = normalizer_x*(w1*(u_xpy0 - u_xny0) + w2*(u_xpyn +u_xpyp -u_xnyn -u_xnyp));
		mathtype_solver uy = normalizer_y*(w1*(u_x0yp - u_x0yn) + w2*(-u_xpyn +u_xpyp -u_xnyn +u_xnyp));
		mathtype_solver vx = normalizer_x*(w1*(v_xpy0 - v_xny0) + w2*(v_xpyn +v_xpyp -v_xnyn -v_xnyp));
		mathtype_solver vy = normalizer_y*(w1*(v_x0yp - v_x0yn) + w2*(-v_xpyn +v_xpyp -v_xnyn +v_xnyp));
		//////////////////////////////////////////////////////////////

		//Rotate vectors to edge aligned basis
		//////////////////////////////////////////////////////////////
		if (adaptive_smoothness)
		{
			mathtype_solver theta = adaptivity[pos];
			mathtype_solver sintheta, costheta;

			__sincosf(theta, &sintheta, &costheta);

			float tmp_val = ux*costheta - uy*sintheta;
					   uy = ux*sintheta + uy*costheta;
					   ux = tmp_val;

			tmp_val = vx*costheta - vy*sintheta;
				 vy = vx*sintheta + vy*costheta;
				 vx = tmp_val;
		}
		//////////////////////////////////////////////////////////////

		//calculate smoothness term
		//////////////////////////////////////////////////////////////
		if (!decoupled)
		{
			mathtype_solver value;

			if(!anisotropic) value = 0.5f/sqrtf(ux*ux + vx*vx + uy*uy + vy*vy + epsilon_phi_squared); //Isotropic flow driven := phi(tr(nabla_u1*nabla_u1^T + nabla_u2^T)
			else if(!adaptive_smoothness || !complementary_smoothness) value = 0.5f/(sqrtf(ux*ux + vx*vx + epsilon_phi_squared)+sqrtf(uy*uy + vy*vy + epsilon_phi_squared)); //Anisotropic flow driven := tr(phi(nabla_u1*nabla_u1^T + nabla_u2^T)
			else value = 0.25f/sqrtf(ux*ux + vx*vx + epsilon_phi_squared) + 0.5f*sqrtf(uy*uy + vy*vy + epsilon_phi_squared); //following Zimmer2011:Optical flow in Harmony

			__syncthreads();
			phi[pos] = value;
		}
		else
		{
			mathtype_solver value1, value2;

			if(!anisotropic){
				//decoupled isotropic:
				value1 = 0.5f/sqrtf(ux*ux + uy*uy + epsilon_phi_squared);
				value2 = 0.5f/sqrtf(vx*vx + vy*vy + epsilon_phi_squared);
			}
			else if (!adaptive_smoothness || !complementary_smoothness){
				//decoupled anisotropic:
				value1 = 0.5f/(sqrtf(ux*ux + epsilon_phi_squared)+sqrtf(uy*uy + epsilon_phi_squared));
				value2 = 0.5f/(sqrtf(vx*vx + epsilon_phi_squared)+sqrtf(vy*vy + epsilon_phi_squared));
			}
			else
			{
				//decoupled complementary
				value1 = 0.25f/sqrtf(ux*ux + epsilon_phi_squared)+0.5f*sqrtf(uy*uy + epsilon_phi_squared);
				value2 = 0.25f/sqrtf(vx*vx + epsilon_phi_squared)+0.5f*sqrtf(vy*vy + epsilon_phi_squared);
			}

			__syncthreads();
			phi[pos] = value1;
			phi[nstack+pos] = value2;
		}
		//////////////////////////////////////////////////////////////

		return;
	}
	__global__ void update_smoothnessterm_Sobel(optflow_type *u, optflow_type *du,  optflow_type *phi, optflow_type *adaptivity)
	{
		//Reference: https://github.com/xpharry/optical-flow-algorithms

		//acquire constants and position
		/////////////////////////////////////////////
		int nx = gpu_const::nx_c;
		int ny = gpu_const::ny_c;
		int nz = gpu_const::nz_c;

		mathtype_solver epsilon_phi_squared = gpu_const::epsilon_phi_squared_c;
		mathtype_solver hx = gpu_const::hx_c;
		mathtype_solver hy = gpu_const::hy_c;

		bool anisotropic = gpu_const::anisotropic_smoothness_c;
		bool decoupled = gpu_const::decoupled_smoothness_c;
		bool adaptive_smoothness = gpu_const::adaptive_smoothness_c;
		bool complementary_smoothness = gpu_const::complementary_smoothness_c;

		idx_type nslice = nx*ny;
		idx_type nstack = nz*nslice;

		idx_type pos = (blockIdx.x*blockDim.x+threadIdx.x);
		if (pos >= nstack) pos = threadIdx.x;

		int y = pos/nx;
		int x = pos-y*nx;

		int yp = y+1; int yn = y-1;
		int xp = x+1; int xn = x-1;

		//Reflective boundary conditions (mirrored on first/last value)
		if (yp == ny) yp -= 2; if (yn < 0) yn = 1;
		if (xp == nx) xp -= 2; if (xn < 0) xn = 1;

		mathtype_solver w1 = 0.6666666667f;
		mathtype_solver w2 = 0.3333333333f;

		mathtype_solver normalizer_x = 0.5f/hx;
		mathtype_solver normalizer_y = 0.5f/hy;
		/////////////////////////////////////////////

		//acquire from global memory
		//////////////////////////////////////////////////////////////
		__syncthreads();
		optflow_type u_x0yn =  u[yn*nx + x ];
		optflow_type u_x0yp =  u[yp*nx + x];
		optflow_type u_xpy0 =  u[y*nx  + xp];
		optflow_type u_xpyn =  u[yn*nx + xp];
		optflow_type u_xpyp =  u[yp*nx + xp];
		optflow_type u_xny0 =  u[y*nx  + xn];
		optflow_type u_xnyn =  u[yn*nx + xn];
		optflow_type u_xnyp =  u[yp*nx + xn];

		optflow_type v_x0yn =  u[nstack + yn*nx + x ];
		optflow_type v_x0yp =  u[nstack + yp*nx + x];
		optflow_type v_xpy0 =  u[nstack + y*nx  + xp];
		optflow_type v_xpyn =  u[nstack + yn*nx + xp];
		optflow_type v_xpyp =  u[nstack + yp*nx + xp];
		optflow_type v_xny0 =  u[nstack + y*nx  + xn];
		optflow_type v_xnyn =  u[nstack + yn*nx + xn];
		optflow_type v_xnyp =  u[nstack + yp*nx + xn];

		u_x0yn += du[yn*nx + x ];
		u_x0yp += du[yp*nx + x];
		u_xpy0 += du[y*nx  + xp];
		u_xpyn += du[yn*nx + xp];
		u_xpyp += du[yp*nx + xp];
		u_xny0 += du[y*nx  + xn];
		u_xnyn += du[yn*nx + xn];
		u_xnyp += du[yp*nx + xn];

		v_x0yn += du[nstack + yn*nx + x ];
		v_x0yp += du[nstack + yp*nx + x];
		v_xpy0 += du[nstack + y*nx  + xp];
		v_xpyn += du[nstack + yn*nx + xp];
		v_xpyp += du[nstack + yp*nx + xp];
		v_xny0 += du[nstack + y*nx  + xn];
		v_xnyn += du[nstack + yn*nx + xn];
		v_xnyp += du[nstack + yp*nx + xn];
		//////////////////////////////////////////////////////////////

		 //partial derivatives
		//////////////////////////////////////////////////////////////
		mathtype_solver ux = normalizer_x*(w1*(u_xpy0 - u_xny0) + w2*(u_xpyn +u_xpyp -u_xnyn -u_xnyp));
		mathtype_solver uy = normalizer_y*(w1*(u_x0yp - u_x0yn) + w2*(-u_xpyn +u_xpyp -u_xnyn +u_xnyp));
		mathtype_solver vx = normalizer_x*(w1*(v_xpy0 - v_xny0) + w2*(v_xpyn +v_xpyp -v_xnyn -v_xnyp));
		mathtype_solver vy = normalizer_y*(w1*(v_x0yp - v_x0yn) + w2*(-v_xpyn +v_xpyp -v_xnyn +v_xnyp));
		//////////////////////////////////////////////////////////////

		//Rotate vectors to edge aligned basis
		//////////////////////////////////////////////////////////////
		if (adaptive_smoothness)
		{
			mathtype_solver theta = adaptivity[pos];
			mathtype_solver sintheta, costheta;

			__sincosf(theta, &sintheta, &costheta);

			float tmp_val = ux*costheta - uy*sintheta;
					   uy = ux*sintheta + uy*costheta;
					   ux = tmp_val;

			tmp_val = vx*costheta - vy*sintheta;
				 vy = vx*sintheta + vy*costheta;
				 vx = tmp_val;
		}
		//////////////////////////////////////////////////////////////

		//calculate smoothness term
		//////////////////////////////////////////////////////////////
		if (!decoupled)
		{
			mathtype_solver value;

			if(!anisotropic) value = 0.5f/sqrtf(ux*ux + vx*vx + uy*uy + vy*vy + epsilon_phi_squared); //Isotropic flow driven := phi(tr(nabla_u1*nabla_u1^T + nabla_u2^T)
			else if(!adaptive_smoothness || !complementary_smoothness) value = 0.5f/(sqrtf(ux*ux + vx*vx + epsilon_phi_squared)+sqrtf(uy*uy + vy*vy + epsilon_phi_squared)); //Anisotropic flow driven := tr(phi(nabla_u1*nabla_u1^T + nabla_u2^T)
			else value = 0.25f/sqrtf(ux*ux + vx*vx + epsilon_phi_squared) + 0.5f*sqrtf(uy*uy + vy*vy + epsilon_phi_squared); //following Zimmer2011:Optical flow in Harmony

			__syncthreads();
			phi[pos] = value;
		}
		else
		{
			mathtype_solver value1, value2;

			if(!anisotropic){
				//decoupled isotropic:
				value1 = 0.5f/sqrtf(ux*ux + uy*uy + epsilon_phi_squared);
				value2 = 0.5f/sqrtf(vx*vx + vy*vy + epsilon_phi_squared);
			}
			else if (!adaptive_smoothness || !complementary_smoothness){
				//decoupled anisotropic:
				value1 = 0.5f/(sqrtf(ux*ux + epsilon_phi_squared)+sqrtf(uy*uy + epsilon_phi_squared));
				value2 = 0.5f/(sqrtf(vx*vx + epsilon_phi_squared)+sqrtf(vy*vy + epsilon_phi_squared));
			}
			else
			{
				//decoupled complementary
				value1 = 0.25f/sqrtf(ux*ux + epsilon_phi_squared)+0.5f*sqrtf(uy*uy + epsilon_phi_squared);
				value2 = 0.25f/sqrtf(vx*vx + epsilon_phi_squared)+0.5f*sqrtf(vy*vy + epsilon_phi_squared);
			}

			__syncthreads();
			phi[pos] = value1;
			phi[nstack+pos] = value2;
		}
		//////////////////////////////////////////////////////////////

		return;
	}
}
}
#endif //SMOOTHNESSTERM_GPU2D_CUH
