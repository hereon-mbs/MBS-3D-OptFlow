#ifndef SMOOTHNESSTERM_GPU2D_RS_CUH
#define SMOOTHNESSTERM_GPU2D_RS_CUH

#include <iostream>
#include <cuda.h>
#include "../Solver/optflow_base.h"
#include "../Solver/gpu_constants.cuh"

namespace optflow
{
namespace gpu2d_rs
{
	__device__ __inline__ idx_type xy2idx(int &x, int &y, int &nx, idx_type &n_even){
		idx_type idx = y*(nx*0.5f)+(x/2);

		if((nx%2) != 0 && (y%2) != 0 && (x%2) != 0) idx++;

		if (     (y%2) == 0 && (x%2) != 0) idx += n_even;
		else if ((y%2) != 0 && (x%2) == 0) idx += n_even;

		return idx;
	}
	__device__ __inline__ idx_type idx2pos2D(idx_type &idx, int &nx, idx_type &n_even){
		idx_type pos;

		if(idx < n_even)
		{
			pos = 2*idx;
			int y = pos/nx;
			if((nx%2) == 0 && (y%2) != 0) pos++;
		}
		else
		{
			pos = (idx-n_even)*2;
			int y = pos/nx;
			if((nx%2)==0 && (y%2)==0) pos++;
			else if ((nx%2) != 0) pos++;
		}

		return pos;
	}

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
		idx_type n_even = nstack-(nstack/2);

		idx_type idx = (blockIdx.x*blockDim.x+threadIdx.x);
		if (idx >= nstack) {idx = threadIdx.x;}
		idx_type pos = idx2pos2D(idx, nx, n_even);

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

		mathtype_solver normalizer_x = -1.f/(12.f*hx);
		mathtype_solver normalizer_y = -1.f/(12.f*hy);
		/////////////////////////////////////////////

		//acquire from global memory
		//////////////////////////////////////////////////////////////
		idx_type idx_xn2 = xy2idx(xn2, y, nx, n_even);
		idx_type idx_xn  = xy2idx(xn, y, nx, n_even);
		idx_type idx_xp  = xy2idx(xp, y, nx, n_even);
		idx_type idx_xp2 = xy2idx(xp2, y, nx, n_even);
		idx_type idx_yn2 = xy2idx(x, yn2, nx, n_even);
		idx_type idx_yn  = xy2idx(x, yn, nx, n_even);
		idx_type idx_yp  = xy2idx(x, yp, nx, n_even);
		idx_type idx_yp2 = xy2idx(x, yp2, nx, n_even);

		__syncthreads();
		optflow_type u_xn2 = u[idx_xn2];
		optflow_type u_xn =  u[idx_xn];
		optflow_type u_xp =  u[idx_xp];
		optflow_type u_xp2 = u[idx_xp2];
		optflow_type u_yn2 = u[idx_yn2];
		optflow_type u_yn =  u[idx_yn];
		optflow_type u_yp =  u[idx_yp];
		optflow_type u_yp2 = u[idx_yp2];

		optflow_type v_xn2 = u[nstack+idx_xn2];
		optflow_type v_xn =  u[nstack+idx_xn];
		optflow_type v_xp =  u[nstack+idx_xp];
		optflow_type v_xp2 = u[nstack+idx_xp2];
		optflow_type v_yn2 = u[nstack+idx_yn2];
		optflow_type v_yn =  u[nstack+idx_yn];
		optflow_type v_yp =  u[nstack+idx_yp];
		optflow_type v_yp2 = u[nstack+idx_yp2];

		u_xn2 += du[idx_xn2];
		u_xn +=  du[idx_xn];
		u_xp +=  du[idx_xp];
		u_xp2 += du[idx_xp2];
		u_yn2 += du[idx_yn2];
		u_yn +=  du[idx_yn];
		u_yp +=  du[idx_yp];
		u_yp2 += du[idx_yp2];

		v_xn2 += du[nstack+idx_xn2];
		v_xn +=  du[nstack+idx_xn];
		v_xp +=  du[nstack+idx_xp];
		v_xp2 += du[nstack+idx_xp2];
		v_yn2 += du[nstack+idx_yn2];
		v_yn +=  du[nstack+idx_yn];
		v_yp +=  du[nstack+idx_yp];
		v_yp2 += du[nstack+idx_yp2];
		//////////////////////////////////////////////////////////////

		//partial derivatives
		//////////////////////////////////////////////////////////////
		mathtype_solver ux = normalizer_x*(-u_xn2 +8.f*u_xn -8.f*u_xp +u_xp2);
		mathtype_solver uy = normalizer_y*(-u_yn2 +8.f*u_yn -8.f*u_yp +u_yp2);
		mathtype_solver vx = normalizer_x*(-v_xn2 +8.f*v_xn -8.f*v_xp +v_xp2);
		mathtype_solver vy = normalizer_y*(-v_yn2 +8.f*v_yn -8.f*v_yp +v_yp2);
		//////////////////////////////////////////////////////////////

		//Rotate vectors to edge aligned basis
		//////////////////////////////////////////////////////////////
		if (adaptive_smoothness)
		{
			mathtype_solver theta = adaptivity[idx];
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
			phi[idx] = value;
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
			phi[idx] = value1;
			phi[nstack+idx] = value2;
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
		idx_type n_even = nstack-(nstack/2);

		idx_type idx = (blockIdx.x*blockDim.x+threadIdx.x);
		if (idx >= nstack) idx = threadIdx.x;
		idx_type pos = idx2pos2D(idx, nx, n_even);

		int y = pos/nx;
		int x = pos-y*nx;

		int yp = y+1; int yn = y-1;
		int xp = x+1; int xn = x-1;

		//Reflective boundary conditions (mirrored on first/last value)
		if (yp == ny) yp -= 2; if (yn < 0) yn = 1;
		if (xp == nx) xp -= 2; if (xn < 0) xn = 1;

		mathtype_solver normalizer_x = -1.f/(2.f*hx);
		mathtype_solver normalizer_y = -1.f/(2.f*hy);
		/////////////////////////////////////////////

		//acquire from global memory
		//////////////////////////////////////////////////////////////
		idx_type idx_xn  = xy2idx(xn, y, nx, n_even);
		idx_type idx_xp  = xy2idx(xp, y, nx, n_even);
		idx_type idx_yn  = xy2idx(x, yn, nx, n_even);
		idx_type idx_yp  = xy2idx(x, yp, nx, n_even);

		__syncthreads();
		optflow_type u_xn =  u[idx_xn];
		optflow_type u_xp =  u[idx_xp];
		optflow_type u_yn =  u[idx_yn];
		optflow_type u_yp =  u[idx_yp];

		optflow_type v_xn =  u[nstack+idx_xn];
		optflow_type v_xp =  u[nstack+idx_xp];
		optflow_type v_yn =  u[nstack+idx_yn];
		optflow_type v_yp =  u[nstack+idx_yp];

		u_xn +=  du[idx_xn];
		u_xp +=  du[idx_xp];
		u_yn +=  du[idx_yn];
		u_yp +=  du[idx_yp];

		v_xn +=  du[nstack+idx_xn];
		v_xp +=  du[nstack+idx_xp];
		v_yn +=  du[nstack+idx_yn];
		v_yp +=  du[nstack+idx_yp];
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
			mathtype_solver theta = adaptivity[idx];
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
			phi[idx] = value;
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
			phi[idx] = value1;
			phi[nstack+idx] = value2;
		}
		//////////////////////////////////////////////////////////////

		return;
	}
	__global__ void update_smoothnessterm_forwardDiff(optflow_type *u, optflow_type *du,  optflow_type *phi, optflow_type *adaptivity)
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
		idx_type n_even = nstack-(nstack/2);

		idx_type idx = (blockIdx.x*blockDim.x+threadIdx.x);
		if (idx >= nstack) idx = threadIdx.x;
		idx_type pos = idx2pos2D(idx, nx, n_even);

		int y = pos/nx;
		int x = pos-y*nx;

		int yp = y+1; int yn = y-1;
		int xp = x+1; int xn = x-1;

		//Reflective boundary conditions (mirrored on first/last value)
		if (yp == ny) yp -= 2; if (yn < 0) yn = 1;
		if (xp == nx) xp -= 2; if (xn < 0) xn = 1;

		mathtype_solver normalizer_x = -1.f/hx;
		mathtype_solver normalizer_y = -1.f/hy;
		/////////////////////////////////////////////

		//acquire from global memory
		//////////////////////////////////////////////////////////////
		idx_type idx_xn  = xy2idx(xn, y, nx, n_even);
		idx_type idx_xp  = xy2idx(xp, y, nx, n_even);
		idx_type idx_yn  = xy2idx(x, yn, nx, n_even);
		idx_type idx_yp  = xy2idx(x, yp, nx, n_even);

		__syncthreads();
		optflow_type u0   =  u[idx];
		optflow_type u_xp =  u[idx_xp];
		optflow_type u_yp =  u[idx_yp];

		optflow_type v0   =  u[nstack+idx];
		optflow_type v_xp =  u[nstack+idx_xp];
		optflow_type v_yp =  u[nstack+idx_yp];

		u0   +=  du[idx];
		u_xp +=  du[idx_xp];
		u_yp +=  du[idx_yp];

		v0   +=  du[nstack+idx];
		v_xp +=  du[nstack+idx_xp];
		v_yp +=  du[nstack+idx_yp];
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
			mathtype_solver theta = adaptivity[idx];
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
			phi[idx] = value;
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
			phi[idx] = value1;
			phi[nstack+idx] = value2;
		}
		//////////////////////////////////////////////////////////////

		return;
	}
}
}
#endif //SMOOTHNESSTERM_GPU2D_RS_CUH
