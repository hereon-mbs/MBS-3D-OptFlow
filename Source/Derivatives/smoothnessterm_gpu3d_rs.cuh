#ifndef SMOOTHNESSTERM_GPU3D_RS_CUH
#define SMOOTHNESSTERM_GPU3D_RS_CUH

#include <iostream>
#include <cuda.h>
#include "../Solver/optflow_base.h"
#include "../Solver/gpu_constants.cuh"

namespace optflow
{
namespace gpu3d_rs
{
	__device__ __inline__ idx_type xyz2idx(int &x, int &y, int &z, int &nx, idx_type &nslice, idx_type &n_even){

		int idx = z*(nslice*0.5f)+y*(nx*0.5f)+(x/2);

		if((nslice%2) == 0 && (nx%2) != 0 && (y%2) != 0 && (x%2) != 0) idx++;
		else if((nslice%2) != 0 && (z%2) == 0 &&(y%2) != 0 && (x%2) != 0) idx++;
		else if((nslice%2) != 0 && (z%2) != 0 && (y%2) == 0 && (x%2) != 0) idx++;

		if (     (z%2) == 0 && (y%2) == 0 && (x%2) != 0) idx += n_even;
		else if ((z%2) == 0 && (y%2) != 0 && (x%2) == 0) idx += n_even;
		else if ((z%2) != 0 && (y%2) == 0 && (x%2) == 0) idx += n_even;
		else if ((z%2) != 0 && (y%2) != 0 && (x%2) != 0) idx += n_even;

		return idx;
	}
	__device__ __inline__ idx_type idx2pos3D(idx_type &idx, int &nx, idx_type &nslice, idx_type &n_even){
		idx_type pos;

		if(idx < n_even)
		{
			pos = 2*idx;
			int z = pos/nslice;
			int y = (pos-z*nslice)/nx;

				 if ((nx%2) == 0 && (y%2) != 0 && (z%2) == 0) pos++;
			else if ((nx%2) == 0 && (y%2) == 0 && (z%2) != 0) pos++;
			else if ((nx%2) != 0 && (nslice%2) == 0 && (z%2) != 0) pos++;
		}
		else
		{
			pos = (idx-n_even)*2;
			int z = pos/nslice;
			int y = (pos-z*nslice)/nx;

				 if((nx%2) != 0 && (z%2) == 0) pos++;
			else if((nslice%2) != 0 && (z%2) != 0) pos++;
			else if((nx%2) == 0 && (y%2) == 0 && (z%2) == 0) pos++;
			else if((nx%2) == 0 && (y%2) != 0 && (z%2) != 0) pos++;
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
		mathtype_solver hz = gpu_const::hz_c;

		bool anisotropic = gpu_const::anisotropic_smoothness_c;
		bool decoupled = gpu_const::decoupled_smoothness_c;
		bool adaptive_smoothness = gpu_const::adaptive_smoothness_c;
		bool complementary_smoothness = gpu_const::complementary_smoothness_c;

		idx_type nslice = nx*ny;
		idx_type nstack = nz*nslice;
		idx_type nstack2 = 2*nstack;
		idx_type n_even = nstack-(nstack/2);

		idx_type idx = (blockIdx.x*blockDim.x+threadIdx.x);
		if (idx >= nstack) {idx = threadIdx.x;}
		idx_type pos = idx2pos3D(idx, nx, nslice, n_even);

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

		mathtype_solver normalizer_x = -1.f/(12.f*hx);
		mathtype_solver normalizer_y = -1.f/(12.f*hy);
		mathtype_solver normalizer_z = -1.f/(12.f*hz);
		/////////////////////////////////////////////

		//acquire from global memory
		//////////////////////////////////////////////////////////////
		idx_type idx_xn2 = xyz2idx(xn2, y, z, nx, nslice, n_even);
		idx_type idx_xn  = xyz2idx(xn, y, z, nx, nslice, n_even);
		idx_type idx_xp  = xyz2idx(xp, y, z, nx, nslice, n_even);
		idx_type idx_xp2 = xyz2idx(xp2, y, z, nx, nslice, n_even);
		idx_type idx_yn2 = xyz2idx(x, yn2, z, nx, nslice, n_even);
		idx_type idx_yn  = xyz2idx(x, yn, z, nx, nslice, n_even);
		idx_type idx_yp  = xyz2idx(x, yp, z, nx, nslice, n_even);
		idx_type idx_yp2 = xyz2idx(x, yp2, z, nx, nslice, n_even);
		idx_type idx_zn2 = xyz2idx(x, y, zn2, nx, nslice, n_even);
		idx_type idx_zn  = xyz2idx(x, y, zn , nx, nslice, n_even);
		idx_type idx_zp  = xyz2idx(x, y, zp , nx, nslice, n_even);
		idx_type idx_zp2 = xyz2idx(x, y, zp2, nx, nslice, n_even);

		__syncthreads();
		optflow_type u_xn2 = u[idx_xn2];
		optflow_type u_xn =  u[idx_xn];
		optflow_type u_xp =  u[idx_xp];
		optflow_type u_xp2 = u[idx_xp2];
		optflow_type u_yn2 = u[idx_yn2];
		optflow_type u_yn =  u[idx_yn];
		optflow_type u_yp =  u[idx_yp];
		optflow_type u_yp2 = u[idx_yp2];
		optflow_type u_zn2 = u[idx_zn2];
		optflow_type u_zn =  u[idx_zn];
		optflow_type u_zp =  u[idx_zp];
		optflow_type u_zp2 = u[idx_zp2];

		optflow_type v_xn2 = u[nstack+idx_xn2];
		optflow_type v_xn =  u[nstack+idx_xn];
		optflow_type v_xp =  u[nstack+idx_xp];
		optflow_type v_xp2 = u[nstack+idx_xp2];
		optflow_type v_yn2 = u[nstack+idx_yn2];
		optflow_type v_yn =  u[nstack+idx_yn];
		optflow_type v_yp =  u[nstack+idx_yp];
		optflow_type v_yp2 = u[nstack+idx_yp2];
		optflow_type v_zn2 = u[nstack+idx_zn2];
		optflow_type v_zn =  u[nstack+idx_zn];
		optflow_type v_zp =  u[nstack+idx_zp];
		optflow_type v_zp2 = u[nstack+idx_zp2];

		optflow_type w_xn2 = u[nstack2+idx_xn2];
		optflow_type w_xn =  u[nstack2+idx_xn];
		optflow_type w_xp =  u[nstack2+idx_xp];
		optflow_type w_xp2 = u[nstack2+idx_xp2];
		optflow_type w_yn2 = u[nstack2+idx_yn2];
		optflow_type w_yn =  u[nstack2+idx_yn];
		optflow_type w_yp =  u[nstack2+idx_yp];
		optflow_type w_yp2 = u[nstack2+idx_yp2];
		optflow_type w_zn2 = u[nstack2+idx_zn2];
		optflow_type w_zn =  u[nstack2+idx_zn];
		optflow_type w_zp =  u[nstack2+idx_zp];
		optflow_type w_zp2 = u[nstack2+idx_zp2];

		u_xn2 += du[idx_xn2];
		u_xn +=  du[idx_xn];
		u_xp +=  du[idx_xp];
		u_xp2 += du[idx_xp2];
		u_yn2 += du[idx_yn2];
		u_yn +=  du[idx_yn];
		u_yp +=  du[idx_yp];
		u_yp2 += du[idx_yp2];
		u_zn2 += du[idx_zn2];
		u_zn +=  du[idx_zn];
		u_zp +=  du[idx_zp];
		u_zp2 += du[idx_zp2];

		v_xn2 += du[nstack+idx_xn2];
		v_xn +=  du[nstack+idx_xn];
		v_xp +=  du[nstack+idx_xp];
		v_xp2 += du[nstack+idx_xp2];
		v_yn2 += du[nstack+idx_yn2];
		v_yn +=  du[nstack+idx_yn];
		v_yp +=  du[nstack+idx_yp];
		v_yp2 += du[nstack+idx_yp2];
		v_zn2 += du[nstack+idx_zn2];
		v_zn +=  du[nstack+idx_zn];
		v_zp +=  du[nstack+idx_zp];
		v_zp2 += du[nstack+idx_zp2];

		w_xn2 += du[nstack2+idx_xn2];
		w_xn +=  du[nstack2+idx_xn];
		w_xp +=  du[nstack2+idx_xp];
		w_xp2 += du[nstack2+idx_xp2];
		w_yn2 += du[nstack2+idx_yn2];
		w_yn +=  du[nstack2+idx_yn];
		w_yp +=  du[nstack2+idx_yp];
		w_yp2 += du[nstack2+idx_yp2];
		w_zn2 += du[nstack2+idx_zn2];
		w_zn +=  du[nstack2+idx_zn];
		w_zp +=  du[nstack2+idx_zp];
		w_zp2 += du[nstack2+idx_zp2];
		//////////////////////////////////////////////////////////////

		//partial derivatives
		//////////////////////////////////////////////////////////////
		mathtype_solver ux = normalizer_x*(-u_xn2 +8.f*u_xn -8.f*u_xp +u_xp2);
		mathtype_solver uy = normalizer_y*(-u_yn2 +8.f*u_yn -8.f*u_yp +u_yp2);
		mathtype_solver uz = normalizer_z*(-u_zn2 +8.f*u_zn -8.f*u_zp +u_zp2);
		mathtype_solver vx = normalizer_x*(-v_xn2 +8.f*v_xn -8.f*v_xp +v_xp2);
		mathtype_solver vy = normalizer_y*(-v_yn2 +8.f*v_yn -8.f*v_yp +v_yp2);
		mathtype_solver vz = normalizer_z*(-v_zn2 +8.f*v_zn -8.f*v_zp +v_zp2);
		mathtype_solver wx = normalizer_x*(-w_xn2 +8.f*w_xn -8.f*w_xp +w_xp2);
		mathtype_solver wy = normalizer_y*(-w_yn2 +8.f*w_yn -8.f*w_yp +w_yp2);
		mathtype_solver wz = normalizer_z*(-w_zn2 +8.f*w_zn -8.f*w_zp +w_zp2);
		//////////////////////////////////////////////////////////////

		//Rotate vectors to edge aligned basis
		//////////////////////////////////////////////////////////////
		if (adaptive_smoothness)
		{
			__syncthreads();
			mathtype_solver theta = adaptivity[idx];
			mathtype_solver ksi = adaptivity[idx+nstack];
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
			phi[idx] = value;
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
			phi[idx] = value1;
			phi[nstack+idx] = value2;
			phi[2*nstack+idx] = value3;
		}
		//////////////////////////////////////////////////////////////

		return;
	}
	__global__ void update_smoothnessterm_centralDiff(optflow_type *u, optflow_type *du,  optflow_type *phi, optflow_type *adaptivity)
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
		idx_type nstack2 = 2*nstack;
		idx_type n_even = nstack-(nstack/2);

		idx_type idx = (blockIdx.x*blockDim.x+threadIdx.x);
		if (idx >= nstack) {idx = threadIdx.x;}
		idx_type pos = idx2pos3D(idx, nx, nslice, n_even);

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

		mathtype_solver normalizer_x = -1.f/(2.f*hx);
		mathtype_solver normalizer_y = -1.f/(2.f*hy);
		mathtype_solver normalizer_z = -1.f/(2.f*hz);
		/////////////////////////////////////////////

		//acquire from global memory
		//////////////////////////////////////////////////////////////
		idx_type idx_xn  = xyz2idx(xn, y, z, nx, nslice, n_even);
		idx_type idx_xp  = xyz2idx(xp, y, z, nx, nslice, n_even);
		idx_type idx_yn  = xyz2idx(x, yn, z, nx, nslice, n_even);
		idx_type idx_yp  = xyz2idx(x, yp, z, nx, nslice, n_even);
		idx_type idx_zn  = xyz2idx(x, y, zn , nx, nslice, n_even);
		idx_type idx_zp  = xyz2idx(x, y, zp , nx, nslice, n_even);

		__syncthreads();
		optflow_type u_xn =  u[idx_xn];
		optflow_type u_xp =  u[idx_xp];
		optflow_type u_yn =  u[idx_yn];
		optflow_type u_yp =  u[idx_yp];
		optflow_type u_zn =  u[idx_zn];
		optflow_type u_zp =  u[idx_zp];

		optflow_type v_xn =  u[nstack+idx_xn];
		optflow_type v_xp =  u[nstack+idx_xp];
		optflow_type v_yn =  u[nstack+idx_yn];
		optflow_type v_yp =  u[nstack+idx_yp];
		optflow_type v_zn =  u[nstack+idx_zn];
		optflow_type v_zp =  u[nstack+idx_zp];

		optflow_type w_xn =  u[nstack2+idx_xn];
		optflow_type w_xp =  u[nstack2+idx_xp];
		optflow_type w_yn =  u[nstack2+idx_yn];
		optflow_type w_yp =  u[nstack2+idx_yp];
		optflow_type w_zn =  u[nstack2+idx_zn];
		optflow_type w_zp =  u[nstack2+idx_zp];

		u_xn +=  du[idx_xn];
		u_xp +=  du[idx_xp];
		u_yn +=  du[idx_yn];
		u_yp +=  du[idx_yp];
		u_zn +=  du[idx_zn];
		u_zp +=  du[idx_zp];

		v_xn +=  du[nstack+idx_xn];
		v_xp +=  du[nstack+idx_xp];
		v_yn +=  du[nstack+idx_yn];
		v_yp +=  du[nstack+idx_yp];
		v_zn +=  du[nstack+idx_zn];
		v_zp +=  du[nstack+idx_zp];

		w_xn +=  du[nstack2+idx_xn];
		w_xp +=  du[nstack2+idx_xp];
		w_yn +=  du[nstack2+idx_yn];
		w_yp +=  du[nstack2+idx_yp];
		w_zn +=  du[nstack2+idx_zn];
		w_zp +=  du[nstack2+idx_zp];
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
			mathtype_solver theta = adaptivity[idx];
			mathtype_solver ksi = adaptivity[idx+nstack];
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
			phi[idx] = value;
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
			phi[idx] = value1;
			phi[nstack+idx] = value2;
			phi[2*nstack+idx] = value3;
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
		mathtype_solver hz = gpu_const::hz_c;

		bool anisotropic = gpu_const::anisotropic_smoothness_c;
		bool decoupled = gpu_const::decoupled_smoothness_c;
		bool adaptive_smoothness = gpu_const::adaptive_smoothness_c;
		bool complementary_smoothness = gpu_const::complementary_smoothness_c;

		idx_type nslice = nx*ny;
		idx_type nstack = nz*nslice;
		idx_type nstack2 = 2*nstack;
		idx_type n_even = nstack-(nstack/2);

		idx_type idx = (blockIdx.x*blockDim.x+threadIdx.x);
		if (idx >= nstack) {idx = threadIdx.x;}
		idx_type pos = idx2pos3D(idx, nx, nslice, n_even);

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

		mathtype_solver normalizer_x = -1.f/hx;
		mathtype_solver normalizer_y = -1.f/hy;
		mathtype_solver normalizer_z = -1.f/hz;
		/////////////////////////////////////////////

		//acquire from global memory
		//////////////////////////////////////////////////////////////
		idx_type idx_xp  = xyz2idx(xp, y, z, nx, nslice, n_even);
		idx_type idx_yp  = xyz2idx(x, yp, z, nx, nslice, n_even);
		idx_type idx_zp  = xyz2idx(x, y, zp , nx, nslice, n_even);

		__syncthreads();
		optflow_type u0 =  u[idx];
		optflow_type u_xp =  u[idx_xp];
		optflow_type u_yp =  u[idx_yp];
		optflow_type u_zp =  u[idx_zp];

		optflow_type v0 =  u[nstack+idx];
		optflow_type v_xp =  u[nstack+idx_xp];
		optflow_type v_yp =  u[nstack+idx_yp];
		optflow_type v_zp =  u[nstack+idx_zp];

		optflow_type w0 =  u[nstack2+idx];
		optflow_type w_xp =  u[nstack2+idx_xp];
		optflow_type w_yp =  u[nstack2+idx_yp];
		optflow_type w_zp =  u[nstack2+idx_zp];


		u0 +=  du[idx];
		u_xp +=  du[idx_xp];
		u_yp +=  du[idx_yp];
		u_zp +=  du[idx_zp];

		v0 +=  du[nstack+idx];
		v_xp +=  du[nstack+idx_xp];
		v_yp +=  du[nstack+idx_yp];
		v_zp +=  du[nstack+idx_zp];

		w0 +=  du[nstack2+idx];
		w_xp +=  du[nstack2+idx_xp];
		w_yp +=  du[nstack2+idx_yp];
		w_zp +=  du[nstack2+idx_zp];
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
			mathtype_solver theta = adaptivity[idx];
			mathtype_solver ksi = adaptivity[idx+nstack];
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
			phi[idx] = value;
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
			phi[idx] = value1;
			phi[nstack+idx] = value2;
			phi[2*nstack+idx] = value3;
		}
		//////////////////////////////////////////////////////////////

		return;
	}
}
}

#endif //SMOOTHNESSTERM_GPU3D_RS_CUH
