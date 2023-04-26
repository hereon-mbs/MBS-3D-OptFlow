#include <iostream>
#include <cuda.h>
#include <omp.h>
#include <typeinfo>
#include <limits>
#include "optflow_gpu2d.h"
#include "gpu_constants.cuh"
#include "../Derivatives/smoothnessterm_gpu2d.cuh"

/*********************************************************************************************************************************************************
 * Location: Helmholtz-Zentrum fuer Material und Kuestenforschung, Max-Planck-Strasse 1, 21502 Geesthacht
 * Author: Stefan Bruns
 * Contact: bruns@nano.ku.dk
 *
 * License: TBA
 *********************************************************************************************************************************************************/

namespace optflow
{
	namespace gpu2d
	{
		__global__ void gaussianfilter2D_x(optflow_type *input, optflow_type *output)
		{
			//acquire constants
			/////////////////////////////////////////////
			float sigma = gpu_const::filter_sigma_c;

			int nx = gpu_const::nx_c;
			int ny = gpu_const::ny_c;
			int nz = gpu_const::nz_c;

			idx_type nslice = nx*ny;
			idx_type nstack = nz*nslice;

			bool outofbounds = false;
			idx_type pos = (blockIdx.x*blockDim.x+threadIdx.x);
			if (pos >= nstack) {outofbounds = true; pos = threadIdx.x;}

			int y = pos/nx;
			int x = pos-y*nx;
			/////////////////////////////////////////////

			//Create Gaussian kernel
			///////////////////////////////////////////////////
			int fsize = (int) (3*sigma);
			float kernelsum = 0.0f;
			float valuesum = 0.0f;
			//////////////////////////////////////////////////

			for(int xi=-fsize; xi<=fsize; xi++)
			{
				int x0 = x+xi;

				//reflective boundaries
				if (x0 < 0) x0 = -x0;
				else if (x0 >= nx) x0 = 2*nx-x0-2;

				float kernel_val = expf(-(xi*xi)/(sigma*sigma*2));
				kernelsum += kernel_val;

				__syncthreads();
				valuesum += kernel_val*input[y*nx+x0];
			}

			if(!outofbounds)
				output[pos] = valuesum/kernelsum;

			return;
		}
		__global__ void gaussianfilter2D_y(optflow_type *input, optflow_type *output)
		{
			//acquire constants
			/////////////////////////////////////////////
			float sigma = gpu_const::filter_sigma_c;

			int nx = gpu_const::nx_c;
			int ny = gpu_const::ny_c;
			int nz = gpu_const::nz_c;

			idx_type nslice = nx*ny;
			idx_type nstack = nz*nslice;

			bool outofbounds = false;
			idx_type pos = (blockIdx.x*blockDim.x+threadIdx.x);
			if (pos >= nstack) {outofbounds = true; pos = threadIdx.x;}

			int y = pos/nx;
			int x = pos-y*nx;
			/////////////////////////////////////////////

			//Create Gaussian kernel
			///////////////////////////////////////////////////
			int fsize = (int) (3*sigma);
			float kernelsum = 0.0f;
			float valuesum = 0.0f;
			//////////////////////////////////////////////////

			for(int yi=-fsize; yi<=fsize; yi++)
			{
				int y0 = y+yi;

				//reflective boundaries
				if (y0 < 0) y0 = -y0;
				else if (y0 >= ny) y0 = 2*ny-y0-2;

				float kernel_val = expf(-(yi*yi)/(sigma*sigma*2));
				kernelsum += kernel_val;

				__syncthreads();
				valuesum += kernel_val*input[y0*nx+x];
			}

			if(!outofbounds)
				output[pos] = valuesum/kernelsum;

			return;
		}

		__device__ __inline__ float interpolate_cubic(float &y0, float &y1, float &y2, float &y3, float &mu)
		{
			float mu2 = mu*mu;

			float a0 = y3-y2-y0+y1;
			float a1 = y0-y1-a0;
			float a2 = y2-y0;
			float a3 = y1;

			return a0*mu*mu2+a1*mu2+a2*mu+a3;
		}

		__global__ void calculate_sorUpdate(int iter, img_type *frame0, img_type *warped1, optflow_type *phi, optflow_type *psi, optflow_type *u, optflow_type *du,
				optflow_type *confidencemap)
		{
			//acquire constants
			/////////////////////////////////////////////
			int nx = gpu_const::nx_c;
			int ny = gpu_const::ny_c;
			int nz = gpu_const::nz_c;

			mathtype_solver epsilon_psi_squared = gpu_const::epsilon_psi_squared_c;
			mathtype_solver hx = gpu_const::hx_c;
			mathtype_solver hy = gpu_const::hy_c;
			mathtype_solver alpha = gpu_const::alpha_c;
			mathtype_solver omega = gpu_const::omega_c;

			bool precalculated_psi = gpu_const::precalculated_psi_c;
			bool decoupled_smoothness = gpu_const::decoupled_smoothness_c;
			int slip_depth = gpu_const::slip_depth_c;

			float minIntensity = gpu_const::lowerIntensityCutoff_c;
			float maxIntensity = gpu_const::upperIntensityCutoff_c;

			mathtype_solver alphax = alpha/(hx*hx);
			mathtype_solver alphay = alpha/(hy*hy);

			int spatiotemporalderivative_id = gpu_const::spatiotemporalderivative_id_c;
			bool use_confidencemap = gpu_const::use_confidencemap_c;

			idx_type nslice = nx*ny;
			idx_type nstack = nz*nslice;

			//Adjust for even/odd updates in 2D
			///////////////////////////////////
			bool outofbounds = false;
			idx_type pos = 2*(blockIdx.x*blockDim.x+threadIdx.x);
			int y = pos/nx;
			int x = pos-y*nx;

			if((iter%2) == 0){
				if ((nx%2) == 0 && (y%2) != 0 ) {x++; pos++;}}
			else{
				if((nx%2) != 0) {x++; pos++;}
				else if((y%2) == 0){x++; pos++;}}

			if (x >= nx) {x = 0; y++; pos = y*nx+x;}

			if (pos >= nstack || x >= nx){
				outofbounds = true; pos = threadIdx.x;
				y = pos/nx; x = pos-y*nx;
			}
			///////////////////////////////////

			mathtype_solver confidence = 1.0f;
			mathtype_solver psi0 = 0.0f;
			mathtype_solver normalizer_x1 = 0.25f/hx;
			mathtype_solver normalizer_y1 = 0.25f/hy;
			/////////////////////////////////////////////

			//Define the neighbourhood
			/////////////////////////////////////////////
			int yp = y+1;
			int yn = y-1;
			int xp = x+1;
			int xn = x-1;

			mathtype_solver xp_active = 1.0f;
			mathtype_solver xn_active = 1.0f;
			mathtype_solver yp_active = 1.0f;
			mathtype_solver yn_active = 1.0f;

			bool boundary_voxel = false;
			if (xp == nx) {xp_active = 0.0f; xp = x; boundary_voxel = true;}
			else if (xn < 0) {xn_active = 0.0f; xn = x; boundary_voxel = true;}
			if (yp == ny) {yp_active = 0.0f; yp = y; boundary_voxel = true;}
			else if (yn < 0) {yn_active = 0.0f; yn = y; boundary_voxel = true;}

			idx_type npos0 = y*nx + xp;
			idx_type npos1 = y*nx + xn;
			idx_type npos2 = yp*nx + x;
			idx_type npos3 = yn*nx + x;

			mathtype_solver phi_neighbour[8] = {0.0f, 0.0f, 0.0f, 0.0f,0.0f, 0.0f, 0.0f, 0.0f,};
			mathtype_solver du_neighbour[4]  = {0.0f, 0.0f, 0.0f, 0.0f,};
			mathtype_solver dv_neighbour[4]  = {0.0f, 0.0f, 0.0f, 0.0f,};
			/////////////////////////////////////////////

			//Switch to reflective boundary conditions
			/////////////////////////////////////////////
			yp = y+1;
			yn = y-1;
			xp = x+1;
			xn = x-1;

			if (yp == ny) yp -= 2;
			else if (yn == -1) yn = 1;
			if (xp == nx) xp -= 2;
			else if (xn == -1) xn = 1;
			/////////////////////////////////////////////

			/////////////////////////////////////////////
			__syncthreads();
			mathtype_solver phi0 = phi[pos];
			mathtype_solver u0 = u[pos];
			mathtype_solver v0 = u[pos+nstack];
			mathtype_solver du0 = du[pos];
			mathtype_solver dv0 = du[pos+nstack];
			mathtype_solver frame0_val = frame0[pos];

			if (use_confidencemap) confidence = confidencemap[pos];
			if (precalculated_psi) psi0 = psi[pos];
			/////////////////////////////////////////////

			//Read in neighbours with 0-boundaries
			/////////////////////////////////////////////
			phi_neighbour[0] = xp_active*0.5f*(phi[npos0] + phi0);
			du_neighbour[0]  = u[npos0] + du[npos0] - u0;
			dv_neighbour[0]  = u[npos0 + nstack] + du[npos0 + nstack] - v0;

			phi_neighbour[1] = xn_active*0.5f*(phi[npos1] + phi0);
			du_neighbour[1]  = u[npos1] + du[npos1] - u0;
			dv_neighbour[1]  = u[npos1 + nstack] + du[npos1 + nstack] - v0;

			phi_neighbour[2] = yp_active*0.5f*(phi[npos2] + phi0);
			du_neighbour[2]  = u[npos2] + du[npos2] - u0;
			dv_neighbour[2]  = u[npos2 + nstack] + du[npos2 + nstack] - v0;

			phi_neighbour[3] = yn_active*0.5f*(phi[npos3] + phi0);
			du_neighbour[3]  = u[npos3] + du[npos3] - u0;
			dv_neighbour[3]  = u[npos3 + nstack] + du[npos3 + nstack] - v0;

			if(decoupled_smoothness)
			{
				mathtype_solver phi1 = phi[nstack+pos];
				phi_neighbour[4] = xp_active*0.5f*(phi[nstack+npos0] + phi1);
				phi_neighbour[5] = xn_active*0.5f*(phi[nstack+npos1] + phi1);
				phi_neighbour[6] = yp_active*0.5f*(phi[nstack+npos2] + phi1);
				phi_neighbour[7] = yn_active*0.5f*(phi[nstack+npos3] + phi1);
			}
			/////////////////////////////////////////////

			mathtype_solver Idx, Idy, Idt;

			//Calculate spatiotemporal derivatives on the fly
			/////////////////////////////////////////////
			if (spatiotemporalderivative_id < 0)
			{
				Idx = psi[pos+nstack];
				Idy = psi[pos+2*nstack];
				Idt = psi[pos+3*nstack];
			}
			else if (spatiotemporalderivative_id == 1){
				//Horn-Schunck: average of frame1 and frame2, dx-kernel := [-1,1; -1,1], dt: local average
				//////////////////////////////////////////////////////////////////////////////////////////
				//mathtype_solver val00a = frame0[y*nx + x];
				mathtype_solver val10a = frame0[y*nx + xp];
				mathtype_solver val01a = frame0[yp*nx + x];
				mathtype_solver val11a = frame0[yp*nx + xp];

				mathtype_solver val00b = warped1[y*nx + x];
				mathtype_solver val10b = warped1[y*nx + xp];
				mathtype_solver val01b = warped1[yp*nx + x];
				mathtype_solver val11b = warped1[yp*nx + xp];

				Idx = normalizer_x1*((-frame0_val + val10a - val01a + val11a) + (-val00b + val10b - val01b + val11b));
				Idy = normalizer_y1*((-frame0_val - val10a + val01a + val11a) + (-val00b - val10b + val01b + val11b));
				Idt = 0.25f*((val00b+val10b+val01b+val11b)-(frame0_val+val10a+val01a+val11a));
				//////////////////////////////////////////////////////////////////////////////////////////
			}
			else if (spatiotemporalderivative_id == 2){
				//Ershov: average of frame1 and frame2, central difference, dt: forward difference
				//////////////////////////////////////////////////////////////////////////////////////////

				mathtype_solver val_xn_a = frame0[y*nx + xn];
				//mathtype_solver val0a    = frame0[pos];
				mathtype_solver val_xp_a = frame0[y*nx + xp];
				mathtype_solver val_yn_a = frame0[yn*nx + x];
				mathtype_solver val_yp_a = frame0[yp*nx + x];

				mathtype_solver val_xn_b = warped1[y*nx + xn];
				mathtype_solver val0b    = warped1[pos];
				mathtype_solver val_xp_b = warped1[y*nx + xp];
				mathtype_solver val_yn_b = warped1[yn*nx + x];
				mathtype_solver val_yp_b = warped1[yp*nx + x];

				Idx = normalizer_x1*((val_xp_a-val_xn_a)+(val_xp_b-val_xn_b));
				Idy = normalizer_y1*((val_yp_a-val_yn_a)+(val_yp_b-val_yn_b));
				Idt = val0b-frame0_val;
				//////////////////////////////////////////////////////////////////////////////////////////
			}
			else if (spatiotemporalderivative_id == 3){
				//Fourth Order Finite Difference
				//////////////////////////////////////////////////////////////////////////////////////////

				int yp2 = y+2; int yn2 = y-2; int xp2 = x+2; int xn2 = x-2;
				if (yp2 >= ny) yp2 = 2*ny-yp2-2; if (yn2 < 0) yn2 = -yn2;
				if (xp2 >= nx) xp2 = 2*nx-xp2-2; if (xn2 < 0) xn2 = -xn2;

				__syncthreads();
				mathtype_solver val_xn2_a = frame0[y*nx + xn2];
				mathtype_solver val_xn_a = frame0[y*nx + xn];
				//mathtype_solver val0a    = frame0[pos];
				mathtype_solver val_xp_a = frame0[y*nx + xp];
				mathtype_solver val_xp2_a = frame0[y*nx + xp2];
				mathtype_solver val_yn2_a = frame0[yn2*nx + x];
				mathtype_solver val_yn_a = frame0[yn*nx + x];
				mathtype_solver val_yp_a = frame0[yp*nx + x];
				mathtype_solver val_yp2_a = frame0[yp2*nx + x];

				mathtype_solver val_xn2_b = warped1[y*nx + xn2];
				mathtype_solver val_xn_b = warped1[y*nx + xn];
				mathtype_solver val0b    = warped1[pos];
				mathtype_solver val_xp_b = warped1[y*nx + xp];
				mathtype_solver val_xp2_b = warped1[y*nx + xp2];
				mathtype_solver val_yn2_b = warped1[yn2*nx + x];
				mathtype_solver val_yn_b = warped1[yn*nx + x];
				mathtype_solver val_yp_b = warped1[yp*nx + x];
				mathtype_solver val_yp2_b = warped1[yp2*nx + x];

				Idx = normalizer_x1/6.f*((val_xn2_a-8.f*val_xn_a+8.f*val_xp_a-val_xp2_a)+(val_xn2_b-8.f*val_xn_b+8.f*val_xp_b-val_xp2_b));
				Idy = normalizer_y1/6.f*((val_yn2_a-8.f*val_yn_a+8.f*val_yp_a-val_yp2_a)+(val_yn2_b-8.f*val_yn_b+8.f*val_yp_b-val_yp2_b));
				Idt = (val0b-frame0_val);
				//////////////////////////////////////////////////////////////////////////////////////////
			}
			else if (spatiotemporalderivative_id == 4){
				//Sixth Order Finite Difference
				//////////////////////////////////////////////////////////////////////////////////////////

				int yp2 = y+2; int yn2 = y-2; int xp2 = x+2; int xn2 = x-2;
				if (yp2 >= ny) yp2 = 2*ny-yp2-2; if (yn2 < 0) yn2 = -yn2;
				if (xp2 >= nx) xp2 = 2*nx-xp2-2; if (xn2 < 0) xn2 = -xn2;

				int yp3 = y+3; int yn3 = y-3; int xp3 = x+3; int xn3 = x-3;
				if (yp3 >= ny) yp3 = 2*ny-yp3-3; if (yn3 < 0) yn3 = -yn3;
				if (xp3 >= nx) xp3 = 2*nx-xp3-3; if (xn3 < 0) xn3 = -xn3;

				__syncthreads();
				mathtype_solver val_xn3_a = frame0[y*nx + xn3];
				mathtype_solver val_xn2_a = frame0[y*nx + xn2];
				mathtype_solver val_xn_a = frame0[y*nx + xn];
				//mathtype_solver val0a    = frame0[pos];
				mathtype_solver val_xp_a = frame0[y*nx + xp];
				mathtype_solver val_xp2_a = frame0[y*nx + xp2];
				mathtype_solver val_xp3_a = frame0[y*nx + xp3];
				mathtype_solver val_yn3_a = frame0[yn3*nx + x];
				mathtype_solver val_yn2_a = frame0[yn2*nx + x];
				mathtype_solver val_yn_a = frame0[yn*nx + x];
				mathtype_solver val_yp_a = frame0[yp*nx + x];
				mathtype_solver val_yp2_a = frame0[yp2*nx + x];
				mathtype_solver val_yp3_a = frame0[yp3*nx + x];

				mathtype_solver val_xn3_b = warped1[y*nx + xn3];
				mathtype_solver val_xn2_b = warped1[y*nx + xn2];
				mathtype_solver val_xn_b = warped1[y*nx + xn];
				mathtype_solver val0b    = warped1[pos];
				mathtype_solver val_xp_b = warped1[y*nx + xp];
				mathtype_solver val_xp2_b = warped1[y*nx + xp2];
				mathtype_solver val_xp3_b = warped1[y*nx + xp3];
				mathtype_solver val_yn3_b = warped1[yn3*nx + x];
				mathtype_solver val_yn2_b = warped1[yn2*nx + x];
				mathtype_solver val_yn_b = warped1[yn*nx + x];
				mathtype_solver val_yp_b = warped1[yp*nx + x];
				mathtype_solver val_yp2_b = warped1[yp2*nx + x];
				mathtype_solver val_yp3_b = warped1[yp3*nx + x];

				Idx = normalizer_x1/30.f*((-val_xn3_a+9*val_xn2_a-45.f*val_xn_a+45.f*val_xp_a-9*val_xp2_a+val_xp3_a)+(-val_xn3_b+9*val_xn2_b-45.f*val_xn_b+45.f*val_xp_b-9.f*val_xp2_b+val_xp3_b));
				Idy = normalizer_y1/30.f*((-val_yn3_a+9*val_yn2_a-45.f*val_yn_a+45.f*val_yp_a-9*val_yp2_a+val_yp3_a)+(-val_yn3_b+9*val_yn2_b-45.f*val_yn_b+45.f*val_yp_b-9.f*val_yp2_b+val_yp3_b));
				Idt = val0b-frame0_val;
				//////////////////////////////////////////////////////////////////////////////////////////
			}
			else if (spatiotemporalderivative_id == 5){
				//LBM
				//////////////////////////////////////////////////////////////////////////////////////////
				mathtype_solver w2 = 0.02777777777777777f;
				mathtype_solver w1 = 4*w2;

				mathtype_solver val_xnyn_a = frame0[yn*nx + xn];
				mathtype_solver val_xny0_a = frame0[y*nx + xn];
				mathtype_solver val_xnyp_a = frame0[yp*nx + xn];
				mathtype_solver val_x0yn_a = frame0[yn*nx + x];
				mathtype_solver val_x0yp_a = frame0[yp*nx + x];
				mathtype_solver val_xpyn_a = frame0[yn*nx + xp];
				mathtype_solver val_xpy0_a = frame0[y*nx + xp];
				mathtype_solver val_xpyp_a = frame0[yp*nx + xp];

				mathtype_solver val_xnyn_b = warped1[yn*nx + xn];
				mathtype_solver val_xny0_b = warped1[y*nx + xn];
				mathtype_solver val_xnyp_b = warped1[yp*nx + xn];
				mathtype_solver val_x0yn_b = warped1[yn*nx + x];
				mathtype_solver val_x0y0_b = warped1[pos];
				mathtype_solver val_x0yp_b = warped1[yp*nx + x];
				mathtype_solver val_xpyn_b = warped1[yn*nx + xp];
				mathtype_solver val_xpy0_b = warped1[y*nx + xp];
				mathtype_solver val_xpyp_b = warped1[yp*nx + xp];

				Idx = 1.5f*(w1*(val_xpy0_a - val_xny0_a) + w2*( val_xpyn_a +val_xpyp_a -val_xnyn_a -val_xnyp_a));
				Idy = 1.5f*(w1*(val_x0yp_a - val_x0yn_a) + w2*(-val_xpyn_a +val_xpyp_a -val_xnyn_a +val_xnyp_a));

				Idx += 1.5f*(w1*(val_xpy0_b - val_xny0_b) + w2*( val_xpyn_b +val_xpyp_b -val_xnyn_b -val_xnyp_b));
				Idy += 1.5f*(w1*(val_x0yp_b - val_x0yn_b) + w2*(-val_xpyn_b +val_xpyp_b -val_xnyn_b +val_xnyp_b));
				Idt = val_x0y0_b-frame0_val;
				//////////////////////////////////////////////////////////////////////////////////////////
			}
			/////////////////////////////////////////////

			//Intensity constancy:
			/////////////////////////////////////////////
			mathtype_solver J11 = Idx*Idx;
			mathtype_solver J22 = Idy*Idy;
			mathtype_solver J12 = Idx*Idy;
			mathtype_solver J13 = Idx*Idt;
			mathtype_solver J23 = Idy*Idt;
			/////////////////////////////////////////////

			//Calculating data term on the fly doesn't hurt much and saves memory
			//(doesn't work for local global approach)
			////////////////////////////////////////////////////////////////
			if(!precalculated_psi)
			{
				//assuming inner_iterations = 1
				psi0 = Idt;//+Idx*du0+Idy*dv0;
				psi0 *= psi0;
			}
			psi0 = 0.5f/sqrtf(psi0+epsilon_psi_squared);

			if(use_confidencemap) psi0 *= max(0.0f, min(1.0f, confidence));

			//deactivate data term for background:
			if (frame0_val < minIntensity || frame0_val > maxIntensity) psi0 = 0.0f;
			if (slip_depth > 0 && (x < slip_depth || x >= nx-slip_depth || y < slip_depth || y >= ny-slip_depth)) psi0 = 0.0f; //avoid objects getting pinned to the boundary
			////////////////////////////////////////////////////////////////

			//Calculate SOR update
			/////////////////////////////////////////////
			mathtype_solver sumH = alphax*(phi_neighbour[0]+phi_neighbour[1]) + alphay*(phi_neighbour[2]+phi_neighbour[3]);
			mathtype_solver sumU = alphax*(phi_neighbour[0]*du_neighbour[0] + phi_neighbour[1]*du_neighbour[1]) + alphay*(phi_neighbour[2]*du_neighbour[2] + phi_neighbour[3]*du_neighbour[3]);

			mathtype_solver sumH2 = sumH;
			mathtype_solver sumV;
			if(!decoupled_smoothness)
				sumV = alphax*(phi_neighbour[0]*dv_neighbour[0] + phi_neighbour[1]*dv_neighbour[1]) + alphay*(phi_neighbour[2]*dv_neighbour[2] + phi_neighbour[3]*dv_neighbour[3]);
			else
			{
				sumV = alphax*(phi_neighbour[4]*dv_neighbour[0] + phi_neighbour[5]*dv_neighbour[1]) + alphay*(phi_neighbour[6]*dv_neighbour[2] + phi_neighbour[7]*dv_neighbour[3]);
				sumH2 = alphax*(phi_neighbour[4]+phi_neighbour[5]) + alphay*(phi_neighbour[6]+phi_neighbour[7]);
			}
			mathtype_solver next_du, next_dv;

			//SOR-step unless Dirichlet boundary conditions
			////////////////////////////////////////////////////////
			if (boundary_voxel)
			{
				if (    (x == 0 && gpu_const::fixedDirichletBoundary_c[2] == 1) || (x == nx-1 && gpu_const::fixedDirichletBoundary_c[3] == 1)
				     || (y == 0 && gpu_const::fixedDirichletBoundary_c[0] == 1) || (y == ny-1 && gpu_const::fixedDirichletBoundary_c[1] == 1))
				{
					next_du = 0.0f;
					next_dv = 0.0f;
				}
				else
				{
					if ((x == 0 && gpu_const::zeroDirichletBoundary_c[2] == 1) || (x == nx-1 && gpu_const::zeroDirichletBoundary_c[3] == 1))
						next_du = 0.0f; //boundary condition set
					else
						next_du = (1.f-omega)*du0 + omega*(psi0 *(-J13 -J12 * dv0) + sumU)/(psi0*J11 + sumH);

					if ((y == 0 && gpu_const::zeroDirichletBoundary_c[0] == 1) || (y == ny-1 && gpu_const::zeroDirichletBoundary_c[1] == 1))
						next_dv = 0.0f;
					else
						next_dv = (1.f-omega)*dv0 + omega*(psi0 *(-J23 -J12 * next_du) + sumV)/(psi0*J22 + sumH2);
				}
			}
			else
			{
				next_du = (1.f-omega)*du0 + omega*(psi0 *(-J13 -J12 * dv0) + sumU)/(psi0*J11 + sumH);
				next_dv = (1.f-omega)*dv0 + omega*(psi0 *(-J23 -J12 * next_du) + sumV)/(psi0*J22 + sumH2);
			}

			if (gpu_const::protect_overlap_c)
			{
				//extend the Dirichlet boundary inwards for mosaic processing
				int half_overlap = gpu_const::overlap_c/2;

				if (    (x < half_overlap && gpu_const::fixedDirichletBoundary_c[2] == 1) || (x >= nx-1-half_overlap && gpu_const::fixedDirichletBoundary_c[3] == 1)
					 || (y < half_overlap && gpu_const::fixedDirichletBoundary_c[0] == 1) || (y >= ny-1-half_overlap && gpu_const::fixedDirichletBoundary_c[1] == 1))
				{
					next_du = 0.0f;
					next_dv = 0.0f;
				}
			}

			////////////////////////////////////////////////////////

			/////////////////////////////////////////////
			__syncthreads();
			if(!outofbounds)
			{
				du[pos] = next_du;
				du[pos+nstack] = next_dv;
			}
			/////////////////////////////////////////////

			return;
		}

		__global__ void addsolution_warpFrame1_xy(bool rewarp, img_type *warped1, img_type *frame0, img_type *frame1, optflow_type *u, optflow_type *du, optflow_type *confidence){

			//acquire constants
			/////////////////////////////////////////////
			int nx = gpu_const::nx_c;
			int ny = gpu_const::ny_c;
			int nz = gpu_const::nz_c;

			int outOfBounds_id = gpu_const::outOfBounds_id_c;
			int interpolation_id = gpu_const::warpInterpolation_id_c;
			bool use_confidencemap = gpu_const::use_confidencemap_c;

			idx_type nslice = nx*ny;
			idx_type nstack = nz*nslice;

			bool outofbounds = false;
			idx_type pos = (blockIdx.x*blockDim.x+threadIdx.x);
			if (pos >= nstack) {outofbounds = true; pos = threadIdx.x;}

			int z = pos/nslice;
			int y = (pos-z*nslice)/nx;
			int x = pos-z*nslice-y*nx;
			/////////////////////////////////////////////

			/////////////////////////////////////////////
			__syncthreads();
			mathtype_solver u0  = u[pos];
			mathtype_solver v0  = u[pos+nstack];
			mathtype_solver w0 = 0.0f; //should already be warped
			mathtype_solver du0 = du[pos];
			mathtype_solver dv0 = du[pos+nstack];

			if(nz > 1)
				w0 = u[pos+2*nstack]; //for out of bounds checking

			u0 += du0;
			v0 += dv0;

			float x0 = x + u0;
			float y0 = y + v0;
			float z0 = z + w0;

			if(rewarp)
			{
				x0 = x + du0;
				y0 = y + dv0;
				z0 = z;
			}

			//out of bounds?
			////////////////////////
			img_type replace_val = 0.0f;
			bool moved_out = false;

			if (y0 < 0 || x0 < 0 || z0 < 0 || x0 > (nx-1) || y0 > (ny-1) || z0 > (nz-1))
			{
				moved_out = true;
				if (outOfBounds_id == 0) replace_val = frame0[pos];
				else replace_val = gpu_const::nanf_c;

				if (use_confidencemap) confidence[pos] = 0.0f;

				x0 = x; y0 = y; //z0 = z;
			}
			////////////////////////

			int xf = floor(x0);
			int xc = ceil(x0);
			int yf = floor(y0);
			int yc = ceil(y0);

			//extrapolate with zero-gradient
			int xf2 = max(0, xf-1);
			int xc2 = min(xc+1, nx-1);
			int yf2 = max(0, yf-1);
			int yc2 = min(yc+1, ny-1);

			float wx = x0-xf;
			float wy = y0-yf;

			img_type value = 0.0f;

			__syncthreads();
			/////////////////////////////////////////////
			img_type P11 = frame1[z*nslice+ yf*nx + xf];
			img_type P21 = frame1[z*nslice+ yf*nx + xc];
			img_type P12 = frame1[z*nslice+ yc*nx + xf];
			img_type P22 = frame1[z*nslice+ yc*nx + xc];

			if (interpolation_id == 1)
			{
				img_type P10 = frame1[z*nslice + yf2*nx + xf];
				img_type P20 = frame1[z*nslice + yf2*nx + xc];
				img_type P01 = frame1[z*nslice + yf*nx  + xf2];
				img_type P31 = frame1[z*nslice + yf*nx  + xc2];
				img_type P02 = frame1[z*nslice + yc*nx  + xf2];
				img_type P32 = frame1[z*nslice + yc*nx  + xc2];
				img_type P13 = frame1[z*nslice + yc2*nx + xf];
				img_type P23 = frame1[z*nslice + yc2*nx + xc];

				float gtu = gpu2d::interpolate_cubic(P01,P11,P21,P31,wx);
				float gbu = gpu2d::interpolate_cubic(P02,P12,P22,P32,wx);

				float glv = gpu2d::interpolate_cubic(P10,P11,P12,P13,wy);
				float grv = gpu2d::interpolate_cubic(P20,P21,P22,P23,wy);

				float sigma_lr = (1.f-wx)*glv + wx*grv;
				float sigma_bt = (1.f-wy)*gtu + wy*gbu;
				float corr_lrbt = P11*(1.f-wy)*(1.f-wx) + P12*wy*(1.f-wx) + P21*(1.f-wy)*wx + P22*wx*wy;

				value = sigma_lr+sigma_bt-corr_lrbt;
			}
			else
			{
				float glv = (P12-P11)*wy+P11; //left
				float grv = (P22-P21)*wy+P21; //right
				float gtu = (P21-P11)*wx+P11; //top
				float gbu = (P22-P12)*wx+P12; //bottom

				float sigma_lr = (1.f-wx)*glv + wx*grv;
				float sigma_bt = (1.f-wy)*gtu + wy*gbu;
				float corr_lrbt = P11*(1.f-wy)*(1.f-wx) + P12*wy*(1.f-wx) + P21*(1.f-wy)*wx + P22*wx*wy;

				value = sigma_lr+sigma_bt-corr_lrbt;
			}

			if(moved_out) value = replace_val;

			__syncthreads();
			////////////////////////////
			if(!outofbounds)
			{
			warped1[pos] = value;
			u[pos] = u0;
			u[pos+nstack] = v0;
			du[pos] = 0.0f;
			du[pos+nstack] = 0.0f;
			}
			////////////////////////////

			return;
		}
		__global__ void addsolution(optflow_type *u, optflow_type *du)
		{
			//acquire constants
			/////////////////////////////////////////////
			int nx = gpu_const::nx_c;
			int ny = gpu_const::ny_c;
			int nz = gpu_const::nz_c;

			idx_type nslice = nx*ny;
			idx_type nstack = nz*nslice;

			idx_type pos = (blockIdx.x*blockDim.x+threadIdx.x);
			if (pos >= nstack) {pos = threadIdx.x;}
			/////////////////////////////////////////////

			/////////////////////////////////////////////
			__syncthreads();
			mathtype_solver u0  = u[pos];
			mathtype_solver v0  = u[pos+nstack];
			mathtype_solver du0 = du[pos];
			mathtype_solver dv0 = du[pos+nstack];

			u0 += du0;
			v0 += dv0;

			u[pos] = u0;
			u[pos+nstack] = v0;
			du[pos] = 0.0f;
			du[pos+nstack] = 0.0f;
			////////////////////////////

			return;
		}

		__global__ void update_dataterm(img_type *frame0, img_type *warped1, optflow_type *du, optflow_type *psi)
		{
			//acquire constants and position
			/////////////////////////////////////////////
			int nx = gpu_const::nx_c;
			int ny = gpu_const::ny_c;
			int nz = gpu_const::nz_c;

			int spatiotemporalderivative_id = gpu_const::spatiotemporalderivative_id_c;
			mathtype_solver hx = gpu_const::hx_c;
			mathtype_solver hy = gpu_const::hy_c;

			mathtype_solver normalizer_x1 = 0.25f/hx;
			mathtype_solver normalizer_y1 = 0.25f/hy;

			idx_type nslice = nx*ny;
			idx_type nstack = nz*nslice;

			idx_type pos = (blockIdx.x*blockDim.x+threadIdx.x);
			if (pos >= nstack) pos = threadIdx.x;

			int y = pos/nx;
			int x = pos-y*nx;

			int yp = y+1; int yn = y-1;
			int xp = x+1; int xn = x-1;

			//Reflective boundary conditions
			if (yp == ny) yp -= 2; if (yn < 0) yn = 1;
			if (xp == nx) xp -= 2; if (xn < 0) xn = 1;

			__syncthreads();
			/////////////////////////////////////////////

			float Idx, Idy, Idt;

			mathtype_solver du0 = du[pos];
			mathtype_solver dv0 = du[pos+nstack];
			mathtype_solver frame0_val = frame0[pos];

			//Precalculate spatiotemporal derivatives for local-global
			/////////////////////////////////////////////
			if (abs(spatiotemporalderivative_id) == 1){

				//Horn-Schunck: average of frame1 and frame2, dx-kernel := [-1,1; -1,1], dt: local average
				//////////////////////////////////////////////////////////////////////////////////////////
				mathtype_solver val10a = frame0[y*nx + xp];
				mathtype_solver val01a = frame0[yp*nx + x];
				mathtype_solver val11a = frame0[yp*nx + xp];

				mathtype_solver val00b = warped1[y*nx + x];
				mathtype_solver val10b = warped1[y*nx + xp];
				mathtype_solver val01b = warped1[yp*nx + x];
				mathtype_solver val11b = warped1[yp*nx + xp];

				Idx = normalizer_x1*((-frame0_val + val10a - val01a + val11a) + (-val00b + val10b - val01b + val11b));
				Idy = normalizer_y1*((-frame0_val - val10a + val01a + val11a) + (-val00b - val10b + val01b + val11b));
				Idt = 0.25f*((val00b+val10b+val01b+val11b)-(frame0_val+val10a+val01a+val11a));
				//////////////////////////////////////////////////////////////////////////////////////////

			}
			else if (abs(spatiotemporalderivative_id) == 2){

				//Ershov: average of frame1 and frame2, central difference, dt: forward difference
				//////////////////////////////////////////////////////////////////////////////////////////

				mathtype_solver val_xn_a = frame0[y*nx + xn];
				mathtype_solver val_xp_a = frame0[y*nx + xp];
				mathtype_solver val_yn_a = frame0[yn*nx + x];
				mathtype_solver val_yp_a = frame0[yp*nx + x];

				mathtype_solver val_xn_b = warped1[y*nx + xn];
				mathtype_solver val0b    = warped1[pos];
				mathtype_solver val_xp_b = warped1[y*nx + xp];
				mathtype_solver val_yn_b = warped1[yn*nx + x];
				mathtype_solver val_yp_b = warped1[yp*nx + x];

				Idx = normalizer_x1*((val_xp_a-val_xn_a)+(val_xp_b-val_xn_b));
				Idy = normalizer_y1*((val_yp_a-val_yn_a)+(val_yp_b-val_yn_b));
				Idt = val0b-frame0_val;
				//////////////////////////////////////////////////////////////////////////////////////////

			}
			else if (abs(spatiotemporalderivative_id) == 3){
				//Fourth Order Finite Difference
				//////////////////////////////////////////////////////////////////////////////////////////

				int yp2 = y+2; int yn2 = y-2; int xp2 = x+2; int xn2 = x-2;
				if (yp2 >= ny) yp2 = 2*ny-yp2-2; if (yn2 < 0) yn2 = -yn2;
				if (xp2 >= nx) xp2 = 2*nx-xp2-2; if (xn2 < 0) xn2 = -xn2;

				mathtype_solver val_xn2_a = frame0[y*nx + xn2];
				mathtype_solver val_xn_a = frame0[y*nx + xn];
				//mathtype_solver val0a    = frame0[pos];
				mathtype_solver val_xp_a = frame0[y*nx + xp];
				mathtype_solver val_xp2_a = frame0[y*nx + xp2];
				mathtype_solver val_yn2_a = frame0[yn2*nx + x];
				mathtype_solver val_yn_a = frame0[yn*nx + x];
				mathtype_solver val_yp_a = frame0[yp*nx + x];
				mathtype_solver val_yp2_a = frame0[yp2*nx + x];

				mathtype_solver val_xn2_b = warped1[y*nx + xn2];
				mathtype_solver val_xn_b = warped1[y*nx + xn];
				mathtype_solver val0b    = warped1[pos];
				mathtype_solver val_xp_b = warped1[y*nx + xp];
				mathtype_solver val_xp2_b = warped1[y*nx + xp2];
				mathtype_solver val_yn2_b = warped1[yn2*nx + x];
				mathtype_solver val_yn_b = warped1[yn*nx + x];
				mathtype_solver val_yp_b = warped1[yp*nx + x];
				mathtype_solver val_yp2_b = warped1[yp2*nx + x];

				Idx = normalizer_x1/6.f*((val_xn2_a-8.f*val_xn_a+8.f*val_xp_a-val_xp2_a)+(val_xn2_b-8.f*val_xn_b+8.f*val_xp_b-val_xp2_b));
				Idy = normalizer_y1/6.f*((val_yn2_a-8.f*val_yn_a+8.f*val_yp_a-val_yp2_a)+(val_yn2_b-8.f*val_yn_b+8.f*val_yp_b-val_yp2_b));
				Idt = val0b-frame0_val;
				//////////////////////////////////////////////////////////////////////////////////////////
			}
			else if (abs(spatiotemporalderivative_id) == 4){
				//Sixth Order Finite Difference
				//////////////////////////////////////////////////////////////////////////////////////////

				int yp2 = y+2; int yn2 = y-2; int xp2 = x+2; int xn2 = x-2;
				if (yp2 >= ny) yp2 = 2*ny-yp2-2; if (yn2 < 0) yn2 = -yn2;
				if (xp2 >= nx) xp2 = 2*nx-xp2-2; if (xn2 < 0) xn2 = -xn2;

				int yp3 = y+3; int yn3 = y-3; int xp3 = x+3; int xn3 = x-3;
				if (yp3 >= ny) yp3 = 2*ny-yp3-3; if (yn3 < 0) yn3 = -yn3;
				if (xp3 >= nx) xp3 = 2*nx-xp3-3; if (xn3 < 0) xn3 = -xn3;

				__syncthreads();
				mathtype_solver val_xn3_a = frame0[y*nx + xn3];
				mathtype_solver val_xn2_a = frame0[y*nx + xn2];
				mathtype_solver val_xn_a = frame0[y*nx + xn];
				//mathtype_solver val0a    = frame0[pos];
				mathtype_solver val_xp_a = frame0[y*nx + xp];
				mathtype_solver val_xp2_a = frame0[y*nx + xp2];
				mathtype_solver val_xp3_a = frame0[y*nx + xp3];
				mathtype_solver val_yn3_a = frame0[yn3*nx + x];
				mathtype_solver val_yn2_a = frame0[yn2*nx + x];
				mathtype_solver val_yn_a = frame0[yn*nx + x];
				mathtype_solver val_yp_a = frame0[yp*nx + x];
				mathtype_solver val_yp2_a = frame0[yp2*nx + x];
				mathtype_solver val_yp3_a = frame0[yp3*nx + x];

				mathtype_solver val_xn3_b = warped1[y*nx + xn3];
				mathtype_solver val_xn2_b = warped1[y*nx + xn2];
				mathtype_solver val_xn_b = warped1[y*nx + xn];
				mathtype_solver val0b    = warped1[pos];
				mathtype_solver val_xp_b = warped1[y*nx + xp];
				mathtype_solver val_xp2_b = warped1[y*nx + xp2];
				mathtype_solver val_xp3_b = warped1[y*nx + xp3];
				mathtype_solver val_yn3_b = warped1[yn3*nx + x];
				mathtype_solver val_yn2_b = warped1[yn2*nx + x];
				mathtype_solver val_yn_b = warped1[yn*nx + x];
				mathtype_solver val_yp_b = warped1[yp*nx + x];
				mathtype_solver val_yp2_b = warped1[yp2*nx + x];
				mathtype_solver val_yp3_b = warped1[yp3*nx + x];

				Idx = normalizer_x1/30.f*((-val_xn3_a+9*val_xn2_a-45.f*val_xn_a+45.f*val_xp_a-9*val_xp2_a+val_xp3_a)+(-val_xn3_b+9*val_xn2_b-45.f*val_xn_b+45.f*val_xp_b-9.f*val_xp2_b+val_xp3_b));
				Idy = normalizer_y1/30.f*((-val_yn3_a+9*val_yn2_a-45.f*val_yn_a+45.f*val_yp_a-9*val_yp2_a+val_yp3_a)+(-val_yn3_b+9*val_yn2_b-45.f*val_yn_b+45.f*val_yp_b-9.f*val_yp2_b+val_yp3_b));
				Idt = val0b-frame0_val;
				//////////////////////////////////////////////////////////////////////////////////////////
			}
			else if (abs(spatiotemporalderivative_id) == 5){
				//LBM
				//////////////////////////////////////////////////////////////////////////////////////////
				mathtype_solver w2 = 0.02777777777777777f;
				mathtype_solver w1 = 4*w2;

				mathtype_solver val_xnyn_a = frame0[yn*nx + xn];
				mathtype_solver val_xny0_a = frame0[y*nx + xn];
				mathtype_solver val_xnyp_a = frame0[yp*nx + xn];
				mathtype_solver val_x0yn_a = frame0[yn*nx + x];
				mathtype_solver val_x0yp_a = frame0[yp*nx + x];
				mathtype_solver val_xpyn_a = frame0[yn*nx + xp];
				mathtype_solver val_xpy0_a = frame0[y*nx + xp];
				mathtype_solver val_xpyp_a = frame0[yp*nx + xp];

				mathtype_solver val_xnyn_b = warped1[yn*nx + xn];
				mathtype_solver val_xny0_b = warped1[y*nx + xn];
				mathtype_solver val_xnyp_b = warped1[yp*nx + xn];
				mathtype_solver val_x0yn_b = warped1[yn*nx + x];
				mathtype_solver val_x0y0_b = warped1[pos];
				mathtype_solver val_x0yp_b = warped1[yp*nx + x];
				mathtype_solver val_xpyn_b = warped1[yn*nx + xp];
				mathtype_solver val_xpy0_b = warped1[y*nx + xp];
				mathtype_solver val_xpyp_b = warped1[yp*nx + xp];

				Idx = 1.5f*(w1*(val_xpy0_a - val_xny0_a) + w2*( val_xpyn_a +val_xpyp_a -val_xnyn_a -val_xnyp_a));
				Idy = 1.5f*(w1*(val_x0yp_a - val_x0yn_a) + w2*(-val_xpyn_a +val_xpyp_a -val_xnyn_a +val_xnyp_a));

				Idx += 1.5f*(w1*(val_xpy0_b - val_xny0_b) + w2*( val_xpyn_b +val_xpyp_b -val_xnyn_b -val_xnyp_b));
				Idy += 1.5f*(w1*(val_x0yp_b - val_x0yn_b) + w2*(-val_xpyn_b +val_xpyp_b -val_xnyn_b +val_xnyp_b));
				Idt = val_x0y0_b-frame0_val;
				//////////////////////////////////////////////////////////////////////////////////////////
			}

			mathtype_solver psi0 = Idt+Idx*du0+Idy*dv0;
			psi0 *= psi0;

			__syncthreads();
			psi[pos] = psi0;

			if (spatiotemporalderivative_id < 0)
			{
				psi[pos+nstack] = Idx;
				psi[pos+2*nstack] = Idy;
				psi[pos+3*nstack] = Idt;
			}
			return;
		}

		__global__ void zeroinitialize(optflow_type *u, optflow_type *du, optflow_type *confidence)
		{
			int ndim = gpu_const::ndim_c;
			idx_type nstack = gpu_const::nstack_c;
			bool use_confidencemap = gpu_const::use_confidencemap_c;

			idx_type pos = (blockIdx.x*blockDim.x+threadIdx.x);
			if (pos >= nstack) pos = threadIdx.x;
			__syncthreads();

			if(ndim > 2)
			{
				u[pos] = 0.0f;
				u[pos+nstack] = 0.0f;
				u[pos+2*nstack] = 0.0f;
				du[pos] = 0.0f;
				du[pos+nstack] = 0.0f;
				du[pos+2*nstack] = 0.0f;
			}
			else
			{
				u[pos] = 0.0f;
				u[pos+nstack] = 0.0f;
				du[pos] = 0.0f;
				du[pos+nstack] = 0.0f;
			}
			if (use_confidencemap)
			{
				confidence[pos] = 1.0f;
			}

			return;
		}
		__global__ void reset_du(optflow_type *du)
		{
			int ndim = gpu_const::ndim_c;
			idx_type nstack = gpu_const::nstack_c;

			idx_type pos = (blockIdx.x*blockDim.x+threadIdx.x);
			if (pos >= nstack) pos = threadIdx.x;
			__syncthreads();

			if(ndim > 2)
			{
				du[pos] = 0.0f;
				du[pos+nstack] = 0.0f;
				du[pos+2*nstack] = 0.0f;
			}
			else
			{
				du[pos] = 0.0f;
				du[pos+nstack] = 0.0f;
			}
			return;
		}
	}

	int OptFlow_GPU2D::configure_device(int maxshape[3], ProtocolParameters *params){

		deviceID = params->gpu.deviceID;
		cudaSetDevice(deviceID);

		idx_type ndim = 2;
		bool use_confidencemap = params->confidence.use_confidencemap;
		idx_type nstack = maxshape[0]*maxshape[1];
		nstack *= maxshape[2];

		mathtype_solver epsilon_phi_squared = params->smoothness.epsilon_phi;
		epsilon_phi_squared *= epsilon_phi_squared;
		mathtype_solver epsilon_psi_squared = params->solver.epsilon_psi;
		epsilon_psi_squared *= epsilon_psi_squared;
		float nanf = std::numeric_limits<float>::quiet_NaN();

		int outOfBounds_id = 0;
		int warp_interpolation_id = 0;
		int spatiotemporalderivative_id = 0;

			 if (params->warp.outOfBounds_mode == "replace") outOfBounds_id = 0;
		else if (params->warp.outOfBounds_mode == "NaN") outOfBounds_id = 1;
		else std::cout << "Warning! Unknown outOfBounds_mode!" << std::endl;

			 if (params->warp.interpolation_mode == "cubic") warp_interpolation_id = 1;
		else if (params->warp.interpolation_mode == "linear") warp_interpolation_id = 0;
		else std::cout << "Warning! Unknow warp interpolation mode!" << std::endl;

			 if (params->solver.spatiotemporalDerivative_type == "HornSchunck") spatiotemporalderivative_id = 1;
		else if (params->solver.spatiotemporalDerivative_type == "Ershov") spatiotemporalderivative_id = 2; //2nd order
		else if (params->solver.spatiotemporalDerivative_type == "centraldifference") spatiotemporalderivative_id = 2;
		else if (params->solver.spatiotemporalDerivative_type == "Barron") spatiotemporalderivative_id = 3; //4th order
		else if (params->solver.spatiotemporalDerivative_type == "Weickert") spatiotemporalderivative_id = 4; //6th order
		else if (params->solver.spatiotemporalDerivative_type == "LBM") spatiotemporalderivative_id = 5; //6th order
		else {std::cout << "Warning! Unknown spatiotemporal derivative type!" << std::endl;}

		//identify precalculated spatiotemporal derivatives:
		if (params->solver.precalculate_derivatives) spatiotemporalderivative_id *= -1;

		bool anisotropic_smoothness = params->smoothness.anisotropic_smoothness;
		bool decoupled_smoothness = params->smoothness.decoupled_smoothness;
		bool adaptive_smoothness = params->smoothness.adaptive_smoothness;
		bool complementary_smoothness = params->smoothness.complementary_smoothness;

		int slip_depth = params->confidence.slip_depth;

		//check memory requirements
		////////////////////////////////////////////////////
		size_t free_byte, total_byte ;
		cudaMemGetInfo( &free_byte, &total_byte ) ;

		int n_optflow = 5;
		int n_img = 2;

		double free_db = (double)free_byte ;
		double expected_usage = 5.*nstack *sizeof(optflow_type);
		expected_usage += 2.*nstack *sizeof(img_type);
		if(params->confidence.use_confidencemap) {expected_usage += nstack *sizeof(optflow_type); n_optflow++;}
		if (params->solver.precalculate_derivatives) {expected_usage += (4*nstack)*sizeof(optflow_type); n_optflow+=4;}
		else if(params->solver.precalculate_psi) {expected_usage += nstack *sizeof(optflow_type); n_optflow++;}
		if(params->warp.rewarp_frame1 == false) {expected_usage += nstack *sizeof(img_type); n_img++;}
		if(params->smoothness.decoupled_smoothness) {expected_usage += nstack *sizeof(optflow_type); n_optflow++;}
		if(params->smoothness.adaptive_smoothness)  {expected_usage += nstack *sizeof(optflow_type); n_optflow++;}

		if (params->mosaicing.mosaic_decomposition && params->mosaicing.max_nstack == -1)
		{
			params->mosaicing.max_nstack = (free_db-(params->mosaicing.memory_buffer*1024*1024))/(n_optflow*sizeof(optflow_type)+n_img*sizeof(img_type));

			if (nstack > params->mosaicing.max_nstack)
			{
				//set nstack to available memory
				expected_usage = expected_usage/nstack*params->mosaicing.max_nstack;
				nstack = params->mosaicing.max_nstack;
				std::cout << "\033[1;31mmax allowed nstack: " << nstack << "\033[0m" << std::endl;
				std::cout << "\033[1;32mGPU memory: " << round(expected_usage/(1024.*1024.)) << " MB out of " << round(free_db/(1024.*1024.)) << " MB\033[0m" << std::endl;
			}
			else
			{
				//deactivate
				params->mosaicing.mosaic_decomposition = false;
				std::cout << "\033[1;32mGPU memory: " << round(expected_usage/(1024.*1024.)) << " MB out of " << round(free_db/(1024.*1024.)) << " MB\033[0m" << std::endl;
			}
		}
		else
		{
			if (expected_usage > free_db){std::cout << "\033[1;31mError! Expected to run out of GPU memory!\033[0m" << std::endl;return 2;}
			else std::cout << "\033[1;32mGPU memory: " << round(expected_usage/(1024.*1024.)) << " MB out of " << round(free_db/(1024.*1024.)) << " MB\033[0m" << std::endl;
		}
		////////////////////////////////////////////////////

		if (params->mosaicing.mosaic_decomposition && params->mosaicing.sequential_approximation == false && params->gpu.n_gpus == 1)
			params->warp.rewarp_frame1 = true; //no reason to keep frame1 in GPU memory (with single GPU)

		//allocate memory and set constant memory
		////////////////////////////////////////////////////
		(optflow_type*) cudaMalloc((void**)&u, (ndim*nstack)*sizeof(*u));
		(optflow_type*) cudaMalloc((void**)&du, (ndim*nstack)*sizeof(*du));
		if(params->smoothness.decoupled_smoothness) (optflow_type*) cudaMalloc((void**)&phi, 2*nstack*sizeof(*phi));
		else (optflow_type*) cudaMalloc((void**)&phi, nstack*sizeof(*phi));
		//confidence map or background mask:
		if(params->confidence.use_confidencemap) (optflow_type*) cudaMalloc((void**)&confidence, nstack*sizeof(*confidence));
		else (optflow_type*) cudaMalloc((void**)&confidence, 0);
		if (params->solver.precalculate_derivatives) (optflow_type*) cudaMalloc((void**)&psi, (4*nstack)*sizeof(*psi));
		else if(params->solver.precalculate_psi) (optflow_type*) cudaMalloc((void**)&psi, nstack*sizeof(*psi));
		else (optflow_type*) cudaMalloc((void**)&psi, 0);
		if(params->smoothness.adaptive_smoothness) (optflow_type*) cudaMalloc((void**)&adaptivity, nstack*sizeof(*adaptivity));
		else (optflow_type*) cudaMalloc((void**)&adaptivity, 0);

		//using an extra copy to warp from source (rewarp would save a copy)
		(img_type*) cudaMalloc((void**)&warped1, nstack*sizeof(*warped1));
		(img_type*) cudaMalloc((void**)&dev_frame0, nstack*sizeof(*dev_frame0));
		if (params->warp.rewarp_frame1 == false) (img_type*) cudaMalloc((void**)&dev_frame1, nstack*sizeof(*dev_frame1));
		else (img_type*) cudaMalloc((void**)&dev_frame1, 0);

		cudaMemcpyToSymbol(gpu_const::ndim_c, &ndim, sizeof(gpu_const::ndim_c));
		cudaMemcpyToSymbol(gpu_const::nstack_c, &nstack, sizeof(gpu_const::nstack_c));
		cudaMemcpyToSymbol(gpu_const::use_confidencemap_c, &use_confidencemap, sizeof(gpu_const::use_confidencemap_c));
		cudaMemcpyToSymbol(gpu_const::epsilon_phi_squared_c, &epsilon_phi_squared, sizeof(gpu_const::epsilon_phi_squared_c));
		cudaMemcpyToSymbol(gpu_const::epsilon_psi_squared_c, &epsilon_psi_squared, sizeof(gpu_const::epsilon_psi_squared_c));
		cudaMemcpyToSymbol(gpu_const::outOfBounds_id_c, &outOfBounds_id, sizeof(gpu_const::outOfBounds_id_c));
		cudaMemcpyToSymbol(gpu_const::warpInterpolation_id_c, &warp_interpolation_id, sizeof(gpu_const::warpInterpolation_id_c));
		cudaMemcpyToSymbol(gpu_const::spatiotemporalderivative_id_c, &spatiotemporalderivative_id, sizeof(gpu_const::spatiotemporalderivative_id_c));
		cudaMemcpyToSymbol(gpu_const::nanf_c, &nanf, sizeof(gpu_const::nanf_c));
		cudaMemcpyToSymbol(gpu_const::anisotropic_smoothness_c, &anisotropic_smoothness, sizeof(gpu_const::anisotropic_smoothness_c));
		cudaMemcpyToSymbol(gpu_const::decoupled_smoothness_c, &decoupled_smoothness, sizeof(gpu_const::decoupled_smoothness_c));
		cudaMemcpyToSymbol(gpu_const::adaptive_smoothness_c, &adaptive_smoothness, sizeof(gpu_const::adaptive_smoothness_c));
		cudaMemcpyToSymbol(gpu_const::complementary_smoothness_c, &complementary_smoothness, sizeof(gpu_const::complementary_smoothness_c));
		cudaMemcpyToSymbol(gpu_const::slip_depth_c, &slip_depth, sizeof(gpu_const::slip_depth_c));
		cudaDeviceSynchronize();
		////////////////////////////////////////////////////

		//Initialize arrays
		////////////////////////////////////////////////////
		int threadsPerBlock(params->gpu.threadsPerBlock);
		int blocksPerGrid = (nstack + threadsPerBlock - 1) / (threadsPerBlock);

		gpu2d::zeroinitialize<<<blocksPerGrid,threadsPerBlock>>>(u,du,confidence);
		cudaDeviceSynchronize();
		////////////////////////////////////////////////////

		std::string error_string = (std::string) cudaGetErrorString(cudaGetLastError());
		if (error_string != "no error")
		{
			std::cout << "Device Variable Copying: " << error_string << std::endl;
			return 1;
		}

		return 0;
	}
	void OptFlow_GPU2D::free_device(){
		cudaSetDevice(deviceID);

		cudaFree(u);
		cudaFree(du);
		cudaFree(phi);
		cudaFree(psi);
		cudaFree(confidence);
		cudaFree(dev_frame0);
		cudaFree(dev_frame1);
		cudaFree(warped1);
		cudaFree(adaptivity);
	}

	void OptFlow_GPU2D::run_outeriterations(int level, img_type *frame0, img_type *frame1, int shape[3], ProtocolParameters *params, bool resumed_state, bool frames_set)
	{
		cudaSetDevice(deviceID);

		//Set constant memory
		////////////////////////////////////////////////////////////////////////////////////////
		int nx = shape[0]; int ny = shape[1]; int nz = shape[2];
		idx_type nslice = shape[0]*shape[1];
		idx_type nstack = shape[2]*nslice;
		idx_type n_odd = nstack/2;
		idx_type n_even = nstack-n_odd;

		int threadsPerBlock(params->gpu.threadsPerBlock);
		int blocksPerGrid = (nstack + threadsPerBlock - 1) / (threadsPerBlock);
		int blocksPerGrid2 = (n_even + threadsPerBlock -1) / (threadsPerBlock); //iterate over every second voxel

		idx_type asize1 = nstack*sizeof(*dev_frame0);

		mathtype_solver hx = params->scaling.hx;
		mathtype_solver hy = params->scaling.hy;
		mathtype_solver alpha = params->alpha;
		mathtype_solver omega = params->solver.sor_omega;
		bool precalculate_psi = params->solver.precalculate_psi;
		float localglobal_sigma_data = params->special.localglobal_sigma_data;
		bool rewarp = params->warp.rewarp_frame1;
		bool use_confidencemap = params->confidence.use_confidencemap;
		bool protect_overlap = params->mosaicing.protect_overlap;
		int overlap = params->mosaicing.overlap;

		if (params->pyramid.alpha_scaling)
			alpha = alpha/pow(params->pyramid.scaling_factor, level);

		int smoothness_id = 0;

			 if (params->solver.flowDerivative_type == "Barron") smoothness_id = 0;
		else if (params->solver.flowDerivative_type == "centraldifference") smoothness_id = 1; //Ershov style
		else if (params->solver.flowDerivative_type == "forwarddifference") smoothness_id = 2; //Liu style
		else if (params->solver.flowDerivative_type == "LBM") smoothness_id = 3; //Lattice Boltzmann style
		else if (params->solver.flowDerivative_type == "Weickert") smoothness_id = 4;
		else if (params->solver.flowDerivative_type == "Leclaire_FIII") smoothness_id = 5;
		else if (params->solver.flowDerivative_type == "Leclaire_FIV") smoothness_id = 6;
		else if (params->solver.flowDerivative_type == "Scharr3") smoothness_id = 7;
		else if (params->solver.flowDerivative_type == "Sobel") smoothness_id = 8;
		else std::cout << "Warning! Unknown flowDerivative_type!" << std::endl;

		if (!resumed_state)
		{
			cudaMemcpyToSymbol(gpu_const::nx_c, &nx, sizeof(gpu_const::nx_c));
			cudaMemcpyToSymbol(gpu_const::ny_c, &ny, sizeof(gpu_const::ny_c));
			cudaMemcpyToSymbol(gpu_const::nz_c, &nz, sizeof(gpu_const::nz_c));
			cudaMemcpyToSymbol(gpu_const::nstack_c, &nstack, sizeof(gpu_const::nstack_c));
			cudaMemcpyToSymbol(gpu_const::hx_c, &hx, sizeof(gpu_const::hx_c));
			cudaMemcpyToSymbol(gpu_const::hy_c, &hy, sizeof(gpu_const::hy_c));
			cudaMemcpyToSymbol(gpu_const::alpha_c, &alpha, sizeof(gpu_const::alpha_c));
			cudaMemcpyToSymbol(gpu_const::omega_c, &omega, sizeof(gpu_const::omega_c));
			cudaMemcpyToSymbol(gpu_const::zeroDirichletBoundary_c, &params->constraint.zeroDirichletBoundary,  6*sizeof(int), 0);
			cudaMemcpyToSymbol(gpu_const::fixedDirichletBoundary_c, &params->constraint.fixedDirichletBoundary,  6*sizeof(int), 0);
			cudaMemcpyToSymbol(gpu_const::lowerIntensityCutoff_c, &(params->constraint.intensityRange[0]), sizeof(gpu_const::lowerIntensityCutoff_c));
			cudaMemcpyToSymbol(gpu_const::upperIntensityCutoff_c, &(params->constraint.intensityRange[1]), sizeof(gpu_const::upperIntensityCutoff_c));
			cudaMemcpyToSymbol(gpu_const::use_confidencemap_c, &use_confidencemap, sizeof(gpu_const::use_confidencemap_c));
			cudaMemcpyToSymbol(gpu_const::precalculated_psi_c, &precalculate_psi, sizeof(gpu_const::precalculated_psi_c));
			cudaMemcpyToSymbol(gpu_const::filter_sigma_c, &localglobal_sigma_data, sizeof(gpu_const::filter_sigma_c));
			cudaMemcpyToSymbol(gpu_const::protect_overlap_c, &protect_overlap, sizeof(gpu_const::protect_overlap_c));
			cudaMemcpyToSymbol(gpu_const::overlap_c, &overlap, sizeof(gpu_const::overlap_c));

			if(!frames_set)
			{
				cudaMemcpy(dev_frame0, frame0, asize1, cudaMemcpyHostToDevice);
				if(!rewarp) cudaMemcpy(dev_frame1, frame1, asize1, cudaMemcpyHostToDevice);
				else cudaMemcpy(phi, frame1, asize1, cudaMemcpyHostToDevice); //keep dev_frame1 in phi for initial warp
				cudaDeviceSynchronize();
			}
			////////////////////////////////////////////////////////////////////////////////////////

			//initial warp for frame 1
			if(!rewarp) gpu2d::addsolution_warpFrame1_xy<<<blocksPerGrid,threadsPerBlock>>>(false, warped1, dev_frame0, dev_frame1, u, du, confidence);
			else gpu2d::addsolution_warpFrame1_xy<<<blocksPerGrid,threadsPerBlock>>>(false, warped1, dev_frame0, phi, u, du, confidence);
			cudaDeviceSynchronize();
		}

		for (int i_outer = 0; i_outer < params->solver.outerIterations; i_outer++)
		{
			std::cout << "level " << level << " (" << nx << "," << ny << "," << nz << "): " << (i_outer+1) << " \r";
			std::cout.flush();

			for (int i_inner = 0; i_inner < params->solver.innerIterations; i_inner++)
			{
				//gpu2d::reset_du<<<blocksPerGrid,threadsPerBlock>>>(du);
				//cudaDeviceSynchronize();

				if(precalculate_psi)
				{
					gpu2d::update_dataterm<<<blocksPerGrid,threadsPerBlock>>>(dev_frame0, warped1, du, psi);
					cudaDeviceSynchronize();
				}
				if (params->special.localglobal_dataterm)
				{
					gpu2d::gaussianfilter2D_x<<<blocksPerGrid,threadsPerBlock>>>(psi, phi);
					cudaDeviceSynchronize();
					gpu2d::gaussianfilter2D_y<<<blocksPerGrid,threadsPerBlock>>>(phi, psi);
					cudaDeviceSynchronize();
				}

				//Calculate the smoothness term
				//////////////////////////////////////////////////////////////////////////////
				if      (smoothness_id == 0) gpu2d::update_smoothnessterm_Barron<<<blocksPerGrid,threadsPerBlock>>>(u, du,  phi, adaptivity);
				else if (smoothness_id == 1) gpu2d::update_smoothnessterm_centralDiff<<<blocksPerGrid,threadsPerBlock>>>(u, du,  phi, adaptivity);
				else if (smoothness_id == 2) gpu2d::update_smoothnessterm_forwardDiff<<<blocksPerGrid,threadsPerBlock>>>(u, du,  phi, adaptivity);
				else if (smoothness_id == 3) gpu2d::update_smoothnessterm_LBM<<<blocksPerGrid,threadsPerBlock>>>(u, du, phi, adaptivity);
				else if (smoothness_id == 4) gpu2d::update_smoothnessterm_Weickert<<<blocksPerGrid,threadsPerBlock>>>(u, du, phi, adaptivity);
				else if (smoothness_id == 5) gpu2d::update_smoothnessterm_Leclaire_FIII<<<blocksPerGrid,threadsPerBlock>>>(u, du, phi, adaptivity);
				else if (smoothness_id == 6) gpu2d::update_smoothnessterm_Leclaire_FIV<<<blocksPerGrid,threadsPerBlock>>>(u, du, phi, adaptivity);
				else if (smoothness_id == 7) gpu2d::update_smoothnessterm_Scharr3<<<blocksPerGrid,threadsPerBlock>>>(u, du, phi, adaptivity);
				else if (smoothness_id == 8) gpu2d::update_smoothnessterm_Sobel<<<blocksPerGrid,threadsPerBlock>>>(u, du, phi, adaptivity);
				cudaDeviceSynchronize();
				//////////////////////////////////////////////////////////////////////////////

				//SOR-Updates with psi calculated on the fly
				//////////////////////////////////////////////////////////////////////////////
				//switching between even and odd
				for (int i_sor = 0; i_sor < 2*params->solver.sorIterations; i_sor++)
				{
					//reset on first sor is optional... deprecated for now, previous result is a better guess
					gpu2d::calculate_sorUpdate<<<blocksPerGrid2,threadsPerBlock>>>(i_sor, dev_frame0, warped1, phi, psi, u, du, confidence);
					cudaDeviceSynchronize();
				}
				//////////////////////////////////////////////////////////////////////////////
			}

			if (rewarp)
			{
				gpu2d::addsolution_warpFrame1_xy<<<blocksPerGrid,threadsPerBlock>>>(true, phi, dev_frame0, warped1, u, du, confidence);
				cudaDeviceSynchronize();
				cudaMemcpy(warped1, phi, asize1, cudaMemcpyDeviceToDevice);
			}
			else gpu2d::addsolution_warpFrame1_xy<<<blocksPerGrid,threadsPerBlock>>>(false, warped1, dev_frame0, dev_frame1, u, du, confidence);
			cudaDeviceSynchronize();
		}

		return;
	}
	void OptFlow_GPU2D::run_singleiteration(int level, img_type *frame0, img_type *frame1, int shape[3], ProtocolParameters *params, bool frames_set)
	{
		cudaSetDevice(deviceID);
		//for multiGPU rewarp option needs to be reactivated!!

		//Set constant memory
		////////////////////////////////////////////////////////////////////////////////////////
		int nx = shape[0]; int ny = shape[1]; int nz = shape[2];
		idx_type nslice = shape[0]*shape[1];
		idx_type nstack = shape[2]*nslice;
		idx_type n_odd = nstack/2;
		idx_type n_even = nstack-n_odd;

		int threadsPerBlock(params->gpu.threadsPerBlock);
		int blocksPerGrid = (nstack + threadsPerBlock - 1) / (threadsPerBlock);
		int blocksPerGrid2 = (n_even + threadsPerBlock -1) / (threadsPerBlock); //iterate over every second voxel

		idx_type asize1 = nstack*sizeof(*dev_frame0);

		mathtype_solver hx = params->scaling.hx;
		mathtype_solver hy = params->scaling.hy;
		mathtype_solver alpha = params->alpha;
		mathtype_solver omega = params->solver.sor_omega;
		bool precalculate_psi = params->solver.precalculate_psi;
		float localglobal_sigma_data = params->special.localglobal_sigma_data;
		bool use_confidencemap = params->confidence.use_confidencemap;
		bool protect_overlap = params->mosaicing.protect_overlap;
		int overlap = params->mosaicing.overlap;

		if (params->pyramid.alpha_scaling)
			alpha = alpha/pow(params->pyramid.scaling_factor, level);

		int smoothness_id = 0;

			 if (params->solver.flowDerivative_type == "Barron") smoothness_id = 0;
		else if (params->solver.flowDerivative_type == "centraldifference") smoothness_id = 1; //Ershov style
		else if (params->solver.flowDerivative_type == "forwarddifference") smoothness_id = 2; //Liu style
		else if (params->solver.flowDerivative_type == "LBM") smoothness_id = 3; //Lattice Boltzmann style
		else if (params->solver.flowDerivative_type == "Weickert") smoothness_id = 4;
		else if (params->solver.flowDerivative_type == "Leclaire_FIII") smoothness_id = 5;
		else if (params->solver.flowDerivative_type == "Leclaire_FIV") smoothness_id = 6;
		else if (params->solver.flowDerivative_type == "Scharr3") smoothness_id = 7;
		else if (params->solver.flowDerivative_type == "Sobel") smoothness_id = 8;
		else std::cout << "Warning! Unknown flowDerivative_type!" << std::endl;

		cudaMemcpyToSymbol(gpu_const::nx_c, &nx, sizeof(gpu_const::nx_c));
		cudaMemcpyToSymbol(gpu_const::ny_c, &ny, sizeof(gpu_const::ny_c));
		cudaMemcpyToSymbol(gpu_const::nz_c, &nz, sizeof(gpu_const::nz_c));
		cudaMemcpyToSymbol(gpu_const::nstack_c, &nstack, sizeof(gpu_const::nstack_c));
		cudaMemcpyToSymbol(gpu_const::hx_c, &hx, sizeof(gpu_const::hx_c));
		cudaMemcpyToSymbol(gpu_const::hy_c, &hy, sizeof(gpu_const::hy_c));
		cudaMemcpyToSymbol(gpu_const::alpha_c, &alpha, sizeof(gpu_const::alpha_c));
		cudaMemcpyToSymbol(gpu_const::omega_c, &omega, sizeof(gpu_const::omega_c));
		cudaMemcpyToSymbol(gpu_const::zeroDirichletBoundary_c, &params->constraint.zeroDirichletBoundary,  6*sizeof(int), 0);
		cudaMemcpyToSymbol(gpu_const::fixedDirichletBoundary_c, &params->constraint.fixedDirichletBoundary,  6*sizeof(int), 0);
		cudaMemcpyToSymbol(gpu_const::lowerIntensityCutoff_c, &(params->constraint.intensityRange[0]), sizeof(gpu_const::lowerIntensityCutoff_c));
		cudaMemcpyToSymbol(gpu_const::upperIntensityCutoff_c, &(params->constraint.intensityRange[1]), sizeof(gpu_const::upperIntensityCutoff_c));
		cudaMemcpyToSymbol(gpu_const::use_confidencemap_c, &use_confidencemap, sizeof(gpu_const::use_confidencemap_c));
		cudaMemcpyToSymbol(gpu_const::precalculated_psi_c, &precalculate_psi, sizeof(gpu_const::precalculated_psi_c));
		cudaMemcpyToSymbol(gpu_const::filter_sigma_c, &localglobal_sigma_data, sizeof(gpu_const::filter_sigma_c));
		cudaMemcpyToSymbol(gpu_const::protect_overlap_c, &protect_overlap, sizeof(gpu_const::protect_overlap_c));
		cudaMemcpyToSymbol(gpu_const::overlap_c, &overlap, sizeof(gpu_const::overlap_c));

		if (!frames_set)
		{
			cudaMemcpy(dev_frame0, frame0, asize1, cudaMemcpyHostToDevice);
			cudaMemcpy(phi, frame1, asize1, cudaMemcpyHostToDevice); //keep dev_frame1 in phi for initial warp
		}
		cudaDeviceSynchronize();
		////////////////////////////////////////////////////////////////////////////////////////

		//initial warp for frame 1
		gpu2d::addsolution_warpFrame1_xy<<<blocksPerGrid,threadsPerBlock>>>(false, warped1, dev_frame0, phi, u, du, confidence);
		cudaDeviceSynchronize();

		for (int i_inner = 0; i_inner < params->solver.innerIterations; i_inner++)
		{
			if(precalculate_psi)
			{
				gpu2d::update_dataterm<<<blocksPerGrid,threadsPerBlock>>>(dev_frame0, warped1, du, psi);
				cudaDeviceSynchronize();
			}
			if (params->special.localglobal_dataterm)
			{
				gpu2d::gaussianfilter2D_x<<<blocksPerGrid,threadsPerBlock>>>(psi, phi);
				cudaDeviceSynchronize();
				gpu2d::gaussianfilter2D_y<<<blocksPerGrid,threadsPerBlock>>>(phi, psi);
				cudaDeviceSynchronize();
			}

			//Calculate the smoothness term
			//////////////////////////////////////////////////////////////////////////////
			if      (smoothness_id == 0) gpu2d::update_smoothnessterm_Barron<<<blocksPerGrid,threadsPerBlock>>>(u, du,  phi, adaptivity);
			else if (smoothness_id == 1) gpu2d::update_smoothnessterm_centralDiff<<<blocksPerGrid,threadsPerBlock>>>(u, du,  phi, adaptivity);
			else if (smoothness_id == 2) gpu2d::update_smoothnessterm_forwardDiff<<<blocksPerGrid,threadsPerBlock>>>(u, du,  phi, adaptivity);
			else if (smoothness_id == 3) gpu2d::update_smoothnessterm_LBM<<<blocksPerGrid,threadsPerBlock>>>(u, du, phi, adaptivity);
			else if (smoothness_id == 4) gpu2d::update_smoothnessterm_Weickert<<<blocksPerGrid,threadsPerBlock>>>(u, du, phi, adaptivity);
			else if (smoothness_id == 5) gpu2d::update_smoothnessterm_Leclaire_FIII<<<blocksPerGrid,threadsPerBlock>>>(u, du, phi, adaptivity);
			else if (smoothness_id == 6) gpu2d::update_smoothnessterm_Leclaire_FIV<<<blocksPerGrid,threadsPerBlock>>>(u, du, phi, adaptivity);
			else if (smoothness_id == 7) gpu2d::update_smoothnessterm_Scharr3<<<blocksPerGrid,threadsPerBlock>>>(u, du, phi, adaptivity);
			else if (smoothness_id == 8) gpu2d::update_smoothnessterm_Sobel<<<blocksPerGrid,threadsPerBlock>>>(u, du, phi, adaptivity);
			cudaDeviceSynchronize();
			//////////////////////////////////////////////////////////////////////////////

			//SOR-Updates with psi calculated on the fly
			//////////////////////////////////////////////////////////////////////////////
			//switching between even and odd
			for (int i_sor = 0; i_sor < 2*params->solver.sorIterations; i_sor++)
			{
				//reset on first sor is optional... deprecated for now, previous result is a better guess
				gpu2d::calculate_sorUpdate<<<blocksPerGrid2,threadsPerBlock>>>(i_sor, dev_frame0, warped1, phi, psi, u, du, confidence);
				cudaDeviceSynchronize();
			}
			//////////////////////////////////////////////////////////////////////////////
		}

		gpu2d::addsolution<<<blocksPerGrid,threadsPerBlock>>>(u, du);
		cudaDeviceSynchronize();

		return;
	}

	void OptFlow_GPU2D::set_frames(float* frame0, float *frame1, int shape[3], std::vector<int> &boundaries, bool rewarp)
	{
		cudaSetDevice(deviceID);

		int nx = shape[0];

		if (boundaries[0] == 0 && boundaries[3] == nx)
		{
			idx_type offset = boundaries[1]*nx;
			idx_type nslice = boundaries[3]*(boundaries[4]-boundaries[1]);
			idx_type asize1 = nslice*sizeof(*dev_frame0);

			cudaMemcpyAsync(dev_frame0, frame0 + offset, asize1, cudaMemcpyHostToDevice);
			if(!rewarp) cudaMemcpyAsync(dev_frame1, frame1 + offset, asize1, cudaMemcpyHostToDevice);
			else cudaMemcpyAsync(phi, frame1 + offset, asize1, cudaMemcpyHostToDevice);
		}
		else
		{
			idx_type nrow_target = (boundaries[3]-boundaries[0]);
			idx_type nrows = boundaries[4]-boundaries[1];
			idx_type asize1 = nrow_target*sizeof(*dev_frame0);

			for (int y = 0; y < nrows; y++)
			{
				idx_type offset_target = y*nrow_target;
				idx_type offset_source = (y+boundaries[1])*nx + boundaries[0];

				cudaMemcpyAsync(dev_frame0+offset_target, frame0+offset_source, asize1, cudaMemcpyHostToDevice);
				if(!rewarp) cudaMemcpyAsync(dev_frame1+offset_target, frame1+offset_source, asize1, cudaMemcpyHostToDevice);
				else cudaMemcpyAsync(phi+offset_target, frame1+offset_source, asize1, cudaMemcpyHostToDevice);
			}
		}
		cudaDeviceSynchronize();
		return;
	}
	void OptFlow_GPU2D::set_flowvector(float* in_vector, int shape[3])
	{
		idx_type nslice = shape[0]*shape[1];
		idx_type nstack = nslice*shape[2];
		idx_type asize1 = 2*nstack*sizeof(*u);
		optflow_type *u_tmp;

		if(typeid(float) != typeid(optflow_type))
		{
			u_tmp = (optflow_type*) malloc(2*nstack*sizeof(*u_tmp));

			#pragma omp parallel for
			for (idx_type pos = 0; pos < nstack; pos++)
			{
				u_tmp[pos] = in_vector[pos];
				u_tmp[pos+nstack] = in_vector[pos+nstack];
			}
		}
		else
			u_tmp = in_vector;

		cudaSetDevice(deviceID);
		cudaMemcpy(u, u_tmp, asize1, cudaMemcpyHostToDevice);
		cudaDeviceSynchronize();

		return;
	}
	void OptFlow_GPU2D::set_flowvector(optflow_type* in_vector, int shape[3], std::vector<int> &boundaries)
	{
		cudaSetDevice(deviceID);

		int nx = shape[0];
		long long int nslice_full = nx*shape[1];
		long long int nstack_full = shape[2]*nslice_full;

		if (boundaries[0] == 0 && boundaries[3] == nx)
		{
			idx_type offset = boundaries[1]*nx;
			idx_type nslice = boundaries[3]*(boundaries[4]-boundaries[1]);
			idx_type asize1 = nslice*sizeof(*u);

			cudaMemcpyAsync(u, in_vector + offset, asize1, cudaMemcpyHostToDevice);
			cudaMemcpyAsync(u+nslice, in_vector + offset +nstack_full, asize1, cudaMemcpyHostToDevice);
		}
		else
		{
			idx_type nrow_target = (boundaries[3]-boundaries[0]);
			idx_type nrows = boundaries[4]-boundaries[1];
			idx_type nslice_target = nrow_target*nrows;
			idx_type asize1 = nrow_target*sizeof(*u);

			for (int y = 0; y < nrows; y++)
			{
				idx_type offset_target = y*nrow_target;
				idx_type offset_source = (y+boundaries[1])*nx + boundaries[0];

				cudaMemcpyAsync(u+offset_target, in_vector+offset_source, asize1, cudaMemcpyHostToDevice);
				cudaMemcpyAsync(u+offset_target+nslice_target, in_vector+offset_source+nstack_full, asize1, cudaMemcpyHostToDevice);
			}
		}
		cudaDeviceSynchronize();
		return;
	}
	void OptFlow_GPU2D::set_confidencemap(float* confidencemap, int shape[3])
	{
		idx_type nslice = shape[0]*shape[1];
		idx_type nstack = nslice*shape[2];
		idx_type asize1 = nstack*sizeof(*confidence);
		optflow_type *c_tmp;

		if(typeid(float) != typeid(optflow_type))
		{
			c_tmp = (optflow_type*) malloc(nstack*sizeof(*c_tmp));

			#pragma omp parallel for
			for (idx_type pos = 0; pos < nstack; pos++)
			{
				c_tmp[pos] = confidencemap[pos];
			}
		}
		else
			c_tmp = confidencemap;

		cudaSetDevice(deviceID);
		cudaMemcpy(confidence, c_tmp, asize1, cudaMemcpyHostToDevice);
		cudaDeviceSynchronize();

		return;
	}
	void OptFlow_GPU2D::set_confidencemap(optflow_type* confidencemap, int shape[3], std::vector<int> &boundaries)
	{
		cudaSetDevice(deviceID);

		int nx = shape[0];

		if (boundaries[0] == 0 && boundaries[3] == nx)
		{
			idx_type offset = boundaries[1]*nx;
			idx_type nslice = boundaries[3]*(boundaries[4]-boundaries[1]);
			idx_type asize1 = nslice*sizeof(*confidence);

			cudaMemcpy(confidence, confidencemap + offset, asize1, cudaMemcpyHostToDevice);
		}
		else
		{
			idx_type nrow_target = (boundaries[3]-boundaries[0]);
			idx_type nrows = boundaries[4]-boundaries[1];
			idx_type asize1 = nrow_target*sizeof(*confidence);

			for (int y = 0; y < nrows; y++)
			{
				idx_type offset_target = y*nrow_target;
				idx_type offset_source = (y+boundaries[1])*nx + boundaries[0];

				cudaMemcpyAsync(confidence+offset_target, confidencemap+offset_source, asize1, cudaMemcpyHostToDevice);
			}
		}
		cudaDeviceSynchronize();
		return;
	}
	void OptFlow_GPU2D::set_adaptivitymap(float* adaptivitymap, int shape[3])
	{
		idx_type nslice = shape[0]*shape[1];
		idx_type nstack = nslice*shape[2];
		idx_type asize1 = (nstack)*sizeof(*adaptivity);
		optflow_type *c_tmp;

		if(typeid(float) != typeid(optflow_type))
		{
			c_tmp = (optflow_type*) malloc(nstack*sizeof(*c_tmp));

			#pragma omp parallel for
			for (idx_type pos = 0; pos < nstack; pos++)
			{
				c_tmp[pos] = adaptivitymap[pos];
			}
		}
		else
			c_tmp = adaptivitymap;

		cudaSetDevice(deviceID);
		cudaMemcpy(adaptivity, c_tmp, asize1, cudaMemcpyHostToDevice);
		cudaDeviceSynchronize();

		return;
	}
	void OptFlow_GPU2D::set_adaptivitymap(optflow_type* adaptivitymap, int shape[3], std::vector<int> &boundaries)
	{
		cudaSetDevice(deviceID);

		int nx = shape[0];

		if (boundaries[0] == 0 && boundaries[3] == nx)
		{
			idx_type offset = boundaries[1]*nx;
			idx_type nslice = boundaries[3]*(boundaries[4]-boundaries[1]);
			idx_type asize1 = nslice*sizeof(*adaptivity);

			cudaMemcpy(adaptivity, adaptivitymap + offset, asize1, cudaMemcpyHostToDevice);
		}
		else
		{
			idx_type nrow_target = (boundaries[3]-boundaries[0]);
			idx_type nrows = boundaries[4]-boundaries[1];
			idx_type asize1 = nrow_target*sizeof(*adaptivity);

			for (int y = 0; y < nrows; y++)
			{
				idx_type offset_target = y*nrow_target;
				idx_type offset_source = (y+boundaries[1])*nx + boundaries[0];

				cudaMemcpyAsync(adaptivity+offset_target, adaptivitymap+offset_source, asize1, cudaMemcpyHostToDevice);
			}
		}
		cudaDeviceSynchronize();
		return;
	}
	void OptFlow_GPU2D::get_resultcopy(float* out_vector, int shape[3])
	{
		cudaSetDevice(deviceID);

		idx_type nslice = shape[0]*shape[1];
		idx_type nstack = nslice*shape[2];
		idx_type asize1 = 2*nstack*sizeof(*u);

		if(typeid(float) != typeid(optflow_type))
		{
			optflow_type *u_tmp = (optflow_type*) malloc(2*nstack*sizeof(*u_tmp));

			cudaMemcpy(u_tmp,u, asize1, cudaMemcpyDeviceToHost);

			#pragma omp parallel for
			for (idx_type pos = 0; pos < nstack; pos++)
			{
				out_vector[pos] = u_tmp[pos];
				out_vector[pos+nstack] = u_tmp[pos+nstack];
			}
		}
		else
			cudaMemcpy(out_vector,u, asize1, cudaMemcpyDeviceToHost);

		cudaDeviceSynchronize();
		return;
	}
	void OptFlow_GPU2D::get_resultcopy(optflow_type* out_vector, int shape[3], std::vector<int> &boundaries)
	{
		cudaSetDevice(deviceID);

		int nx = shape[0];
		long long int nslice_full = nx*shape[1];
		long long int nstack_full = shape[2]*nslice_full;

		if (boundaries[0] == 0 && boundaries[3] == nx)
		{
			idx_type offset = boundaries[1]*nx;
			idx_type nslice = boundaries[3]*(boundaries[4]-boundaries[1]);
			idx_type asize1 = nslice*sizeof(*u);

			cudaMemcpyAsync(out_vector + offset, u, asize1, cudaMemcpyDeviceToHost);
			cudaMemcpyAsync(out_vector + offset + nstack_full, u + nslice, asize1, cudaMemcpyDeviceToHost);
		}
		else
		{
			idx_type nrow_source = (boundaries[3]-boundaries[0]);
			idx_type nrows = boundaries[4]-boundaries[1];
			idx_type nslice_source = nrow_source*nrows;
			idx_type asize1 = nrow_source*sizeof(*u);

			for (int y = 0; y < nrows; y++)
			{
				idx_type offset_source = y*nrow_source;
				idx_type offset_target = (y+boundaries[1])*nx + boundaries[0];

				cudaMemcpyAsync(out_vector+offset_target,u+offset_source,  asize1, cudaMemcpyDeviceToHost);
				cudaMemcpyAsync(out_vector+offset_target+nstack_full,u+offset_source+nslice_source, asize1, cudaMemcpyDeviceToHost);
			}
		}
		cudaDeviceSynchronize();

		return;
	}
	void OptFlow_GPU2D::get_psimap(float* outimg, int shape[3])
	{
		cudaSetDevice(deviceID);

		idx_type nslice = shape[0]*shape[1];
		idx_type nstack = nslice*shape[2];
		idx_type asize1 = nstack*sizeof(*psi);

		if(typeid(float) != typeid(optflow_type))
		{
			optflow_type *u_tmp = (optflow_type*) malloc(2*nstack*sizeof(*u_tmp));

			cudaMemcpy(u_tmp,psi, asize1, cudaMemcpyDeviceToHost);

			#pragma omp parallel for
			for (idx_type pos = 0; pos < nstack; pos++)
			{
				outimg[pos] = u_tmp[pos];
				outimg[pos+nstack] = u_tmp[pos+nstack];
			}
		}
		else
			cudaMemcpy(outimg,psi, asize1, cudaMemcpyDeviceToHost);

		cudaDeviceSynchronize();
		return;
	}
}
