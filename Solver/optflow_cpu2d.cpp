#include <iostream>
#include <math.h>
#include <omp.h>
#include <limits>
#include <cmath>
#include "optflow_cpu2d.h"
#include "../Scaling/resampling.h"
#include "../Derivatives/smoothnessterm_cpu.h"

namespace optflow
{
	namespace cpu2d
	{
	void calculate_sorUpdate(int iter, img_type *frame0, img_type *warped1, int shape[3], ProtocolParameters *params, optflow_type *phi, optflow_type *u, optflow_type *du,
			optflow_type *confidencemap)
	{
		//Set what would be in constant GPU-memory
		/////////////////////////////////////////////////////////////////
		int spatiotemporalderivative_id = 0;

		     if (params->solver.spatiotemporalDerivative_type == "HornSchunck") spatiotemporalderivative_id = 0;
		else if (params->solver.spatiotemporalDerivative_type == "Ershov") spatiotemporalderivative_id = 1;
		else {std::cout << "Warning! Unknown spatiotemporal derivative type!" << std::endl;}

		mathtype_solver epsilon_psi_squared = params->solver.epsilon_psi;
		epsilon_psi_squared *= epsilon_psi_squared;

		mathtype_solver hx = params->scaling.hx;
		mathtype_solver hy = params->scaling.hy;

		mathtype_solver alphax = params->alpha/(hx*hx);
		mathtype_solver alphay = params->alpha/(hy*hy);

		mathtype_solver omega = params->solver.sor_omega;

		bool use_confidencemap = params->confidence.use_confidencemap;

		int nx = shape[0];
		int ny = shape[1];
		int nz = shape[2];
		/////////////////////////////////////////////////////////////////
		idx_type nslice = shape[0]*shape[1];
		idx_type nstack = shape[2]*nslice;

		mathtype_solver confidence = 1.0f;
		mathtype_solver normalizer_x1 = 0.25f/hx;
		mathtype_solver normalizer_y1 = 0.25f/hy;

		#pragma omp parallel for
		for (idx_type idx = 0; idx < nstack/2+1; idx++)
		{
			idx_type pos = 2*idx;
			int y = pos/nx;
			int x = pos-y*nx;

			if((iter%2) == 0)
			{
				if ((nx%2) == 0 && (y%2) != 0) {x++; pos++;}
			}
			else
			{
				if((nx%2) != 0) {x++; pos++;}
				else if((y%2) == 0){x++; pos++;}
			}
			if (x >= nx) {x = 0; y++; pos = y*nx+x;}

			if (pos >= nstack) continue;

			int yp = y+1;
			int yn = y-1;
			int xp = x+1;
			int xn = x-1;

			mathtype_solver phi0 = phi[pos];
			mathtype_solver u0 = u[pos];
			mathtype_solver v0 = u[pos+nstack];
			mathtype_solver du0 = du[pos];
			mathtype_solver dv0 = du[pos+nstack];

			if (use_confidencemap) confidence = confidencemap[pos];

			mathtype_solver phi_neighbour[4] = {0.0f, 0.0f, 0.0f, 0.0f,};
			mathtype_solver du_neighbour[4]  = {0.0f, 0.0f, 0.0f, 0.0f,};
			mathtype_solver dv_neighbour[4]  = {0.0f, 0.0f, 0.0f, 0.0f,};

			//Read in neighbours with 0-boundaries
			///////////////////////////////////////////////////////////
			if (xp < nx){
				//Ershov style
				idx_type npos = y*nx + xp;
				phi_neighbour[0] = 0.5f*(phi[npos] + phi0);
				du_neighbour[0]  = u[npos] + du[npos] - u0;
				dv_neighbour[0]  = u[npos + nstack] + du[npos + nstack] - v0;
			}
			if (xn >= 0){
				idx_type npos = y*nx + xn;
				phi_neighbour[1] = 0.5f*(phi[npos] + phi0);
				du_neighbour[1]  = u[npos] + du[npos] - u0;
				dv_neighbour[1]  = u[npos + nstack] + du[npos + nstack] - v0;
			}
			if (yp < ny){
				idx_type npos = yp*nx + x;
				phi_neighbour[2] = 0.5f*(phi[npos] + phi0);
				du_neighbour[2]  = u[npos] + du[npos] - u0;
				dv_neighbour[2]  = u[npos + nstack] + du[npos + nstack] - v0;
			}
			if (yn >= 0){
				idx_type npos = yn*nx + x;
				phi_neighbour[3] = 0.5f*(phi[npos] + phi0);
				du_neighbour[3]  = u[npos] + du[npos] - u0;
				dv_neighbour[3]  = u[npos + nstack] + du[npos + nstack] - v0;
			}
			///////////////////////////////////////////////////////////

			//Switch to reflective boundary conditions
			if (yp == ny) yp -= 2;
			else if (yn == -1) yn = 1;
			if (xp == nx) xp -= 2;
			else if (xn == -1) xn = 1;

			mathtype_solver Idx, Idy, Idt;
			mathtype_solver frame0_val = frame0[y*nx + x];

			//Calculate spatiotemporal derivatives on the fly
			/////////////////////////////////////////////
			if (spatiotemporalderivative_id == 0){
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
			else if (spatiotemporalderivative_id == 1){
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
			mathtype_solver psi0 = Idt+Idx*du0+Idy*dv0;

			psi0 *= psi0;
			psi0 = 0.5f/sqrtf(psi0+epsilon_psi_squared);

			if(use_confidencemap) psi0 *= confidence;
			////////////////////////////////////////////////////////////////

			//Calculate SOR update
			////////////////////////////////////////////////////////
			mathtype_solver sumH = alphax*(phi_neighbour[0]+phi_neighbour[1]) + alphay*(phi_neighbour[2]+phi_neighbour[3]);
			mathtype_solver sumU = alphax*(phi_neighbour[0]*du_neighbour[0] + phi_neighbour[1]*du_neighbour[1]) + alphay*(phi_neighbour[2]*du_neighbour[2] + phi_neighbour[3]*du_neighbour[3]);
			mathtype_solver sumV = alphax*(phi_neighbour[0]*dv_neighbour[0] + phi_neighbour[1]*dv_neighbour[1]) + alphay*(phi_neighbour[2]*dv_neighbour[2] + phi_neighbour[3]*dv_neighbour[3]);

			mathtype_solver next_du, next_dv;

			if (frame0_val >= params->constraint.intensityRange[0] && frame0_val <= params->constraint.intensityRange[1])
			{
				if ((x == 0 && params->constraint.zeroDirichletBoundary[2] == 1) || (x == nx-1 && params->constraint.zeroDirichletBoundary[3] == 1))
					next_du = 0.0f;
				else
					next_du = (1.f-omega)*du0 + omega*(psi0 *(-J13 -J12 * dv0) + sumU)/(psi0*J11 + sumH);

				if ((y == 0 && params->constraint.zeroDirichletBoundary[0] == 1) || (y == ny-1 && params->constraint.zeroDirichletBoundary[1] == 1))
					next_dv = 0.0f;
				else
					next_dv = (1.f-omega)*dv0 + omega*(psi0 *(-J23 -J12 * next_du) + sumV)/(psi0*J22 + sumH);
			}
			else next_du = next_dv = 0.0f;
			////////////////////////////////////////////////////////

			du[pos] = next_du;
			du[pos+nstack] = next_dv;
		}

		return;
	}

	void addsolution_warpFrame1(img_type *warped1, img_type *frame0, img_type *frame1, optflow_type *u, optflow_type *du, int shape[3], ProtocolParameters *params)
	{
		//////////////////////////////////////////////////////////////////
		int outOfBounds_id = 0;
		int interpolation_id = 0;

		if (params->warp.outOfBounds_mode == "replace") outOfBounds_id = 0;
		else if (params->warp.outOfBounds_mode == "NaN") outOfBounds_id = 1;
		else std::cout << "Warning! Unknown outOfBounds_mode!" << std::endl;

		if (params->warp.interpolation_mode == "cubic") interpolation_id = 1;
		else if (params->warp.interpolation_mode == "linear") interpolation_id = 0;
		else std::cout << "Warning! Unknow warp interpolation mode!" << std::endl;

		int nx = shape[0];
		int ny = shape[1];

		optflow_type maxchange = 0.0f;
		//////////////////////////////////////////////////////////////////

		idx_type nslice = shape[0]*shape[1];
		idx_type nstack = shape[2]*nslice;

		#pragma omp parallel for
		for (idx_type pos = 0; pos < nstack; pos++)
		{
			mathtype_solver u0 = u[pos];
			mathtype_solver v0 = u[pos+nstack];
			mathtype_solver du0 = du[pos];
			mathtype_solver dv0 = du[pos+nstack];
			u0 += du0;
			v0 += dv0;

			int z = 0;
			int y = pos/nx;
			int x = pos-y*nx;

			float x0 = x + u0;
			float y0 = y + v0;

			int xf = floor(x0);
			int xc = ceil(x0);
			int yf = floor(y0);
			int yc = ceil(y0);

			float wx = x0-xf;
			float wy = y0-yf;

			float value = 0.0;

			if (y0 < 0.0f || x0 < 0.0f || x0 > (nx-1) || y0 > (ny-1))
			{
				if (outOfBounds_id == 0)
					value = frame0[pos];
				else
					value = std::numeric_limits<float>::quiet_NaN();
			}
			else if (interpolation_id == 1)//cubic interpolation
			{
				//extrapolate with zero-gradient
				int xf2 = std::max(0, xf-1);
				int xc2 = std::min(xc+1, shape[0]-1);
				int yf2 = std::max(0, yf-1);
				int yc2 = std::min(yc+1, shape[1]-1);

				float P10 = frame1[z*nslice+yf2*nx + xf];
				float P20 = frame1[z*nslice+yf2*nx + xc];

				float P01 = frame1[z*nslice+yf*nx + xf2];
				float P11 = frame1[z*nslice+yf*nx + xf];
				float P21 = frame1[z*nslice+yf*nx + xc];
				float P31 = frame1[z*nslice+yf*nx + xc2];

				float P02 = frame1[z*nslice+yc*nx + xf2];
				float P12 = frame1[z*nslice+yc*nx + xf];
				float P22 = frame1[z*nslice+yc*nx + xc];
				float P32 = frame1[z*nslice+yc*nx + xc2];

				float P13 = frame1[z*nslice+yc2*nx + xf];
				float P23 = frame1[z*nslice+yc2*nx + xc];

				float gtu = resample::interpolate_cubic(P01,P11,P21,P31,wx);
				float gbu = resample::interpolate_cubic(P02,P12,P22,P32,wx);

				float glv = resample::interpolate_cubic(P10,P11,P12,P13,wy);
				float grv = resample::interpolate_cubic(P20,P21,P22,P23,wy);

				float sigma_lr = (1.-wx)*glv + wx*grv;
				float sigma_bt = (1.-wy)*gtu + wy*gbu;
				float corr_lrbt = P11*(1.-wy)*(1.-wx) + P12*wy*(1.-wx) + P21*(1.-wy)*wx + P22*wx*wy;

				value = sigma_lr+sigma_bt-corr_lrbt;
			}
			else //linear_interpolation
			{
				float P00 = frame1[z*nslice+yf*nx + xf];
				float P10 = frame1[z*nslice+yf*nx + xc];
				float P01 = frame1[z*nslice+yc*nx + xf];
				float P11 = frame1[z*nslice+yc*nx + xc];

				float glv = (P01-P00)*wy+P00; //left
				float grv = (P11-P10)*wy+P10; //right
				float gtu = (P10-P00)*wx+P00; //top
				float gbu = (P11-P01)*wx+P01; //bottom

				float sigma_lr = (1.-wx)*glv + wx*grv;
				float sigma_bt = (1.-wy)*gtu + wy*gbu;
				float corr_lrbt = P00*(1.-wy)*(1.-wx) + P01*wy*(1.-wx) + P10*(1.-wy)*wx + P11*wx*wy;

				value = sigma_lr+sigma_bt-corr_lrbt;
			}

			warped1[pos] = value;
			u[pos] = u0;
			u[pos+nstack] = v0;
			du[pos] = 0.0f;
			du[pos+nstack] = 0.0f;
		}

		return;
	}
	}

	int OptFlow_CPU2D::configure_device(int maxshape[3], ProtocolParameters *params)
	{
		idx_type ndim = 2;
		idx_type nstack = maxshape[0]*maxshape[1];
		nstack *= maxshape[2];

		u = (optflow_type*) malloc(ndim*nstack*sizeof(*u));
		du = (optflow_type*) malloc(ndim*nstack*sizeof(*du));
		phi = (optflow_type*) malloc(nstack*sizeof(*phi));
		if(params->confidence.use_confidencemap) confidence = (optflow_type*) malloc(nstack*sizeof(*confidence));
		else confidence = (optflow_type*) malloc(0);

		//using an extra copy to warp from source (rewarp would save a copy)
		warped1 = (img_type*) malloc(nstack*sizeof(*warped1));

		#pragma omp parallel for
		for (idx_type pos = 0; pos < nstack; pos++)
		{
			u[pos] = 0.0f;
			u[pos+nstack] = 0.0f;
			du[pos] = 0.0f;
			du[pos+nstack] = 0.0f;
			//phi[pos] = 0.0f;

			if (params->confidence.use_confidencemap) confidence[pos] = 1.0f;
		}

		return 0;
	}
	void OptFlow_CPU2D::free_device()
	{
		free(u);
		free(du);
		free(phi);
		free(confidence);
		free(warped1);
	}

	void OptFlow_CPU2D::run_outeriterations(int level, img_type *frame0, img_type *frame1, int shape[3], ProtocolParameters *params)
	{
		float mean_distance = 0.0f;
		float last_distance = 1.0f;

		idx_type nslice = shape[0]*shape[1];
		idx_type nstack = shape[2]*nslice;

		mathtype_solver epsilon_phi_squared = params->smoothness.epsilon_phi;
		epsilon_phi_squared *= epsilon_phi_squared;

		mathtype_solver hx = params->scaling.hx;
		mathtype_solver hy = params->scaling.hy;

		int smoothness_id = 0;
		     if (params->solver.flowDerivative_type == "Barron") smoothness_id = 0;
		else if (params->solver.flowDerivative_type == "centraldifference") smoothness_id = 1; //Ershov style
		else if (params->solver.flowDerivative_type == "forwarddifference") smoothness_id = 2; //Liu style

		//initial warp for frame 1
		cpu2d::addsolution_warpFrame1(warped1, frame0, frame1, u, du, shape, params);
		float du_max = 0.0;
		for (int i_outer = 0; i_outer < params->solver.outerIterations; i_outer++)
		{
			std::cout << "level " << level << ": " << "iter " << (i_outer+1) << "       \r";
			std::cout.flush();

			for (int i_inner = 0; i_inner < params->solver.innerIterations; i_inner++)
			{
				//Calculate the smoothness term
				//////////////////////////////////////////////////////////////////////////////
				if      (smoothness_id == 0) cpu2d::update_smoothnessterm_Barron(u, du,  phi, epsilon_phi_squared, shape, hx, hy);
				else if (smoothness_id == 1) cpu2d::update_smoothnessterm_centralDiff(u, du,  phi, epsilon_phi_squared, shape, hx, hy);
				else if (smoothness_id == 2) cpu2d::update_smoothnessterm_forwardDiff(u, du,  phi, epsilon_phi_squared, shape, hx, hy);
				//////////////////////////////////////////////////////////////////////////////

				//////////////////////////////////////////////////////////////////////////////
				//No need to calculate the data term (psi) now unless we want to use a local-global approach...
				//...let's do that in a separate solver module
				//////////////////////////////////////////////////////////////////////////////

				for (int i_sor = 0; i_sor < 2*params->solver.sorIterations; i_sor++)
				{
					cpu2d::calculate_sorUpdate(i_sor, frame0, warped1, shape, params, phi, u, du, confidence);
					//return;
				}
			}

			cpu2d::addsolution_warpFrame1(warped1, frame0, frame1, u, du, shape, params);
		}

		//std::cout << std::endl;

		return;
	}

	void OptFlow_CPU2D::set_flowvector(float* in_vector, int shape[3])
	{
		idx_type nslice = shape[0]*shape[1];
		idx_type nstack = nslice*shape[2];

		#pragma omp parallel for
		for (idx_type pos = 0; pos < nstack; pos++)
		{
			u[pos] = in_vector[pos];
			u[pos+nstack] = in_vector[pos+nstack];
		}

		return;
	}
	void OptFlow_CPU2D::get_resultcopy(float* out_vector, int shape[3])
	{
		idx_type nslice = shape[0]*shape[1];
		idx_type nstack = nslice*shape[2];

		#pragma omp parallel for
		for (idx_type pos = 0; pos < nstack; pos++)
		{
			out_vector[pos] = u[pos];
			out_vector[pos+nstack] = u[pos+nstack];
		}

		return;
	}
}
