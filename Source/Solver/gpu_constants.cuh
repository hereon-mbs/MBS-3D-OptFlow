#ifndef GPU_CONSTANTS_CUH
#define GPU_CONSTANTS_CUH

#include <iostream>
#include <cuda.h>

namespace optflow
{
	namespace gpu_const
	{
		__constant__ int nx_c, ny_c, nz_c;
		__constant__ int ndim_c = 3;
		__constant__ idx_type nstack_c;

		__constant__ int outOfBounds_id_c = 0;
		__constant__ int warpInterpolation_id_c = 0;
		__constant__ int spatiotemporalderivative_id_c = 0;

		__constant__ mathtype_solver alpha_c = 1.0f;
		__constant__ mathtype_solver omega_c = 1.8f;
		__constant__ mathtype_solver max_sorupdate = 0.2f; //limits the stepsize the displacements can make in one sor-update. Stabilizes convergences but doesn't improve final scale.

		__constant__ mathtype_solver epsilon_phi_squared_c = 1.e-6;
		__constant__ mathtype_solver epsilon_psi_squared_c = 1.e-6;
		__constant__ float nanf_c;

		__constant__ mathtype_solver hx_c = 1.0f;
		__constant__ mathtype_solver hy_c = 1.0f;
		__constant__ mathtype_solver hz_c = 1.0f;

		//Constraints:
		__constant__ int zeroDirichletBoundary_c[6] = {0,0,0,0,0,0}; //north,south,left,right,top,bottom (zero change in that dimension)
		__constant__ int fixedDirichletBoundary_c[6] = {0,0,0,0,0,0}; //(zero change in no dimension)
		__constant__ float lowerIntensityCutoff_c = -2000000000.f;
		__constant__ float upperIntensityCutoff_c =  2000000000.f;

		//Program flow
		__constant__ bool precalculated_psi_c = false;

		//LocalGlobal
		__constant__ float filter_sigma_c = 1.5f;

		//Optimal Derivatives
		__constant__ int farid_samples_c = 4;

		//Smoothness term
		__constant__ bool anisotropic_smoothness_c = false;
		__constant__ bool decoupled_smoothness_c = false; //separate smoothness term for u,v and w
		__constant__ bool adaptive_smoothness_c = true; //adapt smoothing direction to image edges
		__constant__ bool complementary_smoothness_c = false; //stronger smoothing along edges

		//data term
		__constant__ bool use_confidencemap_c = false;
		//without confidencemap we still need to make sure that voxels moving out of the domain have no dataterm:
		__constant__ int slip_depth_c = 1; //activate with a value > 0

		//Mosaicing
		__constant__ bool protect_overlap_c = false;
		__constant__ int overlap_c = 0;

		//Kernels
	}
}

#endif //GPU_CONSTANTS_CUH
