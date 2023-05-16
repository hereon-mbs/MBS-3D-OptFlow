#ifndef MOSAIC_OVERLAP
#define MOSAIC_OVERLAP

#include <iostream>
#include <string.h>
#include <cstdint>

#include "../protocol_parameters.h"
#include "../Solver/optflow_base.h"

#include "../Geometry/hdcommunication.h"
#include "../Geometry/filtering.h"
#include "../Geometry/warping.h"
#include "../Geometry/mosaicing.h"

#include "../Scaling/resampling.h"

namespace protocol
{
	typedef float img_type;
	typedef float optflow_type;

	using namespace optflow;
	using namespace std;

	/* run_sequential_mosaic:
	 * 	    Does not communicate between patches. Patches have a Dirichlet boundary to previously solved patches.
	 * 	    Should work if there are only small objects and is reasonable fast.
	 * 	    Direction alternates by level.
	 *
	 * run_singleGPU_mosaic:
	 *	    Updates every patch within an outer iteration. Dirichlet boundaries maintain the previous result. (better fit but bigger deviation from default and slow)
	 */

	float run_sequential_mosaic(int pylevel, OptFlowSolver *optflow_solver, ProtocolParameters *params, img_type *frame0, img_type *frame1, optflow_type *&result, int shape[3],
				img_type *background_mask, img_type *confidencemap, img_type *adaptivitymap, int next_shape[3])
	{
		//
		//Faster Approximation!
		//

		float outval = 0.0;

		int nx = shape[0]; int ny = shape[1]; int nz = shape[2];
		long long int nslice = nx*ny;
		long long int nstack = nz*nslice;

		int ndims = (shape[2] > 1) ? 3 : 2;

		params->mosaicing.protect_overlap = true; //movements >1 across a patch boundary need protection from out of bounds warping

		//Update result (on GPU)
		///////////////////////////////////////////////////////////////////////////////////
		int overlap = params->mosaicing.overlap;
		long long int max_nstack = params->mosaicing.max_nstack;

		int maxshape[3];
		std::vector<std::vector<int>> patches = mosaic::get_mosaic_coordinates(shape, params, maxshape);

		for (int step = 0; step < patches.size(); step++)
		{
			int outeriter = params->solver.outerIterations;

			int patch_id = step;
			if (params->mosaicing.alternating_directions && (pylevel%2) != 0){patch_id = patches.size()-step-1;} //alternate between directions

			int active_shape[3];
			active_shape[0] = patches[patch_id][3]-patches[patch_id][0];
			active_shape[1] = patches[patch_id][4]-patches[patch_id][1];
			active_shape[2] = patches[patch_id][5]-patches[patch_id][2];

			long long int active_nslice = active_shape[0]*active_shape[1];
			long long int active_nstack = active_nslice*active_shape[2];

			//set confidencemap and adaptivitymap
			///////////////////////////////////////////////////////////////////////////////////
			if (params->confidence.use_confidencemap)
				optflow_solver->set_confidencemap(confidencemap, shape, patches[patch_id]);
			if (params->smoothness.adaptive_smoothness)
				optflow_solver->set_adaptivitymap(adaptivitymap, shape, patches[patch_id]);
			///////////////////////////////////////////////////////////////////////////////////

			//set the result
			///////////////////////////////////////////////////////////////////////////////////
			optflow_solver->set_flowvector(result, shape, patches[patch_id]);
			///////////////////////////////////////////////////////////////////////////////////

			//This should work if there are only small volumes that can be covered by the overlap:
			//set boundary conditions (with stupid order) for active patch
			//if not stack boundary, lower boundary is Dirichlet (non-zero but zero change)
			///////////////////////////////////////////////////////////////////////////////////
			if (params->mosaicing.alternating_directions)
			{
				params->constraint.fixedDirichletBoundary[2] = ((pylevel%2) == 0 && patches[patch_id][0] != 0) ? 1 : 0;
				params->constraint.fixedDirichletBoundary[0] = ((pylevel%2) == 0 && patches[patch_id][1] != 0) ? 1 : 0;
				params->constraint.fixedDirichletBoundary[4] = ((pylevel%2) == 0 && patches[patch_id][2] != 0) ? 1 : 0;

				params->constraint.fixedDirichletBoundary[3] = ((pylevel%2) != 0 && patches[patch_id][3] != nx) ? 1 : 0;
				params->constraint.fixedDirichletBoundary[1] = ((pylevel%2) != 0 && patches[patch_id][4] != ny) ? 1 : 0;
				params->constraint.fixedDirichletBoundary[5] = ((pylevel%2) != 0 && patches[patch_id][5] != nz) ? 1 : 0;
			}
			else{
				params->constraint.fixedDirichletBoundary[2] = (patches[patch_id][0] != 0) ? 1 : 0;
				params->constraint.fixedDirichletBoundary[0] = (patches[patch_id][1] != 0) ? 1 : 0;
				params->constraint.fixedDirichletBoundary[4] = (patches[patch_id][2] != 0) ? 1 : 0;
			}
			///////////////////////////////////////////////////////////////////////////////////

			//set the frames and run iterations
			///////////////////////////////////////////////////////////////////////////////////
			optflow_solver->set_frames(frame0, frame1, shape, patches[patch_id], params->warp.rewarp_frame1);
			optflow_solver->run_outeriterations(pylevel, frame0, frame1, active_shape, params, false, true);
			optflow_solver->get_resultcopy(result, shape, patches[patch_id]);
			///////////////////////////////////////////////////////////////////////////////////

			//Additional dynamic updates
			///////////////////////////////////////////////////////////////////////////////////
			if (params->special.dynamic_outerIterations)
			{
				//Instead of running a set amount of iterations we track the convergence.
				//Advantage is that we run more iterations at high pyramid level and less on the lower levels
				float last_result_ev = 0.0f; //EV of displacement of foreground voxels
				float counter = 0.0f;

				#pragma omp parallel for reduction(+: last_result_ev, counter)
				for (long long int pos = 0; pos < active_nstack; pos++)
				{
					int z0 = pos/active_nslice;
					int y0 = (pos-z0*active_nslice)/active_shape[0];
					int x0 = pos-z0*active_nslice-y0*active_shape[0];
					long long int pos_full = (z0+patches[patch_id][2])*nslice+(y0+patches[patch_id][1])*nx+(x0+patches[patch_id][0]);

					if (!params->confidence.background_mask || background_mask[pos_full] > 0.0f)
					{
						for (int i = 0; i < ndims; i++)
						{
							last_result_ev += fabs(result[i*nstack+pos_full]);
							counter++;
				}}}

				last_result_ev /= counter*params->special.doI_stepsize;

				while (outeriter < params->special.doI_maxOuterIter)
				{
					optflow_solver-> run_outeriterations(pylevel, frame0, frame1, active_shape, params, true, true);
					optflow_solver->get_resultcopy(result, shape, patches[patch_id]);
					outeriter += params->solver.outerIterations;

					float next_result_ev = 0.0;

					#pragma omp parallel for reduction(+: next_result_ev)
					for (long long int pos = 0; pos < active_nstack; pos++)
					{
						int z0 = pos/active_nslice;
						int y0 = (pos-z0*active_nslice)/active_shape[0];
						int x0 = pos-z0*active_nslice-y0*active_shape[0];
						long long int pos_full = (z0+patches[patch_id][2])*nslice+(y0+patches[patch_id][1])*nx+(x0+patches[patch_id][0]);

						if (!params->confidence.background_mask || background_mask[pos_full] > 0.0f){
							for (int i = 0; i < ndims; i++) next_result_ev += fabs(result[i*nstack+pos_full]);
					}}

					next_result_ev /= counter*params->special.doI_stepsize; //check on a per iteration basis
					float rel_change = fabs(next_result_ev/std::max(1e-12f, last_result_ev)-1.f);
					last_result_ev = next_result_ev;

					std::cout << "level " << pylevel << ", patch " << (step+1) << "/" << patches.size() << " (" << active_shape[0] << "," << active_shape[1] << "," << active_shape[2] << "): " << outeriter << ", " << rel_change << "           \r";
					std::cout.flush();

					if (rel_change < params->special.doi_convergence){
						std::cout << "level " << pylevel << ", patch " << (step+1) << "/" << patches.size() << " (" << active_shape[0] << "," << active_shape[1] << "," << active_shape[2] << "): " << outeriter << ", " << rel_change;
						std::cout.flush();
						break;}
				}
			}
			///////////////////////////////////////////////////////////////////////////////////

			if(step+1 < patches.size())
				std::cout << std::endl;
		}
		///////////////////////////////////////////////////////////////////////////////////

		//Apply median filter (on host)
		/////////////////////////////////
		if (params->special.medianfilter_flow && params->special.flowfilter_radius > 0.0)
			filter::apply2vector_3DMedianFilter_spheric(result, ndims, shape ,params->special.flowfilter_radius);
		else if (pylevel == 0 && params->postprocessing.median_filter)
			filter::apply2vector_3DMedianFilter_spheric(result,ndims, shape, params->postprocessing.median_radius);
		/////////////////////////////////

		if (params->special.track_correlation)
		{
			float* warped1 = (float*) malloc(nstack*sizeof(*warped1));
			warp::warpFrame1_xyz(warped1, frame0, frame1, result, shape, params);
			std::vector<float> quality = anal::get_qualitymeasures(frame0, warped1, result, shape, background_mask, params->confidence.background_mask);
			std::cout << ", " << quality[1];
			outval = quality[1];
			free(warped1);
		}
		std::cout << std::endl;

		 //resample and rescale flow-vector to next level (on host)
		/////////////////////////////////
		resample::upscalevector(result, shape, next_shape, ndims, params->scaling.upscaling_interpolation_mode);
		/////////////////////////////////
		return outval;
	}

	float run_singleGPU_mosaic(int pylevel, OptFlowSolver *optflow_solver, ProtocolParameters *params, img_type *frame0, img_type *frame1, optflow_type *&result, int shape[3],
					img_type *background_mask, img_type *confidencemap, img_type *adaptivitymap, int next_shape[3])
	{
		float outval = 0.0;

		int nx = shape[0]; int ny = shape[1]; int nz = shape[2];
		long long int nslice = nx*ny;
		long long int nstack = nz*nslice;

		int ndims = (shape[2] > 1) ? 3 : 2;
		int outeriter = params->solver.outerIterations;

		params->mosaicing.protect_overlap = true; //movements >1 across a patch boundary need protection from out of bounds warping

		//Update result (on GPU)
		///////////////////////////////////////////////////////////////////////////////////
		int overlap = params->mosaicing.overlap;
		long long int max_nstack = params->mosaicing.max_nstack;

		int maxshape[3];
		std::vector<std::vector<int>> patches = mosaic::get_mosaic_coordinates(shape, params, maxshape);

		params->mosaicing.overlap -= params->solver.sorIterations+1; //provide some wiggling room

		for (int i_outer = 0; i_outer < params->solver.outerIterations; i_outer++)
		{
			for (int patch_id = 0; patch_id < patches.size(); patch_id++)
			{
				int active_shape[3];
				active_shape[0] = patches[patch_id][3]-patches[patch_id][0];
				active_shape[1] = patches[patch_id][4]-patches[patch_id][1];
				active_shape[2] = patches[patch_id][5]-patches[patch_id][2];

				long long int active_nslice = active_shape[0]*active_shape[1];
				long long int active_nstack = active_nslice*active_shape[2];

				//set confidencemap and adaptivitymap
				///////////////////////////////////////////////////////////////////////////////////
				if (params->confidence.use_confidencemap)
					optflow_solver->set_confidencemap(confidencemap, shape, patches[patch_id]);
				if (params->smoothness.adaptive_smoothness)
					optflow_solver->set_adaptivitymap(adaptivitymap, shape, patches[patch_id]);
				///////////////////////////////////////////////////////////////////////////////////

				//set the result
				///////////////////////////////////////////////////////////////////////////////////
				optflow_solver->set_flowvector(result, shape, patches[patch_id]);
				///////////////////////////////////////////////////////////////////////////////////

				//This should work if there are only small volumes that can be covered by the overlap:
				//set boundary conditions (with stupid order) for active patch
				//if not stack boundary, lower boundary is Dirichlet (non-zero but zero change)
				///////////////////////////////////////////////////////////////////////////////////
				params->constraint.fixedDirichletBoundary[2] = (patches[patch_id][0] != 0) ? 1 : 0;
				params->constraint.fixedDirichletBoundary[0] = (patches[patch_id][1] != 0) ? 1 : 0;
				params->constraint.fixedDirichletBoundary[4] = (patches[patch_id][2] != 0) ? 1 : 0;

				if(i_outer > 0)
				{
					params->constraint.fixedDirichletBoundary[3] = (patches[patch_id][3] != nx) ? 1 : 0;
					params->constraint.fixedDirichletBoundary[1] = (patches[patch_id][4] != ny) ? 1 : 0;
					params->constraint.fixedDirichletBoundary[5] = (patches[patch_id][5] != nz) ? 1 : 0;
				}
				///////////////////////////////////////////////////////////////////////////////////

				//set the frames and run iterations
				///////////////////////////////////////////////////////////////////////////////////
				optflow_solver->set_frames(frame0, frame1, shape, patches[patch_id], params->warp.rewarp_frame1);
				optflow_solver->run_singleiteration(pylevel, frame0, frame1, active_shape, params, true);
				optflow_solver->get_resultcopy(result, shape, patches[patch_id]);
				///////////////////////////////////////////////////////////////////////////////////
			}

			if (((i_outer+1)%10) == 0)
			{
				std::cout << "level " << pylevel << ": " << (i_outer+1) << "    \r";
				std::cout.flush();
			}
		}

		//Additional dynamic updates
		///////////////////////////////////////////////////////////////////////////////////
		if (params->special.dynamic_outerIterations)
		{
			//Instead of running a set amount of iterations we track the convergence.
			//Advantage is that we run more iterations at high pyramid level and less on the lower levels
			int outeriter = params->solver.outerIterations;
			float last_result_ev = 0.0f; //EV of displacement of foreground voxels
			float counter = 0.0f;

			#pragma omp parallel for reduction(+: last_result_ev, counter)
			for (long long int pos = 0; pos < nstack; pos++)
			{
				if (!params->confidence.background_mask || background_mask[pos] > 0.0f)
				{
					for (int i = 0; i < ndims; i++)
					{
						last_result_ev += fabs(result[i*nstack+pos]);
						counter++;
			}}}

			last_result_ev /= counter*params->special.doI_stepsize;

			while (outeriter < params->special.doI_maxOuterIter)
			{
				for (int i_outer = 0; i_outer < params->solver.outerIterations; i_outer++)
				{
					for (int patch_id = 0; patch_id < patches.size(); patch_id++)
					{
						int active_shape[3];
						active_shape[0] = patches[patch_id][3]-patches[patch_id][0];
						active_shape[1] = patches[patch_id][4]-patches[patch_id][1];
						active_shape[2] = patches[patch_id][5]-patches[patch_id][2];

						long long int active_nslice = active_shape[0]*active_shape[1];
						long long int active_nstack = active_nslice*active_shape[2];

						//set confidencemap and adaptivitymap
						///////////////////////////////////////////////////////////////////////////////////
						if (params->confidence.use_confidencemap)
							optflow_solver->set_confidencemap(confidencemap, shape, patches[patch_id]);
						if (params->smoothness.adaptive_smoothness)
							optflow_solver->set_adaptivitymap(adaptivitymap, shape, patches[patch_id]);
						///////////////////////////////////////////////////////////////////////////////////

						//set the result
						///////////////////////////////////////////////////////////////////////////////////
						optflow_solver->set_flowvector(result, shape, patches[patch_id]);
						///////////////////////////////////////////////////////////////////////////////////

						//This should work if there are only small volumes that can be covered by the overlap:
						//set boundary conditions (with stupid order) for active patch
						//if not stack boundary, lower boundary is Dirichlet (non-zero but zero change)
						///////////////////////////////////////////////////////////////////////////////////
						params->constraint.fixedDirichletBoundary[2] = (patches[patch_id][0] != 0) ? 1 : 0;
						params->constraint.fixedDirichletBoundary[0] = (patches[patch_id][1] != 0) ? 1 : 0;
						params->constraint.fixedDirichletBoundary[4] = (patches[patch_id][2] != 0) ? 1 : 0;

						if(i_outer > 0)
						{
							params->constraint.fixedDirichletBoundary[3] = (patches[patch_id][3] != nx) ? 1 : 0;
							params->constraint.fixedDirichletBoundary[1] = (patches[patch_id][4] != ny) ? 1 : 0;
							params->constraint.fixedDirichletBoundary[5] = (patches[patch_id][5] != nz) ? 1 : 0;
						}
						///////////////////////////////////////////////////////////////////////////////////

						//set the frames and run iterations
						///////////////////////////////////////////////////////////////////////////////////
						optflow_solver->set_frames(frame0, frame1, shape, patches[patch_id], params->warp.rewarp_frame1);
						optflow_solver->run_singleiteration(pylevel, frame0, frame1, active_shape, params, true);
						optflow_solver->get_resultcopy(result, shape, patches[patch_id]);
						///////////////////////////////////////////////////////////////////////////////////
					}

					if (((i_outer+1)%10) == 0)
					{
						std::cout << "level " << pylevel << ": " << i_outer << "    \r";
						std::cout.flush();
					}
				}
				outeriter += params->solver.outerIterations;

				float next_result_ev = 0.0;

				#pragma omp parallel for reduction(+: next_result_ev)
				for (long long int pos = 0; pos < nstack; pos++)
				{
					if (!params->confidence.background_mask || background_mask[pos] > 0.0f){
						for (int i = 0; i < ndims; i++) next_result_ev += fabs(result[i*nstack+pos]);
				}}

				next_result_ev /= counter*params->special.doI_stepsize; //check on a per iteration basis
				float rel_change = fabs(next_result_ev/std::max(1e-12f, last_result_ev)-1.f);
				last_result_ev = next_result_ev;

				if (patches.size() == 1)
					std::cout << "level " << pylevel << ", " << patches.size() << " patch (" << nx << "," << ny << "," << nz << "): " << outeriter << ", " << rel_change << "           \r";
				else
					std::cout << "level " << pylevel << ", " << patches.size() << " patches (" << nx << "," << ny << "," << nz << "): " << outeriter << ", " << rel_change << "           \r";
				std::cout.flush();

				if (rel_change < params->special.doi_convergence) {
					if (patches.size() == 1)
						std::cout << "level " << pylevel << ", " << patches.size() << " patch (" << nx << "," << ny << "," << nz << "): " << outeriter << ", " << rel_change;
					else
						std::cout << "level " << pylevel << ", " << patches.size() << " patches (" << nx << "," << ny << "," << nz << "): " << outeriter << ", " << rel_change;
					std::cout.flush();
					break;}
			}
		}

		///////////////////////////////////////////////////////////////////////////////////

		params->mosaicing.overlap +=  params->solver.sorIterations+1; //and reset

		///////////////////////////////////////////////////////////////////////////////////

		//Apply median filter (on host)
		/////////////////////////////////
		if (params->special.medianfilter_flow && params->special.flowfilter_radius > 0.0)
			filter::apply2vector_3DMedianFilter_spheric(result, ndims, shape ,params->special.flowfilter_radius);
		else if (pylevel == 0 && params->postprocessing.median_filter)
			filter::apply2vector_3DMedianFilter_spheric(result,ndims, shape, params->postprocessing.median_radius);
		/////////////////////////////////

		if (params->special.track_correlation)
		{
			float* warped1 = (float*) malloc(nstack*sizeof(*warped1));
			warp::warpFrame1_xyz(warped1, frame0, frame1, result, shape, params);
			std::vector<float> quality = anal::get_qualitymeasures(frame0, warped1, result, shape, background_mask, params->confidence.background_mask);
			std::cout << ", " << quality[1];
			outval = quality[1];
			free(warped1);
		}
		std::cout << std::endl;

		 //resample and rescale flow-vector to next level (on host)
		/////////////////////////////////
		resample::upscalevector(result, shape, next_shape, ndims, params->scaling.upscaling_interpolation_mode);
		/////////////////////////////////

		return outval;
	}
}

#endif //MOSAIC_OVERLAP
