#ifndef DEFAULT_EVALUATION_H
#define DEFAULT_EVALUATION_H

#include <iostream>
#include <string.h>
#include <cstdint>

#include "../protocol_parameters.h"
#include "../Solver/optflow_base.h"

#include "../Geometry/hdcommunication.h"
#include "../Geometry/filtering.h"
#include "../Geometry/warping.h"
#include "../Geometry/derivatives.h"

#include "../Scaling/pyramid.h"
#include "../Scaling/resampling.h"

#include "mosaic_evaluation.h"
#include "../analysis.h"

namespace protocol
{
	typedef float img_type;
	typedef float optflow_type;

	using namespace optflow;
	using namespace std;

	float run_default_evaluation(int pylevel, OptFlowSolver *optflow_solver, ProtocolParameters *params, img_type *frame0, img_type *frame1, optflow_type *&result, int shape[3],
			img_type *background_mask, img_type *confidencemap, img_type *adaptivity_map, int next_shape[3])
	{
		float outval = 0.0;

		int nx = shape[0]; int ny = shape[1]; int nz = shape[2];
		long long int nslice = nx*ny;
		long long int nstack = nz*nslice;

		int ndims = (shape[2] > 1) ? 3 : 2;
		int outeriter = params->solver.outerIterations;

		float tmp_convergence = params->special.doi_convergence;
		if (pylevel == 0) params->special.doi_convergence = params->special.doi_convergence_level0;

		//Set supporting maps
		///////////////////////////////////////////////////////////////////////////////////
		if (params->smoothness.adaptive_smoothness) optflow_solver->set_adaptivitymap(adaptivity_map, shape);
		if (params->confidence.use_confidencemap)   optflow_solver->set_confidencemap(confidencemap,  shape);

		float* result_backup = (float*) malloc((3*nstack)*sizeof(*result_backup));
		#pragma omp parallel for
		for (long long int idx = 0; idx < 3*nstack; idx++) result_backup[idx] = result[idx];
		///////////////////////////////////////////////////////////////////////////////////

		//Update result (on GPU)
		///////////////////////////////////////////////////////////////////////////////////
		optflow_solver->set_flowvector(result, shape);
		optflow_solver->run_outeriterations(pylevel, frame0, frame1, shape, params, false);
		optflow_solver->get_resultcopy(result, shape);
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
				#pragma omp parallel for
				for (long long int idx = 0; idx < 3*nstack; idx++) result_backup[idx] = result[idx];

				optflow_solver->run_outeriterations(pylevel, frame0, frame1, shape, params, true);
				optflow_solver->get_resultcopy(result, shape);
				outeriter += params->solver.outerIterations;

				double next_result_ev = 0.0;
				float maxchange = 0.0;

				#pragma omp parallel for reduction(+: next_result_ev), reduction(max: maxchange)
				for (long long int pos = 0; pos < nstack; pos++)
				{
					if (!params->confidence.background_mask || background_mask[pos] > 0.0f){
						for (int i = 0; i < ndims; i++) next_result_ev += fabs(result[i*nstack+pos]);

						for (int i = 0; i < ndims; i++)
							if (fabs(result[i*nstack+pos]-result_backup[i*nstack+pos]) > maxchange) maxchange = fabs(result[i*nstack+pos]-result_backup[i*nstack+pos]);
				}}

				next_result_ev /= counter*params->special.doI_stepsize; //check on a per iteration basis
				float rel_change = fabs(next_result_ev/std::max(1e-12f, last_result_ev)-1.f);
				last_result_ev = next_result_ev;

				std::cout << "level " << pylevel << " (" << nx << "," << ny << "," << nz << "): " << outeriter << ", " << rel_change <<","<<maxchange << "           \r";
				std::cout.flush();

				if (rel_change < params->special.doi_convergence) {
					std::cout << "level " << pylevel<< " (" << nx << "," << ny << "," << nz << "): " << outeriter << ", " << rel_change<<","<<maxchange;
					break;}
			}
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

		 //resample and rescale flow-vector to next level (on host)
		/////////////////////////////////
		resample::upscalevector(result, shape, next_shape, ndims, params->scaling.upscaling_interpolation_mode);
		/////////////////////////////////

		if (pylevel == 0) params->special.doi_convergence = tmp_convergence;

		std::cout << std::endl;

		free(result_backup);

		return outval;
	}

	void run_evaluation_mode_postprocessing(OptFlowSolver *optflow_solver, ProtocolParameters *params, img_type *frame0, img_type *frame1, optflow_type *&result,
			optflow_type *&result2, img_type *background_mask, img_type *confidencemap, img_type *adaptivity_map, int shape[3], std::string outpath)
	{
		int nx = shape[0]; int ny = shape[1]; int nz = shape[2];
		long long int nslice = nx*ny;
		long long int nstack = nz*nslice;
		int ndims = (shape[2] > 1) ? 3 : 2;

		if (params->special.evaluation_mode.find((string) "forward") != string::npos && params->special.evaluation_mode.find((string) "backward") != string::npos)
		{
			//"forward-backward" = warp and invert result2 and average
			warp::warpVector1_xyz(result, result2, result, shape, params);

			if (params->special.evaluation_mode.find((string) "confidence") != string::npos)
			{
				//"forward-backward-confidence" = generate an additional confidence output
				confidencemap = anal::measure_confidence(result, result2, shape, params->confidence.confidence_mode, params->confidence.confidence_beta);

				hdcom::HdCommunication hdcom;
				if (shape[2] == 1) hdcom.SaveTif_unknowndim_32bit(confidencemap, shape, outpath, "confidence");
				else hdcom.SaveTif_unknowndim_32bit(confidencemap, shape, outpath+"/confidence/", "confidence");
			}

			//take the average
			#pragma omp parallel for
			for(uint64_t pos = 0; pos < ndims*nstack; pos++) result[pos] = (result[pos]-result2[pos])/2.f;

			if (params->special.evaluation_mode.find((string) "confidence") != string::npos && params->special.evaluation_mode.find((string) "filter") != string::npos)
			{
				//"forward-backward-confidence-filter" = use confidence map for adaptive median filtering
				#pragma omp parallel for
				for (uint64_t pos = 0; pos < nstack; pos++)
				{
					//create a filter mask
					if(confidencemap[pos] < params->confidence.confidence_filter_cutoff) confidencemap[pos] = 1.0f;
					else confidencemap[pos] = 0.0f;
				}

				filter::apply2vector_3DMedianFilter_spheric(result,confidencemap, ndims,shape,params->special.flowfilter_radius);
			}
		}
		else if (params->special.evaluation_mode.find((string) "backward") != string::npos)
		{
			//backward only = warp from itself and invert
			swap(result, result2);
			#pragma omp parallel for
			for(uint64_t pos = 0; pos < ndims*nstack; pos++) result[pos] *= -1.f;

			warp::warpVector1_xyz(result, result, result, shape, params);
		}

		if (params->special.evaluation_mode == "forward-backward-confidence-apply")
		{
			//reapply mask
			if (params->confidence.background_mask)
			{
				#pragma omp parallel for
				for(uint64_t pos = 0; pos < nstack; pos++) confidencemap[pos] *= std::max(0.0f, std::min(1.f, background_mask[pos]));
			}

			params->confidence.use_confidencemap = true;

			if(!params->mosaicing.mosaic_decomposition)
				protocol::run_default_evaluation(0, optflow_solver, params, frame0, frame1, result, shape, background_mask, confidencemap, adaptivity_map, shape);
			else if (!params->mosaicing.sequential_approximation)
				protocol::run_singleGPU_mosaic(0, optflow_solver, params, frame0, frame1, result, shape, background_mask, confidencemap, adaptivity_map, shape);
			else
				protocol::run_sequential_mosaic(0, optflow_solver, params, frame0, frame1, result, shape, background_mask, confidencemap, adaptivity_map, shape);
		}

		return;
	}
}

#endif //DEFAULT_EVALUATION_H
