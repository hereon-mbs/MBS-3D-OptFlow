#include <iostream>
#include <unistd.h>
#include <chrono>
#include <omp.h>
#include <vector>

#include "Geometry/hdcommunication.h"
#include "Geometry/auxiliary.h"
#include "Geometry/filtering.h"
#include "Geometry/warping.h"
#include "Geometry/derivatives.h"
#include "Geometry/mosaicing.h"
#include "Geometry/rigid_body_motion.h"
#include "Geometry/histogram.h"

#include "protocol_parameters.h"
#include "analysis.h"
#include "Scaling/pyramid.h"
#include "Scaling/resampling.h"

#include "Solver/optflow_base.h"
#include "Solver/optflow_cpu2d.h"
#include "Solver/optflow_cpu3d.h"
#include "Solver/optflow_gpu2d.h"
#include "Solver/optflow_gpu2d_reshape.h"
#include "Solver/optflow_gpu3d.h"
#include "Solver/optflow_gpu3d_reshape.h"

#include "Preprocessing/histomatching.h"

#include "Protocols/default_evaluation.h"
#include "Protocols/mosaic_evaluation.h"

//incomplete correlation window preprocessing for (future) LDOF-Mode
#include "Preprocessing/cornerdetection_cpu.h"
#include "Preprocessing/register_correlationwindow.h"
#include "Preprocessing/guess_interpolation.h"

using namespace std;

/*********************************************************************************************************************************************************
 *
 * --- Framework for calculating dense optical flow on 4D-Xray CT reconstructions ---
 *
 * Location: Helmholtz-Zentrum hereon GmbH, Max-Planck-Strasse 1, 21502 Geesthacht
 * Author: Stefan Bruns
 * Contact: stefan.bruns@hereon.de
 *
 * License: TBA
 *
 * References:
 * 	[1] dos Santos Rolo, Ershov, van de Kamp and Baumbach, PNAS 2014, 111, 3921-3926: "In vivo X-ray cine-tomography for tracking morphological dynamics".
 * 	[2] Brox, Bruhn, Papenberg and Weickert, Computer Vision - ECCV 2004, 25-36: "High Accuracy Optical Flow Estimation Based on a Theory for Warping".
 * 	[3] Brox, PhD thesis 2005: "Von Pixeln zu Regionen: Partielle Differentialgleichungen in der Bildanalyse".
 * 	[4] Ershov, PhD thesis 2015: "Automated Analysis of Time-Resolved X-ray Data using Optical Flow Methods".
 * 	[5] Liu, PhD thesis 2009: "Beyond Pixels: Exploring New Representations and Applications for Motion Analysis"
 *
 * Helpful 2D-repositories:
 * 	[6] https://github.com/pathak22/pyflow
 * 	[7] https://github.com/axruff/cuda-flow2d
 *
 *********************************************************************************************************************************************************/

int main(int argc, char* argv[])
{
	//Program Parameters
	//////////////////////////////////////////////////////////////////////////////////////
	//////////////////////////////////////////////////////////////////////////////////////
	std::string inpath0 = "";
	std::string inpath1 = "";
	std::string outpath = "";

	std::string inpath_mask0 = "none";
	std::string analysis_mask = "none";

	bool check_quality = true;
	bool check_from_backup = true;
	bool export_warped = false;
	bool export_error_image = false;
	//////////////////////////////////////////////////////////////////////////////////////
	//////////////////////////////////////////////////////////////////////////////////////

	//Variables in program scope
	///////////////////////////////////////////////////////////////////////////////////
	optflow::ProtocolParameters params;

	int shape[3]; int shape_backup[3];
	float *frame0, *frame1, *frame0_backup, *frame1_backup;
	float *result, *result2;
	float *confidencemap, *background_mask;
	float *adaptivity_map;

	bool mask_output = false;
	bool warp_mask = false;
	bool skip_vector_export = false;

	bool debug = false;
	bool empty_confidence = false;

	pair<int,int> zrange = {-1,-1};
	float recomask_diameter = -1; //when > -1 set confidence to 0 for everything outside a circular region (max_diameter when 0, else provided value)

	int n_threads = 128; //number of CPU_threads (-1 to use all)

	bool use_initial_guess = false;
	float initial_guess[3] = {0.0f, 0.0f, 0.0f};
	string previous_result_path = "none";

	//guessing the strain
	float prescribed_zstrain = 0.0;
	float relative_strain_origin = 0.5;
	bool prewarp_frame0 = false;
	bool sparse_translation_search = false;

	///////////////////////////////////////////////////////////////////////////////////
	if ("extract command line arguments"){
		for (uint16_t i = 1; i < argc; i++)
		{
			     if ((string(argv[i]) == "-i0") || (string(argv[i]) == "-input0"))
			{
				i++;
				inpath0 = string(argv[i]);
			}
			else if ((string(argv[i]) == "-i1") || (string(argv[i]) == "-input1"))
			{
				i++;
				inpath1 = string(argv[i]);
			}
			else if ((string(argv[i]) == "-m") || (string(argv[i]) == "-mask"))
			{
				i++;
				inpath_mask0 = string(argv[i]);
			}
			else if ((string(argv[i]) == "-o") || (string(argv[i]) == "-output"))
			{
				i++;
				outpath = string(argv[i]);
			}
			else if ((string(argv[i]) == "-analysis_mask"))
			{
				i++;
				analysis_mask = string(argv[i]);
			}
			else if (string(argv[i]) == "-range")
			{
				i++;
				zrange.first = atoi(argv[i]);
				i++;
				zrange.second = atoi(argv[i]);
			}
			else if (string(argv[i]) == "-prefilter")
			{
				i++;
				params.preprocessing.prefilter = string(argv[i]);
				if (string(argv[i]) != "none")
				{
					i++;
					params.preprocessing.prefilter_sigma = atof(argv[i]);
				}
			}
			else if (string(argv[i]) == "-prefilter2")
			{
				i++;
				params.preprocessing.prefilter2 = string(argv[i]);
				if (string(argv[i]) != "none")
				{
					i++;
					params.preprocessing.prefilter2_sigma = atof(argv[i]);
				}
			}
			else if (string(argv[i]) == "-n_cpu" || string(argv[i]) == "-n_threads")
			{
				i++;
				n_threads = atoi(argv[i]);
			}
			else if (string(argv[i]) == "-flowfilter")
			{
				i++;
				params.special.medianfilter_flow = true;
				params.special.flowfilter_radius = atof(argv[i]);
			}
			else if (string(argv[i]) == "-alpha")
			{
				i++;
				params.alpha = atof(argv[i]);
			}
			else if (string(argv[i]) == "-norm")
			{
				i++;
				params.preprocessing.normalization = string(argv[i]);
			}
			else if (string(argv[i]) == "-level")
			{
				i++;
				params.pyramid.nLevels = atoi(argv[i]);
			}
			else if (string(argv[i]) == "-scale")
			{
				i++;
				params.pyramid.scaling_factor = atof(argv[i]);
			}
			else if (string(argv[i]) == "-pyramid")
			{
				i++;
				params.pyramid.scaling_mode = string(argv[i]);
				i++;
				params.pyramid.interpolation_mode = string(argv[i]);
			}
			else if (string(argv[i]) == "-omega")
			{
				i++;
				params.solver.sor_omega = atof(argv[i]);
			}
			else if (string(argv[i]) == "-iter_sor")
			{
				i++;
				params.solver.sorIterations = atoi(argv[i]);
			}
			else if (string(argv[i]) == "-iter_inner")
			{
				i++;
				params.solver.innerIterations = atoi(argv[i]);
			}
			else if (string(argv[i]) == "-iter_outer")
			{
				i++;
				params.solver.outerIterations = atoi(argv[i]);
			}
			else if (string(argv[i]) == "-doI" || string(argv[i]) == "-doi" || string(argv[i]) == "-dynamic_outeriterations")
			{
				params.special.dynamic_outerIterations = true;
				i++;
				params.special.doi_convergence = atof(argv[i]);
				i++;
				params.special.doI_stepsize = atoi(argv[i]);
			}
			else if (string(argv[i]) == "-smoothness_stencil" || string(argv[i]) == "-flow_derivative" || string(argv[i]) == "-smoothness")
			{
				i++;
				params.solver.flowDerivative_type = string(argv[i]);
			}
			else if (string(argv[i]) == "-st_stencil" || string(argv[i]) == "-spatiotemporal_stencil")
			{
				i++;
				params.solver.spatiotemporalDerivative_type = string(argv[i]);
			}
			else if (string(argv[i]) == "-mode" || string(argv[i]) == "-eval_mode")
			{
				i++;
				params.special.evaluation_mode = string(argv[i]);
			}
			else if (string(argv[i]) == "-n_gpu" || string(argv[i]) == "-n_gpus")
			{
				i++;
				params.gpu.n_gpus = atoi(argv[i]);
			}
			else if (string(argv[i]) == "-gpu0")
			{
				i++;
				params.gpu.deviceID = atoi(argv[i]);
			}
			else if (string(argv[i]) == "-guess")
			{
				use_initial_guess = true;
				i++;
				initial_guess[0] = atof(argv[i]);
				i++;
				initial_guess[1] = atof(argv[i]);
				i++;
				initial_guess[2] = atof(argv[i]);
			}
			else if (string(argv[i]) == "-prevres")
			{
				use_initial_guess = true;
				i++;
				previous_result_path = string(argv[i]);
			}
			else if (string(argv[i]) == "-zstrain")
			{
				use_initial_guess = true;
				prewarp_frame0 = false;
				i++;
				prescribed_zstrain = atof(argv[i]);
				i++;
				relative_strain_origin = atof(argv[i]);
				//params.constraint.zeroDirichletBoundary[4] = 1;
				//params.constraint.zeroDirichletBoundary[5] = 1;
			}
			else if (string(argv[i]) == "-prestrain_ref")
			{
				use_initial_guess = false;
				prewarp_frame0 = true;
				i++;
				prescribed_zstrain = atof(argv[i]);
				i++;
				relative_strain_origin = atof(argv[i]);
			}
			else if (string(argv[i]) == "-prestrain_search")
			{
				use_initial_guess = false;
				prewarp_frame0 = true;
				sparse_translation_search = true;
				i++;
				prescribed_zstrain = atof(argv[i]);
				i++;
				relative_strain_origin = atof(argv[i]);
			}
			else if (string(argv[i]) == "-prestrain_from_prevres")
			{
				use_initial_guess = false;
				prewarp_frame0 = true;
				i++;
				previous_result_path = string(argv[i]);
				prescribed_zstrain = -99;
			}
			else if (string(argv[i]) == "-lb")
			{
				i++;
				params.constraint.intensityRange[0] = atof(argv[i]);
			}
			else if (string(argv[i]) == "-ub")
			{
				i++;
				params.constraint.intensityRange[1] = atof(argv[i]);
			}
			else if (string(argv[i]) == "-slip")
			{
				i++;
				params.confidence.slip_depth = atoi(argv[i]);
			}
			else if (string(argv[i]) == "-c" || string(argv[i]) == "-confidence")
			{
				i++;
				params.confidence.confidence_mode = string(argv[i]);
				i++;
				params.confidence.confidence_beta = atof(argv[i]);
			}
			else if (string(argv[i]) == "-median")
			{
				params.postprocessing.median_filter = true;
				i++;
				params.postprocessing.median_radius = atof(argv[i]);
			}
			else if (string(argv[i]) == "--median")
			{
				params.postprocessing.median_filter = true;
			}
			else if (string(argv[i]) == "-upscale")
			{
				i++;
				params.scaling.upscaling_interpolation_mode = string(argv[i]);
			}
			else if (string(argv[i]) == "-localglobal")
			{
				params.special.localglobal_dataterm = true;
				i++;
				params.special.localglobal_sigma_data = atof(argv[i]);
				params.solver.precalculate_derivatives = true; //needs to be precalculated for applying the Gaussian on the whole Tensor
			}
			else if (string(argv[i]) == "-localglobal_gauss")
			{
				params.special.localglobal_dataterm = true;
				params.special.localglobal_mode = "Gaussian";
				i++;
				params.special.localglobal_sigma_data = atof(argv[i]);
				params.solver.precalculate_derivatives = true; //needs to be precalculated for applying the Gaussian on the whole Tensor
			}
			else if (string(argv[i]) == "-gradientmask" || string(argv[i]) == "-gradient_mask")
			{
				params.confidence.gradient_mask = true;
				i++;
				params.confidence.sigma_blur_gradient = atof(argv[i]);
				i++;
				params.confidence.used_percentage_gradient = atof(argv[i]);
			}
			else if (string(argv[i]) == "--recomask")
				recomask_diameter = 0;
			else if (string(argv[i]) == "-recomask")
			{
				i++;
				recomask_diameter = atof(argv[i]);
			}
			else if (string(argv[i]) == "-adaptivity")
			{
				params.smoothness.adaptive_smoothness = true;
				i++;
				params.smoothness.adaptivity_mode = string(argv[i]);
			}
			else if (string(argv[i]) == "--reshape")
			{
				params.gpu.reshape4coalescence = true;
				cout << "Warning! Reshape solvers are deprecated. Overhead is bigger than speed gain in 3D" << endl;
			}
			else if (string(argv[i]) == "--nogridscale")
				params.scaling.use_gridscaling = false;
			else if (string(argv[i]) == "--scaled_alpha")
				params.pyramid.alpha_scaling = true;
			else if (string(argv[i]) == "--rewarp")
				params.warp.rewarp_frame1 = true;
			else if (string(argv[i]) == "--confidence")
				params.confidence.use_confidencemap = true;
			else if (string(argv[i]) == "--isotropic")
				params.smoothness.anisotropic_smoothness = false;
			else if (string(argv[i]) == "--decoupled")
				params.smoothness.decoupled_smoothness = true;
			else if (string(argv[i]) == "--adaptive")
				params.smoothness.adaptive_smoothness = true;
			else if (string(argv[i]) == "--complementary")
				params.smoothness.complementary_smoothness = true;
			else if (string(argv[i]) == "--mask_output")
				mask_output = true;
			else if (string(argv[i]) == "--gradientmask" || string(argv[i]) == "--gradient_mask")
				params.confidence.gradient_mask = true;
			else if (string(argv[i]) == "-advanced_gradient")
			{
				i++;
				params.confidence.advancedgradient = string(argv[i]);
			}
			else if (string(argv[i]) == "--exportmask" || string(argv[i]) == "--export_mask")
				params.confidence.export_mask = true;
			else if (string(argv[i]) == "--empty_confidence")
				empty_confidence = true;
			else if (string(argv[i]) == "--psimap")
				params.solver.precalculate_psi = true;
			else if (string(argv[i]) == "--fixed_iters")
				params.special.dynamic_outerIterations = false;
			else if (string(argv[i]) == "--precalculate")
			{
				params.solver.precalculate_psi = true;
				params.solver.precalculate_derivatives = true;
			}
			else if (string(argv[i]) == "--fixed_z")
			{
				params.constraint.zeroDirichletBoundary[4] = 1;
				params.constraint.zeroDirichletBoundary[5] = 1;
			}
			else if (string(argv[i]) == "--fixed_xyz")
			{
				params.constraint.zeroDirichletBoundary[0] = 1;
				params.constraint.zeroDirichletBoundary[1] = 1;
				params.constraint.zeroDirichletBoundary[2] = 1;
				params.constraint.zeroDirichletBoundary[3] = 1;
				params.constraint.zeroDirichletBoundary[4] = 1;
				params.constraint.zeroDirichletBoundary[5] = 1;
			}
			else if (string(argv[i]) == "--export_warped" || string(argv[i]) == "--export_warp")
				export_warped = true;
			else if (string(argv[i]) == "--skip_vectors")
				skip_vector_export = true;
			else if (string(argv[i]) == "--mosaic")
				params.mosaicing.mosaic_decomposition = true;
			else if (string(argv[i]) == "-mosaic")
			{
				params.mosaicing.mosaic_decomposition = true;
				i++;
				params.mosaicing.max_nstack = atoi(argv[i]);
				i++;
				params.mosaicing.overlap = atoi(argv[i]);
			}
			else if (string(argv[i]) == "--mosaic_approximation"){
				params.mosaicing.mosaic_decomposition = true;
				params.mosaicing.sequential_approximation = true;
			}
			else if (string(argv[i]) == "-overlap")
			{
				i++;
				params.mosaicing.overlap = atoi(argv[i]);
			}
			else if (string(argv[i]) == "-cut_dim" || string(argv[i]) == "-preferential_cut")
			{
				i++;params.mosaicing.preferential_cut_dimension = atoi(argv[i]);
			}
			else if (string(argv[i]) == "--reorder" || string(argv[i]) == "--transpose")
				params.mosaicing.reorder_axis = true;
			else if (string(argv[i]) == "--warp_mask")
				warp_mask = true;
			else if (string(argv[i]) == "-binning")
			{
				i++;
				params.special.binning = atof(argv[i]);
			}
			else if (string(argv[i]) == "--binning")
				params.special.binning = 2;
			else if (string(argv[i]) == "--binned_output")
				params.special.binned_output = true;
			else if (string(argv[i]) == "--sqrt_equalize")
				params.preprocessing.sqrt_equalization = true;
			else if (string(argv[i]) == "-transform1")
			{i++; params.preprocessing.intensity_transform1 = string(argv[i]);}
			else if (string(argv[i]) == "-transform2")
			{i++;params.preprocessing.intensity_transform2 = string(argv[i]);}
			else if (string(argv[i]) == "-convergence")
			{
				i++;
				params.special.doi_convergence = atof(argv[i]);
			}
			else if (string(argv[i]) == "--extrapolate")
				params.preprocessing.extrapolate_intensities = true;
			else if (string(argv[i]) == "--export_error")
			     export_error_image = true;
			else if (string(argv[i]) == "-zkill_confidence")
			{
				//for SMA-wires that touch top and bottom slice
				i++; params.confidence.zrange_killconfidence[0] = atoi(argv[i]);
				i++; params.confidence.zrange_killconfidence[1] = atoi(argv[i]);
			}
		}

		if (params.special.localglobal_dataterm) params.solver.precalculate_psi = true;

		if(params.pyramid.scaling_factor > 0.98 || params.pyramid.scaling_factor < 0.4)
		{
			std::cout << "Warning! The scaling factor for the Pyramid should be between 0.4 and 0.98. Setting scaling_ratio to 0.75." << std::endl;
			params.pyramid.scaling_factor = 0.75;
		}
		if(params.solver.precalculate_psi == false && params.solver.innerIterations != 1)
			std::cout << "\033[1;31mWarning! The amount of inner iterations needs to be 1 when the data term (psi) is not precalculated.\033[0m" << std::endl;
	}

	if (n_threads > 0) omp_set_num_threads(min(n_threads, omp_get_max_threads()));
	auto time0 = chrono::high_resolution_clock::now();

	std::pair<double,double> histogram_correlation = {0.0, 0.0}; //used to track the success of histomatching option

	//Read data, normalize and preprocess
	///////////////////////////////////////////////////////////////////////////////////
	cout << "--------------------------------------------------" << endl;
	cout << "Frame1: " << inpath1 << endl;
	cout << "Preprocessing:" << endl;
	std::cout << "----------------------------" << std::endl;

	hdcom::HdCommunication hdcom;
	std::string active_path = aux::get_active_directory();

	//test if provided pathes are absolute or relative
	////////////////////////
	if (!hdcom.is_absolute_path(inpath0) && hdcom.path_exists(active_path+"//"+inpath0)) inpath0 = active_path+"//"+inpath0;
	if (!hdcom.is_absolute_path(inpath1) && hdcom.path_exists(active_path+"//"+inpath1)) inpath1 = active_path+"//"+inpath1;
	if (inpath_mask0 != "none" && !hdcom.is_absolute_path(inpath_mask0) && hdcom.path_exists(active_path+"//"+inpath_mask0)) inpath_mask0 = active_path+"//"+inpath_mask0;
	
	std::string rootpath = inpath0.substr(0, inpath0.rfind("/", inpath0.length()-2)+1);
	if (outpath.length() == 0) outpath = rootpath + "/optflow/";
	else if (!hdcom.is_absolute_path(outpath)) outpath = active_path+"//"+outpath;

	bool is_rgb = false;
	std::vector<string> filelist0 = hdcom.GetFilelist_And_ImageSequenceDimensions(inpath0, shape, is_rgb);
	std::vector<string> filelist1 = hdcom.GetFilelist_And_ImageSequenceDimensions(inpath1, shape, is_rgb);

	if (filelist0[0] == "missing")
	{
		cout << "\033[1;31mError! Directory of Frame0 not found! Please provide a valid input with the -i0 argument.\033[0m" << endl;
		cout << "inpath0: " << inpath0 << endl;
		return -1;
	}
	else if (filelist1[0] == "missing")
	{
		cout << "\033[1;31mError! Directory of Frame1 not found! Please provide a valid input with the -i1 argument.\033[0m" << endl;
		cout << "inpath1: " << inpath1 << endl;
		return -1;
	}
	else if (filelist0[0] == "no tif")
	{
		cout << "\033[1;31mError! No tif-file in directory of Frame0! Please provide a valid input with the -i0 argument.\033[0m" << endl;
		cout << "inpath0: " << inpath0 << endl;
		return -1;
	}
	else if (filelist1[0] == "no tif")
	{
		cout << "\033[1;31mError! No tif-file in directory of Frame1! Please provide a valid input with the -i1 argument.\033[0m" << endl;
		cout << "inpath1: " << inpath1 << endl;
		return -1;
	}
	else if (is_rgb)
	{
		cout << "\033[1;31mError! RGB images are currently not supported!\033[0m" << endl;
		return -1;
	}
	////////////////////////

	//read greyscale images from HDD
	frame0 = hdcom.GetTif_unknowndim_32bit(inpath0, shape, zrange, true);
	frame1 = hdcom.GetTif_unknowndim_32bit(inpath1, shape, zrange, true);

	int64_t nslice = shape[0]*shape[1];
	int64_t nstack = shape[2]*nslice;

	//handling large strains by prestraining the reference
	float* prewarp_vector;
	if (sparse_translation_search)
	{
		//We detect corner points and perform a brute force translation matching (to be changed at least for gradient ascent).
		//The expected overall strain is provided for the wires to initialize close to the minimum.
		//Corner points are then subjected to nearest neighbour interpolation and Gaussian filtering and flipped
		//which is then used as a prewarp vector on frame 0;

		float sigma_sparse = 10.;
		int n_nearest = 10;
		int window_shape[3] = {7,7,7};

		lk::NobleCornerDetector corners;
		float* support_image = corners.detectcorners(frame0, shape, 0.5);
		std::vector<std::vector<int>> support_points = corners.corners2coordinatelist(support_image, shape);
		hdcom.SaveTifSequence_32bit(support_image, shape, outpath+"/sparse_translation/support/", "support", true);

		corrwindow::NaiveOptimizer initial_opt(frame0, frame1, shape, window_shape, params.gpu.deviceID);
		std::vector<std::vector<float>> sparse_guess = initial_opt.run_integertranslation_prestrained_cpu(support_points, prescribed_zstrain, frame0, frame1);
		initial_opt.free_device();

		prewarp_vector = guess_interpolate::sparseresult2image(sparse_guess,support_points,n_nearest,sigma_sparse,shape);
		hdcom.SaveTifSequence_32bit(prewarp_vector, shape, outpath+"/sparse_translation/dx/", "ux", true);
		hdcom.SaveTifSequence_32bit(&prewarp_vector[nstack], shape, outpath+"/sparse_translation/dy/", "uy", true);
		hdcom.SaveTifSequence_32bit(&prewarp_vector[2*nstack], shape, outpath+"/sparse_translation/dz/", "uz", true);

		#pragma omp parallel for
		for (long long int idx = 0; idx < nstack; idx++)
		{
			prewarp_vector[idx] = 0.0;
			prewarp_vector[idx+nstack] = 0.0;
			prewarp_vector[idx+2*nstack] = -prewarp_vector[idx+2*nstack];
		}

		frame0 = warp::warpFrame1_xyz(frame1, frame0, prewarp_vector, shape, &params);
		hdcom.SaveTifSequence_32bit(frame0, shape, outpath+"/warped_reference/", "warped_ref", false);
	}
	else if (prescribed_zstrain != 0.0 && use_initial_guess == false && prewarp_frame0 == true)
	{
		prewarp_vector = (float*) calloc(3*nstack,sizeof(*prewarp_vector));
		float applied_strain[3] = {relative_strain_origin,0.0f,-prescribed_zstrain};
		if (prescribed_zstrain != -99)
		{
			//warp the reference to the expected strain
			aux::set_initialguess(prewarp_vector, shape, shape, applied_strain, "zstrain");
		}
		else
		{
			//we use the previous result to approximate the z-displacement
			float* prevres_dz = hdcom.GetTif_unknowndim_32bit(previous_result_path+"/dz/", shape, zrange, true);
			#pragma omp parallel for
			for (long long int idx = 0; idx < nstack; idx++)
				prewarp_vector[2*nstack+idx] = -prevres_dz[idx];
			free(prevres_dz);
		}

		frame0 = warp::warpFrame1_xyz(frame1, frame0, prewarp_vector, shape, &params);
		hdcom.SaveTifSequence_32bit(frame0, shape, outpath+"/warped_reference/", "warped_ref", false);
		//free(prewarp_vector);
	}

	if (params.special.binning > 1)
	{
		int newshape[3] = {max(1,(int) roundf(shape[0]/params.special.binning)), max(1,(int) roundf(shape[1]/params.special.binning)), max(1,(int) roundf(shape[2]/params.special.binning))};
		float* tmp = (float*) malloc((((long long int) newshape[0]*newshape[1])*newshape[2])*sizeof(*tmp));
		resample::linear_coons(frame0, shape, tmp, newshape); swap(tmp, frame0); free(tmp);
		tmp = (float*) malloc((((long long int) newshape[0]*newshape[1])*newshape[2])*sizeof(*tmp));
		resample::linear_coons(frame1, shape, tmp, newshape); swap(tmp, frame1); free(tmp);
		params.special.old_shape[0] = shape[0]; params.special.old_shape[1] = shape[1]; params.special.old_shape[2] = shape[2];
		shape[0] = newshape[0]; shape[1] = newshape[1]; shape[2] = newshape[2];
		initial_guess[0] /= params.special.binning; initial_guess[1] /= params.special.binning; initial_guess[2] /= params.special.binning;
	}

	if (inpath_mask0 != "none"){
		//Having a segmented background is beneficial (data term not applied)
		params.confidence.background_mask = true;
		background_mask = hdcom.GetTif_unknowndim_32bit(inpath_mask0, shape, zrange, true);

		if (params.special.binning > 1)
		{
			int newshape[3] = {max(1,(int) roundf(shape[0]/params.special.binning)), max(1,(int) roundf(shape[1]/params.special.binning)), max(1,(int) roundf(shape[2]/params.special.binning))};
			float* tmp = (float*) malloc((((long long int) newshape[0]*newshape[1])*newshape[2])*sizeof(*tmp));
			if(params.special.binning == 2)
				tmp = resample::downsample_majority_bin2(background_mask, shape);
			else{
				resample::linear_coons(background_mask, shape, tmp, newshape);
				#pragma omp parallel for
				for (long long int idx = 0; idx < (((long long int) newshape[0]*newshape[1])*newshape[2]); idx++)
					tmp[idx] = round(tmp[idx]);
			}
			swap(tmp, background_mask); free(tmp);
			//tmp = resample::linear_coons(background_mask, shape, tmp, newshape); swap(tmp, background_mask); free(tmp);
			shape[0] = newshape[0]; shape[1] = newshape[1]; shape[2] = newshape[2];
		}

		if(0==1){
			//test mask dilatation
			for (int iter = 0; iter < 2; iter++)
			{
				long long int nslice = shape[0]*shape[1];
				long long int nstack = shape[2]*nslice;
				float* newmask = (float*) calloc(nstack,sizeof(*newmask));

				#pragma omp parallel for
				for (long long int idx = 0; idx < nstack; idx++)
				{
					int z = idx/nslice;
					int y = (idx-z*nslice)/shape[0];
					int x = (idx-z*nslice-y*shape[0]);

					float val = background_mask[idx];

					for(int i = 0; i < 27; i++)
					{
						int r = i/9;
						int q = (i-r*9)/3;
						int p = i-r*9-q*3;
						r -=1; q-= 1; p -= 1;
						if(z+r >= 0 && z+r < shape[2] && y+q >= 0 && y+q < shape[1] && x+p >= 0 && x+p < shape[0])
							val = std::max(val, background_mask[idx+r*nslice+q*shape[0]+p]);
					}

					newmask[idx] = val;
				}

				std::swap(background_mask, newmask); free(newmask);
			}
		}

		if (warp_mask)
		{
			nslice = shape[0]*shape[1];
			nstack = shape[2]*nslice;

			//round and scale to max 1
			#pragma omp parallel for
			for (long long int pos = 0; pos < nstack; pos++)
				background_mask[pos] = roundf(min(1.0f,background_mask[pos]));
		}
	}

	int ndims = 3; if (shape[2] == 1) ndims = 2;
	nslice = shape[0]*shape[1];
	nstack = shape[2]*nslice;

	if (params.special.evaluation_mode != "backward") result = (float*) calloc(ndims*nstack,sizeof(*result));
	else result = (float*) calloc(0,sizeof(*result));
	if (params.special.evaluation_mode != "forward") result2 = (float*) calloc(ndims*nstack,sizeof(*result2));
	else result2 = (float*) calloc(0,sizeof(*result2));

	if (params.preprocessing.intensity_transform1 != "none")
	{
		aux::transform_values(params.preprocessing.intensity_transform1, frame0, shape);
		aux::transform_values(params.preprocessing.intensity_transform1, frame1, shape);
	}

	if (params.preprocessing.normalization != "none"){
		std::cout << "normalizing...          \r";
		std::cout.flush();

		int n_histobins = 1000;
		int n_bins_equalization = 65536;
		double histo_cutoff = 0.001;
		string normalization_backup = params.preprocessing.normalization;

		if (params.preprocessing.normalization.find("linear") != string::npos)
		{
			//scale to 0 and 1 first
			//then apply linear transform to match histograms
			//////////////////////////////////////////////////
			bool extrapolate_intensities = true;
			string normalization = "height";

			histomatch::HistogramOptimization histoopt;

			aux::normalize2frames_histogram(frame0, frame1, shape, &params, n_histobins, histo_cutoff, params.preprocessing.ignore_zero, extrapolate_intensities, params.preprocessing.rescale_zero);
			int n_pairs = histoopt.map_pairs_of_extrema(frame0,frame1,shape, normalization);
			if (n_pairs > 0)
			{
				histogram_correlation = histoopt.SelectLinearRegressionSubset(frame0,frame1,shape,params.preprocessing.rescale_zero);
				//histogram_correlation = histoopt.RegressMappedExtremaLinearLeastSquares(frame0, frame1, shape, params.preprocessing.rescale_zero);

			}
			std::cout << "normalizing..." << n_pairs << " pairs...r: " << histogram_correlation.first << " -> " << histogram_correlation.second << std::endl;

			hdcom.makedir(outpath);
			histoopt.export_csv(outpath);
			//////////////////////////////////////////////////

			if (normalization_backup.find("histogram") != string::npos) params.preprocessing.normalization = "histogram";
			else if (normalization_backup.find("equalized") != string::npos) params.preprocessing.normalization = "equalized";

			if (normalization_backup.find("mask") != string::npos) params.preprocessing.normalization += "_mask";
		}

			 if (params.preprocessing.normalization == "simple")                aux::normalize2frames_simple(frame0, frame1, shape, &params);
		else if (params.preprocessing.normalization == "histogram")             aux::normalize2frames_histogram(frame0, frame1, shape, &params, n_histobins, histo_cutoff, params.preprocessing.ignore_zero, params.preprocessing.extrapolate_intensities, params.preprocessing.rescale_zero);
		else if (params.preprocessing.normalization == "histogram_independent") aux::normalize2frames_histogram_independent(frame0, frame1, shape, &params, n_histobins, histo_cutoff, params.preprocessing.ignore_zero, params.preprocessing.extrapolate_intensities, params.preprocessing.rescale_zero);
		else if (params.preprocessing.normalization == "equalized")             aux::normalize2frames_histogramequalized(frame0, frame1, shape, &params, n_histobins, n_bins_equalization, histo_cutoff, params.preprocessing.ignore_zero);
		else if (params.preprocessing.normalization == "equalized_independent")
		{
			aux::normalize1frame_histogramequalized(frame0, shape, &params, n_histobins, n_bins_equalization, histo_cutoff, params.preprocessing.ignore_zero);
			aux::normalize1frame_histogramequalized(frame1, shape, &params, n_histobins, n_bins_equalization, histo_cutoff, params.preprocessing.ignore_zero);
		}
		else if (params.preprocessing.normalization == "equalized_mask" || params.preprocessing.normalization == "histogram_mask")
		{
			if (params.preprocessing.normalization == "equalized_mask") aux::normalize2frames_histogramequalized_mask(frame0, frame1, background_mask, shape, &params, n_histobins, n_bins_equalization, histo_cutoff);
			else if (params.preprocessing.normalization == "histogram_mask") aux::normalize2frames_histogram_mask(frame0, frame1, background_mask, shape, &params, n_histobins, histo_cutoff, params.preprocessing.extrapolate_intensities, params.preprocessing.rescale_zero);
		}
		else if (params.preprocessing.normalization == "equalized_independent_mask" || params.preprocessing.normalization == "histogram_independent_mask")
		{
			if (params.preprocessing.normalization == "equalized_independent_mask"){
				aux::normalize1frame_histogramequalized(frame0, shape, &params, n_histobins, n_bins_equalization, histo_cutoff, params.preprocessing.ignore_zero);
				aux::normalize1frame_histogramequalized(frame1, shape, &params, n_histobins, n_bins_equalization, histo_cutoff, params.preprocessing.ignore_zero);
			}
			else if (params.preprocessing.normalization == "histogram_independent_mask")
				aux::normalize2frames_histogram_independent(frame0, frame1, shape, &params, n_histobins, histo_cutoff, params.preprocessing.ignore_zero);

			if (params.preprocessing.normalization == "equalized_independent_mask") aux::normalize2frames_histogramequalized_mask(frame0, frame1, background_mask, shape, &params, n_histobins, n_bins_equalization, histo_cutoff);
			else if (params.preprocessing.normalization == "histogram_independent_mask") aux::normalize2frames_histogram_mask(frame0, frame1, background_mask, shape, &params, n_histobins, histo_cutoff);
		}
		else if (params.preprocessing.normalization == "gradient_magnitude")
		{
			float* temp0 = derive::firstDerivativeMagnitude_fourthOrder(frame0, shape);
			float* temp1 = derive::firstDerivativeMagnitude_fourthOrder(frame1, shape);

			swap(frame0, temp0); swap(frame1, temp1);
			free(temp0); free(temp1);
		}

		else std::cout << "Warning! Unknown normalization mode!" << std::endl;

		params.preprocessing.normalization = normalization_backup;

		if (params.preprocessing.intensity_transform2 != "none")
		{
			aux::transform_values(params.preprocessing.intensity_transform2, frame0, shape);
			aux::transform_values(params.preprocessing.intensity_transform2, frame1, shape);
		}
	}
	else {
		//adjust alpha to intensity range
			 if(hdcom.last_bps == 8) {cout << "adjusting alpha for 8bit!" << endl; params.alpha *= 255;}
		else if(hdcom.last_bps == 16) {cout << "adjusting alpha for 16bit!" << endl; params.alpha *= 65535;}
	}

	//We may want to check the quality after normalization and before smoothing, i.e., we either spare time or memory
	//Memory is plenty. Thus, we just do a backup.
	if(check_from_backup)
	{
		frame0_backup = aux::backup_imagestack(frame0, shape);
		frame1_backup = aux::backup_imagestack(frame1, shape);
		shape_backup[0] = shape[0]; shape_backup[1] = shape[1]; shape_backup[2] = shape[2];
	}

	std::cout << "applying prefilter...           \r"; std::cout.flush();
	filter::apply_3DImageFilter_2frame(frame0, frame1, shape, params.preprocessing.prefilter_sigma, params.preprocessing.prefilter);
	///////////////////////////////////////////////////////////////////////////////////

	//Prepare confidence map
	///////////////////////////////////////////////////////////////////////////////////
	if(params.confidence.use_confidencemap)
	{
		//Activate a confidence run with --confidence
		if (shape[2] == 1) confidencemap = hdcom.GetTif_unknowndim_32bit(outpath+"/confidence.tif", shape, zrange, true);
		else confidencemap = hdcom.GetTif_unknowndim_32bit(outpath+"/confidence/", shape, zrange, true);

		if (params.special.binning > 1)
		{
			int newshape[3] = {max(1,(int) roundf(shape[0]/params.special.binning)), max(1,(int) roundf(shape[1]/params.special.binning)), max(1,(int) roundf(shape[2]/params.special.binning))};
			float* tmp = (float*) malloc((((long long int) newshape[0]*newshape[1])*newshape[2])*sizeof(*tmp));
			resample::linear_coons(confidencemap, shape, tmp, newshape); swap(tmp, confidencemap); free(tmp);
			shape[0] = newshape[0]; shape[1] = newshape[1]; shape[2] = newshape[2];
		}
	}
	else if (empty_confidence)
	{
		confidencemap = (float*) malloc(nstack*sizeof(*confidencemap));
		params.confidence.use_confidencemap = true;

		#pragma omp parallel for
		for(uint64_t pos = 0; pos < nstack; pos++) confidencemap[pos] = 1.0f;
	}
	if (params.confidence.background_mask && params.confidence.use_confidencemap)
	{
		//merge confidencemap and background mask
		#pragma omp parallel for
		for(uint64_t pos = 0; pos < nstack; pos++) confidencemap[pos] *= std::max(0.0f, std::min(1.f, background_mask[pos]));
	}
	else if (params.confidence.background_mask)
	{
		//background mask is binary confidence map (deep copy for reapplication)
		confidencemap = (float*) malloc(nstack*sizeof(*confidencemap));
		params.confidence.use_confidencemap = true;

		#pragma omp parallel for
		for(uint64_t pos = 0; pos < nstack; pos++) confidencemap[pos] = std::max(0.0f, std::min(1.f, background_mask[pos]));
	}
	if (params.confidence.gradient_mask)
	{
		if(!params.confidence.background_mask && !params.confidence.use_confidencemap){
			params.confidence.use_confidencemap = true;
			confidencemap = (float*) malloc(nstack*sizeof(*confidencemap));
			#pragma omp parallel for
			for (long long int pos = 0; pos < nstack; pos++) confidencemap[pos] = 1.0f;
		}

		if (params.confidence.advancedgradient == "gradientweighted")
			derive::add_gradientweightedconfidence(confidencemap,frame0, shape, params.confidence.sigma_blur_gradient);
		if (params.confidence.advancedgradient == "intensitygradientweighted")
			derive::add_intensity_and_gradientweightedconfidence(confidencemap,frame0, shape, params.confidence.sigma_blur_gradient);
		else
		derive::add_gradientmask(confidencemap,frame0, frame1, shape, params.confidence.used_percentage_gradient, params.confidence.sigma_blur_gradient);
	}
	if (recomask_diameter >= 0.0f)
	{
		if (recomask_diameter == 0.0f) recomask_diameter = min(shape[0], shape[1]);

		float sqradius = recomask_diameter*recomask_diameter*0.25f;
		float rc_x = shape[0]/2.f-0.5f;
		float rc_y = shape[1]/2.f-0.5f;

		if (!params.confidence.use_confidencemap){
			params.confidence.use_confidencemap = true;
			confidencemap = (float*) malloc(nstack*sizeof(*confidencemap));
			#pragma omp parallel for
			for (long long int pos = 0; pos < nstack; pos++) confidencemap[pos] = 1.0f;
		}

		#pragma omp parallel for
		for (long long int pos = 0; pos < nstack; pos++){
			int z = pos/nslice;
			int y = (pos-z*nslice)/shape[0];
			int x = pos-z*nslice-y*shape[0];

			if ((x-rc_x)*(x-rc_x)+(y-rc_y)*(y-rc_y) >= sqradius)
				confidencemap[pos] = 0.0f;
		}
	}
	if (params.confidence.zrange_killconfidence[0] != -1 || params.confidence.zrange_killconfidence[1] != -1)
	{
		if (!params.confidence.use_confidencemap){
			params.confidence.use_confidencemap = true;
			confidencemap = (float*) malloc(nstack*sizeof(*confidencemap));
			#pragma omp parallel for
			for (long long int pos = 0; pos < nstack; pos++) confidencemap[pos] = 1.0f;
		}

		#pragma omp parallel for
		for (long long int pos = 0; pos < nstack; pos++){
			int z = pos/nslice;

			if (z < params.confidence.zrange_killconfidence[0] || (params.confidence.zrange_killconfidence[1] != -1 && z >= params.confidence.zrange_killconfidence[1]))
				confidencemap[pos] = 0.0f;
		}
	}

	if (params.confidence.export_mask)
	{
		if (shape[2] == 1) hdcom.SaveTif_unknowndim_32bit(confidencemap, shape, outpath, "confidence");
		else hdcom.SaveTif_unknowndim_32bit(confidencemap, shape, outpath+"/confidence/", "confidence");
	}

	std::cout << "----------------------------" << std::endl;
	///////////////////////////////////////////////////////////////////////////////////

	//Reorder axis for easier copying
	///////////////////////////////////////////////////////////////////////////////////
	int new_axis_order[3] = {0,1,2};
	if (params.mosaicing.reorder_axis && (params.mosaicing.mosaic_decomposition || params.gpu.n_gpus > 1))
	{
		//provide a preferential split dimension for efficient cuts

		//better to have the longest axis as last axis
		mosaic::reorder_axis_bylength(frame0, shape, new_axis_order, params.mosaicing.preferential_cut_dimension);
		mosaic::reorder_axis_bylength(frame1, shape, new_axis_order, params.mosaicing.preferential_cut_dimension);
		if(params.confidence.use_confidencemap) mosaic::reorder_axis_bylength(confidencemap, shape, new_axis_order, params.mosaicing.preferential_cut_dimension);
		if(inpath_mask0 != "none") mosaic::reorder_axis_bylength(background_mask, shape, new_axis_order, params.mosaicing.preferential_cut_dimension);

		if(params.mosaicing.preferential_cut_dimension != -1) params.mosaicing.preferential_cut_dimension = min(shape[2], 2); //should now be the last dimension

		int new_shape[3] = {shape[new_axis_order[0]], shape[new_axis_order[1]], shape[new_axis_order[2]]};
		shape[0] = new_shape[0]; shape[1] = new_shape[1]; shape[2] = new_shape[2];
	}
	///////////////////////////////////////////////////////////////////////////////////

	//Select a solver
	///////////////////////////////////////////////////////////////////////////////////
	std::cout << "GPU configuration:" << std::endl;
	std::cout << "----------------------------" << std::endl;
	optflow::OptFlowSolver        *optflow_solver;
	optflow::OptFlow_CPU2D         cpu2d_solver;
	optflow::OptFlow_CPU3D         cpu3d_solver;
	optflow::OptFlow_GPU2D         gpu2d_solver;
	optflow::OptFlow_GPU2D_Reshape gpu2d_rs_solver;
	optflow::OptFlow_GPU3D         gpu3d_solver;
	optflow::OptFlow_GPU3D_Reshape gpu3d_rs_solver;

	     if (ndims == 2 && params.gpu.n_gpus == 0)                                   optflow_solver = &cpu2d_solver;
	else if (ndims == 2 && params.gpu.n_gpus == 1 && params.gpu.reshape4coalescence) optflow_solver = &gpu2d_rs_solver;
	else if (ndims == 2 && params.gpu.n_gpus == 1)                                   optflow_solver = &gpu2d_solver;
	else if (ndims == 3 && params.gpu.n_gpus == 0)                                   optflow_solver = &cpu3d_solver;
	else if (ndims == 3 && params.gpu.n_gpus == 1 && params.gpu.reshape4coalescence) optflow_solver = &gpu3d_rs_solver;
	else if (ndims == 3 && params.gpu.n_gpus == 1)									 optflow_solver = &gpu3d_solver;
	else {cout << "Error! No matching solver available!" << endl; return -1;}

	if (params.gpu.n_gpus == 0) cout << "Warning! CPU mode is probably horribly outdated." << endl;

	//activate for configuration
	if (params.special.evaluation_mode == "forward-backward-confidence-apply") params.confidence.use_confidencemap = true;

	if (params.mosaicing.mosaic_decomposition && params.mosaicing.max_nstack != -1)
	{
		//set maximal patch volume manually
		int patch_shape[3];
		vector<vector<int>> patches = mosaic::get_mosaic_coordinates(shape, &params, patch_shape);
		params.mosaicing.max_nstack = ((long long int) (patch_shape[0]*patch_shape[1]))*patch_shape[2]; //in case a higher pyramid level could be solved with less cuts

		if (patches.size() == 1) params.mosaicing.mosaic_decomposition = false;

		int error_id = optflow_solver->configure_device(patch_shape, &params);
		if (error_id != 0) return error_id;
	}
	else
	{
		int error_id = optflow_solver->configure_device(shape, &params);
		if (error_id != 0) return error_id;
	}

	//this run first runs without a confidence map
	if (params.special.evaluation_mode == "forward-backward-confidence-apply") params.confidence.use_confidencemap = false;
	///////////////////////////////////////////////////////////////////////////////////

	///////////////////////////////////////////////////////////////////////////////////
	if("console output"){
		if (params.solver.precalculate_psi) cout << "psimap: active" << endl;
		else cout << "psimap: inactive" << endl;
		if (params.confidence.use_confidencemap) cout << "confidencemap: active" << endl;
		else cout << "confidencemap: inactive" << endl;
		if (params.warp.rewarp_frame1 && params.mosaicing.mosaic_decomposition && params.mosaicing.sequential_approximation == false && params.gpu.n_gpus == 1) cout << "rewarp: irrelevant" << endl;
		else if (params.warp.rewarp_frame1) cout << "rewarp: active" << endl;
		else cout << "rewarp: inactive" << endl;
		cout << "derivatives: " << params.solver.flowDerivative_type << ", " << params.solver.spatiotemporalDerivative_type << endl;
		if (params.smoothness.anisotropic_smoothness) cout << "smoothness: anisotropic";
		else cout << "smoothness: isotropic";
		if (params.smoothness.complementary_smoothness) cout << " complementary";
		if (params.smoothness.decoupled_smoothness) cout << " decoupled";
		if (params.smoothness.adaptive_smoothness) cout << " (edge adaptive)";
		else cout << " (non adaptive)";
		cout << endl;
	}

	//Initialize pyramids
	pyramid::ImagePyramid pyramid0(&params, shape, true);
	pyramid::ImagePyramid pyramid1(&params, shape, true);
	pyramid::ImagePyramid pyramid2(&params, shape, params.confidence.use_confidencemap); //pyramid for confidence map
	pyramid::ImagePyramid pyramid3(&params, shape, params.confidence.background_mask && params.special.dynamic_outerIterations); //if we only want to track convergence in the foreground

	//convergence criteria for outer iterations (if applicable)
	int max_outerIterations = params.solver.outerIterations;
	params.special.doI_maxOuterIter = params.solver.outerIterations;
	if (params.special.dynamic_outerIterations) params.solver.outerIterations = params.special.doI_stepsize;
	std::cout << "----------------------------" << std::endl;
	///////////////////////////////////////////////////////////////////////////////////

	//Walk through the Gaussian pyramid
	///////////////////////////////////////////////////////////////////////////////////
	///////////////////////////////////////////////////////////////////////////////////
	float solver_correlation = 0.0;
	for (int pylevel = params.pyramid.nLevels-1; pylevel >= 0; pylevel--)
	{
		//Debugging
		//pylevel = 0;

		//Probably has to be the inverse but seems useless at best for now:
		///////////////////////////////////////////////////////////////////
		if(params.scaling.use_gridscaling)
		{
			params.scaling.hx = (shape[0]/((float) pyramid0.pyramid_shapes[pylevel][0]));
			params.scaling.hy = (shape[1]/((float) pyramid0.pyramid_shapes[pylevel][1]));
			params.scaling.hz = (shape[2]/((float) pyramid0.pyramid_shapes[pylevel][2]));
		}
		///////////////////////////////////////////////////////////////////

		if(pylevel == 0 && params.preprocessing.prefilter2 != "none")
			filter::apply_3DImageFilter_2frame(frame0, frame1, shape, params.preprocessing.prefilter2_sigma, params.preprocessing.prefilter2);

		//resample frame0 and frame1 to current pyramid level
		pyramid0.resample_frame(frame0, shape, pylevel, &params, params.pyramid.interpolation_mode);
		pyramid1.resample_frame(frame1, shape, pylevel, &params, params.pyramid.interpolation_mode);

		int this_nx = pyramid0.pyramid_shapes[pylevel][0]; int this_ny = pyramid0.pyramid_shapes[pylevel][1]; int this_nz = pyramid0.pyramid_shapes[pylevel][2];
		uint64_t this_nslice = this_nx*this_ny;
		uint64_t this_nstack = this_nz*this_nslice;

		//Resampling of supporting maps
		if (params.smoothness.adaptive_smoothness)
			adaptivity_map = derive::calculate_edgeorientation(pyramid0.active_frame, pyramid0.pyramid_shapes[pylevel], params.smoothness.adaptivity_mode, params.smoothness.adaptivity_sigma);
		if (params.confidence.use_confidencemap)
			pyramid2.resample_frame(confidencemap, shape, pylevel, &params, "linear_filtered");
		if (params.special.dynamic_outerIterations && params.confidence.background_mask)
			pyramid3.resample_frame(background_mask, shape, pylevel, &params, "linear_filtered");

		//apply initial guess
		if (pylevel == params.pyramid.nLevels-1 && use_initial_guess)
		{
			aux::set_initialguess(result, pyramid0.pyramid_shapes[pylevel], shape, initial_guess, previous_result_path);
			//cout << "adding elongation" << endl;
			//aux::add_initial_ycompression(result, pyramid0.pyramid_shapes[pylevel], 0.045);
		}
		if (prescribed_zstrain != 0.0 && use_initial_guess && prewarp_frame0 == false)
		{
			initial_guess[2] = prescribed_zstrain;
			aux::set_initialguess(result, pyramid0.pyramid_shapes[pylevel], shape, initial_guess, "zstrain");
		}

		//Forward evaluation
		if (params.special.evaluation_mode.find((string) "forward") != string::npos)
		{
			if (params.special.localglobal_fading_sigma.size() > 0)
			{
				for (int fading = 0; fading < params.special.localglobal_fading_sigma.size(); fading++)
				{
					params.special.localglobal_sigma_data = params.special.localglobal_fading_sigma[fading];
					if(!params.mosaicing.mosaic_decomposition)
					{
						solver_correlation = protocol::run_default_evaluation(pylevel, optflow_solver, &params, pyramid0.active_frame, pyramid1.active_frame, result, pyramid0.pyramid_shapes[pylevel],
								pyramid3.active_frame, pyramid2.active_frame, adaptivity_map, pyramid0.pyramid_shapes[std::max(pylevel-1,0)]);
					}
					else if (!params.mosaicing.sequential_approximation)
					{
						solver_correlation = protocol::run_singleGPU_mosaic(pylevel, optflow_solver, &params, pyramid0.active_frame, pyramid1.active_frame, result, pyramid0.pyramid_shapes[pylevel],
								pyramid3.active_frame, pyramid2.active_frame, adaptivity_map, pyramid0.pyramid_shapes[std::max(pylevel-1,0)]);
					}
					else
					{
						solver_correlation = protocol::run_sequential_mosaic(pylevel, optflow_solver, &params, pyramid0.active_frame, pyramid1.active_frame, result, pyramid0.pyramid_shapes[pylevel],
								pyramid3.active_frame, pyramid2.active_frame, adaptivity_map, pyramid0.pyramid_shapes[std::max(pylevel-1,0)]);
					}
				}
			}
			else
			{
				if(!params.mosaicing.mosaic_decomposition)
				{
					solver_correlation = protocol::run_default_evaluation(pylevel, optflow_solver, &params, pyramid0.active_frame, pyramid1.active_frame, result, pyramid0.pyramid_shapes[pylevel],
							pyramid3.active_frame, pyramid2.active_frame, adaptivity_map, pyramid0.pyramid_shapes[std::max(pylevel-1,0)]);
				}
				else if (!params.mosaicing.sequential_approximation)
				{
					solver_correlation = protocol::run_singleGPU_mosaic(pylevel, optflow_solver, &params, pyramid0.active_frame, pyramid1.active_frame, result, pyramid0.pyramid_shapes[pylevel],
							pyramid3.active_frame, pyramid2.active_frame, adaptivity_map, pyramid0.pyramid_shapes[std::max(pylevel-1,0)]);
				}
				else
				{
					solver_correlation = protocol::run_sequential_mosaic(pylevel, optflow_solver, &params, pyramid0.active_frame, pyramid1.active_frame, result, pyramid0.pyramid_shapes[pylevel],
							pyramid3.active_frame, pyramid2.active_frame, adaptivity_map, pyramid0.pyramid_shapes[std::max(pylevel-1,0)]);
				}
			}
		}

		//Backward evaluation
		if (params.special.evaluation_mode.find((string) "backward") != string::npos)
		{
			if(!params.mosaicing.mosaic_decomposition)
			{
				solver_correlation = protocol::run_default_evaluation(pylevel, optflow_solver, &params, pyramid1.active_frame, pyramid0.active_frame, result2, pyramid0.pyramid_shapes[pylevel],
						pyramid3.active_frame, pyramid2.active_frame, adaptivity_map, pyramid0.pyramid_shapes[std::max(pylevel-1,0)]);
			}
			else if (!params.mosaicing.sequential_approximation)
			{
				solver_correlation = protocol::run_singleGPU_mosaic(pylevel, optflow_solver, &params, pyramid1.active_frame, pyramid0.active_frame, result2, pyramid0.pyramid_shapes[pylevel],
						pyramid3.active_frame, pyramid2.active_frame, adaptivity_map, pyramid0.pyramid_shapes[std::max(pylevel-1,0)]);
			}
			else
			{
				solver_correlation = protocol::run_sequential_mosaic(pylevel, optflow_solver, &params, pyramid1.active_frame, pyramid0.active_frame, result2, pyramid0.pyramid_shapes[pylevel],
						pyramid3.active_frame, pyramid2.active_frame, adaptivity_map, pyramid0.pyramid_shapes[std::max(pylevel-1,0)]);
			}
		}
	}
	///////////////////////////////////////////////////////////////////////////////////
	///////////////////////////////////////////////////////////////////////////////////

	//Reset the axis ordering
	///////////////////////////////////////////////////////////////////////////////////
	if (new_axis_order[0] != 0 || new_axis_order[1] == 1)
	{
		if (shape[2] > 1 && new_axis_order[2] != 2 && params.mosaicing.preferential_cut_dimension != -1) params.mosaicing.preferential_cut_dimension = new_axis_order[2];
		else if (shape[2] == 1 && new_axis_order[1] != 1  && params.mosaicing.preferential_cut_dimension != -1) params.mosaicing.preferential_cut_dimension = new_axis_order[1];

		mosaic::reorder_axis_byorder(frame0, shape, new_axis_order);
		mosaic::reorder_axis_byorder(frame1, shape, new_axis_order);
		if(params.confidence.use_confidencemap) mosaic::reorder_axis_byorder(confidencemap, shape, new_axis_order);
		if(inpath_mask0 != "none") mosaic::reorder_axis_byorder(background_mask, shape, new_axis_order);
		mosaic::reorder_vector_byorder(result, shape, new_axis_order, ndims);

		int new_shape[3] = {shape[new_axis_order[0]], shape[new_axis_order[1]], shape[new_axis_order[2]]};
		shape[0] = new_shape[0]; shape[1] = new_shape[1]; shape[2] = new_shape[2];
	}
	///////////////////////////////////////////////////////////////////////////////////

	//Perform operations specific to the protocol given by evaluation mode
	///////////////////////////////////////////////////////////////////////////////////
	protocol::run_evaluation_mode_postprocessing(optflow_solver, &params, frame0, frame1, result, result2, background_mask, confidencemap, adaptivity_map, shape, outpath);
	if (params.special.binning > 1)
	{
		if(params.special.binned_output)
		{
			hdcom.SaveTif_unknowndim_32bit(frame0, shape, outpath+"/frame0/", "frame0_");
			hdcom.SaveTif_unknowndim_32bit(frame1, shape, outpath+"/frame1/", "frame1_");
		}
		else
		{
			resample::upscalevector(result, shape, params.special.old_shape, ndims, params.scaling.upscaling_interpolation_mode);
			shape[0] = params.special.old_shape[0]; shape[1] = params.special.old_shape[1]; shape[2] = params.special.old_shape[2];
			nslice = shape[0]*shape[1]; nstack = shape[2]*nslice;
			if (inpath_mask0 != "none"){
				background_mask = hdcom.GetTif_unknowndim_32bit(inpath_mask0, shape, zrange, false);

				if (warp_mask)
				{
					#pragma omp parallel for
					for (long long int pos = 0; pos < nstack; pos++) background_mask[pos] = roundf(min(1.0f,background_mask[pos]));
				}
			}
		}
	}
	///////////////////////////////////////////////////////////////////////////////////

	///////////////////////////////////////////////////////////////////////////////////
	vector<float> pre_quality, post_quality, pre_quality_bone, post_quality_bone;
	double relative_bone_error = 0.0; double bone_mean = 0.0; double bone_count = 0.0;

	//change to a separate mask for analysis
	if (analysis_mask != "none")
	{
		if (inpath_mask0 != "none") free(background_mask);
		inpath_mask0 = analysis_mask;
		background_mask = hdcom.GetTif_unknowndim_32bit(inpath_mask0, shape, zrange, true);

		if (params.special.binning > 1 && params.special.binned_output)
		{
			int newshape[3] = {max(1,(int) roundf(shape[0]/params.special.binning)), max(1,(int) roundf(shape[1]/params.special.binning)), max(1,(int) roundf(shape[2]/params.special.binning))};
			float* tmp = (float*) malloc((((long long int) newshape[0]*newshape[1])*newshape[2])*sizeof(*tmp));
			resample::linear_coons(background_mask, shape, tmp, newshape); swap(tmp, background_mask); free(tmp);
			shape[0] = newshape[0]; shape[1] = newshape[1]; shape[2] = newshape[2];
		}
	}

	if (check_quality){
		std::cout << "----------------------------" << std::endl;

		//grab unmodified input
		if (params.special.binning == 1 || (params.special.binning > 1 && !params.special.binned_output))
		{
			if (params.special.binning == 1 && check_from_backup)
			{
				free(frame0); free(frame1);
				shape[0] = shape_backup[0]; shape[1] = shape_backup[1]; shape[2] = shape_backup[2];
				frame0 = aux::backup_imagestack(frame0_backup, shape);
				frame1 = aux::backup_imagestack(frame1_backup, shape);
			}
			else
			{
				frame0 = hdcom.GetTif_unknowndim_32bit(inpath0, shape, zrange, false);
				frame1 = hdcom.GetTif_unknowndim_32bit(inpath1, shape, zrange, false);
			}
		}
		long long int nslice = shape[0]*shape[1];
		long long int nstack = shape[2]*nslice;

		pre_quality = anal::get_qualitymeasures(frame0, frame1, result, shape, background_mask, params.confidence.background_mask);

		//warp to solution
		frame1 = warp::warpFrame1_xyz(frame0, frame1, result, shape, &params);

		if (export_warped)
		{
			if (shape[2] == 1) hdcom.Save2DTifImage_32bit(frame1, shape, outpath, "warped",0);
			else hdcom.SaveTifSequence_32bit(frame1, shape, outpath+"/warped/", "warped", false);
		}

		//get quality measures
		post_quality = anal::get_qualitymeasures(frame0, frame1, result, shape, background_mask, params.confidence.background_mask);

		if (export_error_image)
		{
			float* error_image = (float*) calloc(nstack, sizeof(*error_image));
			#pragma omp parallel for reduction(+: relative_bone_error)
			for (long long int idx = 0; idx < nstack; idx++)
			{
				if(!params.confidence.background_mask || background_mask[idx] != 0)
				{
					double val0 = frame0[idx];
					if(fabs(val0) > 1e-6)
					    error_image[idx] = fabs((frame1[idx]-val0));
				}
			}
			relative_bone_error /= bone_count;

			if (shape[2] == 1) hdcom.Save2DTifImage_32bit(error_image, shape, outpath, "error",0);
			else hdcom.SaveTifSequence_32bit(error_image, shape, outpath+"/abs_error/", "error", false);
		}

		std::cout << "rel_valid: " << post_quality[0] << std::endl;
		std::cout << "cross-corr: " << pre_quality[1] << " --> "  << post_quality[1] << std::endl;
		std::cout << "mean ssd: " << pre_quality[2] << " --> \033[1;36m"  << post_quality[2] << "\033[0m" << std::endl;
		std::cout << "----------------------------" << std::endl;

	}
	///////////////////////////////////////////////////////////////////////////////////

	//Kill background shift for visualization
	///////////////////////////////////////////////////////////////////////////////////
	if(inpath_mask0 != "none" && mask_output)
	{
		#pragma omp parallel for
		for (uint64_t pos = 0; pos < nstack; pos++)
		{
			if(background_mask[pos] == 0)
			{
				result[pos] = 0.0f;
				result[nstack+pos] = 0.0f;
				if(shape[2] > 1) result[2*nstack+pos] = 0.0f;
			}
		}
	}
	///////////////////////////////////////////////////////////////////////////////////

	//Output the result
	///////////////////////////////////////////////////////////////////////////////////
	if(!skip_vector_export)
	{
		if (prewarp_frame0)
		{
			#pragma omp parallel for
			for (long long int idx = 0; idx < min(shape[2]+1, 3)*nstack; idx++)
			{
				prewarp_vector[idx] = -prewarp_vector[idx];
				result[idx] += prewarp_vector[idx];
			}

			warp::warpVector1_xyz(prewarp_vector, result, prewarp_vector, shape, &params);
		}

		if (shape[2] == 1){
			hdcom.SaveTif_unknowndim_32bit(result,        shape, outpath, "ux");
			hdcom.SaveTif_unknowndim_32bit(result+nstack, shape, outpath, "uy");
		}
		else{
			hdcom.SaveTif_unknowndim_32bit(result, shape, outpath+"/dx/", "ux_");
			hdcom.SaveTif_unknowndim_32bit(result+nstack, shape, outpath+"/dy/", "uy_");
			hdcom.SaveTif_unknowndim_32bit(result+2*nstack, shape, outpath+"/dz/", "uz_");
		}
	}
	///////////////////////////////////////////////////////////////////////////////////

	//Warp mask for next timestep (if only the initial mask is available)
	///////////////////////////////////////////////////////////////////////////////////
	if (warp_mask && inpath_mask0 != "none")
	{
		float mask_cutoff = 0.1f;
		//background_mask = hdcom.GetTif_unknowndim_32bit(inpath_mask0, shape, zrange, true);
		background_mask = warp::warpMaskForward_xyz(background_mask, result, shape, mask_cutoff);

		if (shape[2] == 1) hdcom.SaveTif_unknowndim_32bit(background_mask, shape, outpath, "warped_mask");
		else hdcom.SaveTif_unknowndim_32bit(background_mask, shape, outpath+"/warped_mask/", "mask_");
	}
	///////////////////////////////////////////////////////////////////////////////////

	optflow_solver->free_device();

	auto time_final = chrono::high_resolution_clock::now();
	chrono::duration<double> elapsed_total = time_final-time0;
	std::cout << "execution took " << elapsed_total.count() << " s" << std::endl;
	cout << "--------------------------------------------------" << endl;

	if ("append logfile"){
		time_t now = time(0);
		ofstream logfile;
		logfile.open(outpath + "/logfile.txt", fstream::in | fstream::out | fstream::app);
		logfile << ctime(&now);
		logfile << "ran cudaWBBOptFlow_v0.2:\n";
		logfile << "-------------------------------------------------------------------------\n";
		logfile << "    - runtime: " << elapsed_total.count()/60. << " min\n";
		logfile << "    - frame0: " << inpath0 << "\n";
		logfile << "    - frame1: " << inpath1 << "\n";
		logfile << "    - mask: " << inpath_mask0 << "\n\n";
		logfile << "    - alpha: " << params.alpha << "\n";
		logfile << "    - levels: " << params.pyramid.nLevels << "\n";
		logfile << "    - scaling: " << params.pyramid.scaling_factor << "\n";
		logfile << "    - derivatives: " <<params.solver.flowDerivative_type << ", " << params.solver.spatiotemporalDerivative_type << "\n";
		if (params.smoothness.anisotropic_smoothness) logfile << "    - smoothness: anisotropic";
		else logfile << "    - smoothness: isotropic";
		if (params.smoothness.complementary_smoothness) logfile << " complementary";
		if (params.smoothness.decoupled_smoothness) logfile << " decoupled";
		if (params.smoothness.adaptive_smoothness) logfile << " (edge adaptive)";
		else logfile << " (non adaptive)";
		logfile << "\n\n";
		logfile << "    - normalization: " << params.preprocessing.normalization << "\n";
		logfile << "    - prefilter: " << params.preprocessing.prefilter << ", " << params.preprocessing.prefilter_sigma << "\n";
		if (params.mosaicing.mosaic_decomposition && params.mosaicing.sequential_approximation)
			logfile << "    - mosaic: approximation (overlap: " << params.mosaicing.overlap << ")\n";
		else if (params.mosaicing.mosaic_decomposition)
			logfile << "    - mosaic: true (overlap: " << params.mosaicing.overlap << ")\n";
		else
			logfile << "    - mosaic: false\n";
		if (params.postprocessing.median_filter)
			logfile << "    - median: " << params.postprocessing.median_radius << "\n";
		else
			logfile << "    - median: false\n";

		if (params.preprocessing.normalization.find("linear") != string::npos)
			logfile << "\n    histogram_correlation: " << histogram_correlation.first << " --> "  << histogram_correlation.second << "\n";
		if (solver_correlation != 0.0)
			logfile << "solver_correlation: " << solver_correlation << "\n";
		if (check_quality){
			logfile << "\n    cross-corr: " << pre_quality[1] << " --> "  << post_quality[1] << "\n";
			logfile << "          mssd: " << pre_quality[2] << " --> "  << post_quality[2] << "\n";
		}
		logfile << "\narguments:\n";
		for (uint16_t i = 1; i < argc; i++) logfile << argv[i] << " ";
		logfile << "\n";

		logfile << "-------------------------------------------------------------------------\n\n";
		logfile.close();
	}

	///////////////////////////////////////////////////////////////////////////////////
	///////////////////////////////////////////////////////////////////////////////////

	return 0;
}
