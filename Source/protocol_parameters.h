#ifndef PROTOCOL_PARAMETERS_H
#define PROTOCOL_PARAMETERS_H

#include <iostream>
#include <vector>
#include <string.h>

/*********************************************************************************************************************************************************
 * Location: Helmholtz-Zentrum fuer Material und Kuestenforschung, Max-Planck-Strasse 1, 21502 Geesthacht
 * Author: Stefan Bruns
 * Contact: bruns@nano.ku.dk
 *
 * License: TBA
 *
 * Preprocessing:
 * 		normalization: "none"
 * 		               "simple" := rescale to 0-1 interval (same scaling for both frames)
 * 		               "histogram": rescale 99.9% of intensities to 0-1 interval
 * 		               "histogram_independent": perform the histogram scaling (independently on the 2 frames)
 *
 * 		prefilter:     "gaussian"       := 3D Gaussian filtering with specified sigma on frame0 and frame1
 * 					   "median"         := 3D median with spheric kernel where prefilter_sigma is the floating point radius
 * 					   "median_simple"  := 3D median boxfilter where prefilter_sigma is the integer radius
 *
 *     flowDerivative_type:
 *                     - defines how to calculate the derivatives for the smoothness term
 *                     "Barron", "centraldifference", "forwarddifference" <- fully implemented
 *                     "LBM","Weickert","Leclaire_FIII","Leclaire_FIV","Scharr3","Sobel"  <- only implemented for optflow_gpu2d
 *********************************************************************************************************************************************************/

namespace optflow
{
	struct ProtocolParameters
	{
	public:
		///////////////////////////////////////////
		float alpha = 0.07f; //influence of the smoothness term (for 0-1 range)
		///////////////////////////////////////////

		///////////////////////////////////////////
		struct Preprocessing
		{
			std::string normalization = "histogram"; //"none","simple","histogram","histogram_independent", "equalized", "equalized_independent", "equalized_mask"
			//new modes:
			//"histogram_linear", "equalized_linear": applies a linear least square fits to maxima and minima of both histograms to match them
			bool ignore_zero = true; //ignoring zero values during normalization
			bool rescale_zero = false; //and don't rescale them
			bool sqrt_equalization = false; //less extreme equalization (like in ImageJ) by taking the sqrt of histogram values
			int smoothed_equalization = 0; //smooth the equalization with a mean filter when != 0

			//Intensity management
			bool extrapolate_intensities = false; //not implemented for equalization

			std::string intensity_transform1 = "none"; //"sqrt",etc. before normalization
			std::string intensity_transform2 = "none"; //"sqrt",etc. after normalization

			//Filter applied after IO:
			std::string prefilter = "gaussian"; //"none", "gaussian", "median","median_simple"
			float prefilter_sigma = 1.0f; // <- might want to go down to 0.5

			//Filter only applied at zero-level:
			std::string prefilter2 = "none"; //"none","gaussian","median","median_simple"
			float prefilter2_sigma = 0.5f;
		};
		struct Postprocessing
		{
			bool median_filter = false;
			float median_radius = 1.0f;
		};
		///////////////////////////////////////////

		///////////////////////////////////////////
		struct GaussianPyramid
		{
			int nLevels = 10; //<- depending on biggest movement (10 to 15 thus far)
			float scaling_factor = 0.9f; // <- 0.85 can be better sometimes
			bool alpha_scaling = false;
			int min_edge = 4;

			std::string scaling_mode = "custom"; //"Ershov","Liu","custom"
			std::string interpolation_mode = "cubic_unfiltered"; //"linear", "cubic", "linear_filtered", "cubic_filtered", "linear_unfiltered", "cubic_unfiltered", "linear_antialiasing", "cubic_antialiasing"
		};
		///////////////////////////////////////////

		///////////////////////////////////////////
		struct Mosaic
		{
			bool mosaic_decomposition = false; //allows calculating large volumes with limited GPU memory
			bool sequential_approximation = false; //with small objects we don't require communication between the mosaic patches
			bool alternating_directions = true; //for sequential approximation to avoid directional bias

			long long int max_nstack = -1; //maximal allowed amount of voxels in a patch, -1 for auto estimation from available GPU memory
			int memory_buffer = 256; //MB kept free on GPU when auto estimating

			int overlap = 100; //should probably be at least 2*(max_shift+iter_sor+1)
			bool protect_overlap = false; //warped patches may skew the result with voxels from out of bounds. This is avoided by extending the Dirichlet boundary.

			int preferential_cut_dimension = -1; //-1 = deactivated. Set a dimension for making cuts (if possible).
			bool reorder_axis = false; //transpose the axis to have the mosaic cuts in the last dimension (accelerates copying between patches)
		};
		///////////////////////////////////////////

		///////////////////////////////////////////
		struct IterativeSolver
		{
			float sor_omega = 1.8f;

			int outerIterations = 2000;//100; <-- just the maximum when using dynamic iterations
			int innerIterations = 1;
			int   sorIterations = 15; // <-- 10 to 20+ is seems acceptable

			std::string flowDerivative_type = "Farid5"; //"Barron","centraldifference","forwarddifference","Farid3","Farid5","Farid7","Farid9"
			std::string spatiotemporalDerivative_type = "Farid5"; //"HornSchunck", "centraldifference", "Barron"

			bool precalculate_psi = false; //Basically only needed for local-global approach. When false, requires innerIterations to be 1.
			bool precalculate_derivatives = false; //wasteful on memory
			float epsilon_psi = 0.001;
		};
		///////////////////////////////////////////

		///////////////////////////////////////////
		struct SmoothnessTerm
		{
			float epsilon_phi = 0.001;

			bool anisotropic_smoothness = true;
			bool decoupled_smoothness = false;//phi split into x,y,z
			bool adaptive_smoothness = false; //align smoothing with image edges
			bool complementary_smoothness = false; //following Zimmer2011("Optical Flow in Harmony") = strong smoothing along edges, constraint across them

			std::string adaptivity_mode = "gradient"; //mode for calculating edge orientation: "gradient" or "structure_tensor". The latter is buggy.
			float adaptivity_sigma = 2.f; //Gaussian blur applied in detecting edge orientation
		};
		///////////////////////////////////////////

		///////////////////////////////////////////
		struct Constraints
		{
			int zeroDirichletBoundary[6] = {0,0,0,0,0,0}; //1 to activate: north, south, left, right, top, bottom
			int fixedDirichletBoundary[6] = {0,0,0,0,0,0};
			float intensityRange[2] = {-1.e9, 1.e9}; //frame0 values outside the range are considered immobile
		};
		///////////////////////////////////////////

		///////////////////////////////////////////
		struct DataConfidence //conditions for killing the data term
		{
			bool export_mask = false; //save the generated mask for inspection

			bool use_confidencemap = false;//switch used for Ershov-style confidence map
			bool background_mask = false;//allows setting data term to zero in background
			bool gradient_mask = false; //fractures have a temporal gradient, immobile elements have a spatial gradient, bulk voxels with a poor data term have neither
			std::string advancedgradient = "none"; //"gradientweighted" or "intensitygradientweighted";

			//parameters for the gradient mask
			float used_percentage_gradient = 0.4f;
			float sigma_blur_gradient = 4.0f;

			//parameters for Ershov-style confidence map (not used)
			std::string confidence_mode = "gaussian"; //"threshold","Ershov","exponential","gaussian"
			float confidence_beta = 1.5f;
			float confidence_filter_cutoff = 0.9; //values with less confidence will be median filtered

			int slip_depth = 1; //kill dataterm in proximity of the domain boundary

			//For SMA wires we want to (try to) interpolate near the top and bottom slice or Ezz will be zero
			std::vector<int> zrange_killconfidence = {-1,-1};
		};
		///////////////////////////////////////////

		///////////////////////////////////////////
		struct Special
		{
			std::string evaluation_mode = "forward"; //"forward","backward","forward-backward","forward-backward-confidence"

			//when true we check the relative change in the flow field for convergence
			bool dynamic_outerIterations = true;
			int doI_stepsize = 10;
			int doI_maxOuterIter = 2000; //just for passing. Is set from outerIterations.
			float doi_convergence = 0.005f;
			float doi_convergence_level0 = 0.005f; //more rigorous convergence criterion on last level
			//applied before upscaling flow:
			bool medianfilter_flow = false;
			float flowfilter_radius = 1.5f;  //Sun, Roth, Black suggest a 2x2 box

			//
			bool localglobal_dataterm = false;
			std::string localglobal_mode = "Farid"; //How to interpolate the Dataterm-Tensor: with "Gaussian" or "Farid". Farid needs the radius as sigma,i.e. 1,2,3 or 4
			std::vector<float> localglobal_fading_sigma = {}; //providing a sequence of sigmas to be evaluated
			float localglobal_sigma_data = 1;//for Gauss: 0.5f;

			//Downsample the input and upsample the result
			float binning = 1;
			int old_shape[3] = {0,0,0}; //remember the shape before binning
			bool binned_output = false; //might as well keep the binned output for smaller files

			//verbose tracking
			bool track_correlation = true;
			
			//Memory limitation for Demo
			double memlimit_virtual = 10000000; //Limit GPU memory to MB provided here
		};
		///////////////////////////////////////////

		///////////////////////////////////////////
		struct Warping
		{
			bool rewarp_frame1 = false; //save some GPU memory by warping from previous warp instead of a fresh warp from frame1
			std::string outOfBounds_mode = "replace"; //"replace", "NaN"
			std::string interpolation_mode = "cubic"; //"linear", "cubic"
		};
		///////////////////////////////////////////

		///////////////////////////////////////////
		struct ScaleAdjustment
		{
			std::string upscaling_interpolation_mode = "cubic";//"linear","cubic"

			bool use_gridscaling = true;
			float hx = 1.0f;
			float hy = 1.0f;
			float hz = 1.0f;
		};
		///////////////////////////////////////////

		///////////////////////////////////////////
		struct GPU
		{
			int n_gpus = 1;
			int deviceID = 0;
			int threadsPerBlock = 128;

			bool reshape4coalescence = false;
		};
		///////////////////////////////////////////

		///////////////////////////////////////////
		GPU gpu;
		Preprocessing preprocessing;
		Postprocessing postprocessing;
		GaussianPyramid pyramid;
		Mosaic mosaicing;
		IterativeSolver solver;
		ScaleAdjustment scaling;
		Special special;
		Warping warp;
		Constraints constraint;
		SmoothnessTerm smoothness;
		DataConfidence confidence;
	};
}

#endif //PROTOCOL_PARAMETERS_H
