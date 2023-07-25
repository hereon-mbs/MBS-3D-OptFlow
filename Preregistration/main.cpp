#include <iostream>
#include <unistd.h>
#include <chrono>
#include <algorithm>
#include <omp.h>
#include <fstream>

#include "Geometry/hdcommunication.h"
#include "registration_protocol.h"
#include "Geometry/transformation_cpu.h"
#include "sample_transformations.h"
#include "histogram_stretching.h"

using namespace std;


/*********************************************************************************************************************************************************
 *
 * Initial rigid body registrations
 *
 * Location: Helmholtz-Zentrum fuer Material und Kuestenforschung, Max-Planck-Strasse 1, 21502 Geesthacht
 * Author: Stefan Bruns
 * Contact: bruns@nano.ku.dk
 *
 * License: TBA
 *********************************************************************************************************************************************************/


/*
 * Their seems to be a bug when only a rotation without subinteger translation is applied. Seen with roll and with pitch only.
 *
 */

int main(int argc, char* argv[])
{
	///////////////////////////////////////////////////////////////
	///////////////////////////////////////////////////////////////

	string inpath_frame0 = "none"; //reference image sequence
	string inpath_frame1 = "none"; //image sequence to be registered
	string maskpath = "none";
	string outpath = "none";
	string displacement_file = "none";

	bool optimization = false;
	bool stretch_histogram = false;
	bool export_samples = false;
	bool transform_only = false;
	bool apply_solution = false;

	pair<int,int> slice_range0 = {-1,-1}; //second is exclusive
	pair<int,int> circular_mask_radius = {0, 0}; //SynchroLoad2021:{610, 240}; //set positiv to mask out outer part of reconstruction;
	int zoffset_frame1 = 0; //read in a more appropriate range of images

	//in order: dx, dy, dz, jaw, roll, pitch
	float transformation_parameters[6] = {0.0f, 0.0f, 0.0f,
								          0.0f, 0.0f, 0.0f};

	int dofflag[6] = {1,1,1,0,0,0}; //which degree of freedom may be optimized

	int deviceID = 0;
	int max_threads = 8;

	int interpolation_order = 2; //1 = linear, 2 = cubic

	//search step size
	float step_scaling = 0.5f;
	float max_step_translation = 1.0;
	float min_step_translation = 0.1;
	float max_step_rotation = 0.1f;
	float min_step_rotation = 0.025f;

	//continue searching even with decreasing correlation for n_steps
	int n_linesearch_expansions = 3;

	//gamma for gradient ascent
	float gamma = 0.1f;
	///////////////////////////////////////////////////////////////
	///////////////////////////////////////////////////////////////

	///////////////////////////////////////////////////////////////
	std::string rootpath, sample_identifier;
	bool use_mask = false;
	float initial_guess_backup[6] = {0.0,0.0,0.0,0.0,0.0,0.0};

	if ("extract command line arguments"){
		for (uint16_t i = 1; i < argc; i++)
		{
			if ((string(argv[i]) == "-i0") || (string(argv[i]) == "-input0"))
			{
				i++; inpath_frame0 = string(argv[i]);
			}
			else if ((string(argv[i]) == "-i1") || (string(argv[i]) == "-input1"))
			{
				i++; inpath_frame1 = string(argv[i]);
			}
			else if (string(argv[i]) == "-batchapply")
			{
				i++; sample_identifier = string(argv[i]);
				i++; displacement_file = string(argv[i]);
				transform_only = true;
				apply_solution = true;
			}
			else if (string(argv[i]) == "-range")
			{
				i++; slice_range0.first = atoi(argv[i]);
				i++; slice_range0.second = atoi(argv[i]);
			}
			else if (string(argv[i]) == "-zoffset")
			{
				i++; zoffset_frame1 = atoi(argv[i]);
			}
			else if (string(argv[i]) == "-guess")
			{
				i++; transformation_parameters[0] = atof(argv[i]);
				i++; transformation_parameters[1] = atof(argv[i]);
				i++; transformation_parameters[2] = atof(argv[i]);
				i++; transformation_parameters[3] = atof(argv[i]);
				i++; transformation_parameters[4] = atof(argv[i]);
				i++; transformation_parameters[5] = atof(argv[i]);
			}
			else if (string(argv[i]) == "-gpu0")
			{
				i++; deviceID = atoi(argv[i]);
			}
			else if (string(argv[i]) == "-expansions")
			{
				i++; n_linesearch_expansions = atoi(argv[i]);
			}
			else if (string(argv[i]) == "--jaw")
				dofflag[3] = 1;
			else if (string(argv[i]) == "--roll")
				dofflag[4] = 1;
			else if (string(argv[i]) == "--pitch")
				dofflag[5] = 1;
			else if ((string(argv[i]) == "--histostretch"))
				stretch_histogram = true;
			else if (string(argv[i]) == "--optimize")
				optimization = true;
			else if (string(argv[i]) == "--samples")
				export_samples = true;
			else if (string(argv[i]) == "--transform_only")
			{
				transform_only = true;
				apply_solution = true;
			}
			else if (string(argv[i]) == "-circ_mask")
			{
				i++; circular_mask_radius.first = atof(argv[i]);
			}
		}

		if (inpath_frame0.substr(inpath_frame0.size()-1) != "/") inpath_frame0 += "/";
		if (inpath_frame1.substr(inpath_frame1.size()-1) != "/") inpath_frame1 += "/";

		rootpath = inpath_frame1.substr(0, inpath_frame1.rfind("/", inpath_frame1.length()-3)+1);
		if (outpath == "none" || outpath == "" || outpath == "None"){
			outpath = rootpath + "/registration/";}

		///// acquire precalculated displacements /////
		if (displacement_file != "none")
		{
			std::ifstream displacement_list(displacement_file);
			if (displacement_list.is_open())
			{
				std::string line;
				while(std::getline(displacement_list, line)){
					std::stringstream sstream(line);
					std::string this_id, segment;
					std::getline(sstream, this_id,',');
					if(this_id == sample_identifier){
						std::getline(sstream, segment,','); transformation_parameters[0] = stof(segment);
						std::getline(sstream, segment,','); transformation_parameters[1] = stof(segment);
						std::getline(sstream, segment,','); transformation_parameters[2] = stof(segment);
						std::getline(sstream, segment,','); transformation_parameters[3] = stof(segment);
						std::getline(sstream, segment,','); transformation_parameters[4] = stof(segment);
						std::getline(sstream, segment,','); transformation_parameters[5] = stof(segment);
						break;
					}

				}
				displacement_list.close();
			}
			else
			{
				cout << "Error! displacement_file not found!" << endl;
				return 1;
			}
		}
		///////////////////////////////////////////////

		for (int i = 0; i< 6; i++) initial_guess_backup[i] = transformation_parameters[i];

		if (max_threads > 0){
			max_threads = min(max_threads, omp_get_max_threads());
			omp_set_num_threads(max_threads);
		}
		else max_threads = omp_get_max_threads();
	}
	///////////////////////////////////////////////////////////////

	cout << endl;
	cout << "running registration for:" << endl;
	cout << "    " << inpath_frame0 << endl;
	cout << "    " << inpath_frame1 << endl;
	cout << "rootpath: " << rootpath << endl;
	cout << "outpath: " << outpath << endl;
	cout << "-------------------------------------------------" << endl;

	//Data acquisition
	///////////////////////////////////////////////////////////////
	hdcom::HdCommunication hdcom; int shape[3];

	vector<string> filelist0, filelist1;
	hdcom.GetFilelist(inpath_frame0, filelist0);
	hdcom.GetFilelist(inpath_frame1, filelist1);

	if (slice_range0.first < 0) slice_range0.first = 0;
	if (slice_range0.second < 0 || slice_range0.second > filelist0.size()) slice_range0.second = filelist0.size();

	cout << "acquiring image data..." << endl;
	cout << "-------------------------------------------------" << endl;
	float *frame0 = hdcom.Get3DTifSequence_32bitPointer(filelist0, shape, slice_range0);
	cout << "frame0: " << shape[0] << " " << shape[1] << " " << shape[2] << endl;
	pair<int,int> slice_range1 = {slice_range0.first+zoffset_frame1, slice_range0.second+zoffset_frame1};
	float *frame1 = hdcom.Get3DTifSequence_32bitPointer(filelist1, shape, slice_range1);
	cout << "frame1: " << shape[0] << " " << shape[1] << " " << shape[2] << endl;
	long long int nslice = shape[0]*shape[1];
	long long int nstack = shape[2]*nslice;
	float *mask;
	if (maskpath != "none"){
		use_mask = true;
		vector<string> filelist_mask; hdcom.GetFilelist(maskpath, filelist_mask);
		mask = hdcom.Get3DTifSequence_32bitPointer(filelist_mask, shape, slice_range0);

		if (circular_mask_radius.first > 0)
		{
			#pragma omp parallel for
			for (long long int idx = 0; idx < nstack; idx++)
			{
				int z = idx/nslice;
				int y = (idx-z*nslice)/shape[0];
				int x = (idx-z*nslice-y*shape[0]);

				float sqdist = (x-shape[0]/2.f)*(x-shape[0]/2.f)+(y-shape[1]/2.f)*(y-shape[1]/2.f);
				if (sqdist > circular_mask_radius.first*circular_mask_radius.first) mask[idx] = 0;
			}
		}
	}
	else if (circular_mask_radius.first > 0)
	{
		mask = (float*) calloc(nstack, sizeof(nstack));
		use_mask = true;

		#pragma omp parallel for
		for (long long int idx = 0; idx < nstack; idx++)
		{
			int z = idx/nslice;
			int y = (idx-z*nslice)/shape[0];
			int x = (idx-z*nslice-y*shape[0]);

			float sqdist = (x-shape[0]/2.f)*(x-shape[0]/2.f)+(y-shape[1]/2.f)*(y-shape[1]/2.f);
			if (sqdist <= circular_mask_radius.first*circular_mask_radius.first && sqdist > circular_mask_radius.second*circular_mask_radius.second) mask[idx] = 1;
		}

		//hdcom.SaveTifSequence_32bit(mask, shape, outpath+"/mask/", "mask", true);
	}

	cout << "-------------------------------------------------" << endl;

	float rotcenter[3] = {shape[0]/2.f-0.5f, shape[1]/2.f-0.5f, filelist0.size()/2.f-0.5f};
	cout << shape[2] << " of " << filelist0.size() << " slices to be matched" << endl;
	cout << "rotcenter: " << rotcenter[0] << "," << rotcenter[1] << "," << rotcenter[2] << endl;
	///////////////////////////////////////////////////////////////

	auto time0 = chrono::high_resolution_clock::now();
	float bestcorr = -10.0;

	//Execute the registration
	///////////////////////////////////////////////////////////////
	if(!transform_only)
	{
		guess::RigidBodyRegistration protocol(frame0, frame1, mask, shape, deviceID, use_mask);
		protocol.set_guess(transformation_parameters, zoffset_frame1);

		protocol.max_step_translation = max_step_translation;
		protocol.min_step_translation = min_step_translation;
		protocol.max_step_rotation = max_step_rotation;
		protocol.min_step_rotation = min_step_rotation;
		protocol.gamma = gamma;
		protocol.step_scaling = step_scaling;
		protocol.n_linesearch_expansions = n_linesearch_expansions;
		protocol.interpolation_order = interpolation_order;

		//Debugging without rotation
		//protocol.run_translation_separateascent();
		//protocol.run_translation_gradientascent();

		bestcorr = protocol.run_separated_dofs(dofflag, rotcenter);
		bestcorr = protocol.run_gradientascent(dofflag, rotcenter);

		if (optimization)
		{
			protocol.max_step_translation = 0.1;
			protocol.min_step_translation = 0.05;
			protocol.run_gradientascent(dofflag, rotcenter);
		}

		//get the result
		for (int i = 0; i < 6; i++)
			transformation_parameters[i] = protocol.result[i];

		protocol.free_device();
	}
	else
		cout << "applying guess: " << transformation_parameters[0] << " " << transformation_parameters[1] << " " << transformation_parameters[2] << " "
				<< transformation_parameters[3] << " " << transformation_parameters[4] << " " << transformation_parameters[5] << endl;
	///////////////////////////////////////////////////////////////

	auto time1 = chrono::high_resolution_clock::now();
	chrono::duration<double> elapsed_gpu = time1-time0;
	std::cout << "solving took " << elapsed_gpu.count() << " s" << std::endl;
	cout << "-------------------------------------------------" << endl;

	//apply the solution to the ROI
	///////////////////////////////////////////////////////////////
	std::vector<double> stretch_parameters;

	if (export_samples || stretch_histogram)
	{
		cout << "transforming substack..." << endl;
		float *tmp = transform::apply_transformation(frame1, shape, transformation_parameters, rotcenter, interpolation_order);
		free(frame1);
		swap(tmp, frame1);

		if (stretch_histogram)
		{
			std::cout << "stretching histogram of frame1..." << std::endl;
			stretch_parameters = int_transform::stretch_frame1(frame0, frame1, shape, outpath);
		}
		if (export_samples)
		{
			cout << "saving projections..." << endl;
			sampling::export_projections(frame0, frame1, shape, outpath + "/projections/");
		}
	}
	///////////////////////////////////////////////////////////////

	//subtract back zoffset
	transformation_parameters[2] -= zoffset_frame1;

	//apply the solution to the entire stack
	///////////////////////////////////////////////////////////////
	if (apply_solution || export_samples)
	{
		//register the full stack
		/////////////////////////////////////////////////////////////////
		free(frame1);

		cout << "reading in frame1..." << endl;
		frame1 = hdcom.GetTif_unknowndim_32bit(inpath_frame1, shape, false);

		cout << "registering the full stack..." << endl;
		float *tmp = transform::apply_transformation(frame1, shape, transformation_parameters, rotcenter, interpolation_order);
		free(frame1);
		swap(tmp, frame1);

		if (stretch_histogram)
			int_transform::apply_stretch_parameters(frame1, shape, stretch_parameters);

		if (export_samples)
		{
			cout << "reading in frame0..." << endl;
			free(frame0);
			frame0 = hdcom.GetTif_unknowndim_32bit(inpath_frame0, shape, false);

			sampling::export_central_reslice("xz", frame0, shape, outpath+"/projections/", "xz_reslice_frame0");
			sampling::export_central_reslice("xz", frame1, shape, outpath+"/projections/", "xz_reslice_frame1");
			sampling::export_central_reslice("yz", frame0, shape, outpath+"/projections/", "yz_reslice_frame0");
			sampling::export_central_reslice("yz", frame1, shape, outpath+"/projections/", "yz_reslice_frame1");
		}

		if (apply_solution)
			hdcom.SaveTifSequence_32bit(frame1, shape, rootpath+"/registration/", "registered", true);
		/////////////////////////////////////////////////////////////////

	}
	cout << "-------------------------------------------------" << endl;
	///////////////////////////////////////////////////////////////

	if ("append logfile"){
		hdcom.makepath(rootpath);
		time_t now = time(0);
		ofstream logfile;
		logfile.open(rootpath + "/logfile.txt", fstream::in | fstream::out | fstream::app);
		logfile << ctime(&now);
		logfile << "ran WBBRegistration:\n";
		logfile << "-------------------------------------------------------------------------\n";
		logfile << "    - frame0: " << inpath_frame0 << "\n";
		logfile << "    - frame1: " << inpath_frame1 << "\n";
		logfile << "    - mask: " << maskpath << "\n\n";
		logfile << "    - translation: " << transformation_parameters[0] << " " << transformation_parameters[1] << " " << transformation_parameters[2] << "\n";
		logfile << "    - rotation: " << transformation_parameters[3] << " " << transformation_parameters[4] << " " << transformation_parameters[5] << "\n";
		logfile << "    - correlation: " << bestcorr << "\n\n";
		logfile << "    - initial guess: " << initial_guess_backup[0] <<" "<< initial_guess_backup[1]<<" "<< initial_guess_backup[2]
		                              <<" "<< initial_guess_backup[3]<<" "<< initial_guess_backup[4]<<" "<< initial_guess_backup[5]<< "\n";
		logfile << "    - optimization: " << optimization << "\n";
		logfile << "    - dofs: " << dofflag[0]<< dofflag[1]<< dofflag[2]<< dofflag[3]<< dofflag[4]<< dofflag[5] << "\n";
		if (stretch_histogram)
		logfile << "    - intensity adjustment: histogram stretching by modal intensities\n";
		else
		logfile << "    - intensity adjustment: none\n\n";

		//dump arguments
		logfile << "arguments:\n";
		for (uint16_t i = 1; i < argc; i++)
			logfile << string(argv[i]) << " ";
		logfile << "\n";
		logfile << "-------------------------------------------------------------------------\n\n";
		logfile.close();
	}

	return 0;
}
