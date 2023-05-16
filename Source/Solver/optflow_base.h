#ifndef OPTFLOW_BASE_H
#define OPTFLOW_BASE_H

/*********************************************************************************************************************************************************
 * Location: Helmholtz-Zentrum fuer Material und Kuestenforschung, Max-Planck-Strasse 1, 21502 Geesthacht
 * Author: Stefan Bruns
 * Contact: bruns@nano.ku.dk
 *
 * License: TBA
 *********************************************************************************************************************************************************/

#include <cuda.h>
#include <vector>
#include <fstream>
#include "../protocol_parameters.h"

namespace optflow
{
	typedef float optflow_type;
	typedef float img_type;
	typedef float mathtype_solver;
	typedef long long int idx_type;

	class OptFlowSolver
	{
	public:
		//Constructor and Destructor
		OptFlowSolver(){};
		virtual ~OptFlowSolver(){};

		virtual int configure_device(int maxshape[3], ProtocolParameters *params){std::cout << "Warning! No configure_device implemented with this solver!" << std::endl; return 1;}
		virtual void free_device(){}

		virtual void run_outeriterations(int level, float *frame0, float *frame1, int shape[3], ProtocolParameters *params, bool resumed_state = false, bool frames_set = false){std::cout << "Warning! No run_outeriterations implemented with this solver!" << std::endl;}
		virtual void run_singleiteration(int level, img_type *frame0, img_type *frame1, int shape[3], ProtocolParameters *params, bool frames_set = false){std::cout << "Warning! No run_singleiteration implemented with this solver!" << std::endl;}

		virtual void set_frames(float* frame0, float *frame1, int shape[3], std::vector<int> &boundaries, bool rewarp){std::cout << "Warning! No set_frames implemented with this solver!" << std::endl;}
		virtual void set_flowvector(float* in_vector, int shape[3]){std::cout << "Warning! No set_flowvector implemented with this solver!" << std::endl;}
		virtual void set_flowvector(float* in_vector, int shape[3], std::vector<int> &boundaries){std::cout << "Warning! No overloaded set_flowvector implemented with this solver!" << std::endl;}
		virtual void set_confidencemap(float* confidencemap, int shape[3]){std::cout << "Warning! No set_confidencemap implemented with this solver!" << std::endl;}
		virtual void set_confidencemap(optflow_type* confidencemap, int shape[3], std::vector<int> &boundaries){std::cout << "Warning! No overloaded set_confidencemap implemented with this solver!" << std::endl;}
		virtual void set_adaptivitymap(float* adaptivitymap, int shape[3]){std::cout << "Warning! No set_adaptivitymap implemented with this solver!" << std::endl;}
		virtual void set_adaptivitymap(optflow_type* adaptivitymap, int shape[3], std::vector<int> &boundaries){std::cout << "Warning! No overloaded set_adaptivitymap implemented with this solver!" << std::endl;}
		virtual void get_resultcopy(float* out_vector, int shape[3]){std::cout << "Warning! No get_resultcopy implemented with this solver!" << std::endl;}
		virtual void get_resultcopy(float* out_vector, int shape[3], std::vector<int> &boundaries){std::cout << "Warning! No overloaded get_resultcopy implemented with this solver!" << std::endl;}
		virtual void get_psimap(float* outimg, int shape[3]){std::cout << "Warning! No get_psimap implemented with this solver!" << std::endl;}

		//Precalculated kernels for more accurate derivatives according to
		//Farid and Simoncello 2004 "Differentiation of Discrete Multidimensional Signals"
		//deposited in farid_kernels.cpp
		static double faridkernel_3x3x3_dx[27];
		static double faridkernel_5x5x5_dx[125];
		static double faridkernel_7x7x7_dx[343];
		static double faridkernel_9x9x9_dx[729];
	};
}

#endif //OPTFLOW_SOLVER__BASE_H
