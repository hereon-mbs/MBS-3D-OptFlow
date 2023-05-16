#ifndef OPTFLOW_GPU3D_H
#define OPTFLOW_GPU3D_H

#include <iostream>
#include "optflow_base.h"

/*********************************************************************************************************************************************************
 * Location: Helmholtz-Zentrum fuer Material und Kuestenforschung, Max-Planck-Strasse 1, 21502 Geesthacht
 * Author: Stefan Bruns
 * Contact: bruns@nano.ku.dk
 *
 * License: TBA
 *********************************************************************************************************************************************************/

namespace optflow
{
	class OptFlow_GPU3D : public OptFlowSolver
	{
	public:
		virtual int configure_device(int maxshape[3], ProtocolParameters *params);
		virtual void free_device();

		virtual void run_outeriterations(int level, img_type *frame0, img_type *frame1, int shape[3], ProtocolParameters *params, bool resumed_state = false, bool frames_set = false);
		virtual void run_singleiteration(int level, img_type *frame0, img_type *frame1, int shape[3], ProtocolParameters *params, bool frames_set = false);

		virtual void set_frames(float* frame0, float *frame1, int shape[3], std::vector<int> &boundaries, bool rewarp);
		virtual void set_flowvector(float* in_vector, int shape[3]);
		virtual void set_flowvector(optflow_type* in_vector, int shape[3], std::vector<int> &boundaries);
		virtual void set_confidencemap(float* confidencemap, int shape[3]);
		virtual void set_confidencemap(optflow_type* confidencemap, int shape[3], std::vector<int> &boundaries);
		virtual void set_adaptivitymap(float* adaptivitymap, int shape[3]);
		virtual void set_adaptivitymap(optflow_type* adaptivitymap, int shape[3], std::vector<int> &boundaries);
		virtual void get_resultcopy(float* out_vector, int shape[3]);
		virtual void get_resultcopy(optflow_type* out_vector, int shape[3], std::vector<int> &boundaries);

	private:
		optflow_type *u, *du, *phi, *psi, *confidence;
		img_type *warped1, *dev_frame0, *dev_frame1;
		optflow_type *adaptivity;

		int deviceID = 0;
	};
}

#endif //OPTFLOW_GPU3D_H
