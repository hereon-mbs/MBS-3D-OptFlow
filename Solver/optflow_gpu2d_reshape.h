#ifndef OPTFLOW_GPU2D_RESHAPE_H
#define OPTFLOW_GPU2D_RESHAPE_H

#include <iostream>
#include "optflow_base.h"

/*********************************************************************************************************************************************************
 * Location: Helmholtz-Zentrum fuer Material und Kuestenforschung, Max-Planck-Straße 1, 21502 Geesthacht
 * Author: Stefan Bruns
 * Contact: bruns@nano.ku.dk
 *
 * License: TBA
 *
 * Changes compared to OPTFLOW_GPU2D_H:
 * 		- reestablished memory coalescence by reordering arrays in device memory
 *
 *********************************************************************************************************************************************************/

namespace optflow
{
	class OptFlow_GPU2D_Reshape : public OptFlowSolver
	{
	public:
		virtual int configure_device(int maxshape[3], ProtocolParameters *params);
		virtual void free_device();

		virtual void run_outeriterations(int level, img_type *frame0, img_type *frame1, int shape[3], ProtocolParameters *params, bool resumed_state = false, bool frames_set = false);
		virtual void run_singleiteration(int level, img_type *frame0, img_type *frame1, int shape[3], ProtocolParameters *params, bool frames_set = false);

		virtual void set_flowvector(float* in_vector, int shape[3]);
		virtual void set_confidencemap(float* confidencemap, int shape[3]);
		virtual void set_adaptivitymap(float* adaptivitymap, int shape[3]);
		virtual void get_resultcopy(float* out_vector, int shape[3]);

	private:
		void reshape_on_host(float *input, float *output, int shape[3]);

		optflow_type *u, *du, *phi, *psi, *confidence, *adaptivity;
		img_type *warped1, *dev_frame0, *dev_frame1;

		int deviceID = 0;
	};
}

#endif //OPTFLOW_GPU2D_H
