#ifndef OPTFLOW_CPU3D_H
#define OPTFLOW_CPU3D_H

/*********************************************************************************************************************************************************
 * Location: Helmholtz-Zentrum fuer Material und Kuestenforschung, Max-Planck-Straße 1, 21502 Geesthacht
 * Author: Stefan Bruns
 * Contact: bruns@nano.ku.dk
 *
 * License: TBA
 *********************************************************************************************************************************************************/

#include <iostream>
#include "optflow_base.h"

namespace optflow
{
	class OptFlow_CPU3D : public OptFlowSolver
	{
	public:
		virtual int configure_device(int maxshape[3], ProtocolParameters *params);
		virtual void free_device();

		virtual void run_outeriterations(int level, float *frame0, float *frame1, int shape[3], ProtocolParameters *params);

		virtual void set_flowvector(float* in_vector, int shape[3]);
		virtual void get_resultcopy(float* out_vector, int shape[3]);

	private:
		optflow_type *u, *du, *phi, *confidence;
		img_type *warped1;
	};
}

#endif //OPTFLOW_CPU3D_H
