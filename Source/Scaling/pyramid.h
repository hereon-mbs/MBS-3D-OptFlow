#ifndef PYRAMID_H
#define PYRAMID_H

#include <iostream>
#include <vector>
#include <string.h>
#include "../protocol_parameters.h"
#include "resampling.h"

/*********************************************************************************************************************************************************
 * Location: Helmholtz-Zentrum fuer Material und Kuestenforschung, Max-Planck-Straße 1, 21502 Geesthacht
 * Author: Stefan Bruns
 * Contact: bruns@nano.ku.dk
 *
 * License: TBA
 *
 * There's basically two approaches:
 * 			- Ershov does not blur on resampling but scales the smoothness constrained with pyramid level. Downsampling is linear.
 * 			- Liu blurs on downsampling (for anti-aliasing) but does not scale the smoothness constrained. Above level n additional blur is applied but shape shrinkage is slowed.
 * We split this allowing for all combinations.
 *********************************************************************************************************************************************************/

namespace pyramid
{
	class ImagePyramid
	{
	public:
		float *active_frame;
		std::vector<int*> pyramid_shapes;

		ImagePyramid(optflow::ProtocolParameters *params, int shape[3], bool allocate_memory);
		~ImagePyramid()
		{
			for (int level = 1; level < pyramid_shapes.size(); level++)
				free(pyramid_shapes[level]);
			//free(active_frame);
		}

		void resample_frame(float *frame, int shape[3], int level, optflow::ProtocolParameters *params, std::string interpolation_mode);

	private:
		void buildBackupsLiu(float *frame, int shape[3], int n_Liu, float sigma, std::string interpolation_mode, int maxlevel);

		std::vector<float*> liu_frames;
	};
}

#endif //PYRAMID_H
