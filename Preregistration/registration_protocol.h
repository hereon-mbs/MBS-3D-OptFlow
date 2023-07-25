#ifndef INITIAL_GUESS_H
#define INITIAL_GUESS_H

#include <iostream>
#include <string.h>
#include <cstdint>
#include <vector>
#include <math.h>
#include "registration_bruteforce.h"
#include "Geometry/hdcommunication.h"

namespace guess
{
	class RigidBodyRegistration
	{
	public:
		int interpolation_order = 2; //1 = linear, 2 = cubic

		int shape[3];
		bool use_mask = false;

		float result[6] = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};

		float step_scaling = 0.5f;
		float max_step_translation = 1.0;
		float min_step_translation = 0.1;

		float max_step_rotation = 0.1f;
		float min_step_rotation = 0.025f;

		//continue searching even with decreasing correlation for n_steps
		int n_linesearch_expansions = 3;
		int zoffset = 0;
		float gamma = 0.1f;

		RigidBodyRegistration(float* frame0, float* frame1, float* mask, int shape_[3], int deviceID, bool use_mask_)
		{
			shape[0] = shape_[0]; shape[1] = shape_[1]; shape[2] = shape_[2];
			use_mask = use_mask_;

			//prepare GPU
			/////////////////////////////////////////////////////////////////
			int errorcode = bfregister.configure_device(shape, deviceID, use_mask, interpolation_order);
			if (errorcode != 0) std::cout << "Configuration error!" << std::endl;

			if(!use_mask) bfregister.set_frames(frame0, frame1, shape);
			else bfregister.set_frames(frame0, frame1, mask, shape);
			/////////////////////////////////////////////////////////////////
		}
		void free_device()
		{
			bfregister.free_device();
		}

		void set_guess(float guess[6], int zoffset_)
		{
			zoffset = zoffset_;
			for (int i = 0; i < 6; i++) result[i] = guess[i];
			//result[2] -= zoffset_;
			return;
		}

		void run_translation_gradientascent()
		{
			std::vector<float> stepsizes = prepare_stepsizes_(min_step_translation, max_step_translation, step_scaling);

			float best_corr = bfregister.execute_correlation(result[0], result[1], result[2], shape);

			std::cout << "initial correlation: " << best_corr << std::endl;

			for (int step = 0; step < stepsizes.size(); step++)
			{
				float next_corr = 1.0f;
				float active_stepsize = stepsizes[step];
				float next_step[3];

				while (next_corr >= best_corr)
				{
					next_corr = bfregister.next_gradientascentstep_translation(result, active_stepsize, gamma, shape, next_step);

					if (next_corr > best_corr)
					{
						best_corr = next_corr;
						result[0] = result[0]+next_step[0];
						result[1] = result[1]+next_step[1];
						result[2] = result[2]+next_step[2];
					}

					std::cout << active_stepsize << ": " << best_corr << " (" << result[0] << " " << result[1] << " " << result[2]-zoffset << ")     \r";
					std::cout.flush();
				}
			}

			return;
		}
		float run_gradientascent(int dofflag[6], float rotcenter[3])
		{
			std::vector<float> stepsizes = prepare_stepsizes_(min_step_translation, max_step_translation, step_scaling);
			std::vector<float> rotstep_list = prepare_stepsizes_(min_step_rotation, max_step_rotation, step_scaling);

			float best_corr = bfregister.execute_correlation(result[3], result[4], result[5], result[0], result[1], result[2], shape, rotcenter, use_mask);

			std::cout << "initial correlation: " << best_corr << std::endl;

			for (int rotstep = 0; rotstep < rotstep_list.size(); rotstep++)
			{
				float active_rotstep = rotstep_list[rotstep];
				for (int step = 0; step < stepsizes.size(); step++)
				{
					float next_corr = 1.0f;
					float active_stepsize = stepsizes[step];
					float next_step[6];

					while (next_corr >= best_corr)
					{
						next_corr = bfregister.next_gradientascentstep(result, active_stepsize, active_rotstep, gamma, dofflag, shape, next_step, rotcenter, use_mask);

						if (next_corr > best_corr)
						{
							best_corr = next_corr;
							result[0] = result[0]+next_step[0];
							result[1] = result[1]+next_step[1];
							result[2] = result[2]+next_step[2];
							result[3] = result[3]+next_step[3];
							result[4] = result[4]+next_step[4];
							result[5] = result[5]+next_step[5];
						}
					}
					std::cout << active_rotstep << "/" << active_stepsize << ": " << best_corr << " (" << result[0] << " " << result[1] << " " << result[2]-zoffset << " " <<
							result[3] << " " << result[4] << " " << result[5] << ")    \r";
					std::cout.flush();
				}
			}
			std::cout << std::endl;
			return best_corr;
		}


		void run_translation_separateascent()
		{
			std::vector<float> stepsizes = prepare_stepsizes_(min_step_translation, max_step_translation, step_scaling);

			float best_corr = bfregister.execute_correlation(result[0], result[1], result[2], shape);

			std::cout << "initial correlation: " << best_corr << std::endl;

			for (int step = 0; step < stepsizes.size(); step++)
			{
				float active_stepsize = stepsizes[step];

				while (0 == 0)
				{
					float next_corr0 = bfregister.ascent_translation_singledimension(0, best_corr, active_stepsize, result, shape, n_linesearch_expansions);
					float next_corr1 = bfregister.ascent_translation_singledimension(1, next_corr0, active_stepsize, result, shape, n_linesearch_expansions);
					float next_corr2 = bfregister.ascent_translation_singledimension(2, next_corr1, active_stepsize, result, shape, n_linesearch_expansions);

					if (next_corr2 == best_corr) break;
					else best_corr = next_corr2;

					std::cout << active_stepsize << ": " << best_corr << " (" << result[0] << " " << result[1] << " " << result[2]+zoffset << ")"<< std::endl;
				}
			}

			return;
		}
		float run_separated_dofs(int dofflag[6], float rotcenter[3])
		{
			std::vector<float> stepsizes = prepare_stepsizes_(min_step_translation, max_step_translation, step_scaling);
			std::vector<float> rotstep_list = prepare_stepsizes_(min_step_rotation, max_step_rotation, step_scaling);

			float best_corr = bfregister.execute_correlation(result[3], result[4], result[5], result[0], result[1], result[2], shape, rotcenter, use_mask);

			std::cout << "initial correlation: " << best_corr << " (" << result[0] << " " << result[1] << " " << result[2]-zoffset << " " <<
					result[3] << " " << result[4] << " " << result[5] << ")" << std::endl;

			for (int rotstep = 0; rotstep < rotstep_list.size(); rotstep++)
			{
				float active_rotstep = rotstep_list[rotstep];

				for (int step = 0; step < stepsizes.size(); step++)
				{
					float active_stepsize = stepsizes[step];
					bool inner_change = true;

					while (inner_change)
					{
						inner_change = false;

						for (int dim = 0; dim < 6; dim++)
						{
							if (dofflag[dim] == 0) continue;

							float next_corr;

							if (dim < 3) next_corr = bfregister.ascent_singledimension(dim, best_corr, active_stepsize, result, shape, n_linesearch_expansions, rotcenter, use_mask);
							else next_corr = bfregister.ascent_singledimension(dim, best_corr, active_rotstep, result, shape, n_linesearch_expansions, rotcenter, use_mask);

							if (next_corr == best_corr) continue;
							else {best_corr = next_corr; inner_change = true;}
						}
					}

					std::cout << active_rotstep << "/" << active_stepsize << ": " << best_corr << " (" << result[0] << " " << result[1] << " " << result[2]-zoffset << " " <<
							result[3] << " " << result[4] << " " << result[5] << ")     \r";
					std::cout.flush();

				}
			}
			std::cout << std::endl;
			return best_corr;
		}

	private:
		registrate::BruteForce bfregister;

		std::vector<float> prepare_stepsizes_(float min_stepsize, float max_stepsize, float scaling = 0.5f)
		{
			float stepsize = round(max_stepsize*1.e6f)/1.e6f;
			std::vector<float> stepsize_list;

			while(stepsize > min_stepsize)
			{
				stepsize_list.push_back(stepsize);
				stepsize *= scaling;
				stepsize = round(stepsize*1.e6f)/1.e6f;
			}

			stepsize_list.push_back(min_stepsize);

			return stepsize_list;
		}
	};
}

#endif //INITIAL_GUESS_H
